# Methods: estimating and penalizing tr(H) in a looped transformer

## 1. Objective

L(θ) is the mean cross-entropy over answer tokens, and H(θ) = ∇²_θ L(θ) is its Hessian with respect to all 2.37M parameters. The regularized objective is

  J(θ) = L(θ) + λ · tr(H(θ)).

In a looped model the weights are shared across iterations. H therefore collects curvature contributions from every iteration, including cross-iteration terms, through the chain rule over the K applications of the block.

## 2. Hutchinson's estimator (exact in expectation)

For a random vector v with E[v vᵀ] = I, E[vᵀHv] = tr(H) ([Hutchinson 1990](https://doi.org/10.1080/03610919008812866)).

- **Probe distribution.** Rademacher v (entries ±1) minimizes the estimator's variance among i.i.d. probes. The variance is 2(‖H‖_F² − Σ_i H_ii²).
- **Computing Hv without forming H.** Use the gradient's vector–Jacobian product: Hv = ∇_θ(∇_θL · v). This is the "vjp over grad" construction of the [PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html), implemented in `linalg.hvp_from_grads`.
- **Training penalty** (`regularizers.hutchinson_penalty`):
  ```
  g  = autograd.grad(L_sub, θ, create_graph=True)     # first order, graph kept
  Hv = autograd.grad(g · v,  θ, create_graph=True)    # second order, graph kept
  P  = Hv · v                                         # vᵀ H v
  (k λ P).backward()                                  # third order: ∇θ (vᵀ H v)
  ```
  E_v[∇_θ(vᵀH(θ)v)] = ∇_θ tr(H(θ)), so the update is an unbiased stochastic gradient of J.
- **Amortization (Addendum 1).** The penalty is added on every k-th step (k = 8) with weight kλ. Averaged over steps, the expected penalty gradient per step is still λ∇tr(H). Sophia uses the same idea to refresh its curvature estimate every k = 10 steps ([arXiv:2305.14342](https://arxiv.org/abs/2305.14342)).

## 3. NSO is a finite-difference estimator of the same objective (Stage 2 arm)

[NSO](https://arxiv.org/abs/2306.08553) uses the update direction d = ½[∇L(θ+U) + ∇L(θ−U)], with U ~ N(0, σ²I).

Taylor-expand ∇L around θ:

  ∇L(θ±U) = ∇L(θ) ± H U + ½ ∇³L[U, U] ± O(‖U‖³).

The odd terms cancel in the average:

  d = ∇L(θ) + ½ ∇³L[U, U] + O(‖U‖⁴) = ∇L(θ) + ½ ∇_θ(Uᵀ H(θ) U) + O(σ⁴).

Here ∇³L[U, U] = ∇_θ(UᵀH(θ)U), because the third-derivative tensor is symmetric. Since E[UUᵀ] = σ²I, taking the expectation gives

  E_U[d] = ∇L(θ) + (σ²/2) ∇ tr(H(θ)) + O(σ⁴).

So NSO is a first-order, finite-difference estimator of the gradient of the same J, with **λ = σ²/2**, and the Hutchinson penalty is its exact-autodiff counterpart. The difference between the two is only in the estimator: probe distribution (Gaussian vs Rademacher), the O(σ⁴) terms, and variance.

## 4. Third-order autodiff correctness and the LayerNorm finding

The penalty's gradient is a third derivative of L.

**How it was checked** (`tests/test_third_order.py`, `tests/test_regularizers.py`):
- Each op, and the full float64 model, is compared against central finite differences of vᵀHv itself.
- The step sizes are 1e-4 and 1e-5. Their estimates agree to about 1e-6 relative, so truncation error is negligible.

**Finding (torch 2.15.0):**
- `torch.nn.functional.layer_norm` returns a wrong third-order directional derivative. Relative error ranged from 0.16 to 1.3 across random draws.
- An all-ones probe hides the error; Rademacher probes expose it.
- GELU, math SDPA, embedding, the tied head, cross-entropy and `torch.where` are correct.
- Native LayerNorm's first and second derivatives are correct (HVP and trace tests).

**Fix:**
- `model.LayerNorm` uses the native kernel by default.
- It switches to the composite formula (x−μ)·rsqrt(σ²+ε)·w + b inside `third_order_safe()`, which wraps the penalty's sub-batch forward pass.

## 5. Curvature measurements (`curvature.py`)

Every measurement is of the full-batch loss on a fixed measurement batch. The batch is processed in chunks, with token-count weights w_c, so that L = Σ_c w_c L_c holds exactly. Hessians and gradients combine with the same weights.

| Metric | Estimator | Reference |
|---|---|---|
| tr(H), per-group block traces | Hutchinson. The same probes are reused across chunks, so each probe gives vᵀHv of the full batch. SE = sd/√n_probes. Block traces are v_gᵀ(Hv)_g. | Hutchinson 1990; per-block view: [Zhang et al. 2024](https://arxiv.org/abs/2402.16788) |
| Dominant eigenvalue | Power iteration on H, stopping on relative change < tol. Finds the eigenvalue of largest magnitude, as in PyHessian. | [Yao et al. 2020](https://arxiv.org/abs/1912.07145) |
| Fisher (= Gauss–Newton for softmax CE) trace | tr(G) = N·E‖∇L̂‖², where L̂ uses labels sampled from the model. The cross terms vanish because each score has zero mean. | Bartlett identity as used by [Liu et al. 2023](https://arxiv.org/abs/2210.14199); GNB in [Sophia](https://arxiv.org/abs/2305.14342) |
| Adaptive average sharpness | E_ε[L(θ + ε⊙\|θ\|)] − L(θ), with ε ~ N(0, ρ²I) | [Andriushchenko et al. 2023](https://arxiv.org/abs/2302.07011) |
| Weight norms | L2 norm per group | Scale-invariance caveat: [Dinh et al. 2017](https://arxiv.org/abs/1703.04933) |

Each estimator is tested against float64 finite-difference Hessians of a tiny model (`tests/test_curvature.py`).

## 6. Evaluation and stopping rules (`evaluate.py`)

For each test length n, the model runs K = 1…n+5 iterations in one pass, and every iteration's prediction is scored.

- **Oracle:** the K with the best accuracy for that length. This is the primary metric; it uses labels. Source: [Kuo et al. 2026](https://arxiv.org/abs/2606.29983).
- **Max-confidence:** the K with the highest mean log max-probability over answer positions. Label-free; a batch variant of [Fan et al. 2024](https://arxiv.org/abs/2409.15647), Eq. 2.
- **K = n.**

Exact match requires all n+1 sum bits and the EOS token to be correct.
