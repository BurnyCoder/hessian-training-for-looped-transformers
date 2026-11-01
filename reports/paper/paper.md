---
title: "Hessian-Trace Regularization for Looped Transformers: A Preregistered Single-Seed Pilot on Length Generalization"
author: "Libor \"Burny\" Burian"
date: "2026-11-01"
---

# Abstract

Looped (weight-tied, recurrent-depth) transformers can generalize to longer inputs by iterating a shared block more times, but this extrapolation varies strongly across random seeds. Flat-minima arguments suggest that lowering loss curvature might make extrapolating solutions more likely.

We preregistered a test of whether adding an exact Hutchinson estimate of the Hessian trace, λ·tr(H), to the training loss of a looped transformer changes length generalization on binary addition. Models were trained on operands of 1–19 bits and tested on 20–60 bits. This paper reports Stage 1: one penalized and one unpenalized model, with the same seed, initialization and data stream.

<!-- BEGIN:headline -->
Stage 1 is in progress (finished arms: none).
<!-- END:headline -->

We also found that PyTorch 2.15.0's native LayerNorm returns incorrect third derivatives. Any explicit Hessian-trace penalty through LayerNorm therefore needs a composite implementation. With one seed per arm, all conclusions are descriptive. The confirmatory, multi-seed design is preregistered and deferred.

# 1. Introduction and motivation

**Looped transformers and length generalization.**
- A looped transformer applies one block K times, with the input re-injected at each iteration ([Dehghani et al. 2018](https://arxiv.org/abs/1807.03819); [Yang et al. 2024](https://arxiv.org/abs/2311.12424); [Fan et al. 2024](https://arxiv.org/abs/2409.15647)).
- Because depth can grow at test time, such models can solve arithmetic and algorithmic problems longer than any seen in training (Fan et al. 2024).
- [Kuo et al. 2026](https://arxiv.org/abs/2606.29983) show that this extrapolation is brittle. On binary addition, out-of-distribution (OOD) accuracy has an across-seed standard deviation of about 6–10 points, even among runs with near-identical in-distribution accuracy.

**The flat-minima hypothesis and its critics.**
- The hypothesis: among minimizers of the training loss, those in flat regions generalize better ([Hochreiter & Schmidhuber 1997](https://doi.org/10.1162/neco.1997.9.1.1); [Keskar et al. 2017](https://arxiv.org/abs/1609.04836)).
- Sharpness measures are among the better predictors of generalization in large studies ([Jiang et al. 2020](https://arxiv.org/abs/1912.02178)).
- In language models with equal pretraining loss, a lower Hessian trace accompanies better downstream accuracy ([Liu et al. 2023](https://arxiv.org/abs/2210.14199)).
- Counter-evidence and caveats:
  - Sharpness is not invariant to reparametrization ([Dinh et al. 2017](https://arxiv.org/abs/1703.04933)).
  - It often fails to track OOD performance in modern transformers ([Andriushchenko et al. 2023](https://arxiv.org/abs/2302.07011)).
  - The maximum eigenvalue can change without any change in generalization ([Kaur et al. 2023](https://arxiv.org/abs/2206.10654)).
  - For looped models in particular, [Gong et al. 2025](https://arxiv.org/abs/2510.10089) argue that looping helps *because* it biases training toward steep "V-shaped" valleys.

**Question.** Does explicitly penalizing tr(H) during training change the length generalization of a looped transformer?

Our literature sweeps found no prior application of SAM, NSO or explicit Hessian-trace penalties to looped transformers.

# 2. Background: estimating and penalizing tr(H)

**Hutchinson's estimator.**
- tr(H) = E_v[vᵀHv] for any v with E[vvᵀ] = I. With Rademacher v this is [Hutchinson's estimator](https://doi.org/10.1080/03610919008812866).
- Hv is computed without forming H, by double backpropagation, using the vector-Jacobian product of the gradient ([PyTorch tutorial](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html)).
- Using λ·vᵀHv as a training penalty requires its gradient ∇_θ(vᵀH(θ)v), which is a third-order derivative of the loss.
- This penalty was used to regularize CNNs and LSTMs by [Liu, Yu & Lin 2022 (SEHT)](https://arxiv.org/abs/2208.05924).

**Noise Stability Optimization (NSO)** ([Zhang, Li & Ju 2024](https://arxiv.org/abs/2306.08553)).
- Update direction: ½[∇L(θ+U) + ∇L(θ−U)], with U ~ N(0, σ²I).
- Its expectation is ∇L + (σ²/2)∇tr(H) + O(σ⁴). NSO is therefore a first-order, finite-difference estimator of the same objective, with λ = σ²/2.
- It is the tr(H) regularizer with transformer and language-model results (GPT-2 chain-of-thought fine-tuning; CLIP).
- In LLM practice, flatness is otherwise pursued with SAM ([Foret et al. 2021](https://arxiv.org/abs/2010.01412); [Bahri et al. 2022](https://arxiv.org/abs/2110.08529)) or with weight-noise and variational methods ([Flat-LoRA](https://arxiv.org/abs/2409.14396); [IVON](https://arxiv.org/abs/2402.17641)).
- Stage 1 uses the exact Hutchinson penalty. NSO and SAM are Stage 2 arms.

# 3. Method

## 3.1 Task and data

**Format.** Binary addition in the full-output-prediction format of Fan et al. 2024:
- Input: `a + b >` followed by n+2 placeholders.
- Target: the (n+1)-bit sum, most significant bit first, then EOS.
- Loss is applied only on answer positions.

**Training data.**
- Operand lengths n ∈ [1, 19].
- A linear curriculum raises the maximum length from 1 to 19 by the middle of training.

**Test data.**
- In-distribution (ID): n = 1–19. Out-of-distribution (OOD): n ∈ {20, 25, …, 60}.
- 1000 problems per length, each length from its own seed.

## 3.2 Model

A weight-tied block of 3 pre-LayerNorm GPT-2 layers:
- d = 256, 8 heads, 4× MLP.
- No positional embeddings; causal mask; tied embeddings.
- Input injection: h_t = Block(h_{t−1} + e).
- 2.37M parameters.

**Loop counts.**
- Training: per-sample K_i ~ U{n_i−5, …, n_i+5}, clamped to [1, 20] (Kuo et al. 2026).
- Evaluation: every K = 1, …, n+5 is scored.

## 3.3 Training

**Shared settings.**
- AdamW (lr 1e-4, betas 0.9/0.999, weight decay 0.01), cosine decay to 0.
- Batch 64, gradient clipping at 1.0, 30,000 steps.
- fp32 with TF32 matmuls.
- Both arms use seed 0. The data and loop-count stream comes from one generator, so the two models see identical batches from identical initial weights.

**The penalized arm** adds λ·vᵀH_sub v:
- one Rademacher probe per application;
- H_sub is the Hessian of the loss on the first 4 problems of the batch;
- applied every 8th step with weight 8λ, so the expected per-step penalty gradient is λ∇tr(H);
- λ = 5×10⁻⁵ = σ²/2 at NSO's central σ = 0.01. It was fixed in advance, not tuned.

The sub-batch size and the 8-step amortization were forced by the 8 GB GPU. They were recorded in a preregistration addendum before training (Appendix A).

## 3.4 Third-order autodiff correctness

The penalty's gradient was checked against central finite differences of vᵀHv on a float64 copy of the model, with differences that agree across step sizes.

**Finding: PyTorch 2.15.0's native `layer_norm` gives incorrect third derivatives.**
- All other operations in the model agree with finite differences: GELU, math scaled-dot-product attention, embedding, tied head, cross-entropy, and the per-sample read-out.
- An all-ones probe hides the error.
- We therefore build the penalty's graph with a composite LayerNorm (mean, variance, rsqrt). Elsewhere we keep the native kernel, whose first and second derivatives are correct.

## 3.5 Measurements

**Primary metric:** mean OOD exact match at the per-length best loop count ("oracle", Kuo et al.'s metric).

**Secondary metrics:**
- Label-free maximum-confidence stopping (Fan et al.).
- K = n.
- ID accuracy.
- Curvature on a fixed batch of training-distribution problems at K = n:
  - Hutchinson tr(H) with standard error, overall and per parameter group;
  - dominant eigenvalue by power iteration;
  - Fisher trace via sampled labels (Liu et al. 2023);
  - average-case adaptive sharpness (Andriushchenko et al. 2023);
  - weight norms.
- A loop-count study of tr(H) at fixed K.

**Baselines that need no training:** chance, and the untrained initialization.

## 3.6 Preregistered reading of a single-seed difference

With one seed per arm, seed variance cannot be estimated. Using Kuo et al.'s reported SD of about 5.9 points, a difference between two single runs has an SD of about 8.3 points.

- A difference Δ in the primary metric is called "suggestive" only if |Δ| > 2·√2·5.9 ≈ 16.7 points; otherwise it is "inconclusive at n = 1 seed".
- Wilson intervals and a paired McNemar test cover test-set sampling only.
- **Manipulation check:** the penalized model's final tr(H) must be lower than the baseline's, with ±3 SE intervals that do not overlap.

# 4. Results

<!-- BEGIN:results_table -->
Pending.
<!-- END:results_table -->

![Exact-match accuracy vs. operand length](../figures/accuracy_vs_length.png)

![Accuracy vs. loop count](../figures/loop_heatmaps.png)

![Training dynamics](../figures/training_curves.png)

![Per-group Hessian traces](../figures/curvature_groups.png)

![Curvature vs. loop count](../figures/loop_curvature.png)

<!-- BEGIN:examples_table -->
Pending.
<!-- END:examples_table -->

# 5. Discussion

<!-- discussion written after results -->

# 6. Limitations

- **One seed per arm.** Looped models' OOD accuracy varies by 6–10 points across seeds (Kuo et al. 2026), so Stage 1 cannot separate an effect of the penalty from seed luck.
- **λ was not tuned.** It was fixed in advance from NSO's σ grid, which was tuned for fine-tuning ResNets, not for a looped transformer trained from scratch.
- **The penalty is a high-variance estimator, further altered by the compute constraints.**
  - It uses one probe and 4 problems, applied every 8th step with an 8× weight.
  - On penalty steps, gradient clipping at 1.0 lets the penalty direction dominate the update.
  - The effect of curvature regularization is therefore confounded with injected gradient noise. Stage 2's random-direction control is designed to separate the two.
- **30k training steps,** against 100k in Fan et al. and Kuo et al.
- **The primary metric uses labels to pick the loop count per length.** Label-free metrics are reported alongside.
- **A single task** (binary addition).
- **Short ID lengths overlap with training problems,** which is unavoidable for small n.

# 7. Related work

- **Looped and recurrent-depth transformers:**
  - Universal Transformers ([Dehghani et al. 2018](https://arxiv.org/abs/1807.03819));
  - looped transformers as programmable computers ([Giannou et al. 2023](https://arxiv.org/abs/2301.13196));
  - looped transformers for in-context learning ([Yang et al. 2024](https://arxiv.org/abs/2311.12424));
  - looped transformers for length generalization ([Fan et al. 2024](https://arxiv.org/abs/2409.15647));
  - latent-thought reasoning ([Saunshi et al. 2025](https://arxiv.org/abs/2502.17416));
  - recurrent-depth LLMs ([Geiping et al. 2025](https://arxiv.org/abs/2502.05171); [Ouro, 2025](https://arxiv.org/abs/2510.25741));
  - stochastic stopping for stable extrapolation ([Kuo et al. 2026](https://arxiv.org/abs/2606.29983)).
- **Stability regularizers for looped and equilibrium models:**
  - Jacobian regularization of deep equilibrium models ([Bai et al. 2021](https://arxiv.org/abs/2106.14342));
  - path independence ([Anil et al. 2022](https://arxiv.org/abs/2211.09961));
  - spectral-radius regularization ([STARS, 2026](https://arxiv.org/abs/2605.26733)).
  - These target the state-space Jacobian, not parameter-space curvature.
- **Curvature regularization:**
  - SAM ([Foret et al. 2021](https://arxiv.org/abs/2010.01412)) and its link to trace minimization ([Wen et al. 2023](https://arxiv.org/abs/2211.05729));
  - gradient-norm penalties ([Zhao et al. 2022](https://arxiv.org/abs/2202.03599));
  - anticorrelated noise ([Orvieto et al. 2022](https://arxiv.org/abs/2202.02831));
  - SEHT ([Liu et al. 2022](https://arxiv.org/abs/2208.05924));
  - NSO ([Zhang et al. 2024](https://arxiv.org/abs/2306.08553));
  - Hessian-based optimizers for LLMs ([Sophia](https://arxiv.org/abs/2305.14342)).

# 8. Conclusion

<!-- conclusion written after results -->

# Appendix A: Preregistration and deviations

The preregistration ([reports/preregistration.md](../preregistration.md)) was committed before any code or results existed. Addendum 1 (before training) records:
- the benchmark-driven deviations (penalty sub-batch 16 → 4; penalty applied every 8 steps with an 8× weight);
- non-deterministic kernels;
- sequential runs;
- the composite LayerNorm in penalty graphs;
- the step budget S = 30,000.

# Appendix B: Per-length results

<!-- BEGIN:per_length_table -->
Pending.
<!-- END:per_length_table -->

# Appendix C: Curvature details

<!-- BEGIN:curvature_table -->
Curvature measurements pending.
<!-- END:curvature_table -->

<!-- BEGIN:loop_table -->
Loop-count study pending.
<!-- END:loop_table -->

# AI statement

This codebase and paper were written by Claude Code (Anthropic's Claude, model Opus 5.5) from an exact specification provided by the human author, who directed the research question, scope and constraints. Human cleanup and rewriting are in progress.
