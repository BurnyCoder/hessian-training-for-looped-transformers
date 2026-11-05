# Experiment report: Stage 1, one Hessian-penalized vs one plain looped transformer

**Status:** descriptive single-seed comparison, preregistered in [preregistration.md](preregistration.md) (Sections 4 and 6, Addenda 1–4).

All numbers on this page are injected by `uv run python main.py report` from `results/stage1_*`. None are typed by hand.

## Headline
<!-- BEGIN:headline -->
Stage 1 is in progress (finished arms: none).
<!-- END:headline -->

## What was tested and why

**Question.** Does adding λ·tr(H) to the loss of a looped transformer change how well it extrapolates to longer additions?

Here tr(H) is the trace of the Hessian of the training loss with respect to the weights: the sum of the loss curvatures along every weight direction.

**Motivation.** The flat-minima literature suggests low curvature tends to come with better generalization. The links are collected in the [preregistration](preregistration.md), Section 1. This is the cheapest test of whether the idea transfers to looped models. It is not a test of whether any effect is robust; that is Stage 2.

**Scope.** The owner fixed the scope on 2026-11-01: train exactly one penalized model and one plain model today. Seeds, tuning, ablations and the other trained baselines (NSO, SAM, controls) are deferred.

## Methodology

| Component | Choice | Why / source |
|---|---|---|
| Task | Binary addition `a+b>`; the answer is the (n+1)-bit sum, most significant bit first, then EOS. Full-output prediction, loss on the n+2 answer positions only. | Variant of the format of [Fan et al. 2024](https://arxiv.org/abs/2409.15647), Sec. 2.2 and 6.1.1: their code starts the sum at the `>` position and supervises every trailing EOS position. Binary addition is a standard length-generalization testbed for looped transformers, used by Fan et al. and [Kuo et al. 2026](https://arxiv.org/abs/2606.29983). |
| Train / test lengths | Train n ∈ [1,19] with a linear curriculum reaching 19 at the midpoint. Test ID n = 1–19; OOD n ∈ {20, 25, …, 60}; 1000 problems per length. | Fan et al. train on [1,20); Kuo et al. use these OOD lengths. Test sets use their own seeds (10000+n). OOD problems cannot appear in training because their lengths exceed 19; short ID lengths do overlap with training problems (preregistration Sec. 3). |
| Model | One weight-tied block of 3 pre-LayerNorm GPT-2 layers. d=256, 8 heads, MLP 4×, no positional embeddings, causal mask, tied embeddings. Input injection h_t = Block(h_{t−1} + e). | Fan et al. Sec. 4.2.1 (NoPE, input injection) and App. F, Table 2 (d = 256; 3 layers / 8 heads for addition). Layer code follows nanoGPT (MIT). |
| Loop counts | Training: K_i ~ U{n_i−5 … n_i+5}, clamped to [1,20], with each sample read out at its own K_i. Evaluation: every K = 1 … n+5. | Kuo et al.'s randomized-loop schedule, which reported lower across-seed OOD variance than K = n. |
| Optimizer | AdamW: lr 1e-4, betas (0.9, 0.999), weight decay 0.01. Cosine decay to 0, no warmup. Batch 64. Clip 1.0. 30,000 steps. | lr and batch follow Fan et al. and Kuo et al.; the cosine decay over the whole run follows Kuo et al. (Fan et al. start it only after the curriculum ends). The step count was set by the preregistered throughput rule (Addendum 1); both papers used 100k steps. |
| Penalty arm | L + λ·vᵀH_sub v, with one Rademacher probe v. H_sub is the Hessian of the loss on the first 4 problems of the batch. Applied every 8th step with weight 8λ; λ = 5×10⁻⁵. | Hutchinson's trace estimator (E[vᵀHv] = tr H), as in [SEHT](https://arxiv.org/abs/2208.05924). λ = σ²/2 at σ = 0.01, the center of NSO's published σ grid {0.008, 0.01, 0.012} ([Zhang et al. 2024](https://arxiv.org/abs/2306.08553), Sec. 3.1, used for fine-tuning ResNet-34); not tuned. The sub-batch and the 8-step amortization were forced by the 8 GB GPU (Addendum 1). |
| Controls | Same seed: identical initialization, batches and loop counts. Separate RNG for the probes. Untrained-model and chance baselines. | Pairs the two runs so that the penalty is the only difference. |
| Primary metric | Mean OOD exact match at the per-length best ("oracle") loop count. | Adapted from Kuo et al.'s oracle, which picks the best K per example from {1, …, n+2}; ours picks one K per length from {1, …, n+5}, so their published numbers are not directly comparable (Addendum 3). |
| Secondary metrics | Label-free maximum-confidence stopping (Fan et al. Eq. 2, batch variant); K = n; ID accuracy; curvature measures (below). | Preregistration Sec. 3. |
| Curvature | Measured on a fixed 128-problem batch at K = n (512 in the preregistration, reduced for time in Addendum 2): Hutchinson tr(H) (100 probes, standard error, per parameter group); dominant eigenvalue (power iteration, up to 50 iterations); Fisher trace (sampled labels); adaptive average sharpness at ρ ∈ {0.01, 0.02, 0.05}; weight norms. Loop-count study at K ∈ {1, 2, 4, 8, 12, 16, 20} with 10 probes and up to 10 power iterations. | Manipulation check and the Dinh et al. caveat about scale reparametrization ([1703.04933](https://arxiv.org/abs/1703.04933)). [Andriushchenko et al. 2023](https://arxiv.org/abs/2302.07011) for adaptive sharpness. [Gong et al. 2025](https://arxiv.org/abs/2510.10089) for looping and steep valleys. |

## Results

### Main table
<!-- BEGIN:results_table -->
Pending.
<!-- END:results_table -->

### Accuracy by length and loop count

![accuracy vs length](figures/accuracy_vs_length.png)

![loop heatmaps](figures/loop_heatmaps.png)

<!-- BEGIN:per_length_table -->
Pending.
<!-- END:per_length_table -->

### Training dynamics

![training curves](figures/training_curves.png)

### Curvature

![per-group traces](figures/curvature_groups.png)

<!-- BEGIN:curvature_table -->
Curvature measurements pending.
<!-- END:curvature_table -->

### Loop-count curvature study

![loop curvature](figures/loop_curvature.png)

<!-- BEGIN:loop_table -->
Loop-count study pending.
<!-- END:loop_table -->

### Output examples (OOD lengths, oracle loop count)
<!-- BEGIN:examples_table -->
Pending.
<!-- END:examples_table -->

## Interpretation
<!-- interpretation is written after results exist; see the section below -->

## Learnings (what was tried, what broke, how it was fixed)

**1. PyTorch's native LayerNorm has an incorrect third derivative.**
- What happened: the first full-model check of ∇(vᵀHv) against finite differences was off by more than 10%.
- How it was found: testing each operation separately, in float64, with finite differences whose estimates at different step sizes agree far more closely than the size of the error.
  - `F.layer_norm` in torch 2.15.0 is the only operation whose third derivative disagrees.
  - GELU, math scaled-dot-product attention (SDPA), embedding, tied head, cross-entropy and the `torch.where` read-out all agree.
  - An earlier `gradcheck` passed because its all-ones probe, pushed through a weight matrix into a LayerNorm without affine parameters, only shifted LayerNorm's input by a constant. LayerNorm is invariant to that shift, so vᵀHv was identically zero. Rademacher probes expose the error.
- Fix: a composite-op LayerNorm with the same math, used only while building the penalty's graph. The native kernel is used everywhere else, because its first and second derivatives are correct and it is faster and leaner.
- Status: kept as an expected-failure (`xfail`) regression test in `tests/test_third_order.py`.
- Without this check, the penalty would have been silently trained with a wrong gradient.

**2. The third-order graph is the cost bottleneck.**
- At the preregistered sub-batch of 16, the penalty needed 10.05 GB. Windows silently spilled it into system memory, giving 15.6 s/step.
- At a sub-batch of 4, every-step application costs 2.15 s/step, roughly 11× the baseline.
- Amortizing to every 8th step with weight 8λ keeps the expected gradient unchanged and costs 0.446 s/step. The idea follows Sophia's practice of updating the Hessian estimate every k steps ([arXiv:2305.14342](https://arxiv.org/abs/2305.14342)).
- Concurrent runs gave no throughput gain, so the two arms ran one after the other.
- All of this is recorded in Addendum 1 before training, with the benchmark JSON in `results/benchmark/`.

**3. A single-probe penalty gradient is very noisy.**
- In the smoke test, steps with the penalty had total gradient norms of 10²–10³. Steps without it had about 1–40.
- With clipping at 1.0, those steps follow mostly the penalty direction.
- This was recorded before training and left as preregistered. Stage 2's random-direction control is designed to separate curvature-specific effects from injected gradient noise.

**4. Deterministic kernels were dropped.**
- They made the baseline about 25% slower per step.
- Runs are seeded, with identical data and initialization across arms, but not bitwise reproducible.

## Reproduce

```bash
uv run python main.py run --arm baseline --name stage1_baseline
```

```bash
uv run python main.py run --arm hutchinson --name stage1_hutchinson
```

```bash
uv run python main.py report
```
