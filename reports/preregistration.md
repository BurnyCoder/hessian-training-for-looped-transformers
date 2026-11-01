# Preregistration: Hessian-trace regularization for looped transformers

- **Author:** Libor "Burny" Burian
- **Registered:** 2026-11-01, before any code, data or results existed (verify with the git timestamp of this file's first commit).
- **Status:** Stage 1 is scheduled now. Stage 2 is designed here and deferred to a later date.
- **Amendments:** appended at the bottom as timestamped addenda. Nothing above the addenda is edited after the first commit.

## 1. Domain and question

**Looped transformers** apply one weight-tied block K times. Input injection re-adds the input embedding at every iteration. Examples:
- Universal Transformers, [Dehghani et al. 2018](https://arxiv.org/abs/1807.03819).
- [Yang et al. 2024](https://arxiv.org/abs/2311.12424).
- [Fan et al. 2024/ICLR'25](https://arxiv.org/abs/2409.15647).

Looped models can extrapolate to longer inputs by running more iterations. That extrapolation is brittle: [Kuo et al. 2026 (NeurIPS'26)](https://arxiv.org/abs/2606.29983) report an across-seed OOD standard deviation of about 6–10 accuracy points on binary addition, against under 1 point for non-looped models.

The **flat-minima hypothesis** says that minima with low curvature generalize better. Supporting evidence:
- [Hochreiter & Schmidhuber 1997](https://doi.org/10.1162/neco.1997.9.1.1).
- [Keskar et al. 2017](https://arxiv.org/abs/1609.04836).
- [Jiang et al. 2020](https://arxiv.org/abs/1912.02178): sharpness measures were among the best generalization predictors.
- [Liu et al. 2023](https://arxiv.org/abs/2210.14199): among language models with the same pretraining loss, lower tr(H) predicts better downstream accuracy.

The hypothesis is contested:
- [Dinh et al. 2017](https://arxiv.org/abs/1703.04933): reparametrization can make any minimum arbitrarily sharp.
- [Andriushchenko et al. 2023](https://arxiv.org/abs/2302.07011): sharpness often fails to predict OOD performance in transformers.
- [Kaur et al. 2023](https://arxiv.org/abs/2206.10654).

For looped models specifically, [Gong et al. 2025](https://arxiv.org/abs/2510.10089) argue that looping biases optimization toward *steep* "V-shaped" valleys, and that this helps.

**Question:** does adding a Hessian-trace penalty λ·tr(H) to the training loss of a looped transformer improve out-of-distribution (longer-input) generalization?

**Novelty check (2026-11-01).** Three literature sweeps found no prior work applying SAM, NSO or Hessian-trace penalties to looped, weight-tied or recurrent-depth transformers.

## 2. Methods under test

Both estimators target the same objective, **L(θ) + λ·tr(∇²L(θ))**.

**Exact Hutchinson penalty (`hutchinson`).**
- Uses [Hutchinson's estimator](https://doi.org/10.1080/03610919008812866) (Commun. Stat. Simul. Comput. 19(2), 1990): tr(H) = E_v[vᵀHv] with Rademacher v.
- Hv is computed by double backprop: `torch.autograd.grad(create_graph=True)`, as in the [PyTorch Jacobians/Hessians tutorial](https://docs.pytorch.org/tutorials/intermediate/jacobians_hessians.html).
- The penalty's own gradient comes from a further backward pass, so third-order autodiff.
- As a training regularizer this was used by [Liu, Yu & Lin 2022 (SEHT)](https://arxiv.org/abs/2208.05924) on CNNs and LSTMs.

**Noise Stability Optimization (`nso`).**
- Source: [Zhang, Li & Ju, TMLR 2024](https://arxiv.org/abs/2306.08553). Code: [VirtuosoResearch/Noise-stability-optimization](https://github.com/VirtuosoResearch/Noise-stability-optimization), MIT.
- Update direction: ½[∇L(θ+U) + ∇L(θ−U)], with U ~ N(0, σ²I) and k = 1.
- Taylor expansion gives E_U of that direction = ∇L + (σ²/2)·∇tr(H) + O(σ⁴). NSO is therefore a finite-difference estimator of the gradient of the same objective with **λ = σ²/2**.
- It is the only tr(H) regularizer we found with transformer/LM results (GPT-2 chain-of-thought fine-tuning; a CLIP text encoder).

**Why these count as "state of the art."** We found no explicit Hessian penalty used in LLM pretraining at scale. In LLM work, flatness is pursued with SAM ([Bahri et al. 2022](https://arxiv.org/abs/2110.08529)) and with weight-perturbation or variational methods ([Flat-LoRA](https://arxiv.org/abs/2409.14396); [IVON](https://arxiv.org/abs/2402.17641)). Of these, Hutchinson is the exact explicit tr(H) penalty, and NSO is the best-supported approximate one.

## 3. Testbed (shared by all stages)

The task is binary addition with length generalization, following Fan et al. and Kuo et al. Fan's repository has no license, so the code is reimplemented from the papers.

**Data**
- Format: `a + b >` followed by n+2 placeholder tokens. Fan's full-output prediction: there is no next-token shift, and loss is applied only at the answer positions.
- Target: the n+1-bit sum, most significant bit first (not reversed, as in Fan), then EOS.
- Training lengths: n ∈ [1, 19]. A linear curriculum starts at n_max = 1 and reaches 19 at the middle of training; n is drawn uniformly from [1, n_max].
- Test sets are drawn with separate RNG seeds:
  - in-distribution (ID): n = 1…19;
  - OOD: n ∈ {20, 25, …, 60};
  - 1000 problems per length.
- OOD lengths never occur in training. Short ID lengths have tiny problem spaces (n=1 has 4 problems), so ID test problems overlap training problems there. This is unavoidable and is stated as a limitation.

**Model: `LoopedGPT`**
- One block of 3 pre-LayerNorm GPT-2 layers (nanoGPT style, MIT): d_model 256, 8 heads, MLP 4×, dropout 0.
- No positional embeddings (NoPE), causal mask, tied input/output embeddings.
- Input injection: h_t = Block(h_{t−1} + e), with h_0 = 0.

**Loop counts**
- Training uses Kuo et al.'s randomized loops: for each sample, K_i ~ U{n_i−5, …, n_i+5}, clamped to [1, 20]. Each sample's output is read at its own K_i.
- Evaluation records the prediction after every iteration K = 1…n+5.

**Optimization**
- AdamW (lr 1e-4, betas (0.9, 0.999), weight decay 0.01), cosine decay to 0, no warmup, batch 64, gradient-norm clip 1.0.
- fp32 math with TF32 matmuls. The math SDPA backend is used for every arm, so numerics are identical; fused SDPA kernels do not support double backward ([FlashBoB 2026](https://arxiv.org/abs/2609.24089)).

**Metrics**
- **Primary:** mean over the 9 OOD lengths of exact-match accuracy at the best loop count per length, chosen by oracle (Kuo et al.'s metric).
- **Key secondary:**
  - label-free maximum-confidence stopping: per length, the K with the highest mean log max-probability over answer positions (in the spirit of Fan et al. Eq. 2);
  - fixed K = n;
  - ID exact-match;
  - accuracy per length.
- **Curvature, measured at the end of training:**
  - Measurement set: a fixed 512-problem batch from the training distribution (n uniform on 1…19), at K = n, eval mode, identical for all models.
  - Hutchinson tr(H) with 100 Rademacher probes, reported with its standard error, both overall and per parameter group.
  - Dominant Hessian eigenvalue by power iteration.
  - Fisher trace by the Bartlett sampled-label estimator (as in Liu et al. 2023).
  - Average-case adaptive sharpness E[L(θ + ε⊙|θ|)] − L(θ), ε ~ N(0, ρ²I), for ρ ∈ {0.01, 0.02, 0.05} ([Andriushchenko et al. 2023](https://arxiv.org/abs/2302.07011)).
  - Weight norms per parameter group.
- **During training:** a 10-probe tr(H) on a fixed 128-problem batch, logged periodically for both arms.
- **Loop-count curvature study (evaluation only):** tr(H) and the dominant eigenvalue as functions of a fixed K ∈ {1, 2, 4, 8, 12, 16, 20}.

## 4. Stage 1 (now): one penalized model vs one plain model

Scope set by the project owner on 2026-11-01: train exactly two models today. Everything else is deferred to Stage 2.

| Arm | Training objective |
|---|---|
| `baseline` | L (plain looped transformer, AdamW) |
| `hutchinson` | L + λ·vᵀHv, with 1 Rademacher probe per step, computed on the first 16 of the 64 problems in each batch (sub-batch to fit 8 GB); λ = 5×10⁻⁵ |

**Why λ = 5×10⁻⁵.** It equals σ²/2 for σ = 0.01, the central value of NSO's published σ grid {0.008, 0.01, 0.012}. It is fixed now and **not tuned on any data**.

**Controls**
- Both arms use seed 0. The data stream and the loop counts K_i come from a dedicated generator, so both models see identical batches and identical initial weights. The Hutchinson probes use a separate generator.

**Baselines that need no training**
- Chance exact-match: 2^−(n+1) on the n+1 sum bits.
- The untrained (seed-0 initialization) model.

**Step budget S.** S is chosen before training from a throughput benchmark of 200 steps per arm; the benchmark produces no reported model. Rule: the largest S in {50k, 30k, 20k} for which the projected wall-clock of the Hutchinson run is ≤ 7 h when it runs concurrently with the baseline. The chosen S is recorded in an addendum before training starts.

**Stage 1 analysis (descriptive; not a test of H1)**
- With one seed per arm, seed variance cannot be estimated.
- Using Kuo et al.'s σ ≈ 5.9 points for randomized-loop training, the difference between two single runs has SD ≈ √2·5.9 ≈ 8.3 points.
- Pre-specified reading of the primary metric difference Δ = hutchinson − baseline:
  - **"suggestive positive / negative"** only if |Δ| > 2·8.3 ≈ 17 points;
  - otherwise **"inconclusive at n=1 seed"**.
- Wilson 95% intervals for accuracies capture test-set sampling only. They are reported but are not a substitute for seeds.

**Stage 1 manipulation check**
- Passes if the Hutchinson model's final 100-probe tr(H) is lower than the baseline's, with non-overlapping ±3 SE intervals.
- If tr(H) falls while weight norms grow or the dominant eigenvalue rises, this is flagged as possible scale or reparametrization gaming (Dinh et al.).

**Failure handling**
- If the Hutchinson run diverges (non-finite loss), we report it and rerun once with λ/10. The rerun is labelled as a deviation.

## 5. Stage 2 (deferred): confirmatory design

**Arms**
- AdamW baseline.
- AdamW with 2× steps (compute-matched control).
- Hutchinson.
- NSO.
- SAM ([Foret et al. 2021](https://arxiv.org/abs/2010.01412)).
- Random-direction control: g + ‖Δ_t‖·ξ/‖ξ‖, where Δ_t is NSO's per-step gradient deviation, logged in the NSO run with the same seed.
- Ablation: one-point random weight perturbation ∇L(θ+U). It has the same expected objective as NSO but without the two-point variance reduction.

**Seeds**
- Tuning seeds 100–101, kept separate from confirmation seeds 0–9.

**Tuning**
- Grids:
  - Hutchinson and NSO share σ ∈ {0.003, 0.01, 0.03}, with λ = σ²/2;
  - SAM uses ρ ∈ {0.01, 0.05, 0.2}.
- Selection rule: the largest value whose mean ID exact-match is within 1 point of the baseline. If the largest value qualifies, extend the grid by ×3, at most twice. No OOD data is used in tuning.

**Hypotheses**
- **H1a / H1b:** Hutchinson / NSO beats AdamW on the primary OOD metric.
- **H2 (manipulation check):** both lower final tr(H).
- **H3:** both lower the across-seed variance of OOD accuracy. Tested with Brown–Forsythe (`scipy.stats.levene(center='median')`).
- **H4 (exploratory):** Kendall τ between each curvature measure and OOD accuracy across runs (as in Jiang et al. 2020), with partial correlation controlling for final training loss.

**Statistics**
- **Primary family:** one-sided Welch t-tests for H1a and H1b, Holm-corrected at α = 0.05 (`statsmodels.stats.multitest.multipletests(method='holm')`).
- **Reported with each test:**
  - mean difference with a 95% BCa bootstrap confidence interval (`scipy.stats.bootstrap`);
  - Cohen's d;
  - P(A > B) with a bootstrap CI ([Bouthillier et al. 2021](https://arxiv.org/abs/2103.03098)).
- **Robustness checks:** Mann–Whitney U, and the label-free stopping metric.
- **Secondary family (Holm):**
  - each penalty vs the compute-matched control;
  - each penalty vs the random-direction control;
  - each penalty vs SAM;
  - ID non-inferiority with a 1-point margin;
  - the manipulation check.
- **Power:** with n = 10 per arm and SD ≈ 6 points, the minimum detectable effect is about 8 points at 80% power.

**Falsification criteria**
- **H1 supported:** Holm-adjusted p < 0.05, the 95% CI of the difference excludes 0, and the label-free metric has the same sign.
- **H1 refuted:** the upper bound of the 95% CI is below +2 points.
- **Otherwise:** not supported / inconclusive.
- **Gains credited to curvature regularization** only if the arm also beats both the compute-matched control and the random-direction control, and the manipulation check passes.
- **Inconclusive** if tr(H) does not decrease, or decreases only through weight-norm growth or more negative eigenvalue mass.

**Secondary testbed (exploratory)**
- A looped LM on TinyStories ([Eldan & Li 2023](https://arxiv.org/abs/2305.07759)), byte-level, 2-layer block, trained at K = 4.
- Metrics:
  - bits-per-byte;
  - loop extrapolation at K ∈ {1, 2, 8, 16};
  - context extrapolation at 2× the training context.
- Arms: baseline, Hutchinson, NSO, SAM.
- Tuning: 1 seed per value, scored on a select half of the validation split.
- Confirmation: 3 seeds, scored on a report half. No confirmatory claims.

## 6. Confounders handled
- **Compute:** a compute-matched arm in Stage 2; wall-clock reported in every stage.
- **Tuning budget:** same grid size and selection rule for every regularizer.
- **Optimizer settings:** learning rate, schedule and Adam betas are fixed across arms. Sharpness under Adam tracks these settings ([Cohen et al. 2022](https://arxiv.org/abs/2207.14484)).
- **Weight-norm and scale gaming:** weight norms and adaptive (scale-aware) sharpness are logged.
- **Measurement:** fixed measurement batch, probe seeds and loop count; eval mode.
- **Data order and initialization:** identical across arms within a seed.
- **Stopping rule:** primary oracle-K metric plus label-free secondary metrics.

## Addenda

### Addendum 1: 2026-11-01 19:30 CET, before either Stage 1 model was trained

**What has been run so far.** Only throughput benchmarks and a 60-step smoke test.
- Benchmarks: `results/benchmark/*.json`, 16–200 timed steps at the full length range. No model is kept.
- Smoke test: `main.py smoke`. It checks plumbing only; its models reach 0% accuracy and are not reported.
- No Stage 1 model has been trained or evaluated.

**Deviations from Section 4, all forced by the 8 GB GPU**

1. **Hutchinson sub-batch: 16 → 4 problems.**
   - At 16, the third-order graph needed 10.05 GB, which is more than the GPU's 8 GB. Windows spilled it into system memory, giving 15.6 s/step (`benchmark_hutchinson_only.json`).
   - At 4, peak memory is 4.46 GB (`benchmark_hutchinson_sub4.json`).
2. **Penalty schedule: applied every k = 8 steps with weight k·λ = 4×10⁻⁴, instead of every step with λ.**
   - The expected penalty gradient per step is still λ·∇tr(H). This follows the lazy-curvature practice of Sophia ([arXiv:2305.14342](https://arxiv.org/abs/2305.14342)), which estimates curvature every k = 10 steps.
   - Applying it every step costs 2.15 s/step (`benchmark_hutchinson_every1_sub4.json`), so 30k steps would take about 18 h. The amortized arm costs 0.446 s/step (`benchmark_final_alone.json`).
3. **Deterministic kernels are not forced.**
   - `torch.use_deterministic_algorithms(True)` slowed the baseline from about 0.20 to 0.25 s/step.
   - Runs remain seeded, with identical data streams and initializations across arms, but are not bitwise reproducible.
4. **The two arms run sequentially, not concurrently.**
   - Sharing the GPU gave no throughput gain: the baseline slowed from 0.20 to 0.70 s/step when run alongside the Hutchinson arm (`benchmark_concurrent_*.json`).
   - The 7 h rule is therefore applied to the sequential total.
5. **LayerNorm implementation.**
   - In torch 2.15.0, the native `layer_norm` third derivative disagrees with converged finite differences (`tests/test_third_order.py`).
   - The penalty's sub-batch forward pass therefore uses a mathematically identical composite LayerNorm.
   - All other forward passes use the native kernel. Its first and second derivatives agree with finite differences (`tests/test_curvature.py`).

**Step budget, by the preregistered rule.**
- Projected total: (0.201 + 0.446) s × S, using full-length timing, so this is an upper bound.
- S = 50k → 9.0 h, over the 7 h limit. S = 30k → 5.4 h, within it.
- **S = 30,000.**

**Observation recorded before training (smoke test, not a generalization result).**
- On steps where the penalty is applied, the total gradient norm was 10²–10³. Steps without it had norms of about 1–40.
- The single-probe ∇(vᵀHv) has very high variance in about 2.4M dimensions. With the preregistered clip at 1.0, the penalty direction therefore dominates the update on those steps.
- λ and clipping are kept as preregistered. This is flagged as an interpretation caveat: the Stage 2 random-direction control is designed to separate curvature-specific effects from injected gradient noise.


### Addendum 2: 2026-11-01 19:58 WEST, before either Stage 1 model was trained

**What happened since Addendum 1.**
- The baseline run was launched at 19:30 and stopped by me at step ~2,000 of 30,000, before any evaluation.
- Reason: its periodic in-training tr(H) probe spilled past the 8 GB of VRAM into Windows shared system memory. The step-1000 probe took ~100 s.
- The caching allocator had reserved 7.3 GB, against a peak usage of 4.46 GB. On Windows (WDDM), allocations beyond VRAM do not fail; they fall back to slow system memory.
- That run's folder was deleted. Training restarts from scratch with an identical training configuration. Only its training-loss log lines were seen, no accuracy.

**Infrastructure changes (they do not change the training objective or data).**

1. **GPU memory cap.** `torch.cuda.set_per_process_memory_fraction(0.85)`, so the allocator frees its cache and retries instead of spilling. With the cap, a quick evaluation fell from 80 s to 2.8 s.
2. **In-training tr(H) probe.** Now every 2,000 steps (was "periodically", implemented as 1,000), with 16-problem chunks. It still uses 10 probes on the fixed 128-problem batch.
3. **Curvature measurement cut down for time.** On this machine one Hessian-vector product through ≤20 loops costs ~0.58 s regardless of chunk size, which points to kernel-launch overhead. 32-problem chunks at K = 20 exceed memory, and 16-problem chunks peak at 4.6 GB. At these costs the preregistered measurement would take about 2.5 h per model. Changes:
   - measurement batch 512 → **128** problems (same seed and construction; identical for both models);
   - loop-count study probes 20 → **10**;
   - loop-count study power iterations 20 → **10**;
   - the main tr(H) still uses 100 probes;
   - the dominant eigenvalue still uses 50 power iterations.
4. **No WSL fallback.** A WSL environment was tried as the preregistered fallback; its PyTorch download timed out, so runs stay on Windows.
