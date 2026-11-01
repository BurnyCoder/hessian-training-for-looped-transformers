# Hessian-trace regularization for looped transformers

**What this is.** A preregistered experiment testing whether penalizing loss curvature (the trace of the Hessian, tr(H)) while training a **looped transformer** improves its **length generalization**. A looped transformer is a weight-tied block applied K times, with the input re-injected at every iteration.

**Motivation.** Three strands of prior work meet here:
- **Flat minima.** Flatter minima are reported to generalize better: [Hochreiter & Schmidhuber 1997](https://doi.org/10.1162/neco.1997.9.1.1); [Keskar et al. 2017](https://arxiv.org/abs/1609.04836); [Jiang et al. 2020](https://arxiv.org/abs/1912.02178); [Liu et al. 2023](https://arxiv.org/abs/2210.14199), who report lower tr(H) → better downstream accuracy in language models.
- **Looped models.** They can solve longer problems by looping more: [Fan et al. 2024](https://arxiv.org/abs/2409.15647).
- **Brittleness.** That extrapolation varies a lot across random seeds: [Kuo et al. 2026](https://arxiv.org/abs/2606.29983).

Does steering a looped model toward flat minima make its extrapolation better? The skeptical literature is part of the design: [Dinh et al. 2017](https://arxiv.org/abs/1703.04933); [Andriushchenko et al. 2023](https://arxiv.org/abs/2302.07011); [Gong et al. 2025](https://arxiv.org/abs/2510.10089), who argue looping *benefits* from steep valleys.

**Methods.** Two estimators of the same objective, **L + λ·tr(H)**:
1. **Exact Hutchinson penalty.**
   - Penalty term: λ·vᵀHv, with a Rademacher probe v.
   - Hv comes from double backprop, and the penalty's own gradient from a third-order backward pass.
   - Prior use: [SEHT, Liu et al. 2022](https://arxiv.org/abs/2208.05924).
2. **NSO (Noise Stability Optimization)**, [Zhang, Li & Ju, TMLR 2024](https://arxiv.org/abs/2306.08553).
   - Update direction: ½[∇L(θ+U) + ∇L(θ−U)], with U ~ N(0, σ²I).
   - This equals ∇L + (σ²/2)∇tr(H) in expectation. NSO is a finite-difference estimator of the same objective, with λ = σ²/2.

**Testbed.**
- Binary addition with length generalization ([Fan et al.](https://arxiv.org/abs/2409.15647); [Kuo et al.](https://arxiv.org/abs/2606.29983)).
- Train on numbers of 1–19 bits; test on 20–60 bits.
- Model: a 3-layer looped GPT-2 block with no positional embeddings (NoPE) and input injection.

**Status.**
- **Stage 1 (current):** one Hutchinson-penalized looped transformer vs one plain looped transformer, same seed. Descriptive only.
- **Stage 2 (deferred):** 10 seeds, NSO, SAM, compute-matched and random-direction controls, tuning, ablations, plus a looped TinyStories LM.

The full design, hypotheses, metrics and falsification criteria are in the [preregistration](reports/preregistration.md), which was committed before any code or results existed.

## Results

*No results yet.* This section will be generated from `results/` by `main.py report`.

## Architecture

```mermaid
flowchart LR
    X["tokens: a + b > # # …"] --> E["embedding e (no positional encoding)"]
    E --> ADD(("+"))
    H0["h₀ = 0"] --> ADD
    ADD --> B["shared block: 3 × pre-LN GPT-2 layers<br/>causal attention, d=256, 8 heads"]
    B -->|"h_t, loop K times (input injection: h_t + e)"| ADD
    B --> LN["final LayerNorm → tied head"]
    LN --> Y["sum bits (MSB first) + EOS at the answer positions"]
```

**Training objectives**
- **Baseline:** L.
- **Hutchinson arm:** L + λ·vᵀ(∇²L_sub)v. L_sub is the loss on the first 16 problems of each batch; v is a fresh Rademacher probe per step.

## Repository layout

| Path | Purpose |
|---|---|
| `reports/preregistration.md` | Hypotheses, metrics, analysis plan, falsification criteria, addenda |
| `src/looped_hessian/` | Library code: data, model, regularizers, curvature measurement, training, evaluation, statistics, reporting |
| `main.py` | Single entry point that runs the pipeline phases |
| `tests/` | pytest suite (estimator correctness vs exact Hessians, data and model invariants) |
| `results/` | Per-run JSON results and logs (the source of every reported number) |
| `reports/` | Per-experiment reports, summary, paper |
| `docs/` | Method notes |

## Setup

Requirements:
- [uv](https://docs.astral.sh/uv/).
- An NVIDIA GPU with a recent driver. Developed on an RTX 5070 Laptop (8 GB, sm_120) with PyTorch 2.15.0+cu132 on Windows 11.

```bash
uv sync
```

```bash
uv run pytest
```

Usage instructions for `main.py` are added together with the implementation.

## License

MIT (see [LICENSE](LICENSE)).
