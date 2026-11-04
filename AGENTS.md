# AGENTS.md: notes for coding agents working on this repository

## Commands
- Install: `uv sync`
- Tests: `uv run pytest`. Takes about 1–2 min on CPU. It includes the finite-difference checks of the Hessian estimators and of third-order autograd.
- End-to-end smoke run: `uv run python main.py smoke` (writes to `results/smoke/`, which is gitignored)
- Stage 1: `uv run python main.py run --arm {baseline|hutchinson} --name stage1_{arm}`, then `uv run python main.py report`
- Publish: `uv run python main.py publish` (Hugging Face; needs `hf auth login`)

## Conventions
- **Numbers.** Every number in README.md, `reports/*.md` and `reports/paper/paper.md` is generated. `report.py` replaces text between `<!-- BEGIN:key -->` and `<!-- END:key -->` markers. Never type result numbers by hand; add a block in `report.py` instead.
- **Preregistration.** `reports/preregistration.md` is append-only. Changes go in a new timestamped addendum, written *before* the results they could affect.
- **Third-order graphs.** Anything differentiated three times must be built inside `model.third_order_safe()`. The native LayerNorm's third derivative is wrong in torch 2.15.0 (`tests/test_third_order.py`).
- **GPU memory.** On Windows (WDDM), allocations beyond VRAM silently spill to system memory (the driver's "CUDA - Sysmem Fallback Policy") and run much slower: a quick evaluation took 80 s instead of 2.8 s (preregistration Addendum 2). `main.py` caps the allocator at 85% of VRAM (`LH_GPU_MEM_FRACTION`). Hessian-vector products and Hutchinson traces through up to 20 loops need about 0.2–0.3 GB per problem, so use chunks of ≤ 16 problems (32 exceeded memory at K = 20).
- **Paired runs.** Arms must stay paired: the same seed gives the same initialization and data stream. Penalty randomness uses its own generator.
- **Checkpoints.** They live in `checkpoints/` (gitignored) and on Hugging Face. Results JSON and logs live in `results/` and are committed.
