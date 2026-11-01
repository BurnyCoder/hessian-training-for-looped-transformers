"""Training loop shared by all arms (only the step strategy differs).

Local: AdamW + cosine decay to 0 without warmup (Fan et al. 2024; Kuo et al. 2026), batch
64, gradient-norm clipping, linear length curriculum and randomized loop counts. The data
stream (lengths, operand bits, loop counts) comes from one CPU generator seeded with the
run seed, and the model is initialised right after `torch.manual_seed(seed)`, so two arms
with the same seed see identical batches and start from identical weights; the Hutchinson
probes use their own generator (regularizers.HutchinsonStep).

Global: writes `history.jsonl` (loss, penalty, grad norm, lr, curriculum length, in-training
tr(H), quick accuracies) to the run folder and saves `init.pt` (untrained baseline) and
`final.pt` checkpoints for evaluation and curvature measurement.
"""

import json
import math
import os
import time

import torch

from looped_hessian import data
from looped_hessian.config import RunConfig
from looped_hessian.curvature import hutchinson_trace
from looped_hessian.evaluate import evaluate_model
from looped_hessian.model import LoopedGPT
from looped_hessian.regularizers import make_step


def build_model(cfg: RunConfig) -> LoopedGPT:
    # Seeding immediately before construction makes the initialisation a function of the seed.
    torch.manual_seed(cfg.train.seed)
    return LoopedGPT(cfg.model).to(cfg.device)


def fixed_batch(count: int, max_len: int, seed: int):
    """Problems from the training length distribution read out at K = n (curvature batches)."""
    gen = torch.Generator().manual_seed(seed)
    lengths = torch.randint(1, max_len + 1, (count,), generator=gen)
    x, y = data.make_batch(lengths, gen)
    return x, y, lengths.clone()


def save_checkpoint(model, cfg: RunConfig, name: str) -> str:
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    path = os.path.join(cfg.ckpt_dir, name)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, path)
    return path


def load_checkpoint(cfg: RunConfig, name: str) -> LoopedGPT:
    model = LoopedGPT(cfg.model).to(cfg.device)
    state = torch.load(os.path.join(cfg.ckpt_dir, name), map_location=cfg.device)
    model.load_state_dict(state["model"])
    return model


def benchmark(cfg: RunConfig, n_steps: int, logger, warmup: int = 10) -> dict:
    """Seconds per training step and peak memory at the full length range (post-curriculum).

    Lengths are drawn from [1, max_train_len] at every step, the most expensive part of
    the curriculum, so projected run times are upper bounds. No model is kept.
    """
    tc = cfg.train
    model = build_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    step_fn = make_step(tc.arm, lam=tc.lam, sub_batch=tc.sub_batch, seed=tc.seed + 1_000_003,
                        every=tc.penalty_every)
    gen = torch.Generator().manual_seed(tc.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + n_steps):
        if i == warmup:
            # Exclude CUDA context / allocator warm-up from the timing.
            torch.cuda.synchronize()
            t0 = time.time()
        lengths = torch.randint(1, tc.max_train_len + 1, (tc.batch_size,), generator=gen)
        x, y = data.make_batch(lengths, gen)
        loops = data.sample_loop_counts(lengths, gen, tc.loop_jitter, tc.max_train_loops)
        opt.zero_grad(set_to_none=True)
        step_fn(model, x, y, loops)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
    torch.cuda.synchronize()
    sec = (time.time() - t0) / n_steps
    out = {"arm": tc.arm, "seconds_per_step": sec, "steps_timed": n_steps,
           "peak_memory_gb": torch.cuda.max_memory_allocated() / 1e9}
    logger.info(f"benchmark {tc.arm}: {sec:.4f} s/step, peak {out['peak_memory_gb']:.2f} GB")
    return out


def train(cfg: RunConfig, logger) -> dict:
    tc = cfg.train
    os.makedirs(cfg.run_dir, exist_ok=True)
    model = build_model(cfg)
    save_checkpoint(model, cfg, "init.pt")
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"run {cfg.name}: arm={tc.arm} steps={tc.steps} seed={tc.seed} params={n_params}")
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, betas=(tc.beta1, tc.beta2),
                            weight_decay=tc.weight_decay)
    # Cosine decay from lr to 0 over the run, stepped once per optimizer step.
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / tc.steps)))
    data_gen = torch.Generator().manual_seed(tc.seed)
    # Probe seed offset keeps penalty randomness independent of the data stream.
    step_fn = make_step(tc.arm, lam=tc.lam, sub_batch=tc.sub_batch, seed=tc.seed + 1_000_003,
                        every=tc.penalty_every)
    curv_batch = fixed_batch(tc.curvature_batch, tc.max_train_len, seed=30_000)
    history_path = os.path.join(cfg.run_dir, "history.jsonl")
    history = open(history_path, "w", encoding="utf-8")
    t0 = time.time()
    status = "completed"
    for step in range(tc.steps):
        lengths = data.sample_lengths(step, tc.steps, tc.max_train_len, tc.batch_size, data_gen)
        x, y = data.make_batch(lengths, data_gen)
        loops = data.sample_loop_counts(lengths, data_gen, tc.loop_jitter, tc.max_train_loops)
        opt.zero_grad(set_to_none=True)
        metrics = step_fn(model, x, y, loops)
        if not all(math.isfinite(v) for v in metrics.values()):
            # Preregistered failure handling: report divergence instead of silently continuing.
            logger.error(f"non-finite metrics at step {step}: {metrics}")
            status = "diverged"
            break
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip).item()
        opt.step()
        sched.step()
        record = None
        if step % tc.log_every == 0 or step == tc.steps - 1:
            record = {"step": step, **metrics, "grad_norm": grad_norm, "lr": sched.get_last_lr()[0],
                      "n_max": int(lengths.max()), "elapsed_s": time.time() - t0}
        if step % tc.curvature_every == 0 or step == tc.steps - 1:
            tr = hutchinson_trace(model, curv_batch, tc.curvature_probes, seed=40_000, chunk_size=16)
            record = (record or {"step": step}) | {"trace_h": tr["trace"], "trace_h_se": tr["se"]}
        if (step % tc.eval_every == 0 and step > 0) or step == tc.steps - 1:
            quick = evaluate_model(model, (5, 10, 15, 19, 20, 30, 40), tc.eval_count, seed_base=50_000,
                                   extra_loops=5, chunk_size=200, device=cfg.device)
            record = (record or {"step": step}) | {f"acc_n{n}": s["oracle"]["acc"] for n, s in quick.items()}
        if record:
            history.write(json.dumps(record) + "\n")
            history.flush()
            logger.info(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                 for k, v in record.items()))
    history.close()
    train_seconds = time.time() - t0
    save_checkpoint(model, cfg, "final.pt")
    summary = {"name": cfg.name, "status": status, "steps_done": step + 1, "train_seconds": train_seconds,
               "n_params": n_params, "peak_memory_gb": (torch.cuda.max_memory_allocated() / 1e9
                                                        if torch.cuda.is_available() else None)}
    with open(os.path.join(cfg.run_dir, "train_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"training finished: {summary}")
    return summary
