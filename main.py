"""Single entry point: runs the pipeline phases of the experiment.

Phases (each reads/writes the run folder results/<name>/ and checkpoints/<name>/):
  benchmark  time training steps per arm at full length (sets the step budget; no model kept)
  train      train one arm                         -> history.jsonl, train_summary.json, init.pt, final.pt
  evaluate   ID/OOD accuracy per loop + untrained  -> eval.json, eval_init.json, examples.json
  measure    curvature metrics + loop-count study  -> curvature.json
  run        train + evaluate + measure for one arm
  smoke      tiny end-to-end run of both arms + report (results/smoke, minutes)
  report     regenerate every table/figure from results/ into README, reports, paper
  publish    upload checkpoints and results to the Hugging Face collection

Example (Stage 1):
  uv run python main.py run --arm baseline   --name stage1_baseline
  uv run python main.py run --arm hutchinson --name stage1_hutchinson
  uv run python main.py report
"""

import argparse
import json
import os

import torch  # noqa: E402

from looped_hessian.config import RunConfig, TrainConfig  # noqa: E402
from looped_hessian.log import get_logger  # noqa: E402

# TF32 matmuls (fp32 storage), identical for every arm. Deterministic kernels are NOT forced:
# they made the baseline step 34% slower in our benchmark; runs are seeded but not bitwise
# reproducible (https://docs.pytorch.org/docs/stable/notes/randomness.html).
torch.set_float32_matmul_precision("high")


def make_config(args) -> RunConfig:
    # CLI flags override the preregistered defaults in config.py.
    cfg = RunConfig(name=args.name)
    for key in ("arm", "steps", "seed", "lam", "sub_batch", "penalty_every"):
        value = getattr(args, key, None)
        if value is not None:
            setattr(cfg.train, key, value)
    return cfg


def phase_train(cfg: RunConfig):
    from looped_hessian.train import train
    os.makedirs(cfg.run_dir, exist_ok=True)
    with open(os.path.join(cfg.run_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    return train(cfg, get_logger(cfg.name, os.path.join(cfg.run_dir, "train.log")))


def phase_evaluate(cfg: RunConfig):
    from looped_hessian.evaluate import evaluate_model, output_examples
    from looped_hessian.train import load_checkpoint
    logger = get_logger(cfg.name, os.path.join(cfg.run_dir, "evaluate.log"))
    ec = cfg.eval
    lengths = tuple(ec.id_lengths) + tuple(ec.ood_lengths)
    for ckpt, out_name in (("final.pt", "eval.json"), ("init.pt", "eval_init.json")):
        model = load_checkpoint(cfg, ckpt)
        logger.info(f"evaluating {ckpt} on lengths {lengths[0]}..{lengths[-1]}")
        res = evaluate_model(model, lengths, ec.n_per_length, ec.seed_base, ec.extra_loops, ec.chunk_size,
                             cfg.device, logger)
        with open(os.path.join(cfg.run_dir, out_name), "w") as f:
            json.dump({str(k): v for k, v in res.items()}, f)
        if ckpt == "final.pt":
            # Qualitative examples at the oracle loop count of a few OOD lengths.
            examples = {str(n): output_examples(model, n, 4, seed=60_000 + n, k=res[n]["oracle"]["k"],
                                                device=cfg.device)
                        for n in (20, 30, 60) if n in res}
            with open(os.path.join(cfg.run_dir, "examples.json"), "w") as f:
                json.dump(examples, f, indent=2)


def phase_measure(cfg: RunConfig):
    from looped_hessian.measure import measure
    from looped_hessian.train import load_checkpoint
    logger = get_logger(cfg.name, os.path.join(cfg.run_dir, "measure.log"))
    measure(load_checkpoint(cfg, "final.pt"), cfg, logger)


def phase_benchmark(args):
    from looped_hessian.train import benchmark
    out = {}
    for arm in args.arms:
        cfg = make_config(argparse.Namespace(name=f"benchmark_{arm}", arm=arm, steps=None, seed=0, lam=None,
                                             sub_batch=args.sub_batch, penalty_every=args.penalty_every))
        out[arm] = benchmark(cfg, args.bench_steps, get_logger(f"benchmark_{arm}"))
    os.makedirs("results/benchmark", exist_ok=True)
    path = os.path.join("results/benchmark", f"benchmark_{args.tag}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def phase_smoke():
    # Tiny version of the whole pipeline: verifies plumbing, not science.
    for arm in ("baseline", "hutchinson"):
        cfg = RunConfig(name=f"smoke_{arm}", output_dir="results/smoke", checkpoint_dir="checkpoints/smoke",
                        train=TrainConfig(arm=arm, steps=60, log_every=20, curvature_every=30, eval_every=30,
                                          eval_count=20, curvature_probes=2, curvature_batch=16))
        cfg.eval.id_lengths, cfg.eval.ood_lengths, cfg.eval.n_per_length = (1, 5, 19), (20, 30), 20
        cfg.measure.batch, cfg.measure.probes, cfg.measure.power_iters = 32, 3, 3
        cfg.measure.fisher_samples, cfg.measure.sharpness_draws = 2, 2
        cfg.measure.loop_ks, cfg.measure.loop_probes, cfg.measure.loop_power_iters = (1, 4), 2, 2
        phase_train(cfg)
        phase_evaluate(cfg)
        phase_measure(cfg)
    from looped_hessian.report import build_report
    build_report(results_dir="results/smoke", prefix="smoke_", write_docs=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["benchmark", "train", "evaluate", "measure", "run", "smoke",
                                          "report", "publish"])
    parser.add_argument("--name", default="stage1_baseline")
    parser.add_argument("--arm", choices=["baseline", "hutchinson"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lam", type=float)
    parser.add_argument("--sub-batch", dest="sub_batch", type=int)
    parser.add_argument("--penalty-every", dest="penalty_every", type=int)
    parser.add_argument("--arms", nargs="+", default=["baseline", "hutchinson"])
    parser.add_argument("--bench-steps", type=int, default=200)
    parser.add_argument("--tag", default="single")
    args = parser.parse_args()
    if args.phase == "benchmark":
        return phase_benchmark(args)
    if args.phase == "smoke":
        return phase_smoke()
    if args.phase == "report":
        from looped_hessian.report import build_report
        return build_report()
    if args.phase == "publish":
        from looped_hessian.publish import publish
        return publish()
    cfg = make_config(args)
    if args.phase in ("train", "run"):
        phase_train(cfg)
    if args.phase in ("evaluate", "run"):
        phase_evaluate(cfg)
    if args.phase in ("measure", "run"):
        phase_measure(cfg)


if __name__ == "__main__":
    main()
