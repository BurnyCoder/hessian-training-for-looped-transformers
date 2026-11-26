"""Reconstruct the dated audit from immutable records and analytic fixtures. No network is run."""

import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from looped_hessian.artifacts import load_run
from looped_hessian.data import make_eval_set
from looped_hessian.stage2 import paired_power
from looped_hessian.stats import KUO_SEED_SD


def operator_check():
    """CPU tensor operator only. Independently differentiate native and composite LayerNorm.

    The scalar is a cubic function of four normalized coordinates. Central differences of
    a separate second directional derivative check the mixed third directional derivative.
    https://docs.pytorch.org/docs/2.15/generated/torch.nn.LayerNorm.html
    """
    x = torch.tensor([0.3, -0.8, 1.1, 0.6], dtype=torch.float64, requires_grad=True)
    v = x.new_tensor([1, -1, 1, 1])
    u = x.new_tensor([0.2, 0.6, -0.3, 0.8])
    weights = x.new_tensor([0.5, 1.3, -0.7, 0.9])

    def composite(z):
        centered = z - z.mean()
        return centered * torch.rsqrt(centered.square().mean() + 1e-5)

    def second(z, op):
        scalar = (weights * op(z).pow(3)).sum()
        g, = torch.autograd.grad(scalar, z, create_graph=True)
        hv, = torch.autograd.grad(g @ v, z, create_graph=True)
        return hv @ v

    result = {}
    for name, op in (("native", lambda z: F.layer_norm(z, (4,))), ("composite", composite)):
        third, = torch.autograd.grad(second(x, op), x)
        analytic = (third @ u).item()
        differences = {}
        for step in (1e-3, 1e-4, 1e-5):
            high = second((x + step*u).detach().requires_grad_(), op).item()
            low = second((x - step*u).detach().requires_grad_(), op).item()
            differences[str(step)] = (high - low) / (2 * step)
        result[name] = {"autograd": analytic, "finite_differences": differences}
    return {"torch": str(torch.__version__), "device": str(x.device), "dtype": str(x.dtype), **result}


def reconstruct(root=Path("results")):
    """Verify bytes before using frozen records. Regenerate input prefixes without loading weights."""
    root = Path(root)
    archive = root / "audit-2026-11-26"
    hashes = (archive / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    for line in hashes:
        expected, name = line.split("  ", 1)
        if hashlib.sha256((archive / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"historical evidence hash mismatch: {name}")
    baseline = load_run(archive / "stage1_baseline")
    config = baseline["config"]
    history = baseline["history"]
    finished = {a: load_run(archive / ("smoke_" + a)) for a in ("baseline", "hutchinson")}
    historical = json.loads((archive / "historical_smoke_summary.json").read_text())
    small, full = finished["baseline"]["config"]["eval"], config["eval"]
    overlaps = {}
    for n in small["id_lengths"] + small["ood_lengths"]:
        sx, sy = make_eval_set(n, small["n_per_length"], small["seed_base"] + n)
        fx, fy = make_eval_set(n, full["n_per_length"], full["seed_base"] + n)
        overlaps[n] = bool(torch.equal(sx, fx[:len(sx)]) and torch.equal(sy, fy[:len(sy)]))
    bench = {p.stem: json.loads(p.read_text()) for p in (archive / "benchmark").glob("*.json")}
    alone = bench["benchmark_final_alone"]
    base_time = alone["baseline"]["seconds_per_step"]
    sparse = alone["hutchinson"]["seconds_per_step"]
    dense = bench["benchmark_hutchinson_every1_sub4"]["hutchinson"]["seconds_per_step"]
    times = [r["steps_timed"] for value in bench.values() for r in value.values()]
    smoke_scores = {a: {n: [sum(r["oracle"]["correct"]), len(r["oracle"]["correct"])]
                        for n, r in run["eval"].items()} for a, run in finished.items()}
    return {"verified_files": len(hashes), "baseline": {"last_logged_step": history[-1]["step"],
            "steps_planned": config["train"]["steps"], "quick_steps": [r["step"] for r in history if "acc_n20" in r],
            "quick_accuracy": [{"step": r["step"], "rates": {k[5:]: v for k,v in r.items() if k.startswith("acc_n")}}
                               for r in history if "acc_n20" in r], "quick_count": config["train"]["eval_count"]},
            "historical_report_count": historical["eval_config"]["n_per_length"], "smoke_scores": smoke_scores,
            "prefix_matches": overlaps, "step_benchmark": {"baseline": base_time, "dense": dense, "sparse": sparse,
            "dense_ratio": dense/base_time, "sparse_ratio": sparse/base_time, "timed_steps_range": [min(times), max(times)],
            "step_only_hours": config["train"]["steps"] * (base_time + sparse)/3600},
            "layernorm_operator": json.loads((archive / "layernorm_operator.json").read_text()),
            "power": [{"rho": rho, "paired_sd": KUO_SEED_SD * math.sqrt(2*(1-rho)),
                       "detectable_points": paired_power(rho)} for rho in (-0.5, 0.0, 0.5)]}


def evidence_block(facts):
    """Human-readable audit observations generated from the machine-readable evidence."""
    b, bench = facts["baseline"], facts["step_benchmark"]
    rows = [f"The baseline ends at logged step **{b['last_logged_step']:,}** of a "
            f"**{b['steps_planned']:,}**-step budget. Interim accuracy appears at logged steps "
            + ", ".join(f"{n:,}" for n in b["quick_steps"]) + ".",
            f"The historical smoke summary declared **{facts['historical_report_count']:,}** problems per length. "
            "The final smoke oracle counts, reconstructed from per-problem flags, were:",
            "| Operand bits | Baseline correct / scored | Hutchinson correct / scored | Planned test prefix equals smoke inputs and targets |",
            "|---:|---:|---:|---|"]
    for length, match in facts["prefix_matches"].items():
        counts = [" / ".join(map(str, facts["smoke_scores"][arm][str(length)])) for arm in ("baseline", "hutchinson")]
        rows.append(f"| {length} | {counts[0]} | {counts[1]} | {'yes' if match else 'no'} |")
    rows += ["", f"The timing records cover {bench['timed_steps_range'][0]}–{bench['timed_steps_range'][1]} measured steps. "
             f"The isolated baseline averaged {bench['baseline']:.3f} s/step, the every-step penalty {bench['dense']:.3f} "
             f"({bench['dense_ratio']:.2f}× baseline), and the scheduled penalty {bench['sparse']:.3f} "
             f"({bench['sparse_ratio']:.2f}×). Projecting the two isolated scheduled runs gives "
             f"{bench['step_only_hours']:.2f} hours of steps, excluding periodic diagnostics and final measurements."]
    return "\n".join(rows[:1]) + "\n\n" + "\n".join(rows[1:2]) + "\n\n" + "\n".join(rows[2:])


def operator_block(record):
    rows = [f"Isolated scalar operator check on {record['device']}, {record['dtype']}, PyTorch {record['torch']}.",
            "", "| Implementation | Autograd | Central difference h=0.001 | h=0.0001 | h=0.00001 |",
            "|---|---:|---:|---:|---:|"]
    for name in ("native", "composite"):
        r = record[name]
        cells = [f"{v:.6f}" for v in (r["autograd"], *r["finite_differences"].values())]
        rows.append("| " + name + " | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def interim_block(facts):
    """Include every monitored width, including repeated zeroes, from the frozen baseline."""
    data = facts["baseline"]
    widths = sorted(data["quick_accuracy"][0]["rates"], key=int)
    rows = [f"Archived baseline interim oracle accuracy (%), {data['quick_count']} problems per monitored width.",
            "", "| Logged step | " + " | ".join("n=" + n for n in widths) + " |",
            "|" + "---:|" * (len(widths)+1)]
    for row in data["quick_accuracy"]:
        cells = [f"{100*row['rates'][n]:.1f}" for n in widths]
        rows.append(f"| {row['step']:,} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def power_block(facts):
    rows = ["| Assumed within-seed correlation | SD of paired difference (points) | Gain for 80% power (points) |",
            "|---:|---:|---:|"]
    rows.extend(f"| {r['rho']:+.1f} | {r['paired_sd']:.2f} | {r['detectable_points']:.2f} |" for r in facts["power"])
    return "\n".join(rows)
