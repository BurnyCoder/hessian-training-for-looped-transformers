"""Regenerate scientific outputs from saved artifacts, including partial runs. No model is loaded."""

import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path

from looped_hessian import plots
from looped_hessian.artifacts import load_run, metrics, pairing, per_length, pooled, run_status
from looped_hessian.stats import SUGGESTIVE_THRESHOLD, paired_counts, stage1_reading

ARMS = ("baseline", "hutchinson")
DOCS = ("README.md", "reports/stage1_experiment.md", "reports/all_experiments.md", "reports/summary.md",
        "reports/paper/paper.md", "reports/progress.md", "reports/reviews/2026-11-26-scientific-audit.md")


def summarize(results_dir, prefix):
    """Report each usable arm independently; comparisons require a qualified pair."""
    runs = {a: load_run(Path(results_dir) / (prefix + a)) for a in ARMS}
    statuses = {a: run_status(r, a) for a, r in runs.items()}
    out = {"experiment": "Stage 1" if prefix == "stage1_" else prefix.rstrip("_"),
           "run_status": statuses, "arms": {}, "id_lengths": [], "ood_lengths": [],
           "histories": {a: r["history"] for a, r in runs.items() if r["history"]},
            "pairing": {"verified": False, "reason": "two completed final evaluations required"}}
    configs = [r["config"]["eval"] for r in runs.values() if r["config"]]
    out["eval_config"] = configs[0] if configs and all(c == configs[0] for c in configs) else None
    for a, r in runs.items():
        if statuses[a]["state"] != "evaluated":
            continue
        ec = r["config"]["eval"]
        for key in ("id_lengths", "ood_lengths"):
            out[key] = sorted(set(out[key]).union(ec[key]))
        out["arms"][a] = {k: r[k] for k in ("config", "train", "curvature", "examples")}
        out["arms"][a].update(metrics=metrics(r["eval"], ec), per_length=per_length(r["eval"]))
    if not out["arms"]:
        return out
    first = runs[next(iter(out["arms"]))]
    if len(out["arms"]) == 2:
        ok, reason = pairing(*runs.values())
        out["pairing"] = {"verified": ok, "reason": reason}
        if not ok:
            return out
    ec = first["config"]["eval"]
    out["chance"] = {group: sum(2.0 ** -(n + 1) for n in ec[group + "_lengths"]) / len(ec[group + "_lengths"])
                     for group in ("id", "ood")}
    if first["eval_init"]:
        out["untrained"] = {"metrics": metrics(first["eval_init"], ec), "per_length": per_length(first["eval_init"])}
    if out["pairing"]["verified"]:
        b, h = (out["arms"][a]["metrics"] for a in ARMS)
        delta = 100 * (h["ood_oracle"]["acc"] - b["ood_oracle"]["acc"])
        out["comparison"] = {"delta_" + key + "_points": 100 * (h[key]["acc"] - b[key]["acc"]) for key in b}
        out["comparison"].update(oracle_discordances=paired_counts(b["ood_oracle"]["correct"], h["ood_oracle"]["correct"]))
        if prefix == "stage1_":
            out["comparison"].update(reading=stage1_reading(delta), threshold_points=SUGGESTIVE_THRESHOLD)
        cb, ch = (r["curvature"] for r in runs.values())
        design = first["config"]["measure"]
        if cb and ch and design == runs["hutchinson"]["config"]["measure"]:
            tb, th = cb["trace"], ch["trace"]
            if all(t["n_probes"] == design["probes"] and math.isfinite(t["trace"]) and
                   math.isfinite(t["se"]) and t["se"] >= 0 for t in (tb, th)):
                out["manipulation_check"] = {"passed": th["trace"] + 3*th["se"] < tb["trace"] - 3*tb["se"],
                                            "trace_baseline": tb["trace"], "trace_hutchinson": th["trace"],
                                            "probe_count": design["probes"]}
    return out


def _pct(m):
    score = f"{100 * m['acc']:.1f}"
    return score + (f" [{100*m['lo']:.1f}, {100*m['hi']:.1f}]" if "lo" in m else "")


def results_table(s):
    head = "| Model | ID oracle % | OOD oracle % | OOD batch-confidence % | OOD fixed K=n % [95% interval] |\n|---|---:|---:|---:|---:|"
    rows = [head]
    for name in (*s["arms"], "untrained"):
        arm = s.get("untrained") if name == "untrained" else s["arms"][name]
        if arm:
            m = arm["metrics"]
            cells = [_pct(m[key]) for key in ("id_oracle", "ood_oracle", "ood_max_confidence", "ood_fixed_n")]
            rows.append("| " + name + " | " + " | ".join(cells) + " |")
    if "chance" in s:
        rows.append(f"| Fair independent bits, EOS supplied (single guess) | — | — | — | {100*s['chance']['ood']:.3g} |")
    rows.append("\nEach length has equal weight. The fixed-rule interval combines simultaneous exact binomial bounds. "
                "Oracle and batch-confidence scores describe the evaluated examples. Counts appear in the per-length table.")
    for name, arm in s["arms"].items():
        c = arm["curvature"]
        if c:
            sharp = ", ".join(f"ρ={v['rho']:g}: {v['sharpness']:.4g} ± {v['se']:.2g}" for v in c["adaptive_sharpness"])
            rows.append(f"\n**{name}** on the fixed {arm['config']['measure']['batch']}-problem batch. "
                        f"Loss {c['loss']:.4g}, trace {c['trace']['trace']:.4g} ± {c['trace']['se']:.2g} probe SE, "
                        f"signed dominant eigenvalue estimate {c['dominant_eigenvalue']:.4g}, "
                        f"model Fisher trace {c['fisher_trace']['trace']:.4g} ± {c['fisher_trace']['se']:.2g} sampling SE, "
                        f"weight norm {c['weight_norms']['total']:.4g}. Adaptive sharpness (mean ± noise-draw SE) {sharp}.")
        rows.append(f"\n{name} recorded training time including periodic diagnostics: {arm['train']['train_seconds']/3600:.3g} h.")
    return "\n".join(rows)


def headline(s):
    what = s["experiment"]
    if "comparison" not in s:
        state = []
        for arm, status in s["run_status"].items():
            label = "baseline" if arm == "baseline" else "Hutchinson"
            if status["state"] == "absent":
                line = f"No {label} run is recorded"
            elif status["state"] == "partial_training" and status["last_logged_step"] is not None:
                line = (f"The {label} history ends at logged step {status['last_logged_step']:,} "
                        f"of a {status['steps_planned']:,}-step budget")
                if status["quick_steps"]:
                    line += " and includes interim accuracy checks"
            else:
                line = f"The {label} run is marked {status['state'].replace('_', ' ')}"
                if status["reason"]:
                    line += f" ({status['reason']})"
            state.append(line)
        if len(s["arms"]) == 2:
            state.append("Pairing could not be established because " + s["pairing"]["reason"])
        return (f"**{what} accuracy comparison and final trace check are unevaluated.** "
                + ". ".join(state) + ".")
    c = s["comparison"]
    counts = c["oracle_discordances"]
    line = (f"**{what}.** Paired mean OOD oracle difference {c['delta_ood_oracle_points']:+.2f} percentage points. "
            f"Batch-confidence difference {c['delta_ood_max_confidence_points']:+.2f}, fixed K=n difference "
            f"{c['delta_ood_fixed_n_points']:+.2f}, ID oracle difference {c['delta_id_oracle_points']:+.2f}. "
            f"Only baseline correct on {counts['only_a']} OOD cases, only Hutchinson on {counts['only_b']}.")
    if "reading" in c:
        line += f" Registered single-seed heuristic ({c['threshold_points']:.1f} points): {c['reading']}."
    if "manipulation_check" in s:
        line += " Fixed-batch trace criterion " + ("passed." if s["manipulation_check"]["passed"] else "did not pass.")
    else:
        line += " Final trace manipulation check unevaluated."
    return line


def per_length_table(s):
    rows = ["| Arm | Bits | Oracle % (K) | Batch-confidence % (K) | Fixed K=n % [Wilson 95%] | Problems |",
            "|---|---:|---:|---:|---:|---:|"]
    for arm, rec in s["arms"].items():
        for n, row in sorted(rec["per_length"].items(), key=lambda item: int(item[0])):
            cells = [f"{_pct(row[r])} ({row[r]['k']})" for r in ("oracle", "max_confidence")]
            rows.append(f"| {arm} | {n} | " + " | ".join(cells) + f" | {_pct(row['fixed_n'])} | {row['fixed_n']['n']} |")
    return "\n".join(rows)


def curvature_table(s: dict) -> str:
    arms = [a for a in s["arms"] if s["arms"][a]["curvature"]]
    if not arms:
        return "Curvature measurements pending."
    groups = list(s["arms"][arms[0]]["curvature"]["trace"]["groups"])
    head = "| group | " + " | ".join(f"{a}: tr(H_group) ± SE | {a}: weight norm" for a in arms) + " |\n|" + \
           "---|" * (1 + 2 * len(arms)) + "\n"
    rows = []
    for g in groups:
        cells = []
        for a in arms:
            c = s["arms"][a]["curvature"]
            cells.append(f"{c['trace']['groups'][g]['trace']:.4g} ± {c['trace']['groups'][g]['se']:.2g} "
                         f"| {c['weight_norms'][g]:.4g}")
        rows.append(f"| {g} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def loop_table(s: dict) -> str:
    arms = [a for a in s["arms"] if s["arms"][a]["curvature"]]
    if not arms:
        return "Loop-count study pending."
    head = "| K | " + " | ".join(f"{a}: loss | {a}: tr(H) ± SE | {a}: dominant eig." for a in arms) + " |\n|" + \
           "---|" * (1 + 3 * len(arms)) + "\n"
    rows = []
    lookup = {a: {r["k"]: r for r in s["arms"][a]["curvature"]["loop_study"]} for a in arms}
    for k in sorted(set().union(*(set(d) for d in lookup.values()))):
        cells = []
        for a in arms:
            r = lookup[a].get(k)
            cells.append(f"{r['loss']:.3g} | {r['trace']['trace']:.4g} ± {r['trace']['se']:.2g} | "
                         f"{r['dominant_eigenvalue']:.4g}" if r else "— | — | —")
        rows.append(f"| {k} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def examples_table(s: dict) -> str:
    rows = ["| model | n | input (a+b>) | target (sum, EOS) | prediction | correct |", "|---|---|---|---|---|---|"]
    for arm, a in s["arms"].items():
        for n, exs in (a["examples"] or {}).items():
            for e in exs[:2]:
                ok = "yes" if e["prediction"] == e["target"] else "no"
                rows.append(f"| {arm} | {n} | `{e['input']}` | `{e['target']}` | `{e['prediction']}` | {ok} |")
    return "\n".join(rows)


# ----------------------------------------------------------------------------- figures

def make_figures(s, out_dir):
    paths = []
    if s["histories"]:
        paths.append(plots.training_curves(s["histories"], out_dir))
    if s["arms"]:
        curves, grids = {}, {}
        series = dict(s["arms"])
        if "untrained" in s:
            series["untrained"] = s["untrained"]
        for arm, data in series.items():
            ev = data["per_length"]
            lengths = sorted(map(int, ev))
            curves[arm] = (lengths, [ev[str(n)]["oracle"]["acc"] for n in lengths])
            if arm in s["arms"]:
                kmax = max(len(row["acc_per_k"]) for row in ev.values())
                grid = [(ev[str(n)]["acc_per_k"] + [float("nan")] * kmax)[:kmax] for n in lengths]
                grids[arm] = (lengths, kmax, grid)
        limits = {a["config"]["train"]["max_train_len"] for a in s["arms"].values()}
        paths.append(plots.accuracy_vs_length(curves, next(iter(limits)) if len(limits) == 1 else None, out_dir))
        paths.append(plots.loop_heatmaps(grids, out_dir))
    curv = {a: x["curvature"] for a, x in s["arms"].items() if x["curvature"]}
    if curv:
        paths.append(plots.curvature_groups({a: c["trace"]["groups"] for a, c in curv.items()}, out_dir))
        paths.append(plots.loop_curvature({a: c["loop_study"] for a, c in curv.items()}, out_dir))
    return paths


# ----------------------------------------------------------------------------- injection

def inject(path: str, key: str, content: str) -> bool:
    """Replace the block between BEGIN/END markers for `key`; returns whether the file changed."""
    if not os.path.exists(path):
        return False
    with open(path, encoding="utf-8") as f:
        text = f.read()
    pattern = re.compile(rf"(<!-- BEGIN:{key} -->)(.*?)(<!-- END:{key} -->)", re.S)
    if not pattern.search(text):
        return False
    new = pattern.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)
    if new != text:
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new)
    return new != text


def _strip_private(s: dict) -> dict:
    # Drop bulky per-problem arrays and private eval copies before writing the JSON summary.
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if k not in ("correct", "_eval")}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj
    return clean(s)


def build_pdf(md_path: str = "reports/paper/paper.md") -> bool:
    """Render the paper with pandoc + XeLaTeX (https://pandoc.org/MANUAL.html#creating-a-pdf).

    Cambria / Consolas cover the Greek letters and math symbols used in the text on Windows.
    Returns False (and leaves any old PDF) if pandoc or a TeX engine is unavailable.
    """
    if shutil.which("pandoc") is None:
        print("pandoc not found; skipping PDF")
        return False
    folder, name = os.path.split(md_path)
    # lists_without_preceding_blankline: render "**Heading.**\n- item" as a list, as GitHub does
    # (https://pandoc.org/MANUAL.html#extension-lists_without_preceding_blankline).
    cmd = ["pandoc", name, "-f", "markdown+lists_without_preceding_blankline",
           "-o", name.replace(".md", ".pdf"), "--pdf-engine=xelatex",
           "-V", "geometry:margin=2.2cm", "-V", "mainfont=Cambria", "-V", "mainfontoptions=Ligatures=NoCommon", "-V", "monofont=Consolas",
           "-V", "mathfont=Cambria Math", "-V", "colorlinks=true", "-V", "fontsize=10pt",
           "--resource-path=" + os.pathsep.join((".", ".."))]
    res = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print("PDF build failed:", res.stderr[-2000:])
    return res.returncode == 0


def summary_tables(s):
    """Render the same data and uncertainty policy for every publication surface."""
    return {"headline": headline(s), "results_table": results_table(s) if s["arms"] else "Final accuracy measurements pending.",
              "per_length_table": per_length_table(s) if s["arms"] else "Per-length final scores pending.",
              "curvature_table": curvature_table(s), "loop_table": loop_table(s),
              "examples_table": examples_table(s) if s["arms"] else "Final output examples pending."}


def build_report(results_dir: str = "results", prefix: str = "stage1_", write_docs: bool = True) -> dict:
    s = summarize(results_dir, prefix)
    fig_dir = os.path.join("reports", "figures") if write_docs else os.path.join(results_dir, "figures")
    figures = make_figures(s, fig_dir)
    blocks = summary_tables(s)
    public = _strip_private(s)
    with open(os.path.join(results_dir, f"{prefix}summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(public | {"tables": blocks}, f, indent=2)
    if write_docs:
        from looped_hessian.audit import evidence_block, interim_block, operator_block, power_block, reconstruct
        facts = reconstruct()
        Path("results/audit_facts.json").write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8", newline="\n")
        smoke = summarize("results/audit-2026-11-26", "smoke_")
        replay = _strip_private(smoke) | {"tables": summary_tables(smoke)}
        Path("results/audited_smoke_summary.json").write_text(json.dumps(replay, indent=2) + "\n", encoding="utf-8", newline="\n")
        blocks.update(audit_evidence=evidence_block(facts), audit_power=power_block(facts), audit_interim=interim_block(facts),
                      operator_check=operator_block(facts["layernorm_operator"]))
        for doc in DOCS:
            for key, content in blocks.items():
                inject(doc, key, content)
            # Only link evidence-backed figures. Paths are relative to each document's directory.
            images = []
            for figure in figures:
                relative = Path(os.path.relpath(figure, Path(doc).parent)).as_posix()
                title = {"training_curves": "Logged training loss, trace and interim oracle accuracy",
                         "accuracy_vs_length": "Final per-length oracle accuracy", "loop_heatmaps": "Accuracy by loop count",
                         "curvature_groups": "Group trace estimates", "loop_curvature": "Curvature by loop count"}[Path(figure).stem]
                images.append(f"![{title}]({relative})")
            inject(doc, "figures", "\n\n".join(images) or "No recorded figure data.")
        if not build_pdf():
            raise RuntimeError("paper PDF regeneration failed")
        # The static site reads the same summary and figures.
        if os.path.isdir("site"):
            os.makedirs("site/data", exist_ok=True)
            shutil.copy(os.path.join(results_dir, f"{prefix}summary.json"), "site/data/summary.json")
            for p in figures:
                shutil.copy(p, os.path.join("site", "data", os.path.basename(p)))
    print(blocks["headline"])
    return s
