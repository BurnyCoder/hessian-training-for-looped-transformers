"""Regenerate every reported number, table and figure from results/ (one command: main.py report).

Local: loads each run folder (config, history, evaluation, curvature, examples), computes
the preregistered Stage 1 quantities (primary metric = mean OOD exact-match at the oracle
loop count; secondary stopping rules; untrained and chance baselines; paired McNemar on the
same test problems; the preregistered reading of the single-seed difference; the tr(H)
manipulation check), writes them to `results/<prefix>summary.json`, renders the figures and
replaces the text between `<!-- BEGIN:key -->` and `<!-- END:key -->` markers in the docs.

Global: no number in README.md, reports/ or the paper is typed by hand; they are all
injected here, so rerunning this module after new runs updates every document consistently.
"""

import json
import os
import re
import shutil
import subprocess

from looped_hessian import plots
from looped_hessian.config import EvalConfig, TrainConfig
from looped_hessian.stats import SUGGESTIVE_THRESHOLD, paired_mcnemar, stage1_reading, wilson

ARMS = ("baseline", "hutchinson")
DOCS = ("README.md", "reports/stage1_experiment.md", "reports/all_experiments.md", "reports/summary.md",
        "reports/paper/paper.md")


def _load(path, default=None, jsonl=False):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()] if jsonl else json.load(f)


def load_run(run_dir: str) -> dict:
    return {
        "config": _load(os.path.join(run_dir, "config.json")),
        "history": _load(os.path.join(run_dir, "history.jsonl"), [], jsonl=True),
        "train": _load(os.path.join(run_dir, "train_summary.json")),
        "eval": _load(os.path.join(run_dir, "eval.json")),
        "eval_init": _load(os.path.join(run_dir, "eval_init.json")),
        "curvature": _load(os.path.join(run_dir, "curvature.json")),
        "examples": _load(os.path.join(run_dir, "examples.json")),
    }


def pooled(ev: dict, lengths, rule: str) -> dict:
    """Mean accuracy over lengths (equal problems per length => pooled accuracy) + Wilson CI."""
    correct = [c for n in lengths for c in ev[str(n)][rule]["correct"]]
    k, total = sum(correct), len(correct)
    lo, hi = wilson(k, total)
    return {"acc": k / total, "lo": lo, "hi": hi, "k": k, "n": total, "correct": correct}


def _metrics(ev: dict, id_lengths, ood_lengths) -> dict:
    return {"id_oracle": pooled(ev, id_lengths, "oracle"),
            "ood_oracle": pooled(ev, ood_lengths, "oracle"),
            "ood_max_confidence": pooled(ev, ood_lengths, "max_confidence"),
            "ood_fixed_n": pooled(ev, ood_lengths, "fixed_n")}


def summarize(results_dir: str, prefix: str) -> dict:
    runs = {arm: load_run(os.path.join(results_dir, prefix + arm)) for arm in ARMS}
    runs = {arm: r for arm, r in runs.items() if r["eval"] is not None}
    ec = EvalConfig()
    lengths = sorted(int(n) for n in next(iter(runs.values()))["eval"]) if runs else []
    id_lengths = [n for n in lengths if n <= TrainConfig().max_train_len]
    ood_lengths = [n for n in lengths if n > TrainConfig().max_train_len]
    out = {"id_lengths": id_lengths, "ood_lengths": ood_lengths, "arms": {}, "eval_config": ec.__dict__}
    for arm, r in runs.items():
        out["arms"][arm] = {"metrics": _metrics(r["eval"], id_lengths, ood_lengths),
                            "train": r["train"], "curvature": r["curvature"], "examples": r["examples"],
                            "config": r["config"]}
    if runs:
        # Both arms share seed 0, so their init checkpoints are identical; use the first one.
        init = next(iter(runs.values()))["eval_init"]
        out["untrained"] = {"metrics": _metrics(init, id_lengths, ood_lengths)}
        out["chance"] = {"ood": sum(2.0 ** -(n + 1) for n in ood_lengths) / len(ood_lengths),
                         "id": sum(2.0 ** -(n + 1) for n in id_lengths) / len(id_lengths)}
    if all(a in out["arms"] for a in ARMS):
        b, h = (out["arms"][a]["metrics"] for a in ARMS)
        delta = 100 * (h["ood_oracle"]["acc"] - b["ood_oracle"]["acc"])
        out["comparison"] = {
            "delta_ood_oracle_points": delta,
            "delta_ood_max_confidence_points": 100 * (h["ood_max_confidence"]["acc"] - b["ood_max_confidence"]["acc"]),
            "delta_ood_fixed_n_points": 100 * (h["ood_fixed_n"]["acc"] - b["ood_fixed_n"]["acc"]),
            "delta_id_points": 100 * (h["id_oracle"]["acc"] - b["id_oracle"]["acc"]),
            "reading": stage1_reading(delta),
            "threshold_points": SUGGESTIVE_THRESHOLD,
            "mcnemar_ood_oracle": paired_mcnemar(b["ood_oracle"]["correct"], h["ood_oracle"]["correct"]),
        }
        cb, ch = (out["arms"][a]["curvature"] for a in ARMS)
        if cb and ch:
            tb, th = cb["trace"], ch["trace"]
            separated = th["trace"] + 3 * th["se"] < tb["trace"] - 3 * tb["se"]
            out["manipulation_check"] = {
                "trace_baseline": tb["trace"], "trace_baseline_se": tb["se"],
                "trace_hutchinson": th["trace"], "trace_hutchinson_se": th["se"],
                "passed": bool(separated),
                "weight_norm_ratio": ch["weight_norms"]["total"] / cb["weight_norms"]["total"],
                "dominant_eigenvalue_ratio": ch["dominant_eigenvalue"] / cb["dominant_eigenvalue"],
            }
    return out


# ----------------------------------------------------------------------------- tables

def _pct(m: dict) -> str:
    return f"{100 * m['acc']:.1f} [{100 * m['lo']:.1f}, {100 * m['hi']:.1f}]"


def results_table(s: dict) -> str:
    head = ("| Model | ID exact match % (n=1–19) | **OOD exact match %, oracle K** (primary) | OOD %, max-confidence K "
            "| OOD %, K=n | tr(H) ± SE | dominant eigenvalue | Fisher trace ± SE | adaptive sharpness ρ=0.01 / 0.02 / 0.05 "
            "| weight norm | loss on measurement batch | training wall-clock (incl. periodic probes) |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    # Measurement-batch size as recorded in the runs' own config.json (not a hard-coded number).
    batch = next(iter(s["arms"].values()))["config"]["measure"]["batch"]
    if "chance" in s:
        rows.append(f"| chance (guess n+1 bits) | {100 * s['chance']['id']:.2g} | {100 * s['chance']['ood']:.2g} "
                    f"| – | – | – | – | – | – | – | – | – |")
        m = s["untrained"]["metrics"]
        rows.append(f"| untrained (seed-0 init) | {_pct(m['id_oracle'])} | {_pct(m['ood_oracle'])} "
                    f"| {_pct(m['ood_max_confidence'])} | {_pct(m['ood_fixed_n'])} | – | – | – | – | – | – | – |")
    for arm, a in s["arms"].items():
        m, c, t = a["metrics"], a["curvature"], a["train"]
        if c:
            sh = " / ".join(f"{x['sharpness']:.3g}" for x in c["adaptive_sharpness"])
            curv = (f"{c['trace']['trace']:.4g} ± {c['trace']['se']:.2g} | {c['dominant_eigenvalue']:.4g} "
                    f"| {c['fisher_trace']['trace']:.4g} ± {c['fisher_trace']['se']:.2g} | {sh} "
                    f"| {c['weight_norms']['total']:.4g} | {c['loss']:.3g}")
        else:
            curv = "– | – | – | – | – | –"
        # Wall-clock for both arms (the baseline predates the step_seconds field), flagged if diverged.
        hours = (f"{t['train_seconds'] / 3600:.2f} h"
                 + ("" if t.get("status", "completed") == "completed" else f" ({t['status'].upper()})")) if t else "–"
        rows.append(f"| {plots.STYLE[arm]['label']} | {_pct(m['id_oracle'])} | **{_pct(m['ood_oracle'])}** "
                    f"| {_pct(m['ood_max_confidence'])} | {_pct(m['ood_fixed_n'])} | {curv} | {hours} |")
    note = ("\n\nAccuracies: exact match over all n+1 sum bits and EOS, pooled over lengths (1000 problems per "
            "length), with 95% Wilson intervals that cover test-set sampling only, not seed-to-seed variance. "
            "The oracle K is chosen on the same test problems, so oracle accuracies are optimistic and their "
            "intervals too narrow; the label-free max-confidence column has no such selection. "
            f"Curvature metrics: fixed {batch}-problem training-distribution batch, read out at K = n.")
    return head + "\n".join(rows) + note


def headline(s: dict) -> str:
    if "comparison" not in s:
        done = ", ".join(s["arms"]) or "none"
        return f"Stage 1 is in progress (finished arms: {done})."
    c, b, h = s["comparison"], s["arms"]["baseline"]["metrics"], s["arms"]["hutchinson"]["metrics"]
    mc = c["mcnemar_ood_oracle"]
    text = (f"**Stage 1 (one seed per arm):** mean OOD exact match at the oracle loop count was "
            f"{_pct(b['ood_oracle'])}% for the plain looped transformer and {_pct(h['ood_oracle'])}% with the "
            f"Hutchinson tr(H) penalty (Δ = {c['delta_ood_oracle_points']:+.1f} points; label-free max-confidence "
            f"stopping Δ = {c['delta_ood_max_confidence_points']:+.1f}; K = n Δ = {c['delta_ood_fixed_n_points']:+.1f}; "
            f"ID Δ = {c['delta_id_points']:+.1f}). Under the preregistered rule (|Δ| > {c['threshold_points']:.1f} "
            f"points needed with one seed per arm) this reads as **{c['reading']}**. "
            f"Paired McNemar on the same {b['ood_oracle']['n']} OOD problems: only-baseline correct {mc['only_a']}, "
            f"only-Hutchinson correct {mc['only_b']}, p = {mc['p_value']:.3g} (test-set sampling only).")
    if "manipulation_check" in s:
        m = s["manipulation_check"]
        text += (f" Manipulation check (final tr(H) lower with non-overlapping ±3 SE): "
                 f"{'**passed**' if m['passed'] else '**not passed**'} "
                 f"(baseline {m['trace_baseline']:.4g} ± {m['trace_baseline_se']:.2g}, Hutchinson "
                 f"{m['trace_hutchinson']:.4g} ± {m['trace_hutchinson_se']:.2g}; weight-norm ratio "
                 f"{m['weight_norm_ratio']:.3f}, dominant-eigenvalue ratio {m['dominant_eigenvalue_ratio']:.3f}).")
    return text


def per_length_table(s: dict) -> str:
    arms = list(s["arms"])
    head = "| n | " + " | ".join(f"{a} oracle % (K)" for a in arms) + " | " + \
           " | ".join(f"{a} max-conf % (K)" for a in arms) + " |\n|" + "---|" * (1 + 2 * len(arms)) + "\n"
    rows = []
    for n in s["id_lengths"] + s["ood_lengths"]:
        cells = []
        for rule in ("oracle", "max_confidence"):
            for a in arms:
                ev = s["arms"][a]["_eval"][str(n)][rule]
                cells.append(f"{100 * ev['acc']:.1f} ({ev['k']})")
        tag = " (OOD)" if n in s["ood_lengths"] else ""
        rows.append(f"| {n}{tag} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


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
    for i, row in enumerate(s["arms"][arms[0]]["curvature"]["loop_study"]):
        cells = []
        for a in arms:
            r = s["arms"][a]["curvature"]["loop_study"][i]
            cells.append(f"{r['loss']:.3g} | {r['trace']['trace']:.4g} ± {r['trace']['se']:.2g} "
                         f"| {r['dominant_eigenvalue']:.4g}")
        rows.append(f"| {row['k']} | " + " | ".join(cells) + " |")
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

def make_figures(s: dict, runs: dict, out_dir: str) -> list[str]:
    paths = []
    hist = {a: r["history"] for a, r in runs.items() if r["history"]}
    if hist:
        paths.append(plots.training_curves(hist, out_dir))
    if s["arms"]:
        lengths = s["id_lengths"] + s["ood_lengths"]
        curves = {}
        for arm, a in list(s["arms"].items()) + [("untrained", {"_eval": runs[next(iter(runs))]["eval_init"]})]:
            ev = a["_eval"]
            acc = [ev[str(n)]["oracle"]["acc"] for n in lengths]
            ci = [wilson(sum(ev[str(n)]["oracle"]["correct"]), len(ev[str(n)]["oracle"]["correct"])) for n in lengths]
            curves[arm] = (lengths, acc, [c[0] for c in ci], [c[1] for c in ci])
        chance = (lengths, [2.0 ** -(n + 1) for n in lengths])
        per_len = len(next(iter(s["arms"].values()))["_eval"][str(lengths[0])]["oracle"]["correct"])
        paths.append(plots.accuracy_vs_length(curves, TrainConfig().max_train_len, out_dir, per_len, chance))
        grids = {}
        for arm, a in s["arms"].items():
            kmax = max(len(a["_eval"][str(n)]["acc_per_k"]) for n in lengths)
            grid = [[(a["_eval"][str(n)]["acc_per_k"] + [float("nan")] * kmax)[:kmax]] for n in lengths]
            grids[arm] = (lengths, kmax, [g[0] for g in grid])
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
    with open(path, "w", encoding="utf-8") as f:
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
           "-V", "geometry:margin=2.2cm", "-V", "mainfont=Cambria", "-V", "monofont=Consolas",
           "-V", "mathfont=Cambria Math", "-V", "colorlinks=true", "-V", "fontsize=10pt",
           "--resource-path=.:.."]
    res = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode != 0:
        print("PDF build failed:", res.stderr[-2000:])
    return res.returncode == 0


def build_report(results_dir: str = "results", prefix: str = "stage1_", write_docs: bool = True) -> dict:
    s = summarize(results_dir, prefix)
    runs = {arm: load_run(os.path.join(results_dir, prefix + arm)) for arm in s["arms"]}
    for arm in s["arms"]:
        s["arms"][arm]["_eval"] = runs[arm]["eval"]
    fig_dir = os.path.join("reports", "figures") if write_docs else os.path.join(results_dir, "figures")
    figures = make_figures(s, runs, fig_dir) if runs else []
    blocks = {"headline": headline(s), "results_table": results_table(s) if s["arms"] else "Pending.",
              "per_length_table": per_length_table(s) if s["arms"] else "Pending.",
              "curvature_table": curvature_table(s), "loop_table": loop_table(s),
              "examples_table": examples_table(s) if s["arms"] else "Pending."}
    public = _strip_private(s)
    # Compact per-length/per-loop accuracies and training history for the interactive site.
    for arm in s["arms"]:
        public["arms"][arm]["per_length"] = _strip_private({"x": runs[arm]["eval"]})["x"]
        public["arms"][arm]["history"] = runs[arm]["history"]
    if runs:
        public["untrained"]["per_length"] = _strip_private({"x": next(iter(runs.values()))["eval_init"]})["x"]
    with open(os.path.join(results_dir, f"{prefix}summary.json"), "w", encoding="utf-8") as f:
        json.dump(public | {"tables": blocks}, f, indent=2)
    if write_docs:
        for doc in DOCS:
            for key, content in blocks.items():
                inject(doc, key, content)
        build_pdf()
        # The static site reads the same summary and figures.
        if os.path.isdir("site"):
            os.makedirs("site/data", exist_ok=True)
            shutil.copy(os.path.join(results_dir, f"{prefix}summary.json"), "site/data/summary.json")
            for p in figures:
                shutil.copy(p, os.path.join("site", "data", os.path.basename(p)))
    print(blocks["headline"])
    return s
