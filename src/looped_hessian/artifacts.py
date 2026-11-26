"""Reconcile saved run metadata and validate evaluations before making comparisons."""

import json
import math
import re
from pathlib import Path

from looped_hessian.data import evaluation_count
from looped_hessian.stats import stratified_interval, wilson


RULES = ("oracle", "max_confidence", "fixed_n")


def load_run(run_dir) -> dict:
    """Read only experiment artifacts. Missing phases remain distinguishable."""
    root = Path(run_dir)
    run = {}
    for key, name in (("config", "config"), ("train", "train_summary"), ("eval", "eval"),
                      ("eval_init", "eval_init"), ("curvature", "curvature"), ("examples", "examples")):
        path = root / (name + ".json")
        run[key] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    history = root / "history.jsonl"
    run["history"] = ([json.loads(line) for line in history.read_text(encoding="utf-8").splitlines() if line]
                      if history.is_file() else [])
    return run


def validate_eval(ev, cfg):
    """Check length coverage, decisions, denominators and agreement with per-K records."""
    lengths = cfg["id_lengths"] + cfg["ood_lengths"]
    if not cfg["id_lengths"] or not cfg["ood_lengths"] or len(set(lengths)) != len(lengths):
        raise ValueError("the length strata must be nonempty and disjoint")
    if ev is None or set(ev) != {str(n) for n in lengths}:
        raise ValueError("recorded and configured evaluation lengths differ")
    for n in lengths:
        row = ev[str(n)]
        count = evaluation_count(cfg["n_per_length"], n)
        acc, conf = row["acc_per_k"], row["confidence_per_k"]
        if len(acc) != n + cfg["extra_loops"] or len(conf) != len(acc):
            raise ValueError(f"n={n}: loop horizon differs")
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in acc) or not all(map(math.isfinite, conf)):
            raise ValueError(f"n={n}: invalid loop profile")
        choices = (acc.index(max(acc)) + 1, conf.index(max(conf)) + 1, n)
        for rule, k in zip(RULES, choices):
            result = row[rule]
            flags = result["correct"]
            if len(flags) != count or any(type(v) not in (bool, int) or v not in (0, 1) for v in flags):
                raise ValueError(f"n={n}: incomplete or invalid correctness vector")
            expected = sum(flags) / len(flags)
            if result["k"] != k or any(not math.isclose(x, expected, abs_tol=1e-6)
                                        for x in (result["acc"], acc[k-1])):
                raise ValueError(f"n={n}: {rule} disagrees with the decision or score")
        for i, rule in enumerate(RULES):
            for other in RULES[:i]:
                if row[rule]["k"] == row[other]["k"] and row[rule]["correct"] != row[other]["correct"]:
                    raise ValueError(f"n={n}: same-depth rules disagree about individual outcomes")


def run_status(run, arm):
    """Only explicit completion at the configured budget qualifies a final evaluation."""
    hist, cfg, train = run["history"], run["config"], run["train"]
    status = {"state": "absent", "reason": "", "last_logged_step": max((r["step"] for r in hist), default=None),
              "quick_steps": [r["step"] for r in hist if any(k.startswith("acc_n") for k in r)],
              "steps_planned": cfg["train"]["steps"] if cfg else None}
    if not cfg:
        if any(run.values()):
            status.update(state="incomplete", reason="missing run configuration")
    elif cfg["train"]["arm"] != arm:
        status.update(state="incomplete", reason="arm name conflicts with run configuration")
    elif not train:
        status.update(state="partial_training", reason="no training completion record")
    elif train.get("status", "completed") != "completed":
        status.update(state=train["status"], reason="training did not complete successfully")
    elif train.get("name") != cfg["name"] or train.get("steps_done") != cfg["train"]["steps"]:
        status.update(state="partial_training", reason="completion record differs from the planned run")
    elif run["eval"] is None:
        status.update(state="awaiting_evaluation", reason="missing final evaluation")
    else:
        try:
            validate_eval(run["eval"], cfg["eval"])
            if run["eval_init"] is not None:
                validate_eval(run["eval_init"], cfg["eval"])
            status["state"] = "evaluated"
        except (ValueError, KeyError, TypeError) as error:
            status.update(state="incomplete_evaluation", reason=str(error))
    return status


def pairing(baseline, treatment):
    """Match the training design, ordered test problems and the initial decision records."""
    a, b = baseline["config"], treatment["config"]
    exclude = {"arm", "sub_batch", "lam", "penalty_every"}
    shared = lambda cfg: {key: value for key, value in cfg["train"].items() if key not in exclude}
    if a["model"] != b["model"] or shared(a) != shared(b):
        return False, "model, training seed or shared training settings differ"
    if a["eval"] != b["eval"]:
        return False, "evaluation seeds, lengths, counts or settings differ"
    if baseline["eval_init"] is None or treatment["eval_init"] is None:
        return False, "initialization evaluation missing"
    hashed = []
    for length in baseline["eval"]:
        rows = [r[phase][length] for r in (baseline, treatment) for phase in ("eval", "eval_init")]
        hashes = [row.get("dataset_sha256") for row in rows]
        if any(x is not None for x in hashes):
            if not all(isinstance(h, str) and re.fullmatch(r"[a-f0-9]{64}", h) and h == hashes[0] for h in hashes):
                return False, f"ordered input/target digests differ or are missing at n={length}"
            hashed.append(length)
        for rule in RULES:
            if rows[1][rule]["correct"] != rows[3][rule]["correct"] or rows[1][rule]["k"] != rows[3][rule]["k"]:
                return False, f"initial decision records differ at n={length}"
    if hashed and len(hashed) != len(baseline["eval"]):
        return False, "mixed digest coverage across lengths"
    evidence = ("ordered input/target SHA-256 digests" if hashed else
                "legacy deterministic generator, matching length seeds, counts and order")
    return True, evidence + "; shared settings and matching initial decisions"


def pooled(ev, lengths, rule):
    """The estimand weights lengths equally even if their observed counts differ."""
    flags = [ev[str(n)][rule]["correct"] for n in lengths]
    counts = [(sum(row), len(row)) for row in flags]
    acc = math.fsum(k / total for k, total in counts) / len(counts)
    result = {"acc": acc, "k": sum(k for k, _ in counts), "n": sum(n for _, n in counts),
              "correct": [v for row in flags for v in row]}
    if rule == "fixed_n":
        result["lo"], result["hi"] = stratified_interval(counts)
    return result


def metrics(ev, ec):
    return {"id_oracle": pooled(ev, ec["id_lengths"], "oracle"),
            **{"ood_" + rule: pooled(ev, ec["ood_lengths"], rule) for rule in RULES}}


def per_length(ev):
    """Build portable site data directly from binary outcomes, including immutable counts."""
    out = {}
    for n, row in ev.items():
        out[n] = {k: v for k, v in row.items() if k not in RULES}
        for rule in RULES:
            flags = row[rule]["correct"]
            k, total = sum(flags), len(flags)
            cell = {"k": row[rule]["k"], "n": total, "successes": k, "acc": k / total}
            if rule == "fixed_n":
                cell["lo"], cell["hi"] = wilson(k, total)
            out[n][rule] = cell
    return out
