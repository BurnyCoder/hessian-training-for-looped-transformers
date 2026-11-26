"""Model-free regression checks for artifact provenance, fixed strata and paired hypotheses."""

import copy
import hashlib
import json
from pathlib import Path
from shutil import copytree

import pytest

from looped_hessian import report, stats
from looped_hessian.stage2 import analyze_h1, analyze_h3, paired_power


ARCHIVE = Path(__file__).resolve().parents[1] / "results/audit-2026-11-26"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_archived_evidence_and_interim_status():
    for line in (ARCHIVE / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((ARCHIVE / name).read_bytes()).hexdigest() == digest
    result = report.summarize(ARCHIVE, "stage1_")
    assert result["run_status"]["baseline"]["last_logged_step"] == 6000
    assert result["run_status"]["hutchinson"]["state"] == "absent"
    assert not result["arms"] and "comparison" not in result and "manipulation_check" not in result


def test_replay_preserves_actual_counts_and_stopping_rules():
    historical = load_json(ARCHIVE / "historical_smoke_summary.json")
    assert historical["eval_config"]["n_per_length"] == 1000
    replay = report.summarize(ARCHIVE, "smoke_")
    assert replay["eval_config"]["n_per_length"] == 20
    assert replay["comparison"]["oracle_discordances"]["neither"] == 40
    for arm in replay["arms"].values():
        row = arm["per_length"]["1"]
        assert row["oracle"]["successes"] == 3 and row["oracle"]["n"] == 20
        assert "lo" not in row["oracle"] and "lo" not in row["max_confidence"]
        assert row["fixed_n"]["lo"] <= row["fixed_n"]["acc"] <= row["fixed_n"]["hi"]
        assert "lo" not in arm["metrics"]["ood_oracle"]
    assert "p_value" not in replay["comparison"]["oracle_discordances"]


@pytest.mark.parametrize("issue", ["diverged", "seed", "evaluation_seed", "missing_length", "digest", "short_training"])
def test_unqualified_runs_cannot_produce_a_comparison(tmp_path, issue):
    for arm in ("baseline", "hutchinson"):
        copytree(ARCHIVE / ("smoke_" + arm), tmp_path / ("smoke_" + arm))
    run = tmp_path / "smoke_hutchinson"
    name = {"diverged": "train_summary", "short_training": "train_summary", "seed": "config",
            "evaluation_seed": "config", "missing_length": "eval", "digest": "eval"}[issue]
    path = run / (name + ".json")
    artifact = load_json(path)
    if issue == "diverged":
        artifact["status"] = "diverged"
    elif issue == "short_training":
        artifact["steps_done"] -= 1
    elif issue == "seed":
        artifact["train"]["seed"] += 1
    elif issue == "evaluation_seed":
        artifact["eval"]["seed_base"] += 1
    elif issue == "digest":
        artifact["20"]["dataset_sha256"] = "a" * 64
    else:
        del artifact["20"]
    path.write_text(json.dumps(artifact), encoding="utf-8")
    summary = report.summarize(tmp_path, "smoke_")
    assert "comparison" not in summary and "manipulation_check" not in summary
    assert summary["pairing"]["reason"]


def test_equal_length_mean_and_unequal_counts():
    rows = {"1": {"fixed_n": {"correct": [1]}}, "2": {"fixed_n": {"correct": [0] * 5}}}
    fixed = report.pooled(rows, [1, 2], "fixed_n")
    assert fixed["acc"] == 0.5 and fixed["n"] == 6 and fixed["k"] == 1
    assert 0 < fixed["lo"] < fixed["acc"] < fixed["hi"] < 1
    with pytest.raises(ValueError):
        stats.paired_counts([1, 0], [1])


def test_complete_evaluation_with_different_registered_stratum_counts():
    from looped_hessian.artifacts import validate_eval
    cfg = {"id_lengths": [1], "ood_lengths": [2, 3], "extra_loops": 0, "n_per_length": {"1": 4, "2": 1, "3": 5}}
    ev = {}
    for n, flags in ((1, [0]*4), (2, [1]), (3, [0]*5)):
        acc = sum(flags) / len(flags)
        cell = {"correct": flags, "k": 1, "acc": acc}
        ev[str(n)] = {"acc_per_k": [acc]*n, "confidence_per_k": [-1.0]*n, "oracle": dict(cell),
                      "max_confidence": dict(cell), "fixed_n": dict(cell, k=n)}
    validate_eval(ev, cfg)
    assert report.pooled(ev, cfg["ood_lengths"], "oracle")["acc"] == .5
    cfg["n_per_length"]["3"] += 1
    with pytest.raises(ValueError, match="correctness vector"):
        validate_eval(ev, cfg)


def test_stage2_small_positive_gain_can_rule_out_two_points():
    # Fractions represent paired seed-level exact-match scores, not individual examples.
    baseline = {i: (0.25 + 0.01 * i, 0.20 + 0.01 * i) for i in range(10)}
    positive = {i: (x + 0.008 + i * 0.0004, y + 0.005) for i, (x, y) in baseline.items()}
    rows = analyze_h1({"baseline": baseline, "hutchinson": positive, "nso": positive})
    assert rows["hutchinson"]["positive_gain"] and rows["hutchinson"]["rules_out_two_points"]
    assert rows["hutchinson"]["ci95_points"][1] < 2
    mismatched = copy.deepcopy(positive)
    del mismatched[9]
    with pytest.raises(ValueError):
        analyze_h1({"baseline": baseline, "hutchinson": positive, "nso": mismatched})
    assert paired_power(0.5) < paired_power(0) < paired_power(-0.5)


def test_per_length_oracle_and_batch_confidence_on_saved_style_matrix():
    import torch
    from looped_hessian.evaluate import summarize_length
    outcomes = torch.tensor([[True, False], [False, True], [False, False]])
    log_confidence = torch.tensor([[-.1, -3.], [-1., -1.], [-2., -.5]])
    result = summarize_length(outcomes, log_confidence, n=3)
    assert result["oracle"] == {"k": 1, "acc": .5, "correct": [1, 0]}
    assert outcomes.any(dim=0).all()  # A separate oracle for every example would score 100%.
    assert result["max_confidence"] == {"k": 2, "acc": .5, "correct": [0, 1]}
    assert result["fixed_n"] == {"k": 3, "acc": 0., "correct": [0, 0]}


def test_paired_variance_does_not_break_seed_alignment():
    import numpy as np
    base = {i: (i/20, i/20) for i in range(10)}
    tx = {i: (x/2, y) for i, (x, y) in reversed(base.items())}
    out = analyze_h3({"baseline": base, "hutchinson": tx, "nso": base})
    expected = -.75 * np.var([100*x for x, _ in base.values()], ddof=1)
    assert out["hutchinson"]["difference_points_squared"] == pytest.approx(expected)
    assert out["hutchinson"]["ci95_points_squared"][1] < 0
    assert len(out["hutchinson"]["leave_one_pair_out"]) == 10
    assert out["nso"]["ci95_points_squared"] is None


def test_probe_target_token_weighting_and_clipping_counterexamples():
    from fractions import Fraction
    from itertools import product
    # L(theta) = theta_1^3/6 + theta_1*theta_2^2/2 gives H=[[x,y],[y,x]].
    # Enumerating signs gives both tr(H) and its differentiated expectation exactly.
    signs = list(product((-1, 1), repeat=2))
    assert sum(2*3 + 2*5*a*b for a, b in signs) / len(signs) == 6
    assert (2, sum(2*a*b for a, b in signs) / len(signs)) == (2, 0)
    # Two equal-probability problems with 3 and 5 tokens and per-token curvatures 0 and 1.
    assert Fraction(0 + 1, 2) != Fraction(5, 3 + 5)
    # Clipping a unit raw penalty each step differs from clipping one eightfold impulse.
    assert sum(min(1, g) for g in [1]*8) != sum(min(1, g) for g in [8]+[0]*7)


def test_operator_fixture_is_reproducible_without_a_network():
    from looped_hessian.audit import operator_check
    original = load_json(ARCHIVE / "layernorm_operator.json")
    current = operator_check()
    fd = current["composite"]["finite_differences"]["1e-05"]
    assert current["composite"]["autograd"] == pytest.approx(fd, rel=1e-7)
    if current["torch"] == original["torch"]:
        for op in ("native", "composite"):
            assert current[op]["autograd"] == pytest.approx(original[op]["autograd"], rel=1e-7)


def test_one_evaluated_arm_and_missing_initial_controls(tmp_path):
    copytree(ARCHIVE/"smoke_baseline", tmp_path/"smoke_baseline")
    (tmp_path/"smoke_baseline/eval_init.json").unlink()
    result = report.summarize(tmp_path, "smoke_")
    assert list(result["arms"]) == ["baseline"] and "comparison" not in result and "untrained" not in result
    assert "baseline" in report.results_table(result)


def test_off_protocol_completed_pair_only_gets_descriptive_results(tmp_path):
    for arm in ("baseline", "hutchinson"):
        name = "stage1_" + arm
        out = tmp_path / name
        copytree(ARCHIVE / ("smoke_"+arm), out)
        for kind in ("config", "train_summary"):
            path = out / (kind+".json")
            data = load_json(path)
            data["name"] = name
            path.write_text(json.dumps(data), encoding="utf-8")
    result = report.summarize(tmp_path, "stage1_")
    assert result["pairing"]["verified"] and "reading" not in result["comparison"]
    assert "train.steps" in result["registration_deviations"]["accuracy"]
    assert "measure.probes" in result["registration_deviations"]["curvature"]
    assert "manipulation_check" not in result and result["trace_diagnostic"]["probe_count"] == 3


def test_future_smoke_streams_and_historical_input_prefixes():
    from looped_hessian.config import RunConfig, smoke_config
    from looped_hessian.audit import reconstruct
    full, small = RunConfig("stage1_baseline"), smoke_config("baseline")
    for kind, field in (("train", "seed"), ("train", "quick_seed_base"), ("train", "curvature_data_seed"),
                        ("train", "curvature_probe_seed"), ("eval", "seed_base"), ("eval", "example_seed_base"),
                        ("measure", "seed")):
        assert getattr(getattr(small, kind), field) != getattr(getattr(full, kind), field)
    assert small.eval == smoke_config("hutchinson").eval
    assert all(reconstruct(ARCHIVE.parent)["prefix_matches"].values())


def test_existing_evidence_and_standalone_final_measurement_eligibility(tmp_path):
    from looped_hessian.config import RunConfig
    paths = RunConfig("smoke_baseline", output_dir=str(tmp_path), checkpoint_dir=str(tmp_path / "weights"))
    copytree(ARCHIVE / paths.name, paths.run_dir)
    assert RunConfig.from_record(paths).eval.n_per_length == 20
    with pytest.raises(FileExistsError, match="Existing evidence"):
        paths.write_new_config()
    summary = Path(paths.run_dir) / "train_summary.json"
    original = load_json(summary)
    for changed in (dict(original, status="diverged"), dict(original, steps_done=5)):
        summary.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="completed training"):
            RunConfig.from_record(paths)
    summary.unlink()
    with pytest.raises(FileNotFoundError):
        RunConfig.from_record(paths)
