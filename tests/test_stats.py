"""Tests for the Stage 1 statistics helpers."""

from looped_hessian import stats


def test_wilson_interval_contains_estimate_and_is_bounded():
    lo, hi = stats.wilson(30, 100)
    assert 0 <= lo < 0.30 < hi <= 1
    assert stats.wilson(0, 50)[0] == 0.0


def test_mcnemar_counts_discordant_pairs():
    a = [1, 1, 0, 0, 1, 0]
    b = [1, 0, 1, 1, 1, 0]
    res = stats.paired_mcnemar(a, b)
    assert (res["only_a"], res["only_b"], res["both"], res["neither"]) == (1, 2, 2, 1)
    assert 0 <= res["p_value"] <= 1 and res["exact"]


def test_stage1_reading_uses_preregistered_threshold():
    assert abs(stats.SUGGESTIVE_THRESHOLD - 16.69) < 0.01
    assert stats.stage1_reading(20) == "suggestive positive"
    assert stats.stage1_reading(-20) == "suggestive negative"
    assert stats.stage1_reading(10) == "inconclusive at n=1 seed"
