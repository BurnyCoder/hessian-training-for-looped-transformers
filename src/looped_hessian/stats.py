"""Conditional test-sampling uncertainty and the registered single-pair reading rule."""

import math

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

# Kuo et al. 2026 (arXiv:2606.29983, Table 1), randomized K = n ± 5 on addition: OOD 30.9, SD 5.9
# points across 32 runs. Their OOD metric uses a per-example oracle K, ours a per-length one, so the
# SD is an approximate reference for seed variance, not a like-for-like number.
KUO_SEED_SD = 5.9
# Preregistered threshold: |Δ| > 2 · sqrt(2) · SD for a "suggestive" single-seed difference.
SUGGESTIVE_THRESHOLD = 2 * math.sqrt(2) * KUO_SEED_SD


def wilson(successes: int, n: int) -> tuple[float, float]:
    # 95% Wilson score interval for a binomial proportion.
    lo, hi = proportion_confint(successes, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def stratified_interval(counts: list[tuple[int, int]]) -> tuple[float, float]:
    """Equal-weight mean of simultaneous, exact binomial bounds for fixed length strata.

    Clopper–Pearson is statsmodels' beta method. Bonferroni gives joint coverage of
    at least 95% without assuming equal success probabilities across lengths.
    https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm
    """
    bounds = [proportion_confint(k, n, alpha=0.05 / len(counts), method="beta") for k, n in counts]
    return tuple(math.fsum(side) / len(bounds) for side in zip(*bounds))


def paired_counts(a: list[int], b: list[int]) -> dict:
    """Count discordances in aligned binary outcomes. Alignment is checked by the caller."""
    cells = [[0, 0], [0, 0]]
    for x, y in zip(a, b, strict=True):
        if x not in (0, 1) or y not in (0, 1):
            raise ValueError("paired correctness must be binary")
        cells[x][y] += 1
    return dict(neither=cells[0][0], only_b=cells[0][1], only_a=cells[1][0], both=cells[1][1])


def paired_mcnemar(a: list[int], b: list[int]) -> dict:
    """McNemar for prespecified paired predictions. The oracle report uses counts only."""
    cells = paired_counts(a, b)
    b01, b10, both, neither = (cells[k] for k in ("only_b", "only_a", "both", "neither"))
    # Exact binomial version for few discordant pairs, chi-square with continuity correction otherwise.
    exact = (b01 + b10) < 25
    res = mcnemar([[neither, b01], [b10, both]], exact=exact, correction=True)
    return {"only_a": b10, "only_b": b01, "both": both, "neither": neither,
            "statistic": float(res.statistic), "p_value": float(res.pvalue), "exact": exact}


def stage1_reading(delta_points: float) -> str:
    """Preregistered verbal reading of a single-seed difference in the primary metric."""
    if delta_points > SUGGESTIVE_THRESHOLD:
        return "suggestive positive"
    if delta_points < -SUGGESTIVE_THRESHOLD:
        return "suggestive negative"
    return "inconclusive at n=1 seed"
