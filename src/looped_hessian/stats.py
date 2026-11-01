"""Statistics for the Stage 1 (one seed per arm) descriptive comparison.

Local: with one training seed per arm, the only uncertainty we can quantify is test-set
sampling: Wilson intervals for accuracies (statsmodels `proportion_confint`, method
"wilson") and a paired McNemar test on the same test problems (statsmodels `mcnemar`).
Neither captures seed-to-seed variance, which is why the preregistered reading rule
compares the difference against the across-seed SD reported by Kuo et al. 2026.

Global: `report.py` uses these to fill the results table and the Stage 1 verdict.
"""

import math

from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

# Kuo et al. 2026 (arXiv:2606.29983), randomized K = n ± 5 on addition: OOD 30.9 ± 5.9 (points).
KUO_SEED_SD = 5.9
# Preregistered threshold: |Δ| > 2 · sqrt(2) · SD for a "suggestive" single-seed difference.
SUGGESTIVE_THRESHOLD = 2 * math.sqrt(2) * KUO_SEED_SD


def wilson(successes: int, n: int) -> tuple[float, float]:
    # 95% Wilson score interval for a binomial proportion.
    lo, hi = proportion_confint(successes, n, alpha=0.05, method="wilson")
    return float(lo), float(hi)


def paired_mcnemar(a: list[int], b: list[int]) -> dict:
    """McNemar test on paired per-problem correctness of two models on the same problems."""
    b01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)  # only model B correct
    b10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)  # only model A correct
    both = sum(1 for x, y in zip(a, b) if x == 1 and y == 1)
    neither = len(a) - b01 - b10 - both
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
