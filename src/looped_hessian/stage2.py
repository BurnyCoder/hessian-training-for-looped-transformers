"""Prospective paired-seed analysis. Inputs are saved scores; no experiment is executed.

SciPy documents the paired t statistic and BCa bootstrap at
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html and
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html .
"""

import math

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

from looped_hessian.stats import KUO_SEED_SD


def paired_power(rho: float, marginal_sd: float = KUO_SEED_SD) -> float:
    """Detectable superiority difference in points for ten pairs, first Holm cutoff, 80% power."""
    sd_difference = marginal_sd * math.sqrt(2 * (1 - rho))
    effect = TTestPower().solve_power(nobs=10, alpha=0.025, power=0.8, alternative="larger")
    return float(effect * sd_difference)


def _paired_arrays(scores):
    """Align the registered seed IDs before taking any difference or resampling."""
    seeds = list(range(10))
    if set(scores) != {"baseline", "hutchinson", "nso"} or any(set(s) != set(seeds) for s in scores.values()):
        raise ValueError("the two primary contrasts require the same ten registered seed pairs")
    arrays = {arm: np.asarray([s[i] for i in seeds], dtype=float) for arm, s in scores.items()}
    if any(a.shape != (10, 2) or not np.isfinite(a).all() or (a < 0).any() or (a > 1).any()
           for a in arrays.values()):
        raise ValueError("each seed needs two finite accuracy fractions in [0,1]")
    return arrays


def analyze_h1(scores: dict) -> dict:
    """Each arm maps seed to (oracle fraction, batch-confidence fraction).

    Resampling differences keeps both observations within a seed together. The interval
    and practical upper bound are per contrast; Holm applies to the two superiority tests.
    Degenerate resampling distributions are reported as unevaluable.
    """
    arrays = _paired_arrays(scores)
    rows = {}
    for arm in ("hutchinson", "nso"):
        delta, confidence = (100 * (arrays[arm] - arrays["baseline"])).T
        interval = p = None
        if np.ptp(delta) > 1e-12:
            p = float(stats.ttest_1samp(delta, popmean=0, alternative="greater").pvalue)
            ci = stats.bootstrap((delta,), np.mean, method="BCa", n_resamples=19999,
                                 random_state=20261126).confidence_interval
            if np.isfinite(ci).all():
                interval = [float(ci.low), float(ci.high)]
        rows[arm] = {"mean_points": float(delta.mean()), "ci95_points": interval, "seed_deltas_points": delta.tolist(),
                     "paired_d": float(delta.mean()/delta.std(ddof=1)) if p is not None else None,
                     "same_seed_win_fraction": float(np.mean(delta > 0)), "ties": int(np.sum(delta == 0)),
                     "p": p, "confidence_delta_points": float(confidence.mean())}
    # Keep both planned slots in the Holm family even if one contrast is undefined.
    adjusted = multipletests([1 if r["p"] is None else r["p"] for r in rows.values()], method="holm")[1]
    for row, adj in zip(rows.values(), adjusted):
        ci = row["ci95_points"]
        row["holm_p"] = float(adj) if row["p"] is not None else None
        row["positive_gain"] = bool(adj < .05 and ci[0] > 0 and row["confidence_delta_points"] > 0) if ci else None
        row["rules_out_two_points"] = ci[1] < 2 if ci else None
    return rows


def analyze_h3(scores: dict) -> dict:
    """Exploratory marginal variance differences, preserving the dependence within seed.

    Seed-pair percentile bootstrap intervals and leave-one-pair-out estimates describe
    sensitivity. At ten pairs their coverage and tail precision can be poor.
    """
    arrays = _paired_arrays(scores)
    baseline = 100 * arrays["baseline"][:, 0]
    indices = np.random.default_rng(20261126).integers(0, len(baseline), size=(19999, len(baseline)))
    left = np.array([np.delete(np.arange(len(baseline)), i) for i in range(len(baseline))])
    out = {}
    for arm in ("hutchinson", "nso"):
        treatment = 100 * arrays[arm][:, 0]

        def diff(i):
            return treatment[i].var(axis=-1, ddof=1) - baseline[i].var(axis=-1, ddof=1)

        draws = diff(indices)
        out[arm] = {"difference_points_squared": float(diff(np.arange(len(baseline)))),
                    "ci95_points_squared": np.quantile(draws, [.025, .975]).tolist() if np.ptp(draws) > 1e-12 else None,
                    "leave_one_pair_out": diff(left).tolist(), "baseline_points": baseline.tolist(),
                    "treatment_points": treatment.tolist()}
    return out
