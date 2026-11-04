"""Static figures for the README, reports and paper (matplotlib, Agg backend).

Local: colors follow a colorblind-validated categorical order (blue, orange, aqua; checked
with the dataviz skill's validator, all pairs pass CVD and normal-vision floors) and the
model identity is also carried by line style, markers and direct labels, so no chart relies
on color alone. One y-axis per panel; sequential blue ramp for heatmaps.

Global: `report.make_figures(summary, runs, out_dir)` calls these functions with the loaded run
folders; every figure is regenerated from results/ files, never edited by hand.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

# Role-based colors (dataviz reference palette, light mode).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {
    "baseline": {"color": "#2a78d6", "ls": "-", "marker": "o", "label": "baseline (plain looped)"},
    "hutchinson": {"color": "#eb6834", "ls": "--", "marker": "s", "label": "Hutchinson tr(H) penalty"},
    "untrained": {"color": "#1baf7a", "ls": ":", "marker": "^", "label": "untrained (init)"},
}
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5",
                                                    "#1c5cab", "#0d366b"])
# Cells that were not evaluated (K > n + extra loops) are gray, distinct from 0% (near-white).
BLUES.set_bad("#d6d5d0")


def _style_axes(ax, title: str, xlabel: str, ylabel: str) -> None:
    # Recessive grid and axes; text in ink colors, never series colors.
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)


def _figure(ncols: int, width: float = 5.0, height: float = 3.6):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, height), facecolor=SURFACE, squeeze=False)
    return fig, axes[0]


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def training_curves(histories: dict, out_dir: str) -> str:
    """Loss, in-training tr(H) and quick ID/OOD accuracy against training step."""
    fig, (a, b, c) = _figure(3)
    for arm, hist in histories.items():
        st = STYLE[arm]
        rows = [r for r in hist if "loss" in r]
        a.plot([r["step"] for r in rows], [r["loss"] for r in rows], color=st["color"], ls=st["ls"],
               lw=2, label=st["label"])
        tr = [r for r in hist if "trace_h" in r]
        steps = np.array([r["step"] for r in tr])
        val, se = np.array([r["trace_h"] for r in tr]), np.array([r["trace_h_se"] for r in tr])
        b.errorbar(steps, val, yerr=se, color=st["color"], ls=st["ls"], marker=st["marker"], ms=5,
                   lw=2, capsize=2, label=st["label"])
        ev = [r for r in hist if "acc_n19" in r]
        c.plot([r["step"] for r in ev], [r["acc_n19"] for r in ev], color=st["color"], ls="-",
               marker=st["marker"], ms=5, lw=2, label=f"{arm}: n=19 (ID)")
        c.plot([r["step"] for r in ev], [r["acc_n30"] for r in ev], color=st["color"], ls=":",
               marker=st["marker"], ms=5, lw=2, mfc="none", label=f"{arm}: n=30 (OOD)")
    a.set_yscale("log")
    _style_axes(a, "Training loss (log scale)", "step", "cross-entropy on answer tokens")
    _style_axes(b, "tr(H) during training (10 probes, ±1 SE)", "step", "Hutchinson tr(H)")
    _style_axes(c, "Quick-eval exact match (200 problems)", "step", "accuracy (oracle K)")
    c.set_ylim(-0.02, 1.02)
    for ax in (a, b, c):
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    return _save(fig, out_dir, "training_curves.png")


def accuracy_vs_length(curves: dict, train_max: int, out_dir: str, per_length: int, chance=None) -> str:
    """Oracle-K exact match per length with 95% Wilson bands; dashed line = training limit."""
    fig, (ax,) = _figure(1, width=8, height=4.4)
    # Direct labels at the last point, nudged apart so equal values do not overprint.
    ends = sorted(((acc[-1], arm) for arm, (_, acc, _, _) in curves.items()))
    label_y, last = {}, -1.0
    for y, arm in ends:
        last = max(y, last + 0.06)
        label_y[arm] = last
    for arm, (lengths, acc, lo, hi) in curves.items():
        st = STYLE[arm]
        ax.fill_between(lengths, lo, hi, color=st["color"], alpha=0.15, lw=0)
        ax.plot(lengths, acc, color=st["color"], ls=st["ls"], marker=st["marker"], ms=5, lw=2, label=st["label"])
        ax.annotate(arm, (lengths[-1], label_y[arm]), xytext=(6, 0), textcoords="offset points",
                    color=INK_2, fontsize=8, va="center")
    if chance is not None:
        ax.plot(chance[0], chance[1], color=INK_2, ls="-.", lw=1, label="chance (2^-(n+1))")
    ax.axvline(train_max + 0.5, color=INK_2, ls="--", lw=1)
    ax.text(train_max + 0.8, 0.98, "longer than any\ntraining problem →", color=INK_2, fontsize=8, va="top")
    _style_axes(ax, f"Exact-match accuracy vs. operand length n (oracle loop count, {per_length} problems/length)",
                "n (bits per operand)", "exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    return _save(fig, out_dir, "accuracy_vs_length.png")


def loop_heatmaps(grids: dict, out_dir: str) -> str:
    """Accuracy as a function of length (rows) and loop count K (columns), one panel per model."""
    fig, axes = _figure(len(grids), width=5.5, height=4.2)
    for ax, (arm, (lengths, kmax, grid)) in zip(axes, grids.items()):
        im = ax.imshow(np.ma.masked_invalid(np.array(grid, dtype=float)), aspect="auto", origin="lower", cmap=BLUES, vmin=0, vmax=1,
                       extent=(0.5, kmax + 0.5, -0.5, len(lengths) - 0.5))
        ax.set_yticks(range(len(lengths)))
        ax.set_yticklabels(lengths, fontsize=7)
        _style_axes(ax, f"{STYLE[arm]['label']} (gray = not evaluated)", "loop count K", "n (bits)")
        ax.grid(False)
    fig.colorbar(im, ax=list(axes), shrink=0.8, label="exact-match accuracy")
    fig.set_layout_engine("none")
    path = os.path.join(out_dir, "loop_heatmaps.png")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def curvature_groups(groups: dict, out_dir: str) -> str:
    """Per-parameter-group diagonal-block traces with ±1 SE, models side by side."""
    fig, (ax,) = _figure(1, width=7, height=3.8)
    names = list(next(iter(groups.values())).keys())
    width = 0.8 / len(groups)
    for i, (arm, g) in enumerate(groups.items()):
        st = STYLE[arm]
        x = np.arange(len(names)) + (i - (len(groups) - 1) / 2) * width
        ax.bar(x, [g[n]["trace"] for n in names], width=width * 0.92, yerr=[g[n]["se"] for n in names],
               color=st["color"], hatch=None if i == 0 else "//", edgecolor=SURFACE, capsize=3, label=st["label"])
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names)
    ax.axhline(0, color=INK_2, lw=0.8)
    _style_axes(ax, "Diagonal-block Hessian traces at the end of training (±1 SE)", "parameter group",
                "Hutchinson tr(H_group)")
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    return _save(fig, out_dir, "curvature_groups.png")


def loop_curvature(studies: dict, out_dir: str) -> str:
    """tr(H) and dominant eigenvalue when every problem is read out at a fixed loop count K."""
    fig, (a, b) = _figure(2)
    for arm, rows in studies.items():
        st = STYLE[arm]
        ks = [r["k"] for r in rows]
        a.errorbar(ks, [r["trace"]["trace"] for r in rows], yerr=[r["trace"]["se"] for r in rows],
                   color=st["color"], ls=st["ls"], marker=st["marker"], ms=5, lw=2, capsize=2, label=st["label"])
        b.plot(ks, [r["dominant_eigenvalue"] for r in rows], color=st["color"], ls=st["ls"],
               marker=st["marker"], ms=5, lw=2, label=st["label"])
    # Probe count is read from the results (MeasureConfig.loop_probes, 10 after addendum 2).
    probes = next(iter(studies.values()))[0]["trace"]["n_probes"]
    _style_axes(a, f"tr(H) vs. loop count ({probes} probes, ±1 SE)", "loop count K (same for all problems)", "tr(H)")
    _style_axes(b, "Dominant Hessian eigenvalue vs. loop count", "loop count K", "eigenvalue (power iteration)")
    for ax in (a, b):
        ax.legend(fontsize=8, frameon=False, labelcolor=INK_2)
    return _save(fig, out_dir, "loop_curvature.png")
