"""Results slide: what every model scores, and how the spread model got there.

Panel A -- the five models, as skill against the 0.5 coin-flip line. ROC-AUC is
used throughout because it is the one metric comparable across targets whose base
rates range from 5% to 70%. Grouped into the two model families (hourly GOES-16
vs daily INPE) with a shaded band per group, since that grouping is the first
thing worth reading off the chart.

Panel B -- the spread model's progression, which is the part that needs
explaining: the first version scored what it scored because it was predicting the
wrong quantity, not because the model was weak. Delta labels between the bars
carry that story on their own, without a floating annotation competing with the
bars for space.

Usage:
    python make_scoreboard_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
FIG_DIR = BASE_DIR / "results" / "figures"

# One family per data source, so colour alone tells you which instrument.
C_HOUR = "#1565c0"      # GOES-16 hourly models
C_HOUR_2 = "#1e88e5"
C_HOUR_3 = "#42a5f5"
C_DAY = "#6a5f7a"       # INPE daily models -- same hue family as C_HOUR's
C_DAY_2 = "#8a7f99"     #   complement, deliberately muted rather than a clash
GREY = "#9e9e9e"
RED = "#c62828"
INK = "#222222"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 19, "axes.labelsize": 15.5,
    "xtick.labelsize": 14, "ytick.labelsize": 14.5, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.dpi": 110, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": INK, "ytick.color": INK,
})

# (question, roc_auc, base rate, colour, source, group)
MODELS = [
    ("Will it still burn\nin 1 hour?", 0.830, 0.699, C_HOUR, "GOES-16", "hourly"),
    ("Will a new fire start\nnearby in 1 hour?", 0.815, 0.047, C_HOUR_2, "GOES-16", "hourly"),
    ("Will it spread 2 km\nin 1 hour?", 0.745, 0.196, C_HOUR_3, "GOES-16", "hourly"),
    ("Will it still burn\ntomorrow?", 0.670, 0.398, C_DAY, "INPE", "daily"),
    ("Will it spread\nby tomorrow?", 0.652, 0.276, C_DAY_2, "INPE", "daily"),
]

# spread model progression
STEPS = [
    ("v1\ntarget at 28 km\n(next day, INPE)", 0.652, GREY),
    ("v2\ntarget at 28 km\n(next hour, GOES)", 0.715, "#8bb8a8"),
    ("v3\ntarget at 2 km\n+ split from ignition", 0.745, "#2e7d32"),
]


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(17, 7.4),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ==================== Panel A ====================
    n = len(MODELS)
    gap = 0.9                                # extra vertical gap between groups
    ys, cur = [], n - 1 + gap
    for i, m in enumerate(MODELS):
        if i > 0 and m[5] != MODELS[i - 1][5]:
            cur -= gap
        ys.append(cur)
        cur -= 1
    ys = np.array(ys)

    # Shaded band behind each group, spanning the full axis width.
    hourly_y = [y for y, m in zip(ys, MODELS) if m[5] == "hourly"]
    daily_y = [y for y, m in zip(ys, MODELS) if m[5] == "daily"]
    band_pad = 0.55
    axA.axhspan(min(hourly_y) - band_pad, max(hourly_y) + band_pad,
               color=C_HOUR, alpha=0.06, zorder=0)
    axA.axhspan(min(daily_y) - band_pad, max(daily_y) + band_pad,
               color=C_DAY, alpha=0.07, zorder=0)

    axA.text(0.455, max(hourly_y) + band_pad - 0.08,
             "GOES-16  ·  hourly forecast, 2024", fontsize=13, fontweight="bold",
             color=C_HOUR, va="top", style="italic")
    axA.text(0.455, max(daily_y) + band_pad - 0.08,
             "INPE  ·  daily forecast, 2019–2024", fontsize=13, fontweight="bold",
             color=C_DAY, va="top", style="italic")

    # Bars are drawn on their full 0-1 scale (score IS the bar length); the
    # axis view is what crops them to the 0.45-0.97 window below. The earlier
    # bug passed left=0.45 here, which shifted the bar *start* rather than the
    # view -- every bar then ran out to score+0.45 (e.g. 1.28), off the visible
    # axis entirely, which is why the labels vanished.
    scores = [m[1] for m in MODELS]
    colours = [m[3] for m in MODELS]
    axA.barh(ys, scores, color=colours, height=0.58, zorder=3)

    axA.axvline(0.5, color=RED, ls=(0, (5, 3)), lw=2.2, zorder=4)
    axA.text(0.5, max(ys) + band_pad + 0.28, "coin flip (0.5)", color=RED,
             fontsize=13, fontweight="bold", ha="center", va="bottom")

    for yi, (lab, sc, base, col, src, grp) in zip(ys, MODELS):
        # Score and base-rate stack vertically just past the bar end, clear of
        # both the bar itself and the next row -- avoids the earlier version's
        # cramped single-line "0.83GOES-16 · base rate 70%" run-on.
        axA.text(sc + 0.014, yi + 0.13, f"{sc:.2f}", va="center", fontsize=18,
                 fontweight="bold", color=col)
        axA.text(sc + 0.014, yi - 0.17, f"base rate {base:.0%}",
                 va="center", fontsize=11, color="#7a7a7a")

    axA.set_yticks(ys)
    axA.set_yticklabels([m[0] for m in MODELS])
    axA.set_xlim(0.45, 0.975)
    axA.set_ylim(min(ys) - band_pad - 0.15, max(ys) + band_pad + 0.55)
    axA.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
    axA.set_xlabel("skill  (ROC-AUC — 0.5 = chance, 1.0 = perfect)", labelpad=10)
    axA.set_title("What each model predicts, and how well", pad=14,
                  fontweight="bold")
    axA.grid(axis="x", alpha=0.18, zorder=0)
    axA.tick_params(axis="y", length=0)

    # ==================== Panel B ====================
    x = np.arange(len(STEPS))
    vals = [s[1] for s in STEPS]
    cols = [s[2] for s in STEPS]

    axB.bar(x, vals, color=cols, width=0.56, zorder=3, edgecolor="white", linewidth=0)
    axB.axhline(0.5, color=RED, ls=(0, (5, 3)), lw=2.2, zorder=1)
    axB.text(2.32, 0.503, "chance", color=RED, fontsize=12, fontweight="bold",
             va="bottom", ha="right")

    for xi, v, c in zip(x, vals, cols):
        axB.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=18,
                 fontweight="bold", color=INK)

    # Delta arrows + labels between consecutive bars -- this carries the
    # "each step fixed a data problem" story without a floating annotation
    # competing with the bars for the same vertical space.
    for (x0, v0), (x1, v1) in zip(zip(x, vals), list(zip(x, vals))[1:]):
        mid_y = 0.795
        axB.annotate("", xy=(x1 - 0.32, mid_y), xytext=(x0 + 0.32, mid_y),
                     arrowprops=dict(arrowstyle="-|>", lw=2, color="#2e7d32",
                                     shrinkA=0, shrinkB=0))
        axB.text((x0 + x1) / 2, mid_y + 0.012, f"+{v1 - v0:.3f}",
                 ha="center", va="bottom", fontsize=13.5, fontweight="bold",
                 color="#2e7d32")

    axB.set_xticks(x)
    axB.set_xticklabels([s[0] for s in STEPS], fontsize=12.5)
    axB.set_ylim(0.45, 0.86)
    axB.set_ylabel("skill  (ROC-AUC)")
    axB.set_title("Fixing the target, not the model", pad=14, fontweight="bold")
    axB.grid(axis="y", alpha=0.18, zorder=0)
    axB.tick_params(axis="x", length=0)
    axB.text(1, 0.465, "same algorithms at every step", ha="center",
             fontsize=12, color="#7a7a7a", style="italic")

    fig.suptitle("Pantanal fire prediction — model scores", fontsize=23,
                 fontweight="bold", y=1.03)
    fig.tight_layout()
    out = FIG_DIR / "fig3_scoreboard.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
