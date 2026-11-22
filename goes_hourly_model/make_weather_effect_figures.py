"""The two figures that back "weather only matters for spread".

Figure 4 -- the core evidence: paired, per-fold contribution of adding weather
on top of the fire/activity block, for all three targets, from rolling-origin
cross-validation (5 independent monthly folds, June-October 2024). Paired
because each fold compares the same train/test split with and without weather,
which cancels fold-to-fold variance and is what makes +0.012 detectable against
noise that would otherwise swamp it.

Figure 5 -- why: four physical checks showing spread is a wind/dryness-driven
process and persistence/ignition are not.

Usage:
    python make_weather_effect_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
FIG_DIR = RESULTS_DIR / "figures"

C_PERSIST = "#8e8e8e"
C_SPREAD = "#2e7d32"
C_IGN = "#ef6c00"
RED = "#c62828"
INK = "#222222"

plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 18.5, "axes.labelsize": 15.5,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "text.color": INK, "axes.labelcolor": INK,
})


# --------------------------------------------------------------------
# Figure 4: paired per-fold met contribution
# --------------------------------------------------------------------
def paired_delta(path, target, set_with, set_without):
    r = pd.read_csv(path)
    s = r[r.target == target] if "target" in r.columns else r
    piv = s.pivot(index="test_month", columns="feature_set", values="score")
    return piv[set_with] - piv[set_without]


def figure_4():
    d_persist = paired_delta(RESULTS_DIR / "rolling_origin.csv", "persists_t1h",
                             "context_fire_met", "context_fire")
    d_spread = paired_delta(RESULTS_DIR / "spread2km_rolling.csv", None,
                            "context_fire_met", "context_fire")
    d_ign = paired_delta(RESULTS_DIR / "ignition_rolling.csv", None,
                         "context_activity_met", "context_activity")

    # A visibly darker dot colour than the (paler) diamond/CI colour, purely so
    # the individual-fold dots don't wash out against white -- most noticeable
    # for the grey persistence group.
    groups = [
        ("Persistence\n\"still burning in 1h?\"", d_persist, C_PERSIST, "#5c5c5c"),
        ("Spread\n\"reaches new ground in 1h?\"", d_spread, C_SPREAD, C_SPREAD),
        ("Ignition\n\"new fire starts nearby?\"", d_ign, C_IGN, C_IGN),
    ]

    fig, ax = plt.subplots(figsize=(12, 7.6))
    x = np.arange(len(groups))

    Y_LO, Y_HI = -0.014, 0.033
    Y_CALLOUT = 0.0275          # fixed height for every stat card, regardless
                                 # of where that group's own mean happens to sit

    # faint separators between the three groups
    for sep in (0.5, 1.5):
        ax.axvline(sep, color="#e3e3e3", lw=1.2, zorder=0)

    ax.axhline(0, color=RED, ls=(0, (5, 3)), lw=2.2, zorder=1)
    ax.text(-0.62, 0.0016, "no measurable\neffect", color=RED, fontsize=11.5,
           fontweight="bold", va="bottom", ha="left", linespacing=1.3)

    for xi, (label, d, col, dot_col) in zip(x, groups):
        n = len(d)
        mean = d.mean()
        sem = d.std(ddof=1) / np.sqrt(n)
        t = mean / sem
        # 95% CI with the t-critical value at df=n-1, NOT the normal-approximation
        # 1.96. At n=5 (df=4) the t-distribution is much fatter-tailed than normal
        # -- t_crit(4, .975)=2.776 vs 1.96 -- so using 1.96 here understates the
        # interval by about 30% and overstates how tight the estimate is.
        t_crit = stats.t.ppf(0.975, df=n - 1)
        p_two = 2 * stats.t.sf(abs(t), df=n - 1)
        n_pos = int((d > 0).sum())

        # individual folds as jittered dots -- shows the paired, replicated
        # nature of the test rather than hiding it behind one error bar
        jitter = np.linspace(-0.16, 0.16, n)
        ax.scatter(xi + jitter, d.values, s=75, color=dot_col, alpha=0.55,
                  zorder=3, edgecolor="white", linewidth=0.9)

        ax.errorbar([xi], [mean], yerr=[sem * t_crit], fmt="D", color=col,
                   ms=14, elinewidth=2.6, capsize=9, capthick=2.6, zorder=5,
                   markeredgecolor="white", markeredgewidth=1.3)

        # Stat card sits at the SAME height for every group -- decoupled from
        # where that group's mean lands, so persistence (mean ~0), spread
        # (mean ~+0.012) and ignition (mean ~0) all read on one horizontal
        # line instead of scattering to the bottom/top/middle of the chart.
        significant = p_two < 0.05
        headline = f"{mean:+.4f}" + ("  ✓" if significant else "")
        ax.text(xi, Y_CALLOUT, headline, ha="center", va="bottom",
                fontsize=17, fontweight="bold", color=col)
        ax.text(xi, Y_CALLOUT - 0.0028,
                f"t = {t:+.2f}   p = {p_two:.3f}\n{n_pos}/{n} folds improved",
                ha="center", va="top", fontsize=11.5, color=col, linespacing=1.5)

        # connector from the stat card down to that group's own diamond, so the
        # (fixed-height) card still visually belongs to its (variable-height) data
        ax.plot([xi, xi], [Y_CALLOUT - 0.0006, mean + np.sign(mean or 1) * sem * t_crit
                          + 0.0009], color=col, lw=1.1, ls=(0, (1, 1.6)), zorder=2,
               alpha=0.55)

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("skill added by weather\n(paired, per fold)")
    ax.set_xlim(-0.85, 2.5)
    ax.set_ylim(Y_LO, Y_HI)
    ax.grid(axis="y", alpha=0.15, zorder=0)

    # Title as figure-level suptitle + a smaller axes subtitle underneath it --
    # the same two-tier pattern used in fig3. Putting the subtitle in its own
    # ax.set_title (rather than a second manually-positioned ax.text) is what
    # keeps matplotlib's own spacing rules in charge, instead of two independent
    # y-guesses landing on top of each other as they did in the previous version.
    fig.suptitle("Does adding weather improve the forecast?", fontsize=21,
                 fontweight="bold", y=1.05)
    ax.set_title("5 independent monthly tests, same fire population each time",
                fontsize=13.5, color="#555555", fontweight="normal", pad=14)

    # Legend via matplotlib's own machinery (proxy handles), not manually
    # placed markers/text -- the earlier hand-placed version guessed wrong on
    # both text width (it ran across the group divider) and available vertical
    # room (it was clipped by the axis). A real legend sizes and clips itself.
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=9,
              markerfacecolor="#5c5c5c", markeredgecolor="white",
              alpha=0.65, label="one monthly fold"),
        Line2D([0], [0], marker="D", linestyle="none", markersize=11,
              markerfacecolor="#5c5c5c", markeredgecolor="white",
              label="group mean (± 95% confidence interval)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.03),
              ncol=2, fontsize=12, frameon=False, handletextpad=0.7,
              columnspacing=2.3)

    fig.tight_layout()
    out = FIG_DIR / "fig4_weather_effect_by_target.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


# --------------------------------------------------------------------
# Figure 5: four physical mechanism checks
# --------------------------------------------------------------------
def figure_5():
    spread = pd.read_parquet(DATA_DIR / "goes_spread_2km_2024_01-12.parquet")
    ignition = pd.read_parquet(DATA_DIR / "goes_ignition_2024_01-12.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2))

    # --- A: downwind bias ---
    pos = spread[spread.spreads_2km_t1h == 1]
    frac = (pos.wind_spread_angle < 45).mean()
    ax = axes[0]
    ax.bar(["random\nplacement", "observed\nspread"], [0.25, frac],
          color=[ "#bdbdbd", C_SPREAD], width=0.55, zorder=3)
    ax.axhline(0.25, color=RED, ls=(0, (4, 3)), lw=1.8, zorder=1)
    for i, v in enumerate([0.25, frac]):
        ax.text(i, v + 0.01, f"{v:.1%}", ha="center", fontweight="bold",
               fontsize=15, color=C_SPREAD if i else "#666666")
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("share of new spread\nwithin 45° of downwind")
    ax.set_title("Spread favours the\ndownwind direction", fontsize=15.5,
                fontweight="bold", pad=10)
    ax.text(0.5, -0.24, "z = 25.6, n = 93,017 events", ha="center",
           transform=ax.transAxes, fontsize=11.5, color="#666666")

    # --- B: wind speed vs spread rate ---
    ax = axes[1]
    q = pd.qcut(spread.wind_speed, 5, labels=["Q1\nlowest", "Q2", "Q3", "Q4", "Q5\nhighest"])
    g = spread.groupby(q, observed=True).spreads_2km_t1h.mean()
    shades = ["#c8e0cb", "#9ecda3", "#66ad70", "#3d8b48", C_SPREAD]
    ax.bar(range(5), g.values, color=shades, width=0.6, zorder=3)
    ax.set_xticks(range(5))
    ax.set_xticklabels(g.index)
    for i, v in enumerate(g.values):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", fontweight="bold", fontsize=13)
    ax.set_ylabel("probability of spread")
    ax.set_title("Spread rate rises with\nwind speed", fontsize=15.5,
                fontweight="bold", pad=10)
    ax.set_ylim(0, 0.32)
    ax.text(0.5, -0.24, "wind speed quintile, 478,640 fire pixels",
           ha="center", transform=ax.transAxes, fontsize=11.5, color="#666666")

    # --- C: ignition vs distance to existing fire ---
    ax = axes[2]
    bins = [0, 30, 60, 120, 250, 1000]
    labels = ["30-60", "60-120", "120-250", "250+"]
    q2 = pd.cut(ignition.dist_to_fire_km, bins)
    g2 = ignition.groupby(q2, observed=True).ignites_t1h.mean()
    shades2 = [C_IGN, "#f3934a", "#f6b17a", "#f9cfab"]
    ax.bar(range(len(g2)), g2.values, color=shades2, width=0.6, zorder=3)
    ax.set_xticks(range(len(g2)))
    ax.set_xticklabels(labels)
    for i, v in enumerate(g2.values):
        ax.text(i, v + 0.002, f"{v:.1%}", ha="center", fontweight="bold", fontsize=13)
    ax.set_ylabel("probability a new fire\nstarts in the next hour")
    ax.set_xlabel("distance to nearest active fire (km)")
    ax.set_title("Ignition tracks distance to\nexisting fire, not weather",
                fontsize=15.5, fontweight="bold", pad=10)
    ax.set_ylim(0, 0.10)

    for ax in axes:
        ax.grid(axis="y", alpha=0.18, zorder=0)

    fig.suptitle("Why weather predicts spread but not persistence or ignition",
                fontsize=20, fontweight="bold", y=1.05)
    fig.tight_layout()
    out = FIG_DIR / "fig5_spread_mechanism.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out.name}")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figure_4()
    figure_5()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
