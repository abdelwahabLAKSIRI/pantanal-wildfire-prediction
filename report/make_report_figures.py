"""Figures made specifically for the internship report.

fig_study_area.png   -- where the Pantanal is inside the study domain, from the
                        project's own detections (INPE 2019-2024, GOES-16 2024).
fig_resolution.png   -- schematic of why a 28 km "spread" target fails and a
                        2 km one works, for readers without a remote-sensing
                        background.

Usage:
    python make_report_figures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
FIG = HERE / "figures"

FIRE = "#d84315"
FIRE_NEW = "#ffa000"
OTHER = "#b0b0b0"
DOMAIN = "#1565c0"
INK = "#222222"

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def study_area() -> None:
    inpe = pd.read_csv(PROJECT / "hourly_model" / "data" /
                       "fire_events_hourly_regional_2019_2024.csv",
                       usecols=["lat", "lon", "biome"])
    pant = inpe[inpe["biome"] == "Pantanal"]
    rest = inpe[inpe["biome"] != "Pantanal"]

    px = pd.concat([pd.read_parquet(f, columns=["lat", "lon"])
                    for f in sorted(glob.glob(str(PROJECT / "goes_hourly_model" / "data" /
                                                  "goes_fire_pixels_2024*.parquet")))])

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.4),
                             gridspec_kw={"width_ratios": [1.0, 0.92]})

    ax = axes[0]
    ax.scatter(rest["lon"], rest["lat"], s=0.4, color=OTHER, alpha=0.35,
               rasterized=True, label="other biomes (Cerrado, Amazon, ...)")
    ax.scatter(pant["lon"], pant["lat"], s=0.6, color=FIRE, alpha=0.55,
               rasterized=True, label="Pantanal biome")
    ax.set_title("INPE hotspots, 2019–2024\n(Mato Grosso + Mato Grosso do Sul)")

    ax = axes[1]
    hb = ax.hexbin(px["lon"], px["lat"], gridsize=70, bins="log", cmap="YlOrRd",
                   mincnt=1, extent=(-60, -53, -25, -13))
    cb = fig.colorbar(hb, ax=ax, shrink=0.8, pad=0.02)
    cb.set_label("GOES-16 fire pixels (log scale)")
    ax.set_title("GOES-16 fire detections, 2024\n(581,419 fire pixels)")

    for ax in axes:
        ax.add_patch(Rectangle((-60, -25), 7, 12, fill=False, lw=2.2,
                               ls=(0, (6, 3)), edgecolor=DOMAIN, zorder=5))
        ax.set_xlabel("longitude (°)")
        ax.set_ylabel("latitude (°)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.15)
    axes[0].set_xlim(-62, -49)
    axes[0].set_ylim(-25.5, -5.5)
    axes[0].text(-59.8, -12.6, "model domain", color=DOMAIN, fontsize=12,
                 fontweight="bold")
    axes[0].legend(loc="upper right", markerscale=12, fontsize=10, framealpha=0.95)
    axes[1].set_xlim(-60.3, -52.7)
    axes[1].set_ylim(-25.3, -12.7)

    fig.tight_layout()
    fig.savefig(FIG / "fig_study_area.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote fig_study_area.png")


def resolution_schematic() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.6))

    # ---------- left: 28 km cells ----------
    for i in range(3):
        for j in range(3):
            a.add_patch(Rectangle((i, j), 1, 1, fill=False, lw=2, edgecolor=INK))
    rng = np.random.default_rng(3)
    src = np.array([1.5, 1.5]) + rng.normal(0, 0.07, (14, 2))
    a.scatter(src[:, 0], src[:, 1], s=70, marker="s", color=FIRE, zorder=3)
    new = np.array([2.72, 2.78]) + rng.normal(0, 0.03, (3, 2))
    a.scatter(new[:, 0], new[:, 1], s=70, marker="s", color=FIRE_NEW, zorder=3)
    a.annotate("a new, unrelated fire\n~20 km away is counted\nas 'spread'",
               xy=(2.66, 2.72), xytext=(0.08, 2.35), fontsize=12, color=INK,
               bbox=dict(facecolor="white", edgecolor="none", pad=2), zorder=6,
               arrowprops=dict(arrowstyle="->", lw=1.6, color=INK))
    a.text(1.5, 1.08, "burning fire", ha="center", fontsize=12, color=FIRE,
           fontweight="bold")
    a.text(1.5, -0.28, "each square = one 0.25° cell ≈ 28 km", ha="center",
           fontsize=12, color="#555555")
    a.set_title("Old target: 28 km cells\n'spread' would need a 7.7 m/s fire front",
                fontweight="bold")

    # ---------- right: 2 km pixels ----------
    n = 12
    for i in range(n + 1):
        b.plot([i, i], [0, n], color="#cfcfcf", lw=0.8, zorder=1)
        b.plot([0, n], [i, i], color="#cfcfcf", lw=0.8, zorder=1)
    burning = [(4, 5), (4, 6), (5, 5), (5, 6), (5, 7), (6, 6), (4, 7), (6, 5), (3, 6)]
    spread = [(7, 6), (7, 7), (6, 7), (7, 5)]
    for (x, y) in burning:
        b.add_patch(Rectangle((x, y), 1, 1, color=FIRE, zorder=2))
    for (x, y) in spread:
        b.add_patch(Rectangle((x, y), 1, 1, color=FIRE_NEW, zorder=2))
    b.add_patch(FancyArrowPatch((0.8, 2.0), (4.3, 2.0), arrowstyle="-|>",
                                mutation_scale=28, lw=3, color=DOMAIN, zorder=3))
    b.text(2.55, 2.45, "wind", ha="center", fontsize=13, color=DOMAIN,
           fontweight="bold")
    b.add_patch(Rectangle((0.6, 10.4), 0.8, 0.8, color=FIRE))
    b.text(1.7, 10.8, "burning now", va="center", fontsize=12)
    b.add_patch(Rectangle((0.6, 9.3), 0.8, 0.8, color=FIRE_NEW))
    b.text(1.7, 9.7, "adjacent pixel ignites 1 h later", va="center", fontsize=12)
    b.text(n / 2, -0.75, "each square = one satellite pixel ≈ 2 km", ha="center",
           fontsize=12, color="#555555")
    b.set_title("New target: 2 km pixels\n'spread' needs only a 0.56 m/s front",
                fontweight="bold")

    for ax, lim in ((a, (-0.1, 3.1)), (b, (-0.1, n + 0.1))):
        ax.set_xlim(*lim)
        ax.set_ylim(lim[0] - 0.4, lim[1])
        ax.set_aspect("equal")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(FIG / "fig_resolution.png", dpi=200, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("wrote fig_resolution.png")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    study_area()
    resolution_schematic()
