"""Generate the two figures that carry the argument, sized for projection.

1. ``fig1_observing_window.png`` -- INPE's polar orbiter against GOES-16's
   geostationary coverage, by hour of day. The point a meteorologist grasps
   instantly: one instrument sees three hours, the other sees all of them.

2. ``fig2_diurnal_cycle.png`` -- the fire diurnal cycle GOES reveals, with the
   INPE observing window shaded on top of it, so the two figures tell one story.

Both use large type and heavy lines because they will be projected, not read on
a laptop.

Usage:
    python make_presentation_figures.py
"""

from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "results" / "figures"
INPE_FILE = BASE_DIR.parent / "hourly_model" / "data" / "fire_events_hourly_2024.csv"

INPE_C = "#c62828"
GOES_C = "#1565c0"
ACCENT = "#ef6c00"
GREY = "#666666"

plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})


def load_inpe_hours() -> pd.Series:
    d = pd.read_csv(INPE_FILE, parse_dates=["detection_hour"], usecols=[
        "detection_hour", "biome"])
    pa = d[d["biome"] == "Pantanal"]
    return pa["detection_hour"].dt.hour.value_counts().reindex(range(24), fill_value=0)


def load_goes_hours() -> pd.Series:
    parts = [pd.read_parquet(f, columns=["time"])
             for f in sorted(glob.glob(str(DATA_DIR / "goes_fire_pixels_*.parquet")))]
    t = pd.concat(parts, ignore_index=True)["time"]
    return t.dt.hour.value_counts().reindex(range(24), fill_value=0)


def figure_1(inpe: pd.Series, goes: pd.Series) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharex=True)

    axes[0].bar(inpe.index, inpe.values, color=INPE_C, width=0.85)
    axes[0].set_title("INPE reference satellite (polar orbiting)", pad=12)
    axes[0].set_ylabel("fire detections")
    seen = int((inpe > 0).sum())
    axes[0].text(0.5, 0.93, f"sees {seen} of 24 hours",
                 transform=axes[0].transAxes, ha="center", fontsize=17,
                 color=INPE_C, fontweight="bold")

    axes[1].bar(goes.index, goes.values, color=GOES_C, width=0.85)
    axes[1].set_title("GOES-16 (geostationary)", pad=12)
    axes[1].set_ylabel("fire detections")
    axes[1].text(0.5, 0.93, "sees all 24 hours",
                 transform=axes[1].transAxes, ha="center", fontsize=17,
                 color=GOES_C, fontweight="bold")

    for ax in axes:
        ax.set_xlabel("hour of day (UTC)")
        ax.set_xticks(range(0, 24, 3))
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("The same region, the same year — two observing geometries",
                 fontsize=20, y=1.00)
    fig.text(0.5, -0.04,
             "Pantanal, 2024.  The polar orbiter's timestamps are hourly, but its "
             "sampling is daily:\n1,172 of 1,189 fire days carry a single detection hour.",
             ha="center", fontsize=14, color=GREY)

    fig.tight_layout()
    out = FIG_DIR / "fig1_observing_window.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def figure_2(inpe: pd.Series):
    """Two stacked panels: the activity cycle, then the persistence cycle.

    Kept as separate panels rather than twin y-axes -- the shaded observing
    window, the bars and the line all wanted the same corner of the plot, and a
    figure that has to be decoded is no use on a projector.
    """
    tbl = pd.read_parquet(DATA_DIR / "goes_model_table_2024.parquet",
                          columns=["hour_local", "persists_t1h", "n_fire_px"])
    g = tbl.groupby("hour_local").agg(
        persistence=("persists_t1h", "mean"),
        activity=("n_fire_px", "sum"))
    g["activity_share"] = g["activity"] / g["activity"].sum()

    # INPE's observing window, converted from UTC to Pantanal local time (UTC-4)
    inpe_local = sorted(((h - 4) % 24) for h in inpe.index[inpe > 0])
    lo, hi = min(inpe_local) - 0.5, max(inpe_local) + 0.5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.15, 1]})

    # --- top: fire activity ---
    ax1.axvspan(lo, hi, color=INPE_C, alpha=0.13, zorder=0)
    ax1.bar(g.index, g["activity_share"] * 100, color=GOES_C, width=0.85, zorder=2)
    ax1.set_ylabel("share of fire\nactivity (%)")
    night = g.loc[0:5, "activity_share"].sum()
    ax1.set_title("Fire activity through the day — GOES-16, Pantanal 2024", pad=10)
    ax1.annotate("the only hours the\npolar orbiter observes",
                 xy=((lo + hi) / 2, ax1.get_ylim()[1] * 0.93),
                 ha="center", va="top", fontsize=14, color=INPE_C,
                 fontweight="bold")
    ax1.text(2.5, ax1.get_ylim()[1] * 0.80,
             f"{night:.0%} of activity\nhappens 00:00–06:00",
             fontsize=14, color=GOES_C, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    # --- bottom: persistence ---
    ax2.axvspan(lo, hi, color=INPE_C, alpha=0.13, zorder=0)
    ax2.plot(g.index, g["persistence"], color=ACCENT, lw=3.5, marker="o",
             ms=7, zorder=3)
    ax2.set_ylabel("probability fire\nsurvives the next hour")
    ax2.set_xlabel("hour of day (local, UTC−4)")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(-0.7, 23.7)
    ax2.set_ylim(0.54, 0.87)
    ax2.grid(axis="y", alpha=0.2)

    peak, trough = int(g["persistence"].idxmax()), int(g["persistence"].idxmin())
    ax2.annotate(f"{g['persistence'].max():.2f}", xy=(peak, g["persistence"].max()),
                 xytext=(peak - 3.4, 0.83), fontsize=15, color=ACCENT,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax2.annotate(f"{g['persistence'].min():.2f}", xy=(trough, g["persistence"].min()),
                 xytext=(trough + 2.2, 0.57), fontsize=15, color=ACCENT,
                 fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=ACCENT, lw=2))
    ax2.set_title("Whether a fire survives also depends on the hour", pad=10)

    fig.tight_layout()
    out = FIG_DIR / "fig2_diurnal_cycle.png"
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out, g, night


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    inpe = load_inpe_hours()
    goes = load_goes_hours()

    print("INPE detections by UTC hour (non-zero):")
    print(inpe[inpe > 0].to_string())
    print(f"\nGOES detections span {int((goes > 0).sum())} of 24 hours "
          f"(min {goes.min():,} at {goes.idxmin():02d} UTC, "
          f"max {goes.max():,} at {goes.idxmax():02d} UTC)")

    f1 = figure_1(inpe, goes)
    f2, g, night = figure_2(inpe)

    print(f"\nwrote {f1.name}")
    print(f"wrote {f2.name}")
    print(f"\npersistence by local hour: peak {g['persistence'].max():.3f} at "
          f"{int(g['persistence'].idxmax()):02d}:00, trough "
          f"{g['persistence'].min():.3f} at {int(g['persistence'].idxmin()):02d}:00 "
          f"({g['persistence'].max()/g['persistence'].min():.1f}x)")
    print(f"night (00-06 local) share of activity: {night:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
