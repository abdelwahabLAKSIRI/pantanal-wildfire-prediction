"""Cross-check the GOES-16 hourly fire table against INPE's overpass detections.

GOES and INPE are independent instruments with different detection physics, so
they will never agree pixel for pixel. What this checks is that the GOES table is
*sane*, and it does so by asking three questions that would each catch a
different class of pipeline bug:

1. **Geolocation.** During the INPE overpass hour, do the cells GOES calls
   burning coincide with the cells INPE calls burning? A projection or
   grid-indexing error would destroy this agreement while leaving every other
   diagnostic looking healthy.

2. **Diurnal cycle.** Does the hourly fire signal follow the shape fire
   physically has -- rising through the afternoon as fuels dry and winds pick up,
   falling overnight? A flat profile would mean the extraction is picking up
   something other than fire.

3. **Coverage.** Is the "observed" fraction high enough, and roughly uniform
   across the day, for a 0 label to be meaningful? This is the property the whole
   exercise depends on.

The point is not that GOES should reproduce INPE. It is that GOES should agree
with INPE where they overlap, and then keep going for the other 23 hours.

Usage:
    python validate_goes_vs_inpe.py --year 2024 --month 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                      # this model's own files
PROJECT_DIR = BASE_DIR.parent                     # shared project inputs
HOURLY_DATA = PROJECT_DIR / "hourly_model" / "data"   # shared ERA5 + INPE tables
RESULTS_DIR = BASE_DIR / "results"
GRID_STEP = 0.25


def load_inpe(year: int, month: int, cell_lat, cell_lon) -> pd.DataFrame:
    """INPE Pantanal hotspots for the month, snapped to the ERA5 grid."""
    path = HOURLY_DATA / f"fire_events_hourly_{year}.csv"
    if not path.exists():
        path = HOURLY_DATA / "fire_events_hourly_regional_2019_2024.csv"
    if not path.exists():
        raise SystemExit("no INPE file found in hourly_model/data/")

    d = pd.read_csv(path, parse_dates=["date", "detection_hour"])
    d = d[(d.biome == "Pantanal") & (d.date.dt.year == year) & (d.date.dt.month == month)]
    if d.empty:
        raise SystemExit(f"no INPE Pantanal rows for {year}-{month:02d}")

    ilat = np.abs(d.lat.to_numpy()[:, None] - cell_lat[None, :]).argmin(1)
    ilon = np.abs(d.lon.to_numpy()[:, None] - cell_lon[None, :]).argmin(1)
    d = d.assign(cell=ilat * len(cell_lon) + ilon,
                 hour=d.detection_hour.dt.floor("h"))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--month", type=int, default=8)
    args = ap.parse_args()

    goes_path = DATA_DIR / f"goes_hourly_pantanal_{args.year}{args.month:02d}.parquet"
    if not goes_path.exists():
        raise SystemExit(f"{goes_path.name} not found -- run build_goes_hourly.py first.")

    g = pd.read_parquet(goes_path)
    cell_lon = np.sort(g.lon.unique())
    cell_lat = np.sort(g.lat.unique())
    n_lon = len(cell_lon)

    print(f"GOES table : {len(g):,} cell-hours, {g.time.nunique():,} hours, "
          f"{g.cell.nunique():,} cells")
    print(f"             prevalence {g.fire_t.mean():.5f}, "
          f"{int(g.fire_t.sum()):,} positive cell-hours\n")

    inpe = load_inpe(args.year, args.month, cell_lat, cell_lon)
    print(f"INPE       : {len(inpe):,} hotspots, "
          f"{inpe.hour.dt.hour.nunique()} distinct detection hours "
          f"{sorted(inpe.hour.dt.hour.unique())}\n")

    # ---- 1. geolocation agreement during the overpass hour -------------
    print("=" * 68)
    print("1. GEOLOCATION -- agreement during INPE's own overpass hour")
    print("=" * 68)

    rows = []
    for hour, grp in inpe.groupby("hour"):
        gh = g[g.time == hour]
        if gh.empty:
            continue
        inpe_cells = set(grp.cell)
        goes_cells = set(gh.loc[gh.fire_t == 1, "cell"])
        if not inpe_cells:
            continue
        inter = inpe_cells & goes_cells
        rows.append({
            "hour": hour,
            "inpe_cells": len(inpe_cells),
            "goes_cells": len(goes_cells),
            "both": len(inter),
            "recall_of_inpe": len(inter) / len(inpe_cells),
        })

    match = pd.DataFrame(rows)
    if match.empty:
        print("  no overlapping hours -- GOES table does not cover the overpass times")
        return 1

    tot_inpe = match.inpe_cells.sum()
    tot_both = match.both.sum()
    print(f"  overpass hours compared : {len(match)}")
    print(f"  INPE fire cells         : {tot_inpe:,}")
    print(f"  also flagged by GOES     : {tot_both:,}  "
          f"({tot_both / tot_inpe:.1%})")
    print(f"  median per-hour agreement: {match.recall_of_inpe.median():.1%}")

    # A random-cell baseline says how much of that is just "lots of cells burn".
    mean_goes_frac = (g.groupby("time").fire_t.mean()).mean()
    print(f"  chance agreement if GOES fire cells were random: {mean_goes_frac:.1%}")
    lift = (tot_both / tot_inpe) / mean_goes_frac if mean_goes_frac > 0 else np.nan
    print(f"  lift over chance         : {lift:.1f}x")

    # ---- 2. diurnal cycle ---------------------------------------------
    print()
    print("=" * 68)
    print("2. DIURNAL CYCLE -- the thing INPE structurally cannot show")
    print("=" * 68)

    g = g.assign(hour_utc=g.time.dt.hour,
                 hour_local=(g.time.dt.hour - 4) % 24)   # Pantanal is UTC-4
    diel = g.groupby("hour_local").agg(
        cell_hours=("fire_t", "size"),
        fire_cell_hours=("fire_t", "sum"),
        prevalence=("fire_t", "mean"),
        mean_obs_frac=("obs_frac", "mean"),
    )
    diel["share_of_fires"] = diel.fire_cell_hours / diel.fire_cell_hours.sum()
    print(diel.round(4).to_string())

    peak = int(diel.prevalence.idxmax())
    trough = int(diel.prevalence.idxmin())
    print(f"\n  peak local hour   : {peak:02d}:00  (prevalence {diel.prevalence.max():.5f})")
    print(f"  trough local hour : {trough:02d}:00  (prevalence {diel.prevalence.min():.5f})")
    ratio = diel.prevalence.max() / max(diel.prevalence.min(), 1e-12)
    print(f"  peak/trough ratio : {ratio:.1f}x")
    night = diel.loc[[h for h in range(0, 6)], "share_of_fires"].sum()
    print(f"  share of fire cell-hours between 00:00-05:59 local: {night:.1%}")
    print("    (INPE reports exactly 0% here -- it never observes those hours)")

    # ---- 3. coverage ---------------------------------------------------
    print()
    print("=" * 68)
    print("3. COVERAGE -- is a 0 label actually an observation?")
    print("=" * 68)
    print(f"  median obs_frac        : {g.obs_frac.median():.3f}")
    print(f"  cell-hours >= 0.9 obs  : {(g.obs_frac >= 0.9).mean():.1%}")
    print(f"  hours present          : {g.time.nunique():,} of "
          f"{24 * pd.Period(f'{args.year}-{args.month:02d}').days_in_month:,} possible")
    by_hour = g.groupby("hour_local").obs_frac.mean()
    print(f"  obs_frac by local hour : min {by_hour.min():.3f} at {int(by_hour.idxmin()):02d}:00, "
          f"max {by_hour.max():.3f} at {int(by_hour.idxmax()):02d}:00")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    diel.to_csv(RESULTS_DIR / f"goes_diurnal_{args.year}{args.month:02d}.csv")
    match.to_csv(RESULTS_DIR / f"goes_vs_inpe_match_{args.year}{args.month:02d}.csv", index=False)
    print(f"\nwrote diagnostics to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
