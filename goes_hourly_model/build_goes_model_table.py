"""Build the hourly-target training table from GOES-16 fire + ERA5 forcing.

This is the model table the whole GOES exercise was for: unlike the INPE-based
table, the **target** here is hourly, not just the forcing.

Targets
-------
Both are conditioned on a cell that is burning at hour t, mirroring the
INPE-based model's structure one resolution finer:

``persists_t1h``  fire still present in the same cell at t+1h.
``spreads_t1h``   at t+1h a *neighbouring* cell is burning that was not burning
                  at t -- a new front, not the same fire seen again.

Why hour-of-day is a legal feature here
---------------------------------------
In the INPE product, hour-of-day was pure observation geometry: all detections
sat in 4 of 24 hours, so it predicted the target while knowing nothing about
fire. GOES observes every hour with near-flat coverage (obs_frac 0.961-0.978
across the day), so the diurnal cycle it carries is fire physics -- activity
peaks at 14:00 local and troughs at 06:00, a 7.8x swing.

Feature decisions carried over from the August-2024 review
----------------------------------------------------------
* **Retrieval failures are NaN, not 0.** GOES detects fire in a pixel but often
  cannot retrieve its radiative power, area or temperature (11-17% of rows).
  Mapping those to 0 conflated "not measured" with "zero energy", and it was not
  inert: persistence was 0.55 where max_frp was 0 against 0.77 where it was
  positive, so the model was reading a retrieval-failure flag dressed as a
  physical measurement. Now the values stay NaN -- XGBoost handles NaN natively
  -- and ``has_frp`` / ``has_area_temp`` carry the retrieval status explicitly.
* **``day_of_month`` is gone.** It is an index, not a physical quantity, and
  repeats meaninglessly across months. Replaced by a cyclic day-of-year encoding.
* **``obs_frac`` stays in the table but out of the feature list.** It is a
  quality gate; leaving it as a predictor invites the model to learn cloud
  artifacts.
* **``fire_area_km2`` added.** ABI pixels grow off-nadir, so a raw pixel count
  mixes fire size with viewing geometry. Cell area divided by that cell's pixel
  count gives the true per-pixel footprint, which converts the count to km2.
* Unused wind component lags are no longer written.

Lags are computed over the whole requested period at once, so they do not break
at month boundaries.

Usage:
    python build_goes_model_table.py --year 2024
    python build_goes_model_table.py --year 2024 --months 6-10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROJECT_DIR = BASE_DIR.parent
HOURLY_DATA = PROJECT_DIR / "hourly_model" / "data"
ERA5_FILE = HOURLY_DATA / "era5_hourly_2019_2024.nc"

N_LAT, N_LON = 49, 29
N_CELLS = N_LAT * N_LON
FIRE_LAGS = (1, 2, 3, 6, 12, 24)
MET_LAGS = (1, 3, 6, 12)
KM_PER_DEG = 111.32
CELL_DEG = 0.25


def dense(g: pd.DataFrame, hours: pd.DatetimeIndex, column: str) -> np.ndarray:
    out = np.full((N_CELLS, len(hours)), np.nan, dtype="float32")
    hi = pd.Index(hours).get_indexer(g["time"])
    out[g["cell"].to_numpy(), hi] = g[column].to_numpy()
    return out


def shift_hours(a: np.ndarray, lag: int) -> np.ndarray:
    """Shift along time; lag>0 looks back, lag<0 looks forward."""
    out = np.full_like(a, np.nan)
    if lag > 0:
        out[:, lag:] = a[:, :-lag]
    elif lag < 0:
        out[:, :lag] = a[:, -lag:]
    else:
        out[:] = a
    return out


def neighbour_sum(a: np.ndarray) -> np.ndarray:
    """Sum over the 8 surrounding cells, per hour."""
    n_hours = a.shape[1]
    grid = np.nan_to_num(a).reshape(N_LAT, N_LON, n_hours)
    total = np.zeros_like(grid)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                total += np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
    # np.roll wraps; blank the edges so the far side of the domain cannot
    # masquerade as a neighbour.
    total[0, :, :] = total[-1, :, :] = np.nan
    total[:, 0, :] = total[:, -1, :] = np.nan
    return total.reshape(N_CELLS, n_hours)


def load_months(year: int, months: list[int]) -> pd.DataFrame:
    parts = []
    for m in months:
        p = DATA_DIR / f"goes_hourly_pantanal_{year}{m:02d}.parquet"
        if not p.exists():
            print(f"  [{year}-{m:02d}] missing, skipped")
            continue
        parts.append(pd.read_parquet(p))
        print(f"  [{year}-{m:02d}] {len(parts[-1]):,} cell-hours")
    if not parts:
        raise SystemExit("no monthly GOES files found")
    return pd.concat(parts, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", default="1-12")
    ap.add_argument("--min-obs-frac", type=float, default=0.5)
    args = ap.parse_args()

    if "-" in args.months:
        lo, hi = (int(x) for x in args.months.split("-"))
        months = list(range(lo, hi + 1))
    else:
        months = [int(args.months)]

    print(f"loading GOES {args.year} months {months[0]}-{months[-1]}:")
    g = load_months(args.year, months)
    # Build a COMPLETE hourly axis, not just the observed timestamps. Some
    # hours are absent entirely (a failed scan, or every cell below the
    # obs_frac gate). Indexing lags off the observed hours alone would make
    # shift_hours(1) reach back 2 h across each gap and mislabel the lag.
    observed = pd.DatetimeIndex(sorted(g["time"].unique()))
    hours = pd.date_range(observed.min(), observed.max(), freq="h")
    gaps = len(hours) - len(observed)
    print(f"\ntotal {len(g):,} cell-hours across {len(hours):,} hours "
          f"({gaps} with no data, kept as gaps so lags stay aligned)")

    fire = np.nan_to_num(dense(g, hours, "fire_t"))
    obs = dense(g, hours, "obs_frac")
    npx = dense(g, hours, "n_fire_px")
    cellpx = dense(g, hours, "n_cell_px")

    # --- targets and row validity --------------------------------------
    fire_next = shift_hours(fire, -1)
    obs_next = shift_hours(obs, -1)
    nb_new = neighbour_sum(np.clip(fire_next - fire, 0, 1))

    valid = ((fire == 1) & (obs >= args.min_obs_frac)
             & (obs_next >= args.min_obs_frac) & np.isfinite(nb_new))
    valid[:, -1] = False
    valid[:, :24] = False          # 24 h history undefined at the start

    cells, times = np.nonzero(valid)
    print(f"rows: {len(cells):,} burning cell-hours with an observed next hour")

    def at(arr):
        """Sample a dense array at the valid row positions."""
        return arr[cells, times]

    ts = hours[times]
    local_hour = (ts.hour - 4) % 24
    doy = ts.dayofyear
    days_in_year = 366 if (args.year % 4 == 0) else 365

    lat = -25.0 + (cells // N_LON) * CELL_DEG
    # Exact per-pixel footprint: the cell's ground area divided by however many
    # ABI pixels landed in it. Off-nadir stretching is already reflected in that
    # pixel count, so this converts a raw count into real area.
    cell_area = (CELL_DEG * KM_PER_DEG) ** 2 * np.cos(np.radians(lat))
    px_area = cell_area / np.maximum(at(cellpx), 1)

    df = pd.DataFrame({
        "time": ts,
        "cell": cells,
        "lat": lat,
        "lon": -60.0 + (cells % N_LON) * CELL_DEG,
        "hour_local": local_hour,
        "hour_sin": np.sin(2 * np.pi * local_hour / 24),
        "hour_cos": np.cos(2 * np.pi * local_hour / 24),
        "doy_sin": np.sin(2 * np.pi * doy / days_in_year),
        "doy_cos": np.cos(2 * np.pi * doy / days_in_year),
        "obs_frac": at(obs),
        "persists_t1h": at(fire_next).astype(int),
        "spreads_t1h": (at(nb_new) > 0).astype(int),
    })
    del fire_next, obs_next, nb_new

    # --- fire state -----------------------------------------------------
    df["n_fire_px"] = at(npx)
    df["fire_area_km2"] = df["n_fire_px"] * px_area
    df["cell_area_km2"] = cell_area
    df["fire_frac_of_cell"] = df["fire_area_km2"] / df["cell_area_km2"]

    for lag in FIRE_LAGS:
        df[f"fire_lag{lag}"] = at(shift_hours(fire, lag))
    df["fire_hours_last6"] = at(sum(shift_hours(fire, k) for k in range(1, 7)))
    df["fire_hours_last24"] = at(sum(shift_hours(fire, k) for k in range(1, 25)))

    npx_lag1 = shift_hours(npx, 1)
    df["npx_lag1"] = np.nan_to_num(at(npx_lag1))
    df["npx_delta"] = df["n_fire_px"] - df["npx_lag1"]
    del npx_lag1

    nb_fire = neighbour_sum(fire)
    df["nb_cells_burning"] = at(nb_fire)
    df["nb_cells_burning_lag1"] = at(shift_hours(nb_fire, 1))
    df["nb_fire_px"] = at(neighbour_sum(np.nan_to_num(npx)))
    del nb_fire, npx, fire, obs, cellpx

    # --- retrieved fire properties: NaN where not retrieved -------------
    for col, flag in [("max_frp", "has_frp"), ("sum_frp", None),
                      ("max_area", "has_area_temp"), ("max_temp", None)]:
        arr = dense(g, hours, col)
        vals = at(arr)
        # 0 in the source means "detected, but this quantity was not retrieved".
        vals = np.where(vals > 0, vals, np.nan)
        df[col] = vals
        if flag:
            df[flag] = np.isfinite(vals).astype(int)
        if col == "max_frp":
            df["frp_lag1"] = np.where(at(shift_hours(arr, 1)) > 0,
                                      at(shift_hours(arr, 1)), np.nan)
        del arr

    # --- ERA5 forcing ----------------------------------------------------
    print("sampling ERA5 forcing...")
    with xr.open_dataset(ERA5_FILE) as ds:
        axis = pd.DatetimeIndex(ds["time"].values)
        idx = axis.get_indexer(hours)
        if (idx < 0).any():
            raise SystemExit("ERA5 does not cover every requested hour")
        for var in ("t2m", "rh", "vpd", "wind_speed", "u10", "v10"):
            cube = ds[var].values[idx].reshape(len(hours), -1).T.astype("float32")
            if var == "t2m":
                cube = cube - 273.15
            name = {"u10": "wind_u", "v10": "wind_v"}.get(var, var)

            df[f"{name}_t"] = at(cube)
            if name in ("wind_u", "wind_v"):
                del cube
                continue                       # components: instantaneous only
            for lag in MET_LAGS:
                df[f"{name}_lag{lag}"] = at(shift_hours(cube, lag))

            stack = np.stack([shift_hours(cube, k) for k in range(24)])
            if var == "rh":
                df["rh_24h_min"] = at(np.nanmin(stack, axis=0))
                df["rh_24h_mean"] = at(np.nanmean(stack, axis=0))
            else:
                df[f"{var}_24h_max"] = at(np.nanmax(stack, axis=0))
                df[f"{var}_24h_mean"] = at(np.nanmean(stack, axis=0))
            del cube, stack

    df = df.sort_values(["time", "cell"]).reset_index(drop=True)
    suffix = f"{args.year}" if len(months) == 12 else f"{args.year}_{months[0]:02d}-{months[-1]:02d}"
    out = DATA_DIR / f"goes_model_table_{suffix}.parquet"
    df.to_parquet(out, index=False)

    print(f"\nwrote {out.name}: {len(df):,} rows x {df.shape[1]} columns")
    print(f"  {df['time'].min()} -> {df['time'].max()}")
    print(f"  persists_t1h: {df.persists_t1h.mean():.4f} ({int(df.persists_t1h.sum()):,} positive)")
    print(f"  spreads_t1h : {df.spreads_t1h.mean():.4f} ({int(df.spreads_t1h.sum()):,} positive)")
    print(f"  retrieval    : has_frp {df.has_frp.mean():.1%}, "
          f"has_area_temp {df.has_area_temp.mean():.1%}")
    print(f"  fire_area_km2: median {df.fire_area_km2.median():.1f}, "
          f"max {df.fire_area_km2.max():.0f} (cell is ~{df.cell_area_km2.mean():.0f} km2)")
    print("\n  rows and persistence by month:")
    print(df.groupby(df["time"].dt.month).agg(
        rows=("persists_t1h", "size"),
        persists=("persists_t1h", "mean"),
        spreads=("spreads_t1h", "mean")).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
