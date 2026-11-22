"""Build separate SPREAD and IGNITION tables from the native-pixel store.

Why they are separate
---------------------
The first spread target -- "a neighbouring 0.25 deg cell is burning at t+1h" --
was not measuring spread. Three diagnostics said so: a 7.7 m/s implied front
speed, zero directional preference (25.4% of new neighbours within 45 deg of
downwind against 25% for random), and almost no dependence on the source fire's
size (0.20/0.17/0.23/0.25/0.24 across size quintiles, while persistence ran
0.63 -> 0.97). It was really measuring "some new fire appeared within 28 km",
which mixes two different processes:

**Spread** -- a fire front advancing into adjacent unburnt fuel. Fire-front
physics: wind, slope, fuel continuity, intensity of the source. Built here at
native ABI resolution (~2 km), where one-hour adjacency implies 0.55 m/s rather
than an impossible 7.7 m/s.

**Ignition** -- a fire starting where nothing was burning and nothing adjacent
was burning either. In the Pantanal this is largely human, and it is a landscape
question rather than a fire-front one, so it stays on the 0.25 deg cell grid
alongside the meteorology.

Modelling them together means neither is predicted well.

Tables
------
``goes_spread_2km_<period>.parquet``
    One row per fire pixel per hour that *could* spread -- i.e. has at least one
    adjacent pixel not already burning. Target ``spreads_2km_t1h``, plus the
    directional variants ``spreads_downwind`` and ``spread_bearing``.

``goes_ignition_<period>.parquet``
    One row per observed, non-burning 0.25 deg cell-hour with no fire in or
    adjacent to it. Target ``ignites_t1h``. Negatives are subsampled, since the
    raw rate is under 1%.

Observation gating: pixel-level observation status is approximated by the
containing cell's ``obs_frac``. The cell-hour table records what fraction of a
cell's pixels the fire algorithm could judge; a pixel in a cell below the gate is
treated as unobserved rather than fire-free.

Usage:
    python build_goes_split_targets.py --year 2024 --months 6-10
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROJECT_DIR = BASE_DIR.parent
ERA5_FILE = PROJECT_DIR / "hourly_model" / "data" / "era5_hourly_2019_2024.nc"
GRID_META = DATA_DIR / "goes_pixel_grid.json"

N_LAT, N_LON = 49, 29
N_CELLS = N_LAT * N_LON
KM_PER_DEG = 111.32
CELL_DEG = 0.25
HISTORY = 24                      # hours of pixel history to keep buffered

# 8-neighbour offsets with the compass bearing each represents (0 = north).
NEIGHBOURS = {(-1, 0): 0.0, (-1, 1): 45.0, (0, 1): 90.0, (1, 1): 135.0,
              (1, 0): 180.0, (1, -1): 225.0, (0, -1): 270.0, (-1, -1): 315.0}


def load_pixels(year: int, months: list[int]) -> pd.DataFrame:
    parts = []
    for m in months:
        p = DATA_DIR / f"goes_fire_pixels_{year}{m:02d}.parquet"
        if not p.exists():
            print(f"  [{year}-{m:02d}] pixel store missing, skipped")
            continue
        parts.append(pd.read_parquet(p))
        print(f"  [{year}-{m:02d}] {len(parts[-1]):,} fire pixels")
    if not parts:
        raise SystemExit("no pixel stores found -- run build_goes_pixels.py first")
    return pd.concat(parts, ignore_index=True)


def load_cells(year: int, months: list[int]) -> pd.DataFrame:
    parts = []
    for m in months:
        p = DATA_DIR / f"goes_hourly_pantanal_{year}{m:02d}.parquet"
        if p.exists():
            parts.append(pd.read_parquet(p))
    if not parts:
        raise SystemExit("no cell-hour tables found")
    return pd.concat(parts, ignore_index=True)


def dense_hour(px: pd.DataFrame, shape) -> np.ndarray:
    grid = np.zeros(shape, dtype=bool)
    if len(px):
        grid[px["px_row"].to_numpy(), px["px_col"].to_numpy()] = True
    return grid


def neighbour_count(grid: np.ndarray) -> np.ndarray:
    """Number of burning pixels among the 8 neighbours, edges not wrapped."""
    p = np.pad(grid.astype(np.int8), 1)
    out = np.zeros_like(grid, dtype=np.int8)
    for dy, dx in NEIGHBOURS:
        out += p[1 + dy:1 + dy + grid.shape[0], 1 + dx:1 + dx + grid.shape[1]]
    return out


def sample_era5(hours: pd.DatetimeIndex):
    """Per-cell hourly met, as (n_cells, n_hours) arrays."""
    out = {}
    with xr.open_dataset(ERA5_FILE) as ds:
        axis = pd.DatetimeIndex(ds["time"].values)
        idx = axis.get_indexer(hours)
        if (idx < 0).any():
            raise SystemExit("ERA5 does not cover every requested hour")
        for var in ("t2m", "rh", "vpd", "wind_speed", "u10", "v10"):
            cube = ds[var].values[idx].reshape(len(hours), -1).T.astype("float32")
            out[var] = cube - 273.15 if var == "t2m" else cube
    return out


def shift_hours(a, lag):
    out = np.full_like(a, np.nan)
    if lag > 0:
        out[:, lag:] = a[:, :-lag]
    elif lag < 0:
        out[:, :lag] = a[:, -lag:]
    else:
        out[:] = a
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", default="6-10")
    ap.add_argument("--min-obs-frac", type=float, default=0.5)
    ap.add_argument("--negatives-per-positive", type=float, default=20.0,
                    help="ignition table: how many negative cell-hours to keep "
                         "per positive (raw rate is under 1%%)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if "-" in args.months:
        lo, hi = (int(x) for x in args.months.split("-"))
        months = list(range(lo, hi + 1))
    else:
        months = [int(args.months)]

    if not GRID_META.exists():
        raise SystemExit(f"{GRID_META.name} not found -- run build_goes_pixels.py")
    shape = tuple(json.loads(GRID_META.read_text())["window_shape"])
    print(f"pixel window {shape[0]} x {shape[1]}")

    print(f"\nloading pixel stores:")
    px = load_pixels(args.year, months)
    cells = load_cells(args.year, months)

    hours = pd.date_range(cells["time"].min(), cells["time"].max(), freq="h")
    hi = pd.Index(hours)
    print(f"\n{len(hours):,} hours | {len(px):,} fire pixels | {len(cells):,} cell-hours")

    # cell observation gate, as a dense (cell, hour) array
    obs = np.zeros((N_CELLS, len(hours)), dtype="float32")
    obs[cells["cell"].to_numpy(), hi.get_indexer(cells["time"])] = cells["obs_frac"].to_numpy()

    met = sample_era5(hours)
    wind_to = (np.degrees(np.arctan2(met["u10"], met["v10"]))) % 360.0

    by_hour = {t: d for t, d in px.groupby("time")}

    # Which cell each pixel belongs to, from the pixel store itself.
    cell_of = np.full(shape, -1, dtype="int32")
    seen = px.drop_duplicates(["px_row", "px_col"])
    cell_of[seen["px_row"].to_numpy(), seen["px_col"].to_numpy()] = seen["cell"].to_numpy()

    spread_rows, ignition_rows = [], []
    buf: deque[np.ndarray] = deque(maxlen=HISTORY + 1)
    cell_hist: deque[np.ndarray] = deque(maxlen=HISTORY)
    rng = np.random.default_rng(args.seed)

    for ti, t in enumerate(hours):
        grid = dense_hour(by_hour.get(t, px.iloc[:0]), shape)
        buf.append(grid)
        if ti + 1 >= len(hours):
            break
        nxt = dense_hour(by_hour.get(hours[ti + 1], px.iloc[:0]), shape)

        nb_now = neighbour_count(grid)
        new_next = nxt & ~grid                      # pixels that ignite at t+1

        # Connected fire patch each pixel belongs to. An isolated 1-pixel
        # detection and the edge of a 200-pixel front are different objects, and
        # the neighbour count alone cannot tell them apart.
        patch_lbl, n_patch = ndimage.label(grid, structure=np.ones((3, 3), int))
        patch_size = (np.bincount(patch_lbl.ravel())
                      if n_patch else np.zeros(1, dtype=int))

        # ---------- SPREAD: fire pixels with somewhere to go ----------
        fr, fc = np.nonzero(grid)
        if len(fr):
            cell_idx = cell_of[fr, fc]
            ok = cell_idx >= 0
            fr, fc, cell_idx = fr[ok], fc[ok], cell_idx[ok]
            gate = obs[cell_idx, ti] >= args.min_obs_frac
            # A pixel ringed by fire cannot spread; excluding it keeps the target
            # about spreading rather than about being interior.
            room = nb_now[fr, fc] < 8
            keep = gate & room
            fr, fc, cell_idx = fr[keep], fc[keep], cell_idx[keep]

            if len(fr):
                spread = np.zeros(len(fr), dtype=bool)
                bearing = np.full(len(fr), np.nan)
                for (dy, dx), bear in NEIGHBOURS.items():
                    rr, cc = fr + dy, fc + dx
                    inb = ((rr >= 0) & (rr < shape[0]) & (cc >= 0) & (cc < shape[1]))
                    hit = np.zeros(len(fr), dtype=bool)
                    hit[inb] = new_next[rr[inb], cc[inb]]
                    bearing = np.where(hit & np.isnan(bearing), bear, bearing)
                    spread |= hit

                wdir = wind_to[cell_idx, ti]
                ang = np.abs((bearing - wdir + 180.0) % 360.0 - 180.0)

                src = by_hour.get(t)
                lut = {}
                if src is not None:
                    lut = {(r, c): (f, a, tm) for r, c, f, a, tm in zip(
                        src["px_row"], src["px_col"], src["frp"], src["area"], src["temp"])}
                props = np.array([lut.get((r, c), (np.nan,) * 3) for r, c in zip(fr, fc)])

                hist = np.stack([b[fr, fc] for b in list(buf)[-HISTORY - 1:-1]][::-1]) \
                    if len(buf) > 1 else np.zeros((1, len(fr)), dtype=bool)

                spread_rows.append(pd.DataFrame({
                    "time": t, "px_row": fr, "px_col": fc, "cell": cell_idx,
                    "spreads_2km_t1h": spread.astype(int),
                    "spread_bearing": bearing,
                    "spreads_downwind": np.where(spread & (ang < 45), 1, 0),
                    "wind_spread_angle": ang,
                    "nb_burning_now": nb_now[fr, fc],
                    "patch_px": patch_size[patch_lbl[fr, fc]],
                    "region_fire_px": int(grid.sum()),
                    "frp": props[:, 0], "area": props[:, 1], "temp": props[:, 2],
                    "fire_hours_last6": hist[:6].sum(0) if hist.shape[0] >= 6 else np.nan,
                    "fire_hours_last24": hist.sum(0),
                    "hour_local": (t.hour - 4) % 24,
                    "t2m": met["t2m"][cell_idx, ti], "rh": met["rh"][cell_idx, ti],
                    "vpd": met["vpd"][cell_idx, ti],
                    "wind_speed": met["wind_speed"][cell_idx, ti],
                    "wind_u": met["u10"][cell_idx, ti], "wind_v": met["v10"][cell_idx, ti],
                    "wind_dir_to": wdir,
                }))

        # ---------- IGNITION: cells with no fire in or beside them ----------
        cells_with_fire = np.unique(cell_of[grid & (cell_of >= 0)]) if grid.any() else np.array([], int)
        near = np.zeros(N_CELLS, dtype=bool)
        g2 = np.zeros((N_LAT, N_LON), dtype=bool)
        if len(cells_with_fire):
            g2[cells_with_fire // N_LON, cells_with_fire % N_LON] = True
            near = neighbour_count(g2).reshape(-1) > 0
            near |= g2.reshape(-1)
        cell_hist.append(g2.reshape(-1).copy())

        # Distance to the nearest burning cell. Ignition is excluded within one
        # cell of a fire by construction, so this measures how far into quiet
        # ground the new start is -- the useful remaining spatial signal.
        if g2.any():
            dist_cells = ndimage.distance_transform_edt(~g2)
        else:
            dist_cells = np.full((N_LAT, N_LON), np.inf)
        dist_km = (dist_cells * CELL_DEG * KM_PER_DEG).reshape(-1)
        hist24 = (np.sum(cell_hist, axis=0) if len(cell_hist)
                  else np.zeros(N_CELLS))
        region_cells = int(g2.sum())

        cells_next = np.unique(cell_of[nxt & (cell_of >= 0)]) if nxt.any() else np.array([], int)
        fire_next_cell = np.zeros(N_CELLS, dtype=bool)
        fire_next_cell[cells_next] = True

        eligible = ((obs[:, ti] >= args.min_obs_frac)
                    & (obs[:, ti + 1] >= args.min_obs_frac) & ~near)
        if eligible.any():
            idx = np.flatnonzero(eligible)
            y = fire_next_cell[idx].astype(int)
            pos = idx[y == 1]
            neg = idx[y == 0]
            n_keep = int(max(len(pos) * args.negatives_per_positive, 5))
            if len(neg) > n_keep:
                neg = rng.choice(neg, n_keep, replace=False)
            sel = np.concatenate([pos, neg])
            if len(sel):
                ignition_rows.append(pd.DataFrame({
                    "time": t, "cell": sel,
                    "ignites_t1h": fire_next_cell[sel].astype(int),
                    "hour_local": (t.hour - 4) % 24,
                    "lat": -25.0 + (sel // N_LON) * CELL_DEG,
                    "lon": -60.0 + (sel % N_LON) * CELL_DEG,
                    "t2m": met["t2m"][sel, ti], "rh": met["rh"][sel, ti],
                    "vpd": met["vpd"][sel, ti],
                    "wind_speed": met["wind_speed"][sel, ti],
                    "obs_frac": obs[sel, ti],
                    "dist_to_fire_km": np.minimum(dist_km[sel], 999.0),
                    "cell_fire_hours_last24": hist24[sel],
                    "region_fire_cells": region_cells,
                    "n_sampled_neg": len(neg), "n_eligible": len(idx),
                }))

    period = f"{args.year}_{months[0]:02d}-{months[-1]:02d}"

    sp = pd.concat(spread_rows, ignore_index=True)
    sp["doy_sin"] = np.sin(2 * np.pi * sp["time"].dt.dayofyear / 366)
    sp["doy_cos"] = np.cos(2 * np.pi * sp["time"].dt.dayofyear / 366)
    sp["hour_sin"] = np.sin(2 * np.pi * sp["hour_local"] / 24)
    sp["hour_cos"] = np.cos(2 * np.pi * sp["hour_local"] / 24)
    sp_path = DATA_DIR / f"goes_spread_2km_{period}.parquet"
    sp.to_parquet(sp_path, index=False)

    ig = pd.concat(ignition_rows, ignore_index=True)
    ig["doy_sin"] = np.sin(2 * np.pi * ig["time"].dt.dayofyear / 366)
    ig["doy_cos"] = np.cos(2 * np.pi * ig["time"].dt.dayofyear / 366)
    ig["hour_sin"] = np.sin(2 * np.pi * ig["hour_local"] / 24)
    ig["hour_cos"] = np.cos(2 * np.pi * ig["hour_local"] / 24)
    ig_path = DATA_DIR / f"goes_ignition_{period}.parquet"
    ig.to_parquet(ig_path, index=False)

    print(f"\nwrote {sp_path.name}: {len(sp):,} rows")
    print(f"  spreads_2km_t1h prevalence: {sp.spreads_2km_t1h.mean():.4f} "
          f"({int(sp.spreads_2km_t1h.sum()):,} positive)")
    print(f"  of those, downwind (<45 deg): {sp.loc[sp.spreads_2km_t1h == 1, 'spreads_downwind'].mean():.1%} "
          f"(25% would mean no directional preference)")
    print(f"  implied front speed at 2 km / 1 h: {2000/3600:.2f} m/s")

    print(f"\nwrote {ig_path.name}: {len(ig):,} rows "
          f"(negatives subsampled {args.negatives_per_positive:.0f}:1)")
    print(f"  ignites_t1h prevalence: {ig.ignites_t1h.mean():.4f} "
          f"({int(ig.ignites_t1h.sum()):,} positive)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
