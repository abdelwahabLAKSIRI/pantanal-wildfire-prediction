"""Store GOES-16 fire detections at native ABI resolution (~2 km).

Why this exists
---------------
The first pass aggregated fire straight onto the 0.25 deg ERA5 grid, ~28 km
cells. That threw away a factor of ~14 in exactly the dimension where spread
happens, and it broke the spread target: fire reaching an adjacent 0.25 deg cell
within an hour implies a 7.7 m/s front, where extreme crown fires reach 1-3 m/s.
Three diagnostics confirmed the target was not measuring spread at all --

  * zero directional signal (25.4% of new neighbours within 45 deg of downwind,
    against 25% for random placement),
  * no dependence on the source fire's size (spread rate 0.20 / 0.17 / 0.23 /
    0.25 / 0.24 across size quintiles, while persistence went 0.63 -> 0.97),
  * a physically impossible implied rate.

At native 2 km resolution, one-hour adjacency implies 0.55 m/s, which is
squarely physical.

What is stored
--------------
**Fire pixels only.** They are sparse -- a few hundred per scan against 211,324
pixels in the Pantanal window -- so a year costs tens of megabytes rather than
the ~1.8 billion rows a full pixel dump would need. One row per fire pixel per
scan: window indices, geolocation, mask code, and the retrieved fire properties.

The per-cell hourly table (`goes_hourly_pantanal_*.parquet`) is written in the
same pass, because it carries the observation gate (`obs_frac`) that decides
which negatives are real. Both come from one download, so choosing a different
grid later never requires re-fetching anything.

Usage:
    python build_goes_pixels.py --year 2024 --month 8
    python build_goes_pixels.py --year 2024 --months 1-12
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import requests

import goes_grid as gg
from build_goes_hourly import (BASE_DIR, CACHE_DIR, DATA_DIR, Geometry,  # noqa: F401
                               PRODUCT, _max_by_cell, fetch,
                               list_day, pick_scans)


def extract_pixels(path: Path, geom: Geometry, high_conf: bool):
    """Fire pixels for one scan, plus the per-cell observation counts.

    Returns (pixel_frame, cell_aggregates). The pixel frame is empty when
    nothing is burning, which is the common case outside the fire season.
    """
    window = (geom.rows, geom.cols)
    with h5py.File(path, "r") as h5:
        def read(name):
            d = h5[name]
            raw = d[window][geom.inside]
            out = raw.astype("float32")
            a = d.attrs
            if "_FillValue" in a:
                out[raw == np.asarray(a["_FillValue"]).ravel()[0]] = np.nan
            if "scale_factor" in a:
                out = out * np.asarray(a["scale_factor"]).ravel()[0]
            if "add_offset" in a:
                out = out + np.asarray(a["add_offset"]).ravel()[0]
            return out

        mask = read("Mask")
        power = read("Power")
        area = read("Area")
        temp = read("Temp")

    fire, no_fire, _ = gg.classify(mask, high_confidence_only=high_conf)
    observed = fire | no_fire
    n = geom.n_cells
    c = geom.flat_cell

    frp = np.where(fire & np.isfinite(power), power, 0.0)
    cells = {
        "n_fire": np.bincount(c, weights=fire.astype("float64"), minlength=n),
        "n_obs": np.bincount(c, weights=observed.astype("float64"), minlength=n),
        "sum_frp": np.bincount(c, weights=frp, minlength=n),
        "max_frp": _max_by_cell(c, frp, n),
        "max_area": _max_by_cell(c, np.where(fire & np.isfinite(area), area, 0.0), n),
        "max_temp": _max_by_cell(c, np.where(fire & np.isfinite(temp), temp, 0.0), n),
    }

    idx = np.flatnonzero(fire)
    if len(idx) == 0:
        return None, cells

    pixels = pd.DataFrame({
        "px_row": geom.px_row[idx].astype("int16"),
        "px_col": geom.px_col[idx].astype("int16"),
        "lat": geom.px_lat[idx].astype("float32"),
        "lon": geom.px_lon[idx].astype("float32"),
        "cell": geom.flat_cell[idx].astype("int16"),
        "mask_code": mask[idx].astype("int16"),
        "frp": power[idx].astype("float32"),
        "area": area[idx].astype("float32"),
        "temp": temp[idx].astype("float32"),
    })
    return pixels, cells


def add_pixel_coords(geom: Geometry) -> Geometry:
    """Attach window-relative row/col and geolocation for every in-window pixel.

    The ABI fixed grid never moves, so this is computed once and reused for every
    scan of the mission.
    """
    import xarray as xr
    with xr.open_dataset(next(CACHE_DIR.glob("*.nc"))) as ds:
        x_axis = ds["x"].values.astype("float64")
        y_axis = ds["y"].values.astype("float64")

    n_row = geom.rows.stop - geom.rows.start
    n_col = geom.cols.stop - geom.cols.start
    rr, cc = np.meshgrid(np.arange(n_row), np.arange(n_col), indexing="ij")

    X, Y = np.meshgrid(x_axis[geom.cols], y_axis[geom.rows])
    lon, lat = gg.scan_to_lonlat(X, Y)

    geom.px_row = rr[geom.inside]
    geom.px_col = cc[geom.inside]
    geom.px_lat = lat[geom.inside]
    geom.px_lon = lon[geom.inside]
    geom.window_shape = (n_row, n_col)
    return geom


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--months", default="8")
    ap.add_argument("--scans-per-hour", type=int, default=1)
    ap.add_argument("--high-confidence-only", action="store_true")
    ap.add_argument("--min-obs-frac", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if "-" in args.months:
        lo, hi = (int(x) for x in args.months.split("-"))
        months = list(range(lo, hi + 1))
    else:
        months = [int(args.months)]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    for month in months:
        start = datetime(args.year, month, 1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
        days = [start + timedelta(days=i) for i in range((end - start).days)]

        px_out = DATA_DIR / f"goes_fire_pixels_{args.year}{month:02d}.parquet"
        cell_out = DATA_DIR / f"goes_hourly_pantanal_{args.year}{month:02d}.parquet"
        if px_out.exists():
            print(f"[{args.year}-{month:02d}] pixel store exists, skipping")
            continue

        print(f"\n[{args.year}-{month:02d}] {len(days)} days, "
              f"{args.scans_per_hour} scan(s)/hour")

        geom = None
        pixel_parts, cell_records = [], []

        for day in days:
            keys = list_day(args.year, day.timetuple().tm_yday, session)
            scans = pick_scans(keys, args.scans_per_hour)
            if not scans:
                print(f"  {day.date()}: no scans", flush=True)
                continue

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                paths = list(pool.map(lambda s: fetch(s[1], session), scans))

            if geom is None:
                first = next(p for p in paths if p is not None)
                geom = add_pixel_coords(Geometry(first))

            by_hour: dict[datetime, list] = {}
            for (t, _), path in zip(scans, paths):
                if path is None:
                    continue
                try:
                    px, cells = extract_pixels(path, geom, args.high_confidence_only)
                except Exception as exc:          # noqa: BLE001
                    print(f"    read failed {path.name}: {type(exc).__name__}", flush=True)
                    continue
                hour = t.replace(minute=0, second=0, microsecond=0)
                by_hour.setdefault(hour, []).append(cells)
                if px is not None and len(px):
                    px.insert(0, "time", hour)
                    px.insert(1, "scan_time", t)
                    pixel_parts.append(px)
                path.unlink(missing_ok=True)

            for hour, aggs in by_hour.items():
                n_fire = np.sum([a["n_fire"] for a in aggs], axis=0)
                n_obs = np.max([a["n_obs"] for a in aggs], axis=0)
                keep = geom.px_per_cell > 0
                obs_frac = np.divide(n_obs, geom.px_per_cell,
                                     out=np.zeros_like(n_obs), where=keep)
                sel = keep & (obs_frac >= args.min_obs_frac)
                if not sel.any():
                    continue
                idx = np.flatnonzero(sel)
                cell_records.append(pd.DataFrame({
                    "time": hour, "cell": idx,
                    "lat": geom.cell_lat[idx // len(geom.cell_lon)],
                    "lon": geom.cell_lon[idx % len(geom.cell_lon)],
                    "n_fire_px": n_fire[idx], "n_obs_px": n_obs[idx],
                    "n_cell_px": geom.px_per_cell[idx], "obs_frac": obs_frac[idx],
                    "max_frp": np.max([a["max_frp"] for a in aggs], axis=0)[idx],
                    "sum_frp": np.sum([a["sum_frp"] for a in aggs], axis=0)[idx],
                    "max_area": np.max([a["max_area"] for a in aggs], axis=0)[idx],
                    "max_temp": np.max([a["max_temp"] for a in aggs], axis=0)[idx],
                }))

            n_px = sum(len(p) for p in pixel_parts)
            print(f"  {day.date()}: {len(by_hour):2d} h, {n_px:7,} fire px so far",
                  flush=True)

        if not cell_records:
            print(f"[{args.year}-{month:02d}] no data extracted")
            continue

        cells_df = pd.concat(cell_records, ignore_index=True)
        cells_df["fire_t"] = (cells_df["n_fire_px"] > 0).astype(int)
        cells_df.sort_values(["time", "cell"]).to_parquet(cell_out, index=False)

        px_df = (pd.concat(pixel_parts, ignore_index=True)
                 if pixel_parts else pd.DataFrame(columns=["time", "px_row", "px_col"]))
        px_df.attrs = {}
        px_df.sort_values(["time", "px_row", "px_col"]).to_parquet(px_out, index=False)

        print(f"[{args.year}-{month:02d}] wrote {px_out.name}: {len(px_df):,} fire "
              f"pixels ({px_out.stat().st_size/1e6:.1f} MB) | "
              f"{cell_out.name}: {len(cells_df):,} cell-hours")

    # The window shape is needed to rebuild a dense pixel grid later.
    if geom is not None:
        meta = DATA_DIR / "goes_pixel_grid.json"
        import json
        meta.write_text(json.dumps({
            "window_rows": [geom.rows.start, geom.rows.stop],
            "window_cols": [geom.cols.start, geom.cols.stop],
            "window_shape": list(geom.window_shape),
            "n_in_window": int(geom.inside.sum()),
        }, indent=1))
        print(f"\nwrote {meta.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
