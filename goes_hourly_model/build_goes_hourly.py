"""Build a genuinely hourly Pantanal fire table from GOES-16 ABI.

Why this exists
---------------
The INPE reference-satellite product is one afternoon overpass per day: across
2020-2024, 1,172 of 1,189 Pantanal fire days carry a single detection hour. An
hourly target cannot be labelled against it. Filling the unobserved hours with
zeros does not fix that -- it manufactures them. On the real data that scheme
yields prevalence 0.000886 with every positive confined to 4 of 24 hours, so
hour-of-day alone scores ROC-AUC 0.917 while knowing nothing about fire.

GOES-16 is geostationary: it stares at South America continuously and produces a
full-disk fire mask every 10 minutes. Two things follow.

1. An hourly target becomes estimable, because every hour is actually observed.
2. A 0 label becomes an observation. Mask code 100 is
   ``processed_no_fire_pixel`` -- the pixel was examined and nothing was burning.
   Codes 40/50/60 (off-earth, zenith-angle and sun-glint block-out) and the 120+
   family (missing/saturated/invalid input, cloud) mark pixels that could not be
   judged. Those are excluded rather than labelled 0, which is the distinction
   the INPE product could never support.

Output
------
One row per (0.25 deg ERA5 cell, hour), matching the grid the ERA5 forcing is
already on:

    fire_t         1 if any fire pixel in the cell that hour
    n_fire_px      fire pixel count
    n_obs_px       pixels the algorithm could actually judge
    n_cell_px      pixels in the cell
    obs_frac       n_obs_px / n_cell_px  -- the coverage/quality gate
    max_frp        peak Fire Radiative Power (MW)
    sum_frp, max_area, max_temp

Rows are kept only where ``obs_frac >= --min-obs-frac``; below that the cell was
too obscured for its 0 to mean anything.

Usage:
    python build_goes_hourly.py --year 2024 --month 8          # validation month
    python build_goes_hourly.py --year 2024 --month 8 --scans-per-hour 6
"""

from __future__ import annotations

import argparse
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import requests
import xarray as xr

import goes_grid as gg

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"                      # this model's own files
PROJECT_DIR = BASE_DIR.parent                     # shared project inputs
HOURLY_DATA = PROJECT_DIR / "hourly_model" / "data"   # shared ERA5 + INPE tables
CACHE_DIR = DATA_DIR / "goes_cache"

BUCKET = "https://noaa-goes16.s3.amazonaws.com"
PRODUCT = "ABI-L2-FDCF"

LAT_MIN, LAT_MAX = -25.0, -13.0
LON_MIN, LON_MAX = -60.0, -53.0
GRID_STEP = 0.25


def target_grid() -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre latitudes and longitudes of the 0.25 deg ERA5 grid (49 x 29).

    Built from the domain bounds rather than read from a file, so the GOES
    pipeline needs no laboratory-provided input. Identical to the lat/lon axes
    of ``era5_hourly_2019_2024.nc`` and ``xhwi_era5_hourly_2024.nc``.
    """
    lat = np.round(np.arange(LAT_MIN, LAT_MAX + GRID_STEP / 2, GRID_STEP), 4)
    lon = np.round(np.arange(LON_MIN, LON_MAX + GRID_STEP / 2, GRID_STEP), 4)
    return lat, lon

_START_RE = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")


def _get_with_retry(session: requests.Session, url: str, attempts: int = 5,
                    timeout: int = 60) -> requests.Response:
    """GET with exponential backoff.

    S3 drops the occasional connection. Without a retry here a single blip
    aborts the whole month, and because the caller moves on to the next month
    the loss is easy to miss -- exactly how April 2024 went absent on the first
    pass, leaving a silent hole in the year.
    """
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as exc:                  # noqa: BLE001
            if attempt == attempts:
                raise
            print(f"    {type(exc).__name__} on attempt {attempt}/{attempts}, "
                  f"retrying in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def list_day(year: int, doy: int, session: requests.Session) -> list[str]:
    """Every object key for one day of the full-disk fire product."""
    keys, token = [], None
    while True:
        url = f"{BUCKET}/?list-type=2&max-keys=1000&prefix={PRODUCT}/{year}/{doy:03d}/"
        if token:
            url += f"&continuation-token={requests.utils.quote(token, safe='')}"
        r = _get_with_retry(session, url)
        keys += re.findall(r"<Key>([^<]+)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if not m:
            break
        token = m.group(1)
    return keys


def scan_start(key: str) -> datetime:
    y, doy, hh, mm, ss = (int(g) for g in _START_RE.search(key).groups())
    return datetime(y, 1, 1) + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)


def pick_scans(keys: list[str], per_hour: int) -> list[tuple[datetime, str]]:
    """Choose which scans to use, evenly spread within each hour."""
    by_hour: dict[datetime, list[tuple[datetime, str]]] = {}
    for k in keys:
        t = scan_start(k)
        by_hour.setdefault(t.replace(minute=0, second=0, microsecond=0), []).append((t, k))

    chosen = []
    for hour in sorted(by_hour):
        scans = sorted(by_hour[hour])
        if per_hour >= len(scans):
            chosen += scans
        else:
            idx = np.linspace(0, len(scans) - 1, per_hour).round().astype(int)
            chosen += [scans[i] for i in dict.fromkeys(idx)]
    return chosen


def fetch(key: str, session: requests.Session) -> Path | None:
    target = CACHE_DIR / Path(key).name
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        r = _get_with_retry(session, f"{BUCKET}/{key}", timeout=180)
        target.write_bytes(r.content)
        return target
    except Exception as exc:                      # noqa: BLE001
        print(f"    DOWNLOAD FAILED {Path(key).name}: {type(exc).__name__}", flush=True)
        return None


class Geometry:
    """Fixed-grid window, and the ERA5 cell each pixel falls in.

    Built once from a sample file: the ABI fixed grid does not move, so pixel
    geolocation is identical in every scan of the mission.
    """

    def __init__(self, sample: Path):
        with xr.open_dataset(sample) as ds:
            x_axis = ds["x"].values.astype("float64")
            y_axis = ds["y"].values.astype("float64")

        self.rows, self.cols = gg.pantanal_window(
            x_axis, y_axis, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

        X, Y = np.meshgrid(x_axis[self.cols], y_axis[self.rows])
        lon, lat = gg.scan_to_lonlat(X, Y)

        self.cell_lat, self.cell_lon = target_grid()

        inside = (
            np.isfinite(lat) & np.isfinite(lon)
            & (lat >= self.cell_lat.min() - GRID_STEP / 2)
            & (lat <= self.cell_lat.max() + GRID_STEP / 2)
            & (lon >= self.cell_lon.min() - GRID_STEP / 2)
            & (lon <= self.cell_lon.max() + GRID_STEP / 2)
        )
        self.inside = inside

        ilat = np.abs(lat[inside][:, None] - self.cell_lat[None, :]).argmin(1)
        ilon = np.abs(lon[inside][:, None] - self.cell_lon[None, :]).argmin(1)
        self.n_cells = len(self.cell_lat) * len(self.cell_lon)
        self.flat_cell = ilat * len(self.cell_lon) + ilon

        self.px_per_cell = np.bincount(self.flat_cell, minlength=self.n_cells)
        print(f"  window {self.rows.stop - self.rows.start} x "
              f"{self.cols.stop - self.cols.start} px "
              f"({100 * (self.rows.stop - self.rows.start) * (self.cols.stop - self.cols.start) / 5424**2:.2f}% of disk)")
        occupied = int((self.px_per_cell > 0).sum())
        print(f"  {int(inside.sum()):,} ABI pixels mapped onto {occupied} ERA5 cells "
              f"(median {int(np.median(self.px_per_cell[self.px_per_cell > 0]))} px/cell)")


def _read_var(h5, name, window, inside):
    """Windowed read of one ABI variable, unpacked to physical units.

    Deliberately h5py rather than xarray. The variables are chunked 226x226 and
    gzipped, so a windowed read of all four costs about 1 ms -- but xarray's
    per-file decoding of a 5424x5424 array costs ~3.8 s, which at 44k files is
    the difference between hours and days. The unpacking below is what xarray
    would have done: apply _FillValue as NaN, then scale_factor/add_offset.
    """
    dset = h5[name]
    raw = dset[window][inside]
    attrs = dset.attrs

    out = raw.astype("float32")
    if "_FillValue" in attrs:
        fill = np.asarray(attrs["_FillValue"]).ravel()[0]
        out[raw == fill] = np.nan
    if "scale_factor" in attrs:
        out = out * np.asarray(attrs["scale_factor"]).ravel()[0]
    if "add_offset" in attrs:
        out = out + np.asarray(attrs["add_offset"]).ravel()[0]
    return out


def extract(path: Path, geom: Geometry, high_conf: bool):
    """Per-cell fire and observation counts for a single scan."""
    window = (geom.rows, geom.cols)
    with h5py.File(path, "r") as h5:
        mask = _read_var(h5, "Mask", window, geom.inside)
        power = _read_var(h5, "Power", window, geom.inside)
        area = _read_var(h5, "Area", window, geom.inside)
        temp = _read_var(h5, "Temp", window, geom.inside)

    fire, no_fire, _ = gg.classify(mask, high_confidence_only=high_conf)
    observed = fire | no_fire
    n = geom.n_cells
    c = geom.flat_cell

    frp = np.where(fire & np.isfinite(power), power, 0.0)
    return {
        "n_fire": np.bincount(c, weights=fire.astype("float64"), minlength=n),
        "n_obs": np.bincount(c, weights=observed.astype("float64"), minlength=n),
        "sum_frp": np.bincount(c, weights=frp, minlength=n),
        "max_frp": _max_by_cell(c, frp, n),
        "max_area": _max_by_cell(c, np.where(fire & np.isfinite(area), area, 0.0), n),
        "max_temp": _max_by_cell(c, np.where(fire & np.isfinite(temp), temp, 0.0), n),
    }


def _max_by_cell(cell_idx, values, n):
    out = np.zeros(n)
    np.maximum.at(out, cell_idx, values)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--month", type=int, default=8)
    ap.add_argument("--scans-per-hour", type=int, default=1,
                    help="ABI scans to use per hour (6 available); 1 is enough to "
                         "validate the pipeline, 6 catches short-lived fires")
    ap.add_argument("--high-confidence-only", action="store_true",
                    help="count only good/saturated fire pixels, not the "
                         "low/medium probability categories")
    ap.add_argument("--min-obs-frac", type=float, default=0.5,
                    help="drop cell-hours where fewer than this fraction of "
                         "pixels could be judged")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep-files", action="store_true",
                    help="keep the raw .nc files instead of deleting after use")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()

    start = datetime(args.year, args.month, 1)
    end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    days = [(start + timedelta(days=i)) for i in range((end - start).days)]

    print(f"GOES-16 {PRODUCT} | {args.year}-{args.month:02d} "
          f"({len(days)} days, {args.scans_per_hour} scan(s)/hour)")
    print(f"fire codes: {'high-confidence only' if args.high_confidence_only else 'all six categories'}\n")

    sample = next(CACHE_DIR.glob("*.nc"), None)
    if sample is None:
        keys = list_day(args.year, days[0].timetuple().tm_yday, session)
        sample = fetch(sorted(keys)[0], session)
    print("geometry:")
    geom = Geometry(sample)

    records = []
    for day in days:
        doy = day.timetuple().tm_yday
        keys = list_day(args.year, doy, session)
        scans = pick_scans(keys, args.scans_per_hour)
        if not scans:
            print(f"  {day.date()}: no scans listed", flush=True)
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            paths = list(pool.map(lambda s: fetch(s[1], session), scans))

        by_hour: dict[datetime, list] = {}
        for (t, _), path in zip(scans, paths):
            if path is None:
                continue
            try:
                agg = extract(path, geom, args.high_confidence_only)
            except Exception as exc:              # noqa: BLE001
                print(f"    read failed {path.name}: {type(exc).__name__}", flush=True)
                continue
            by_hour.setdefault(t.replace(minute=0, second=0, microsecond=0), []).append(agg)
            if not args.keep_files:
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
            records.append(pd.DataFrame({
                "time": hour,
                "cell": idx,
                "lat": geom.cell_lat[idx // len(geom.cell_lon)],
                "lon": geom.cell_lon[idx % len(geom.cell_lon)],
                "n_fire_px": n_fire[idx],
                "n_obs_px": n_obs[idx],
                "n_cell_px": geom.px_per_cell[idx],
                "obs_frac": obs_frac[idx],
                "max_frp": np.max([a["max_frp"] for a in aggs], axis=0)[idx],
                "sum_frp": np.sum([a["sum_frp"] for a in aggs], axis=0)[idx],
                "max_area": np.max([a["max_area"] for a in aggs], axis=0)[idx],
                "max_temp": np.max([a["max_temp"] for a in aggs], axis=0)[idx],
            }))

        n_hours = len(by_hour)
        n_fires = int(sum(a["n_fire"].sum() for aggs in by_hour.values() for a in aggs))
        print(f"  {day.date()}: {n_hours:2d} h, {len(scans):3d} scans, "
              f"{n_fires:5d} fire px", flush=True)

    if not records:
        raise SystemExit("no data extracted")

    out = pd.concat(records, ignore_index=True)
    out["fire_t"] = (out["n_fire_px"] > 0).astype(int)
    out = out.sort_values(["time", "cell"]).reset_index(drop=True)

    path = DATA_DIR / f"goes_hourly_pantanal_{args.year}{args.month:02d}.parquet"
    out.to_parquet(path, index=False)

    print(f"\nwrote {path.name}: {len(out):,} cell-hours")
    print(f"  {out['time'].min()} -> {out['time'].max()}")
    print(f"  distinct hours   : {out['time'].nunique():,}")
    print(f"  distinct cells   : {out['cell'].nunique():,}")
    print(f"  fire_t prevalence: {out['fire_t'].mean():.5f} "
          f"({int(out['fire_t'].sum()):,} positive)")
    print(f"  median obs_frac  : {out['obs_frac'].median():.3f}")
    print("\n  positives by hour of day (UTC):")
    print(out[out.fire_t == 1].groupby(out["time"].dt.hour).size().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
