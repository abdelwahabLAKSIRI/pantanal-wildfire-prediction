"""Download hourly ERA5 surface fields for the Pantanal box, any year range.

Stage 1 of the hourly persistence/propagation model. Produces
``era5_hourly_<years>.nc`` on exactly the same spatial grid as
``xhwi_era5_hourly_2024.nc``, so the two join on identical coordinates with no
interpolation.

Variables
---------
Raw from CDS: ``u10``, ``v10``, ``t2m``, ``d2m``.
Derived here: ``wind_speed``, ``wind_dir_to_deg``, ``rh`` (relative humidity)
and ``vpd`` (vapour pressure deficit).

Temperature and humidity are here because **XHWI only covers 2024**. Extending
the model back to 2019 needs a thermal forcing available for every year, and
t2m/RH/VPD are the physical quantities XHWI is itself built from -- an
unthresholded version of the same signal. Where the XHWI file does cover a year,
``build_hourly_dataset.py`` still uses it as an extra feature.

Why CDS and not the ARCO Zarr store that XHWI came from: the ARCO chunks are
[1, 721, 1440] -- one full global field per hour -- so a spatial subset still
pulls ~58 GB per year over the wire to keep 25 MB. The CDS API subsets
server-side via ``area``.

Setup (once):

    1. Create a free account at https://cds.climate.copernicus.eu/
    2. Accept the ERA5 single-levels licence on that dataset's page --
       downloads fail with a 403 until you do.
    3. Put your personal API key in %USERPROFILE%\\.cdsapirc:

           url: https://cds.climate.copernicus.eu/api
           key: <your-key>

Then:

    python download_era5_pantanal.py                  # 2019-2024, ~72 requests
    python download_era5_pantanal.py --years 2024
    python download_era5_pantanal.py --check --years 2019-2024

Requests are issued one month at a time and cached under ``era5_cache/``, so an
interrupted run resumes instead of starting over. A six-year pull is a
background job -- expect it to run for hours depending on CDS queue depth.

Note: this supersedes ``download_era5_wind_pantanal.py``, which fetched wind
only. Its cache is not reusable here because those files lack t2m/d2m.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # generated + cached inputs for this model
PROJECT_DIR = BASE_DIR.parent         # shared inputs from the original project
DATA_DIR.mkdir(exist_ok=True)
XHWI_FILE = PROJECT_DIR / "xhwi_era5_hourly_2024.nc"
CACHE_DIR = DATA_DIR / "era5_cache"

DATASET = "reanalysis-era5-single-levels"
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "2m_dewpoint_temperature",
]
RAW_VARS = ["u10", "v10", "t2m", "d2m"]


def parse_years(spec: str) -> list[int]:
    if "-" in spec:
        lo, hi = (int(x) for x in spec.split("-"))
        return list(range(lo, hi + 1))
    return [int(spec)]


def out_path(years: list[int]) -> Path:
    suffix = f"{years[0]}_{years[-1]}" if len(years) > 1 else str(years[0])
    return DATA_DIR / f"era5_hourly_{suffix}.nc"


def read_target_grid():
    """Return the (lat, lon) of the XHWI file -- the spatial grid we must match."""
    if not XHWI_FILE.exists():
        raise SystemExit(
            f"{XHWI_FILE.name} not found next to this script. It defines the target "
            "grid, so the download cannot be aligned without it."
        )
    with xr.open_dataset(XHWI_FILE) as ds:
        return ds["lat"].values.astype("float64"), ds["lon"].values.astype("float64")


def expected_time_axis(years: list[int]) -> np.ndarray:
    """Every hour of every requested year, contiguous."""
    return pd.date_range(f"{years[0]}-01-01 00:00", f"{years[-1]}-12-31 23:00",
                         freq="h").values


def area_from_grid(lat, lon):
    """CDS [N, W, S, E] covering the grid exactly."""
    return [float(lat.max()), float(lon.min()), float(lat.min()), float(lon.max())]


def download_month(client, year: int, month: int, area) -> Path:
    """Fetch one month into the cache. No-op if already cached."""
    target = CACHE_DIR / f"era5_{year}{month:02d}.nc"
    if target.exists() and target.stat().st_size > 0:
        print(f"  [{year}-{month:02d}] cached ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    print(f"  [{year}-{month:02d}] requesting...", flush=True)
    client.retrieve(
        DATASET,
        {
            "product_type": ["reanalysis"],
            "variable": VARIABLES,
            "year": [str(year)],
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": area,
            "grid": [0.25, 0.25],
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(target),
    )
    print(f"  [{year}-{month:02d}] done ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def normalise(ds: xr.Dataset) -> xr.Dataset:
    """Bring a CDS response onto XHWI's naming and dimension layout.

    The CDS ERA5 endpoint has shipped several coordinate spellings over the
    years (``time`` vs ``valid_time``) and sometimes carries singleton
    ``number``/``expver`` dimensions, so normalise defensively rather than
    assuming one layout.
    """
    renames = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        renames["valid_time"] = "time"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        renames["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        renames["longitude"] = "lon"
    ds = ds.rename(renames)

    for extra in ("number", "expver"):
        if extra in ds.dims:
            ds = ds.isel({extra: 0}, drop=True)
        elif extra in ds.coords:
            ds = ds.drop_vars(extra)

    missing = [v for v in RAW_VARS if v not in ds.data_vars]
    if missing:
        raise SystemExit(
            f"expected {RAW_VARS} in the CDS response, got {list(ds.data_vars)}"
        )

    # ERA5 ships latitude descending; XHWI is ascending.
    if ds["lat"].size > 1 and ds["lat"].values[0] > ds["lat"].values[-1]:
        ds = ds.sortby("lat")
    return ds[RAW_VARS]


def saturation_vapour_pressure_hpa(t_celsius):
    """Tetens formula over water, hPa."""
    return 6.1078 * np.exp(17.27 * t_celsius / (t_celsius + 237.3))


def add_derived(ds: xr.Dataset) -> xr.Dataset:
    """Wind speed/direction, relative humidity and vapour pressure deficit."""
    u, v = ds["u10"], ds["v10"]
    ds["wind_speed"] = np.sqrt(u**2 + v**2)

    # Meteorology reports the direction wind blows *from* (a "NE wind" comes out
    # of the northeast); trajectory bearings point the way something travels
    # *to*. Only the "to" convention is stored, because the one consumer --
    # wind_alignment -- compares it against a movement bearing, and keeping both
    # around is how the sign gets flipped by accident.
    dir_from = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    ds["wind_dir_to_deg"] = (dir_from + 180.0) % 360.0

    t_c = ds["t2m"] - 273.15
    td_c = ds["d2m"] - 273.15
    es = saturation_vapour_pressure_hpa(t_c)
    ea = saturation_vapour_pressure_hpa(td_c)
    ds["rh"] = (100.0 * ea / es).clip(0.0, 100.0)
    ds["vpd"] = (es - ea).clip(0.0, None)

    ds["u10"].attrs = {"long_name": "10 metre U wind component", "units": "m s**-1"}
    ds["v10"].attrs = {"long_name": "10 metre V wind component", "units": "m s**-1"}
    ds["t2m"].attrs = {"long_name": "2 metre temperature", "units": "K"}
    ds["d2m"].attrs = {"long_name": "2 metre dewpoint temperature", "units": "K"}
    ds["wind_speed"].attrs = {"long_name": "10 m wind speed", "units": "m s**-1"}
    ds["wind_dir_to_deg"].attrs = {
        "long_name": "10 m wind direction (vector, direction wind blows TOWARDS)",
        "units": "degrees",
        "comment": "Compare against a movement bearing; NOT the meteorological FROM convention.",
    }
    ds["rh"].attrs = {"long_name": "2 m relative humidity", "units": "%"}
    ds["vpd"].attrs = {"long_name": "2 m vapour pressure deficit", "units": "hPa"}
    return ds


def verify(ds: xr.Dataset, lat, lon, time) -> None:
    """Fail loudly if the result is not join-compatible with the XHWI grid."""
    problems = []
    if ds["lat"].size != lat.size or not np.allclose(ds["lat"].values, lat, atol=1e-4):
        problems.append(f"lat mismatch: got {ds['lat'].size} pts, expected {lat.size}")
    if ds["lon"].size != lon.size or not np.allclose(ds["lon"].values, lon, atol=1e-4):
        problems.append(f"lon mismatch: got {ds['lon'].size} pts, expected {lon.size}")
    if ds["time"].size != time.size:
        problems.append(f"time mismatch: got {ds['time'].size} steps, expected {time.size}")
    elif not np.array_equal(ds["time"].values, time):
        problems.append("time values differ from the expected hourly axis")

    for var in ("wind_speed", "t2m", "rh", "vpd"):
        if var not in ds:
            problems.append(f"missing variable {var}")
            continue
        nan_frac = float(np.isnan(ds[var].values).mean())
        if nan_frac > 0:
            problems.append(f"{var} is {nan_frac:.2%} NaN")

    if problems:
        raise SystemExit("verification FAILED:\n  - " + "\n  - ".join(problems))

    print("verification OK")
    print(f"  grid   : {ds['lat'].size} lat x {ds['lon'].size} lon x {ds['time'].size} hours")
    print(f"  period : {pd.Timestamp(ds['time'].values[0])} -> {pd.Timestamp(ds['time'].values[-1])}")
    for var, unit in [("wind_speed", "m/s"), ("t2m", "K"), ("rh", "%"), ("vpd", "hPa")]:
        a = ds[var].values
        print(f"  {var:11s}: mean {a.mean():7.2f}  p1 {np.percentile(a, 1):7.2f}  "
              f"p99 {np.percentile(a, 99):7.2f}  {unit}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2019-2024",
                        help="year or range, e.g. '2024' or '2019-2024'")
    parser.add_argument("--check", action="store_true",
                        help="verify the existing output without downloading")
    args = parser.parse_args()

    years = parse_years(args.years)
    lat, lon = read_target_grid()
    time = expected_time_axis(years)
    out = out_path(years)

    if args.check:
        if not out.exists():
            raise SystemExit(f"{out.name} does not exist yet -- nothing to check.")
        with xr.open_dataset(out) as ds:
            verify(ds, lat, lon, time)
        return 0

    area = area_from_grid(lat, lon)
    print(f"years       : {years[0]}-{years[-1]} ({len(years) * 12} monthly requests)")
    print(f"variables   : {RAW_VARS} (+ wind_speed, wind_dir_to_deg, rh, vpd)")
    print(f"target grid : {lat.size} lat x {lon.size} lon x {time.size} hours")
    print(f"CDS area    : N={area[0]} W={area[1]} S={area[2]} E={area[3]}")
    print(f"output      : {out.name}\n")

    try:
        import cdsapi
    except ImportError:
        raise SystemExit("cdsapi is not installed. Run: python -m pip install cdsapi")

    rc = Path.home() / ".cdsapirc"
    if not rc.exists():
        raise SystemExit(
            f"No CDS credentials found at {rc}.\n"
            "Create a free account at https://cds.climate.copernicus.eu/, accept the\n"
            "ERA5 single-levels licence, then write your key into that file as:\n\n"
            "    url: https://cds.climate.copernicus.eu/api\n"
            "    key: <your-key>\n"
        )

    CACHE_DIR.mkdir(exist_ok=True)
    client = cdsapi.Client()

    paths = []
    for year in years:
        for month in range(1, 13):
            paths.append(download_month(client, year, month, area))

    print("\nconcatenating...")
    parts = [normalise(xr.open_dataset(p)) for p in paths]
    ds = xr.concat(parts, dim="time").sortby("time")

    # Snap to the XHWI coordinates so tiny float32/float64 differences in the
    # grid labels never turn a join into a cartesian product.
    ds = ds.reindex(lat=lat, lon=lon, method="nearest", tolerance=0.01)
    ds = ds.sel(time=time)

    ds = add_derived(ds)
    ds.attrs = {
        "title": "ERA5 hourly surface fields over the Pantanal XHWI domain",
        "source": "ERA5 hourly single-level reanalysis via the Copernicus CDS API.",
        "summary": (
            "10 m wind, 2 m temperature and dewpoint, with derived wind speed and "
            "direction, relative humidity and vapour pressure deficit, on the same "
            "grid as xhwi_era5_hourly_2024.nc."
        ),
        "institution": "Copernicus Climate Change Service (C3S) / ECMWF",
        "license": "Copernicus Licence (ERA5)",
        "Conventions": "CF-1.10",
        "temporal_resolution": "hourly",
        "years": f"{years[0]}-{years[-1]}",
    }

    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out, encoding=encoding)
    for p in parts:
        p.close()

    print(f"wrote {out.name} ({out.stat().st_size / 1e6:.1f} MB)\n")
    with xr.open_dataset(out) as check:
        verify(check, lat, lon, time)
    return 0


if __name__ == "__main__":
    sys.exit(main())
