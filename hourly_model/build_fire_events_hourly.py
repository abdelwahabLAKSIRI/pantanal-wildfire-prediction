"""Rebuild the hourly fire-event file for an arbitrary range of years.

Reproduces the schema of ``fire_events_hourly_2024.csv`` from INPE's public
annual archive, so the hourly model can be extended beyond 2024.

Source
------
``https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/anual/Brasil_sat_ref/``
-- the **reference satellite** series (single afternoon overpass), not
``Brasil_todos_sats``. Verified against the existing 2024 file: 278,299 rows,
identical ``id_bdq`` values, identical lat/lon/biome/state, and
``detection_hour == data_pas.floor('h')`` for every row.

Clustering
----------
``event_id`` groups hotspots seen in the same overpass hour into one fire.
The original file's exact parameters are not recoverable, but DBSCAN with
haversine distance, ``eps = 1.4 km`` and ``min_samples = 1`` reproduces its
partition at ARI 0.974. Rather than bit-match one year and drift on the others,
every year here -- 2024 included -- is re-clustered with this single rule, so
the multi-year table is internally consistent.

Usage:
    python build_fire_events_hourly.py                     # 2019-2024
    python build_fire_events_hourly.py --years 2015-2024
    python build_fire_events_hourly.py --biome Pantanal    # smaller output
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.cluster import DBSCAN

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # generated + cached inputs for this model
PROJECT_DIR = BASE_DIR.parent         # shared inputs from the original project
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR = DATA_DIR / "inpe_cache"
ARCHIVE = ("https://dataserver-coids.inpe.br/queimadas/queimadas/focos/csv/"
           "anual/Brasil_sat_ref/focos_br_ref_{year}.zip")

EARTH_R_KM = 6371.0
CLUSTER_EPS_KM = 1.4
OUT_COLUMNS = ["event_id", "date", "detection_hour", "hotspot_id", "lat", "lon",
               "state", "municipality", "biome", "event_num_hotspots"]


def download_year(year: int) -> Path:
    target = CACHE_DIR / f"focos_br_ref_{year}.zip"
    if target.exists() and target.stat().st_size > 0:
        print(f"  [{year}] cached ({target.stat().st_size / 1e6:.1f} MB)")
        return target

    url = ARCHIVE.format(year=year)
    print(f"  [{year}] downloading...", flush=True)
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    target.write_bytes(r.content)
    print(f"  [{year}] done ({target.stat().st_size / 1e6:.1f} MB)")
    return target


def read_year(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        with z.open(name) as f:
            df = pd.read_csv(f, parse_dates=["data_pas"])

    rename = {"id_bdq": "hotspot_id", "estado": "state",
              "municipio": "municipality", "bioma": "biome"}
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name}: unexpected schema, missing {missing}")

    df = df.rename(columns=rename)
    df["detection_hour"] = df["data_pas"].dt.floor("h")
    df["date"] = df["data_pas"].dt.normalize()
    return df[["hotspot_id", "date", "detection_hour", "lat", "lon",
               "state", "municipality", "biome"]]


def assign_events(df: pd.DataFrame, eps_km: float, start_id: int) -> tuple[pd.DataFrame, int]:
    """Cluster hotspots into events within each overpass hour."""
    labels = np.empty(len(df), dtype=np.int64)
    next_id = start_id
    eps = eps_km / EARTH_R_KM

    positions = {k: df.index.get_indexer(v) for k, v in df.groupby("detection_hour").groups.items()}
    for hour in sorted(positions):
        pos = positions[hour]
        coords = np.radians(df.iloc[pos][["lat", "lon"]].to_numpy())
        if len(pos) == 1:
            local = np.zeros(1, dtype=np.int64)
        else:
            local = DBSCAN(eps=eps, min_samples=1, metric="haversine").fit_predict(coords)
        labels[pos] = local + next_id
        next_id += local.max() + 1

    df = df.copy()
    df["event_id"] = labels
    df["event_num_hotspots"] = df.groupby("event_id")["hotspot_id"].transform("size")
    return df, next_id


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--years", default="2019-2024", help="e.g. '2019-2024' or '2021'")
    p.add_argument("--eps-km", type=float, default=CLUSTER_EPS_KM)
    p.add_argument("--biome", default=None,
                   help="keep only this biome; shrinks the file but see --states")
    p.add_argument("--states", default=None,
                   help="comma-separated states to keep, e.g. "
                        "'MATO GROSSO DO SUL,MATO GROSSO'. Preferred over --biome "
                        "for the Pantanal model: build_hourly_dataset.py uses "
                        "non-Pantanal detections in those states to tell 'the "
                        "satellite saw nothing burning' apart from 'the satellite "
                        "did not observe', and --biome throws exactly those rows away")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if "-" in args.years:
        lo, hi = (int(x) for x in args.years.split("-"))
        years = list(range(lo, hi + 1))
    else:
        years = [int(args.years)]

    states = [s.strip().upper() for s in args.states.split(",")] if args.states else None

    suffix = f"{years[0]}_{years[-1]}" if len(years) > 1 else str(years[0])
    if args.biome:
        suffix = f"{args.biome.lower()}_{suffix}"
    elif states:
        suffix = f"regional_{suffix}"
    out = args.out or DATA_DIR / f"fire_events_hourly_{suffix}.csv"

    CACHE_DIR.mkdir(exist_ok=True)
    scope = args.biome or (", ".join(states) if states else "all")
    print(f"years: {years} | eps = {args.eps_km} km | scope = {scope}\n")

    parts = []
    next_id = 0
    for year in years:
        path = download_year(year)
        df = read_year(path)
        n_raw = len(df)

        if args.biome:
            df = df[df["biome"] == args.biome].reset_index(drop=True)
        elif states:
            df = df[df["state"].isin(states)].reset_index(drop=True)

        df, next_id = assign_events(df, args.eps_km, next_id)
        parts.append(df)
        print(f"  [{year}] {n_raw:,} hotspots -> {len(df):,} kept, "
              f"{df['event_id'].nunique():,} events")

    full = pd.concat(parts, ignore_index=True)
    full = full.sort_values(["detection_hour", "event_id"]).reset_index(drop=True)
    full = full[OUT_COLUMNS]
    full.to_csv(out, index=False)

    print(f"\nwrote {out.name}: {len(full):,} hotspots, {full['event_id'].nunique():,} events")
    print(f"  {full['date'].min().date()} -> {full['date'].max().date()}")
    print("\nhotspots per year:")
    print(full.groupby(full["date"].dt.year).size().to_string())
    if not args.biome:
        print("\nhotspots per biome:")
        print(full["biome"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
