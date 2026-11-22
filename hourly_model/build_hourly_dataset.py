"""Build the hourly-forced Pantanal persistence/propagation training table.

Stages 2 and 3 of the hourly model. Reads the per-overpass fire clusters and the
hourly ERA5 grid, and writes ``hourly_ml_table_pantanal.parquet`` -- one row per
fire cluster per overpass, with labels for the next overpass and features
sampled at the exact detection hour.

What the "hourly" in this model actually means
----------------------------------------------
The fire product is a single afternoon overpass: Pantanal detections land only
at 17, 18 or 19 UTC, and the overwhelming majority of fire days carry exactly
one distinct detection hour. There is no t+1h observation to label against, so
the TARGET stays an overpass-to-overpass (~24 h) transition, exactly as in the
daily model. What becomes hourly is the FORCING: every atmospheric feature is
sampled at the detection timestamp and along its lead-up.

Because the gap between consecutive overpasses is not always 24 h (a 17 UTC
detection followed by an 18 UTC one is 25 h apart), that gap is carried as an
explicit feature rather than assumed away.

Thermal forcing
---------------
XHWI covers **2024 only**, so the multi-year model's thermal forcing comes from
raw ERA5 instead: temperature, relative humidity and vapour pressure deficit,
which are the physical quantities XHWI is built from. Hours above 32 degC
reproduces XHWI's own temperature threshold as a year-independent feature.
Where the XHWI file does cover a year, it is added on top as an extra feature
block; clusters outside its coverage get NaN, and the notebook excludes that
block from the feature sets unless every row has it.

Labels
------
``persists_t1``  1 if any cluster is detected on the next overpass within
                 LINK_KM of this cluster's centroid.
``spreads_t1``   1 if a *new* cluster -- one with no predecessor of its own,
                 so not simply the continuation of a neighbouring fire --
                 appears on the next overpass in the ring (LINK_KM, RING_KM].

Rows whose next day was not observed regionally are dropped rather than
labelled 0, since absence of detection there is not evidence of extinction.

Usage:
    python build_hourly_dataset.py
    python build_hourly_dataset.py --link-km 3 --out tabela_hourly_3km.parquet
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import BallTree

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"          # generated + cached inputs for this model
PROJECT_DIR = BASE_DIR.parent         # shared inputs from the original project
DATA_DIR.mkdir(exist_ok=True)
FIRE_FILE = DATA_DIR / "fire_events_hourly_regional_2019_2024.csv"
ERA5_FILE = DATA_DIR / "era5_hourly_2019_2024.nc"
XHWI_FILE = PROJECT_DIR / "xhwi_era5_hourly_2024.nc"
BLOCKINGS = PROJECT_DIR / "serie_diaria_bloqueios_clima80_10.csv"
SACZ_FILE = PROJECT_DIR / "SACZ_daily_series.csv"
DEFAULT_OUT = DATA_DIR / "hourly_ml_table_pantanal.parquet"

BIOME = "Pantanal"
EARTH_R_KM = 6371.0

# States whose detections prove the platform was operating over the region on a
# given day. Absence of any detection here means "not observed", not "no fire".
OBSERVED_STATES = ["MATO GROSSO DO SUL", "MATO GROSSO"]

BLOCKING_COLS = ["total", "norte", "norte_h1", "norte_h2", "sul", "sul_h1", "sul_h2"]
SACZ_COLS = ["SACZ_AB", "SACZ_C", "SACZ_DE"]

WINDOW_H = 24


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------
def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees (0-360) from point 1 to point 2.

    This is a direction of travel -- where something is heading TO. It must be
    compared against ``wind_dir_to_deg``, never against the meteorological
    "comes FROM" convention, or the sign of any alignment flips.
    """
    lat1r, lat2r = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def radians_xy(df, latcol="lat", loncol="lon"):
    return np.radians(df[[latcol, loncol]].to_numpy(dtype="float64"))


def cluster_geometry(g: pd.DataFrame) -> pd.Series:
    """Shape descriptors for one overpass cluster, from its member hotspots."""
    lat, lon = g["lat"].to_numpy(), g["lon"].to_numpy()
    clat, clon = lat.mean(), lon.mean()

    # Local equirectangular projection to km -- fine over a single cluster.
    dy = (lat - clat) * 111.32
    dx = (lon - clon) * 111.32 * np.cos(np.radians(clat))
    r = np.hypot(dx, dy)

    out = {
        "lat": clat,
        "lon": clon,
        "n_hotspots": len(g),
        "cluster_spread_km": float(r.mean()),
        "cluster_radius_km": float(r.max()),
    }

    if len(g) >= 3:
        cov = np.cov(np.vstack([dx, dy]))
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        major = evecs[:, -1]
        # 1 = circular, 0 = perfectly linear
        out["cluster_elongation"] = float(
            np.sqrt(evals[0] / evals[-1]) if evals[-1] > 1e-12 else 1.0
        )
        out["cluster_axis_deg"] = float(np.degrees(np.arctan2(major[0], major[1])) % 180.0)
    else:
        out["cluster_elongation"] = 1.0
        out["cluster_axis_deg"] = np.nan

    return pd.Series(out)


# --------------------------------------------------------------------------
# stage 2: clusters, chains, labels
# --------------------------------------------------------------------------
def build_clusters(fire: pd.DataFrame) -> pd.DataFrame:
    pa = fire[fire["biome"] == BIOME]
    print(f"  {BIOME} hotspots: {len(pa):,} in {pa['event_id'].nunique():,} overpass clusters")

    geom = pa.groupby("event_id").apply(cluster_geometry, include_groups=False)
    meta = pa.groupby("event_id").agg(
        date=("date", "first"),
        detection_hour=("detection_hour", "first"),
        state=("state", "first"),
        municipality=("municipality", "first"),
    )
    clusters = meta.join(geom).reset_index()
    clusters["n_hotspots"] = clusters["n_hotspots"].astype(int)
    return clusters.sort_values(["date", "event_id"]).reset_index(drop=True)


def observed_days(fire: pd.DataFrame) -> set:
    """Days on which the platform demonstrably observed the region."""
    sub = fire[fire["state"].isin(OBSERVED_STATES)]
    return set(pd.to_datetime(sub["date"].unique()))


def regional_overpass_hour(fire: pd.DataFrame) -> dict:
    """Median detection timestamp per day across the wider region.

    Taken from MS/MT rather than from Pantanal clusters on purpose. Deriving
    the next overpass time from Pantanal detections alone would leave it
    undefined exactly on the days nothing burned there -- so any fallback value
    would encode the answer to persists_t1. The regional pass time is known
    from the orbit regardless of what burned, which is what makes it a legal
    feature.
    """
    sub = fire[fire["state"].isin(OBSERVED_STATES)]
    return sub.groupby("date")["detection_hour"].median().to_dict()


def link_chains(clusters: pd.DataFrame, link_km: float) -> pd.DataFrame:
    """Assign each cluster to a multi-day chain via nearest-predecessor linking.

    Produces ``persistence_accumulated`` (overpasses this fire has already been
    alive) and the bearing of its own displacement since the last overpass --
    causal, since it only uses the event's own past and present position.
    """
    clusters = clusters.sort_values(["date", "event_id"]).reset_index(drop=True)
    n = len(clusters)

    chain_id = np.full(n, -1, dtype=np.int64)
    chain_age = np.zeros(n, dtype=np.int64)
    prev_lat = np.full(n, np.nan)
    prev_lon = np.full(n, np.nan)

    by_day = {d: g for d, g in clusters.groupby("date")}
    next_chain = 0

    for day in sorted(by_day):
        idx = by_day[day].index.to_numpy()
        prev_day = day - pd.Timedelta(days=1)

        if prev_day in by_day:
            pidx = by_day[prev_day].index.to_numpy()
            tree = BallTree(radians_xy(clusters.loc[pidx]), metric="haversine")
            dist, nn = tree.query(radians_xy(clusters.loc[idx]), k=1)
            dist_km = dist[:, 0] * EARTH_R_KM
            matched = dist_km <= link_km
            src = pidx[nn[:, 0]]
        else:
            matched = np.zeros(len(idx), dtype=bool)
            src = np.zeros(len(idx), dtype=np.int64)

        for k, i in enumerate(idx):
            if matched[k]:
                p = src[k]
                chain_id[i] = chain_id[p]
                chain_age[i] = chain_age[p] + 1
                prev_lat[i] = clusters.at[p, "lat"]
                prev_lon[i] = clusters.at[p, "lon"]
            else:
                chain_id[i] = next_chain
                next_chain += 1

    clusters["chain_id"] = chain_id
    clusters["persistence_accumulated"] = chain_age
    clusters["has_prior_position"] = (~np.isnan(prev_lat)).astype(int)

    clusters["movement_bearing_deg"] = np.where(
        clusters["has_prior_position"] == 1,
        bearing_deg(prev_lat, prev_lon, clusters["lat"].to_numpy(), clusters["lon"].to_numpy()),
        np.nan,
    )
    dyk = (clusters["lat"].to_numpy() - prev_lat) * 111.32
    dxk = (clusters["lon"].to_numpy() - prev_lon) * 111.32 * np.cos(np.radians(clusters["lat"]))
    clusters["movement_km"] = np.hypot(dxk, dyk)
    clusters["movement_km"] = clusters["movement_km"].fillna(0.0)

    print(f"  chains: {next_chain:,} distinct fires; "
          f"{(chain_age > 0).mean():.1%} of clusters continue an earlier one")
    return clusters


def label_transitions(clusters: pd.DataFrame, link_km: float, ring_km: float,
                      obs_days: set, pass_hour: dict) -> pd.DataFrame:
    """Label persistence and propagation against the next overpass."""
    clusters = clusters.reset_index(drop=True)
    n = len(clusters)

    persists = np.zeros(n, dtype=np.int64)
    spreads = np.zeros(n, dtype=np.int64)
    next_observed = np.zeros(n, dtype=bool)
    dt_next = np.full(n, np.nan)
    nn_same_day = np.full(n, np.nan)
    n_nb_ring = np.zeros(n, dtype=np.int64)

    by_day = {d: g for d, g in clusters.groupby("date")}

    for day in sorted(by_day):
        cur = by_day[day]
        idx = cur.index.to_numpy()
        cur_rad = radians_xy(clusters.loc[idx])

        # --- same-day neighbourhood ---
        if len(idx) > 1:
            tree_self = BallTree(cur_rad, metric="haversine")
            dist, _ = tree_self.query(cur_rad, k=2)
            nn_same_day[idx] = dist[:, 1] * EARTH_R_KM
            counts = tree_self.query_radius(cur_rad, r=ring_km / EARTH_R_KM, count_only=True)
            n_nb_ring[idx] = counts - 1  # exclude self
        else:
            nn_same_day[idx] = np.nan
            n_nb_ring[idx] = 0

        nxt_day = day + pd.Timedelta(days=1)
        if nxt_day not in obs_days:
            continue  # censored: cannot distinguish "went out" from "not seen"
        next_observed[idx] = True

        # Regional pass time -- defined whether or not Pantanal burned.
        dt_next[idx] = (
            pass_hour[nxt_day].value - cur["detection_hour"].to_numpy().astype("int64")
        ) / 3.6e12

        if nxt_day not in by_day:
            continue  # observed and genuinely empty -> labels stay 0

        nxt = by_day[nxt_day]
        nidx = nxt.index.to_numpy()
        tree_next = BallTree(radians_xy(clusters.loc[nidx]), metric="haversine")

        # A next-day cluster is "new" when it starts its own chain.
        is_new_next = (clusters.loc[nidx, "persistence_accumulated"].to_numpy() == 0)

        near = tree_next.query_radius(cur_rad, r=link_km / EARTH_R_KM)
        ring = tree_next.query_radius(cur_rad, r=ring_km / EARTH_R_KM)

        for k, i in enumerate(idx):
            persists[i] = 1 if len(near[k]) > 0 else 0
            in_ring_only = np.setdiff1d(ring[k], near[k], assume_unique=False)
            spreads[i] = 1 if is_new_next[in_ring_only].any() else 0

    clusters["persists_t1"] = persists
    clusters["spreads_t1"] = spreads
    clusters["next_day_observed"] = next_observed
    clusters["dt_next_overpass_h"] = dt_next
    clusters["n_clusters_ring"] = n_nb_ring

    # An isolated cluster has no same-day neighbour at all. Encode that as a
    # flag plus a far sentinel rather than a NaN, since the scaler and the
    # RandomForest both choke on missing values.
    clusters["has_same_day_neighbor"] = (~np.isnan(nn_same_day)).astype(int)
    clusters["nn_dist_km"] = np.where(np.isnan(nn_same_day), 500.0, nn_same_day)

    kept = clusters[clusters["next_day_observed"]].copy()
    dropped = len(clusters) - len(kept)
    print(f"  labels: kept {len(kept):,} clusters, dropped {dropped:,} with an unobserved next day")
    print(f"    persists_t1 rate = {kept['persists_t1'].mean():.3f} ({kept['persists_t1'].sum():,} positive)")
    print(f"    spreads_t1  rate = {kept['spreads_t1'].mean():.3f} ({kept['spreads_t1'].sum():,} positive)")
    return kept.reset_index(drop=True)


# --------------------------------------------------------------------------
# stage 3: hourly forcing
# --------------------------------------------------------------------------
def grid_index(clusters: pd.DataFrame, lat, lon, time_axis):
    """Nearest grid cell and exact hour index for every cluster.

    ``it`` is -1 where the detection falls outside the grid's time axis, which
    is how a partial-coverage file (XHWI, 2024 only) is handled without
    silently sampling the wrong hour.
    """
    ilat = np.abs(clusters["lat"].to_numpy()[:, None] - lat[None, :]).argmin(1)
    ilon = np.abs(clusters["lon"].to_numpy()[:, None] - lon[None, :]).argmin(1)

    hours = clusters["detection_hour"].to_numpy().astype("datetime64[ns]")
    pos = np.searchsorted(time_axis, hours)
    pos_clipped = np.clip(pos, 0, len(time_axis) - 1)
    it = np.where(time_axis[pos_clipped] == hours, pos_clipped, -1)
    return ilat, ilon, it


def day_start_index(clusters: pd.DataFrame, time_axis):
    """Index of 00:00 on each cluster's calendar day, -1 if out of range."""
    days = clusters["date"].to_numpy().astype("datetime64[ns]")
    pos = np.searchsorted(time_axis, days)
    pos_clipped = np.clip(pos, 0, len(time_axis) - 1)
    return np.where(time_axis[pos_clipped] == days, pos_clipped, -1)


def sample_window(cube, ilat, ilon, it, back_hours=WINDOW_H):
    """(n, back_hours) array ending at the detection hour inclusive.

    Rows with ``it < 0`` come back as NaN.
    """
    offsets = np.arange(back_hours - 1, -1, -1)
    tidx = np.clip(it[:, None] - offsets[None, :], 0, cube.shape[0] - 1)
    out = cube[tidx, ilat[:, None], ilon[:, None]].astype("float64")
    out[it < 0, :] = np.nan
    return out


def sample_calendar_day(cube, ilat, ilon, i0):
    """(n, 24) array covering 00:00-23:00 of each cluster's own day."""
    tidx = np.clip(i0[:, None] + np.arange(24)[None, :], 0, cube.shape[0] - 1)
    out = cube[tidx, ilat[:, None], ilon[:, None]].astype("float64")
    out[i0 < 0, :] = np.nan
    return out


def add_window_features(clusters, win, day, prefix, lags=(1, 3, 6, 12),
                        thresholds=None, want_min=False):
    """Standard lag / window / daily block for one gridded variable."""
    clusters[f"{prefix}_t"] = win[:, -1]
    for lag in lags:
        clusters[f"{prefix}_lag{lag}"] = win[:, -1 - lag]

    clusters[f"{prefix}_24h_max"] = np.nanmax(win, axis=1)
    clusters[f"{prefix}_24h_mean"] = np.nanmean(win, axis=1)
    if want_min:
        clusters[f"{prefix}_24h_min"] = np.nanmin(win, axis=1)
    # Local night and morning for a ~17-18 UTC detection is the preceding 12 h.
    clusters[f"{prefix}_prev12h_min"] = np.nanmin(win[:, :12], axis=1)
    clusters[f"{prefix}_daily_mean"] = np.nanmean(day, axis=1)
    clusters[f"{prefix}_daily_max"] = np.nanmax(day, axis=1)

    for name, thr in (thresholds or {}).items():
        clusters[f"{prefix}_24h_hours_{name}"] = (win > thr).sum(1)
    return clusters


def add_era5_features(clusters: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Thermal, humidity and wind features, sampled one variable at a time.

    Held one variable at a time on purpose: six years of hourly fields is a few
    gigabytes if every variable is materialised at once.
    """
    with xr.open_dataset(path) as ds:
        lat = ds["lat"].values.astype("float64")
        lon = ds["lon"].values.astype("float64")
        time_axis = ds["time"].values

        ilat, ilon, it = grid_index(clusters, lat, lon, time_axis)
        i0 = day_start_index(clusters, time_axis)
        outside = (it < 0).sum()
        if outside:
            raise SystemExit(
                f"{outside:,} clusters fall outside {path.name}'s time axis "
                f"({pd.Timestamp(time_axis[0])} -> {pd.Timestamp(time_axis[-1])}). "
                "Re-run download_era5_pantanal.py for the full year range."
            )

        # --- temperature (degC) ---
        t = ds["t2m"].values - 273.15
        win, day = sample_window(t, ilat, ilon, it), sample_calendar_day(t, ilat, ilon, i0)
        # 32 degC mirrors the XHWI temperature threshold, giving a comparable
        # heat-exposure count for the years XHWI does not cover.
        clusters = add_window_features(clusters, win, day, "t2m",
                                       thresholds={"gt32": 32.0, "gt35": 35.0})
        del t, win, day

        # --- relative humidity (%) ---
        rh = ds["rh"].values
        win, day = sample_window(rh, ilat, ilon, it), sample_calendar_day(rh, ilat, ilon, i0)
        clusters = add_window_features(clusters, win, day, "rh", lags=(3, 6, 12),
                                       want_min=True)
        # Counted here rather than through the threshold helper, which counts
        # exceedances above a level; for humidity the dry tail is what matters.
        clusters["rh_24h_hours_lt30"] = (win < 30.0).sum(1)
        del rh, win, day

        # --- vapour pressure deficit (hPa) ---
        vpd = ds["vpd"].values
        win, day = sample_window(vpd, ilat, ilon, it), sample_calendar_day(vpd, ilat, ilon, i0)
        clusters = add_window_features(clusters, win, day, "vpd", lags=(3, 6, 12),
                                       thresholds={"gt20": 20.0})
        del vpd, win, day

        # --- wind speed (m/s) ---
        spd = ds["wind_speed"].values
        win, day = sample_window(spd, ilat, ilon, it), sample_calendar_day(spd, ilat, ilon, i0)
        clusters = add_window_features(clusters, win, day, "wind_speed", lags=(3, 6, 12))
        del spd, win, day

        # --- wind direction and components ---
        dir_to = ds["wind_dir_to_deg"].values
        dir_win = sample_window(dir_to, ilat, ilon, it)
        wdir_to = dir_win[:, -1]
        rad = np.radians(dir_win)
        # 0 = perfectly steady direction, 1 = uniformly scattered
        clusters["wind_dir_var_24h"] = 1.0 - np.hypot(
            np.cos(rad).mean(1), np.sin(rad).mean(1))
        clusters["wind_dir_to_deg"] = wdir_to
        clusters["wind_dir_sin"] = np.sin(np.radians(wdir_to))
        clusters["wind_dir_cos"] = np.cos(np.radians(wdir_to))
        del dir_to, dir_win, rad

        clusters["wind_u_t"] = ds["u10"].values[it, ilat, ilon]
        clusters["wind_v_t"] = ds["v10"].values[it, ilat, ilon]

    # Both angles are "direction of travel", so +1 really is a tailwind here.
    align = np.cos(np.radians(clusters["movement_bearing_deg"].to_numpy() - wdir_to))
    clusters["wind_alignment"] = np.nan_to_num(align, nan=0.0)
    return clusters


def add_xhwi_features(clusters: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, list]:
    """Optional XHWI block. Returns the columns added and NaNs outside coverage."""
    if not path.exists():
        print(f"  XHWI: {path.name} not found, skipping")
        return clusters, []

    with xr.open_dataset(path) as ds:
        lat = ds["lat"].values.astype("float64")
        lon = ds["lon"].values.astype("float64")
        time_axis = ds["time"].values
        cube = ds["xhwi"].values

    ilat, ilon, it = grid_index(clusters, lat, lon, time_axis)
    i0 = day_start_index(clusters, time_axis)
    covered = (it >= 0)

    win = sample_window(cube, ilat, ilon, it)
    day = sample_calendar_day(cube, ilat, ilon, i0)

    before = set(clusters.columns)
    # Rows outside coverage are all-NaN slices, which the nan-aware reducers
    # warn about before returning the NaN we want. Blanked explicitly below.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        clusters = add_window_features(clusters, win, day, "xhwi",
                                       thresholds={"pos": 0.0, "gt01": 0.1})

        # Hours between the window's peak and the detection. When the whole
        # window is zero -- no heat anywhere in the last 24 h -- argmax lands on
        # index 0 and would report "peak was 23 h ago", making the calmest
        # conditions look identical to a peak that has fully decayed. Those rows
        # get -1 instead.
        hours_since = (win.shape[1] - 1) - np.argmax(np.nan_to_num(win, nan=-1.0), axis=1)
        peak = np.nanmax(win, axis=1)
    clusters["xhwi_hours_since_peak"] = np.where(peak > 0, hours_since, -1.0)

    added = [c for c in clusters.columns if c not in before]
    clusters.loc[~covered, added] = np.nan

    years = sorted(clusters.loc[covered, "date"].dt.year.unique())
    print(f"  XHWI: covers {covered.sum():,} of {len(clusters):,} clusters "
          f"(years {years}); {len(added)} columns")
    return clusters, added


def add_context_features(clusters: pd.DataFrame) -> pd.DataFrame:
    doy = clusters["date"].dt.dayofyear
    month = clusters["date"].dt.month
    days_in_year = np.where(clusters["date"].dt.is_leap_year, 366, 365)
    clusters["day_year_sin"] = np.sin(2 * np.pi * doy / days_in_year)
    clusters["day_year_cos"] = np.cos(2 * np.pi * doy / days_in_year)
    clusters["month_sin"] = np.sin(2 * np.pi * month / 12)
    clusters["month_cos"] = np.cos(2 * np.pi * month / 12)
    clusters["year"] = clusters["date"].dt.year
    clusters["detection_hour_utc"] = clusters["detection_hour"].dt.hour

    # The cluster's major axis is an undirected orientation on [0, 180), so it
    # wraps at 180 rather than 360 -- hence the doubled angle. Undefined for
    # clusters of fewer than three hotspots, which get a flag instead.
    axis = clusters["cluster_axis_deg"]
    clusters["has_axis"] = axis.notna().astype(int)
    clusters["cluster_axis_sin"] = np.sin(2 * np.radians(axis)).fillna(0.0)
    clusters["cluster_axis_cos"] = np.cos(2 * np.radians(axis)).fillna(0.0)

    for path, cols, label in [(BLOCKINGS, BLOCKING_COLS, "blocking"),
                              (SACZ_FILE, SACZ_COLS, "SACZ")]:
        if not path.exists():
            print(f"  {label}: file missing, skipping")
            continue
        aux = pd.read_csv(path, parse_dates=["time"])
        keep = [c for c in cols if c in aux.columns]
        clusters = clusters.merge(
            aux[["time"] + keep].rename(columns={"time": "date"}), on="date", how="left"
        )
        missing = clusters[keep].isna().any(axis=1).sum()
        print(f"  {label}: merged {len(keep)} columns, {missing} rows unmatched")
    return clusters


# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fire", type=Path, default=FIRE_FILE)
    p.add_argument("--era5", type=Path, default=ERA5_FILE)
    p.add_argument("--xhwi", type=Path, default=XHWI_FILE)
    p.add_argument("--link-km", type=float, default=5.0,
                   help="radius that counts as the same fire on the next overpass")
    p.add_argument("--ring-km", type=float, default=20.0,
                   help="outer radius of the propagation ring")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    for f in (args.fire, args.era5):
        if not f.exists():
            raise SystemExit(
                f"missing required input: {f.name}\n"
                "  fire events : python build_fire_events_hourly.py --years 2019-2024 "
                "--states \"MATO GROSSO DO SUL,MATO GROSSO\"\n"
                "  ERA5 fields : python download_era5_pantanal.py"
            )

    print(f"link radius {args.link_km} km | propagation ring ({args.link_km}, {args.ring_km}] km\n")

    print("stage 2: clusters and chains")
    fire = pd.read_csv(args.fire, parse_dates=["date", "detection_hour"])
    obs = observed_days(fire)
    pass_hour = regional_overpass_hour(fire)
    print(f"  region observed on {len(obs):,} days "
          f"({fire['date'].min().date()} -> {fire['date'].max().date()})")
    clusters = build_clusters(fire)
    clusters = link_chains(clusters, args.link_km)
    clusters = label_transitions(clusters, args.link_km, args.ring_km, obs, pass_hour)

    print("\nstage 3: hourly forcing")
    clusters = add_era5_features(clusters, args.era5)
    print("  ERA5: t2m / rh / vpd / wind sampled at detection hour + 24 h lead-up")
    clusters, xhwi_cols = add_xhwi_features(clusters, args.xhwi)
    clusters = add_context_features(clusters)

    clusters["biome"] = BIOME
    clusters["link_km"] = args.link_km
    clusters["ring_km"] = args.ring_km

    clusters = clusters.sort_values("detection_hour").reset_index(drop=True)
    clusters.to_parquet(args.out, index=False)

    print(f"\nwrote {args.out.name}: {len(clusters):,} rows x {clusters.shape[1]} columns")
    print(f"  {clusters['date'].min().date()} -> {clusters['date'].max().date()}")
    print("\nclusters and target rates per year:")
    print(clusters.groupby("year").agg(
        n=("event_id", "size"),
        persists=("persists_t1", "mean"),
        spreads=("spreads_t1", "mean"),
    ).round(3).to_string())

    nan_cols = clusters.isna().sum()
    nan_cols = nan_cols[nan_cols > 0]
    if len(nan_cols):
        expected = set(xhwi_cols) | {"cluster_axis_deg", "movement_bearing_deg"}
        unexpected = [c for c in nan_cols.index if c not in expected]
        print(f"\n  columns with NaN: {len(nan_cols)} "
              f"({len(xhwi_cols)} from partial XHWI coverage, 2 circular diagnostics)")
        if unexpected:
            print("  UNEXPECTED NaN columns:")
            print(nan_cols[unexpected].to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
