"""Measure the feature-set effect against run-to-run noise.

Why this exists
---------------
The notebook tunes each feature set with its own Optuna study. That makes the
comparison between sets confounded: part of any observed delta is just a luckier
hyperparameter draw. Re-running an IDENTICAL baseline configuration with an
unseeded sampler moved persistence ROC-AUC from 0.631 to 0.601 -- as large as the
deltas being claimed.

Seeding the sampler makes runs reproducible, but reproducible is not the same as
significant: it fixes one arbitrary draw rather than telling you how much the
draw mattered.

So this script does the comparison the other way round: **identical fixed
hyperparameters across every feature set**, repeated over several estimator
seeds. Tuning is removed from the comparison entirely, and the spread across
seeds gives an honest noise floor to read the deltas against.

Usage:
    python feature_set_effect_size.py
    python feature_set_effect_size.py --seeds 10
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "hourly_ml_table_pantanal.parquet"
RESULTS_DIR = BASE_DIR / "results"
YEAR_TRAIN_END = 2022

BASELINE = [
    'lat', 'lon', 'n_hotspots', 'persistence_accumulated',
    'day_year_sin', 'day_year_cos', 'month_sin', 'month_cos', 'year',
    't2m_daily_mean', 't2m_daily_max', 'rh_daily_mean', 'rh_daily_max',
    'vpd_daily_mean', 'vpd_daily_max', 'wind_speed_daily_mean', 'wind_speed_daily_max',
    'total', 'norte', 'norte_h1', 'norte_h2', 'sul', 'sul_h1', 'sul_h2',
    'SACZ_AB', 'SACZ_C', 'SACZ_DE',
]
THERMAL = [
    't2m_t', 't2m_lag1', 't2m_lag3', 't2m_lag6', 't2m_lag12',
    't2m_24h_max', 't2m_24h_mean', 't2m_prev12h_min',
    't2m_24h_hours_gt32', 't2m_24h_hours_gt35',
    'rh_t', 'rh_lag3', 'rh_lag6', 'rh_lag12',
    'rh_24h_min', 'rh_24h_mean', 'rh_prev12h_min', 'rh_24h_hours_lt30',
    'vpd_t', 'vpd_lag3', 'vpd_lag6', 'vpd_lag12',
    'vpd_24h_max', 'vpd_24h_mean', 'vpd_prev12h_min', 'vpd_24h_hours_gt20',
]
GEOMETRY = [
    'cluster_spread_km', 'cluster_radius_km', 'cluster_elongation',
    'has_axis', 'cluster_axis_sin', 'cluster_axis_cos',
    'nn_dist_km', 'has_same_day_neighbor', 'n_clusters_ring',
    'movement_km', 'has_prior_position',
]
WIND = [
    'wind_speed_t', 'wind_speed_lag3', 'wind_speed_lag6', 'wind_speed_lag12',
    'wind_speed_24h_max', 'wind_speed_24h_mean', 'wind_speed_prev12h_min',
    'wind_dir_var_24h', 'wind_u_t', 'wind_v_t',
    'wind_dir_sin', 'wind_dir_cos', 'wind_alignment',
]
TIMING = ['dt_next_overpass_h', 'detection_hour_utc']

FEATURE_SETS = {
    "baseline": BASELINE,
    "with_hourly": BASELINE + THERMAL + GEOMETRY,
    "with_hourly_wind": BASELINE + THERMAL + GEOMETRY + WIND,
    # kept as a check that the dropped block really is inert
    "with_hourly_wind_timing": BASELINE + THERMAL + GEOMETRY + WIND + TIMING,
}

# One fixed configuration for every set -- deliberately untuned.
FIXED = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8,
             objective="binary:logistic", tree_method="hist", n_jobs=-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    if not DATA_FILE.exists():
        raise SystemExit(f"{DATA_FILE.name} not found -- run build_hourly_dataset.py first.")

    df = pd.read_parquet(DATA_FILE).sort_values("detection_hour").reset_index(drop=True)
    test_chains = set(df.loc[df.year > YEAR_TRAIN_END, "chain_id"])
    tr = df[(df.year <= YEAR_TRAIN_END) & (~df.chain_id.isin(test_chains))]
    te = df[df.year > YEAR_TRAIN_END]

    print(f"train {len(tr):,} | test {len(te):,} | {args.seeds} seeds per configuration")
    print(f"fixed hyperparameters: {FIXED['n_estimators']} trees, "
          f"max_depth={FIXED['max_depth']}, lr={FIXED['learning_rate']}\n")

    rows = []
    for target in ["persists_t1", "spreads_t1"]:
        spw = (tr[target] == 0).sum() / (tr[target] == 1).sum() if target == "spreads_t1" else 1.0
        for set_name, feats in FEATURE_SETS.items():
            scaler = StandardScaler().fit(tr[feats])
            Xtr, Xte = scaler.transform(tr[feats]), scaler.transform(te[feats])
            for seed in range(args.seeds):
                model = XGBClassifier(**FIXED, scale_pos_weight=spw, random_state=seed)
                model.fit(Xtr, tr[target])
                score = model.predict_proba(Xte)[:, 1]
                rows.append({
                    "target": target, "feature_set": set_name, "seed": seed,
                    "roc_auc": roc_auc_score(te[target], score),
                    "pr_auc": average_precision_score(te[target], score),
                })

    raw = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RESULTS_DIR / "feature_set_effect_size_raw.csv", index=False)

    metric_for = {"persists_t1": "roc_auc", "spreads_t1": "pr_auc"}
    summary = []
    for target, metric in metric_for.items():
        g = raw[raw.target == target].groupby("feature_set")[metric]
        base_mean = g.mean()["baseline"]
        noise = raw[raw.target == target].groupby("feature_set")[metric].std().max()
        for set_name in FEATURE_SETS:
            summary.append({
                "target": target, "metric": metric, "feature_set": set_name,
                "mean": round(g.mean()[set_name], 4),
                "std_across_seeds": round(g.std()[set_name], 4),
                "delta_vs_baseline": round(g.mean()[set_name] - base_mean, 4),
                "delta_over_noise": (round((g.mean()[set_name] - base_mean) / noise, 1)
                                     if noise > 0 else np.nan),
            })

    out = pd.DataFrame(summary)
    out.to_csv(RESULTS_DIR / "feature_set_effect_size.csv", index=False)

    for target, metric in metric_for.items():
        prevalence = te[target].mean()
        print(f"--- {target} ({metric}, prevalence {prevalence:.3f}) ---")
        print(out[out.target == target].drop(columns=["target", "metric"]).to_string(index=False))
        print()

    print("delta_over_noise = delta divided by the largest across-seed std.")
    print("Below ~2 the difference is not distinguishable from seed noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
