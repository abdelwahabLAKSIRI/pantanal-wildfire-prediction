"""Train the hourly-target model on the GOES-16 table.

Unlike `../hourly_model/`, where the target is an overpass-to-overpass (~24 h)
transition, here both target and forcing are hourly.

Feature sets answer one question: **does the weather add anything once you
already know what the fire is doing?** At hourly scale fire state is enormously
informative (P(fire at t+1h | fire at t) ~ 0.75), so meteorology has to earn its
place on top of that rather than against an empty baseline.

    context      position, cyclical hour-of-day and day-of-year
    + fire       GOES fire state, its recent history, and the neighbourhood
    + met        ERA5 temperature / humidity / VPD / wind at t and lags

Two evaluations, because one year supports both and they answer different things:

**Holdout (--mode holdout)** — train Jan-Aug, test Sep-Dec. A single forward
forecast, directly comparable to the INPE-based 2024 model which used the same
cut. Reported with a fixed-hyperparameter multi-seed effect size, because a
single tuned draw has repeatedly proven unable to resolve deltas this small.

**Rolling origin (--mode rolling)** — train through month m, test month m+1, for
m = May..September. Five in-season folds, each strictly forecasting forward. Gives
a mean *and* a spread, which a single 4-month holdout cannot.

Every split embargoes 24 h before the test period: fire at t and t+1h in the same
cell is strongly autocorrelated, so without a gap the last training hour and the
first test hour are the same fire.

Usage:
    python train_goes_hourly.py --mode holdout
    python train_goes_hourly.py --mode rolling
    python train_goes_hourly.py --mode both --trials 20 --seeds 5
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import optuna  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import (average_precision_score, f1_score,  # noqa: E402
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SEED = 42

# `obs_frac` is deliberately absent: it is a quality gate used to filter rows,
# and leaving it as a predictor invites the model to learn cloud artifacts.
# `day_of_month` is gone too -- an index, not a physical quantity.
CONTEXT = ["lat", "lon", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

FIRE_STATE = [
    "n_fire_px", "fire_area_km2", "fire_frac_of_cell",
    "max_frp", "sum_frp", "max_area", "max_temp",
    "has_frp", "has_area_temp",
    "fire_lag1", "fire_lag2", "fire_lag3", "fire_lag6", "fire_lag12", "fire_lag24",
    "fire_hours_last6", "fire_hours_last24",
    "frp_lag1", "npx_lag1", "npx_delta",
    "nb_cells_burning", "nb_fire_px", "nb_cells_burning_lag1",
]

MET = [
    "t2m_t", "t2m_lag1", "t2m_lag3", "t2m_lag6", "t2m_lag12",
    "rh_t", "rh_lag1", "rh_lag3", "rh_lag6", "rh_lag12",
    "vpd_t", "vpd_lag1", "vpd_lag3", "vpd_lag6", "vpd_lag12",
    "wind_speed_t", "wind_speed_lag1", "wind_speed_lag3", "wind_speed_lag6",
    "wind_speed_lag12", "wind_u_t", "wind_v_t",
    "t2m_24h_max", "t2m_24h_mean", "vpd_24h_max", "vpd_24h_mean",
    "wind_speed_24h_max", "wind_speed_24h_mean", "rh_24h_min", "rh_24h_mean",
]

FEATURE_SETS = {
    "context": CONTEXT,
    "context_fire": CONTEXT + FIRE_STATE,
    "context_fire_met": CONTEXT + FIRE_STATE + MET,
}

TARGETS = {
    "persists_t1h": dict(scoring="f1_macro", calibrate=False, headline="roc_auc"),
    "spreads_t1h": dict(scoring="average_precision", calibrate=True, headline="pr_auc"),
}

FIXED = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
             tree_method="hist", n_jobs=-1)


def _ap_scorer(est, X, y):
    return average_precision_score(y, est.predict_proba(X)[:, 1])


def tune_xgb(X, y, trials, cv, scoring, spw=None):
    metric = "aucpr" if spw else "logloss"
    cv_scoring = _ap_scorer if scoring == "average_precision" else scoring
    extra = {"scale_pos_weight": spw} if spw else {}

    def objective(t):
        m = XGBClassifier(
            n_estimators=t.suggest_int("n_estimators", 100, 800),
            max_depth=t.suggest_int("max_depth", 2, 8),
            learning_rate=t.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=t.suggest_float("min_child_weight", 1.0, 10.0),
            gamma=t.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            objective="binary:logistic", eval_metric=metric, tree_method="hist",
            random_state=SEED, n_jobs=-1, **extra)
        return np.mean(cross_val_score(m, X, y, cv=cv, scoring=cv_scoring, n_jobs=-1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = XGBClassifier(**study.best_params, objective="binary:logistic",
                         eval_metric=metric, tree_method="hist",
                         random_state=SEED, n_jobs=-1, **extra)
    best.fit(X, y)
    return "XGBoost", best, study.best_value


def tune_rf(X, y, trials, cv, scoring, balanced=False):
    weights = ["balanced", "balanced_subsample"] if balanced else [None, "balanced_subsample"]

    def objective(t):
        m = RandomForestClassifier(
            n_estimators=t.suggest_int("n_estimators", 100, 400),
            max_depth=t.suggest_int("max_depth", 3, 25),
            min_samples_split=t.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
            max_features=t.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight=t.suggest_categorical("class_weight", weights),
            random_state=SEED, n_jobs=-1)
        return np.mean(cross_val_score(m, X, y, cv=cv, scoring=scoring, n_jobs=1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = RandomForestClassifier(**study.best_params, random_state=SEED, n_jobs=-1)
    best.fit(X, y)
    return "RandomForest", best, study.best_value


def metrics(y_true, score, thr):
    pred = (score >= thr).astype(int)
    return {
        "n": len(y_true), "prevalence": float(np.mean(y_true)), "threshold": float(thr),
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "f1_macro": f1_score(y_true, pred, average="macro", zero_division=0),
        "f1_positive": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "precision_positive": precision_score(y_true, pred, pos_label=1, zero_division=0),
        "recall_positive": recall_score(y_true, pred, pos_label=1, zero_division=0),
    }


def split(df, test_start, test_end=None, embargo_h=24):
    """Forward split with an embargo gap before the test period."""
    test_start = pd.Timestamp(test_start)
    emb = test_start - pd.Timedelta(hours=embargo_h)
    tr = df[df.time < emb]
    te = df[df.time >= test_start]
    if test_end is not None:
        te = te[te.time < pd.Timestamp(test_end)]
    return tr, te


def fit_eval(tr, te, target, info, feats, trials):
    """Tune both model families on one split and score them on the test."""
    scaler = StandardScaler().fit(tr[feats])
    Xtr, ytr = scaler.transform(tr[feats]), tr[target]
    Xte, yte = scaler.transform(te[feats]), te[target]
    cv = TimeSeriesSplit(n_splits=5)
    spw = ((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if info["calibrate"] else None

    out = []
    for tune in (tune_xgb, tune_rf):
        if tune is tune_xgb:
            name, model, cv_score = tune(Xtr, ytr, trials, cv, info["scoring"], spw)
        else:
            name, model, cv_score = tune(Xtr, ytr, trials, cv, info["scoring"],
                                         info["calibrate"])
        if info["calibrate"]:
            cut = int(0.8 * len(tr))
            val = tr.iloc[cut:]
            vs = model.predict_proba(scaler.transform(val[feats]))[:, 1]
            prec, rec, thr = precision_recall_curve(val[target], vs)
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
            threshold = thr[int(np.argmax(f1))] if len(thr) else 0.5
        else:
            threshold = 0.5
        r = metrics(yte, model.predict_proba(Xte)[:, 1], threshold)
        r.update(model=name, cv_score=cv_score)
        out.append((r, model))
    return out


def effect_size(tr, te, target, calibrate, seeds):
    """Fixed hyperparameters across feature sets, repeated over seeds."""
    spw = ((tr[target] == 0).sum() / max((tr[target] == 1).sum(), 1)) if calibrate else 1.0
    rows = []
    for set_name, feats in FEATURE_SETS.items():
        scaler = StandardScaler().fit(tr[feats])
        Xtr, Xte = scaler.transform(tr[feats]), scaler.transform(te[feats])
        for seed in range(seeds):
            m = XGBClassifier(**FIXED, scale_pos_weight=spw, random_state=seed)
            m.fit(Xtr, tr[target])
            s = m.predict_proba(Xte)[:, 1]
            rows.append({"feature_set": set_name, "seed": seed,
                         "roc_auc": roc_auc_score(te[target], s),
                         "pr_auc": average_precision_score(te[target], s)})
    return pd.DataFrame(rows)


def report_effect(ef, target, metric, prevalence):
    gmean = ef.groupby("feature_set")[metric].mean()
    gstd = ef.groupby("feature_set")[metric].std()
    base, noise = gmean["context"], gstd.max()
    print(f"\n  {target} ({metric}, prevalence {prevalence:.3f})")
    for fs in FEATURE_SETS:
        d = gmean[fs] - base
        sig = "reference" if fs == "context" else (
            f"{d / noise:+.1f} sigma" if noise > 0 else "n/a")
        print(f"    {fs:18s} {gmean[fs]:.4f} +- {gstd[fs]:.4f}   delta {d:+.4f}   {sig}")


def run_holdout(df, args):
    print("=" * 70)
    print("HOLDOUT  train Jan-Aug  ->  test Sep-Dec")
    print("=" * 70)
    tr, te = split(df, f"{args.year}-09-01", embargo_h=args.embargo_hours)
    print(f"train {len(tr):,} rows | test {len(te):,} rows "
          f"| embargo {len(df) - len(tr) - len(te):,} dropped")
    for t in TARGETS:
        print(f"  {t}: train {tr[t].mean():.4f} | test {te[t].mean():.4f}")

    rows, imps = [], []
    for target, info in TARGETS.items():
        print(f"\n  --- {target} ---")
        for set_name, feats in FEATURE_SETS.items():
            for r, model in fit_eval(tr, te, target, info, feats, args.trials):
                r.update(target=target, feature_set=set_name, n_features=len(feats))
                rows.append(r)
                imps.append(pd.DataFrame({
                    "target": target, "model": r["model"], "feature_set": set_name,
                    "feature": feats, "importance": model.feature_importances_}))
                print(f"    {set_name:18s} {r['model']:13s} "
                      f"roc_auc={r['roc_auc']:.4f}  pr_auc={r['pr_auc']:.4f}  "
                      f"f1_macro={r['f1_macro']:.4f}")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / "holdout_metrics.csv", index=False)
    pd.concat(imps, ignore_index=True).to_csv(
        RESULTS_DIR / "holdout_importance.csv", index=False)

    print("\n" + "=" * 70)
    print(f"HOLDOUT EFFECT SIZE -- fixed hyperparameters, {args.seeds} seeds")
    print("=" * 70)
    allef = []
    for target, info in TARGETS.items():
        ef = effect_size(tr, te, target, info["calibrate"], args.seeds)
        report_effect(ef, target, info["headline"], te[target].mean())
        allef.append(ef.assign(target=target))
    pd.concat(allef).to_csv(RESULTS_DIR / "holdout_effect_size.csv", index=False)


def run_rolling(df, args):
    print("\n" + "=" * 70)
    print("ROLLING ORIGIN  train through month m -> test month m+1")
    print("=" * 70)
    folds = []
    for m in range(args.rolling_first, args.rolling_last + 1):
        start = pd.Timestamp(f"{args.year}-{m + 1:02d}-01")
        end = (start + pd.DateOffset(months=1))
        tr, te = split(df, start, end, embargo_h=args.embargo_hours)
        if len(te) < 200 or tr[list(TARGETS)].nunique().min() < 2:
            print(f"  fold test {args.year}-{m+1:02d}: skipped ({len(te)} test rows)")
            continue
        folds.append((m + 1, tr, te))
        print(f"  fold test {args.year}-{m+1:02d}: train {len(tr):,} | test {len(te):,}")

    if not folds:
        print("  no usable folds")
        return

    rows = []
    for month, tr, te in folds:
        for target, info in TARGETS.items():
            ef = effect_size(tr, te, target, info["calibrate"], args.seeds)
            g = ef.groupby("feature_set")[info["headline"]].mean()
            for fs in FEATURE_SETS:
                rows.append({"test_month": month, "target": target,
                             "metric": info["headline"], "feature_set": fs,
                             "score": g[fs], "delta_vs_context": g[fs] - g["context"],
                             "n_test": len(te), "prevalence": te[target].mean()})

    roll = pd.DataFrame(rows)
    roll.to_csv(RESULTS_DIR / "rolling_origin.csv", index=False)

    for target, info in TARGETS.items():
        sub = roll[roll.target == target]
        print(f"\n  {target} ({info['headline']}) by fold:")
        piv = sub.pivot(index="test_month", columns="feature_set", values="score")
        piv = piv[[c for c in FEATURE_SETS if c in piv.columns]]
        print(piv.round(4).to_string())
        print(f"\n  mean across folds, and spread:")
        for fs in FEATURE_SETS:
            s = sub[sub.feature_set == fs].score
            d = sub[sub.feature_set == fs].delta_vs_context
            print(f"    {fs:18s} {s.mean():.4f} +- {s.std():.4f}   "
                  f"delta {d.mean():+.4f} +- {d.std():.4f}   "
                  f"({int((d > 0).sum())}/{len(d)} folds positive)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--table", type=Path, default=DATA_DIR / "goes_model_table_2024.parquet")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--mode", choices=["holdout", "rolling", "both"], default="both")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--embargo-hours", type=int, default=24)
    ap.add_argument("--rolling-first", type=int, default=5,
                    help="first training-end month (test is the month after)")
    ap.add_argument("--rolling-last", type=int, default=9)
    args = ap.parse_args()

    if not args.table.exists():
        raise SystemExit(f"{args.table.name} not found -- run build_goes_model_table.py")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.table).sort_values("time").reset_index(drop=True)
    missing = [f for fs in FEATURE_SETS.values() for f in fs if f not in df.columns]
    if missing:
        raise SystemExit(f"table is missing features: {sorted(set(missing))}")
    print(f"table: {len(df):,} rows | {df.time.min()} -> {df.time.max()}\n")

    if args.mode in ("holdout", "both"):
        run_holdout(df, args)
    if args.mode in ("rolling", "both"):
        run_rolling(df, args)

    print(f"\nwrote results to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
