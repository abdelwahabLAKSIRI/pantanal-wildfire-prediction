"""Train the split SPREAD and IGNITION models.

Both replace the single `spreads_t1h` target, which three diagnostics showed was
not measuring spread at all -- see `build_goes_split_targets.py`.

    SPREAD    rows = fire pixels at 2 km with somewhere to go.
              Target: does an adjacent unburnt pixel ignite at t+1h?
    IGNITION  rows = observed 0.25 deg cells with no fire in or beside them.
              Target: does fire appear at t+1h?

Feature sets are nested so each block's contribution is isolated, and every
comparison is scored at fixed hyperparameters across several estimator seeds --
tuned single draws have repeatedly failed to resolve differences this small.

Leakage note
------------
`spread_bearing`, `spreads_downwind` and `wind_spread_angle` are computed *from*
the outcome -- they describe where the fire went, which is only known once it has
gone. They are excluded from every feature set and asserted against at load.

Metric note for IGNITION
------------------------
Its negatives are subsampled (default 20:1), so the table's prevalence is
artificially high. ROC-AUC is invariant to class balance and transfers to the
real rate; PR-AUC does not, and is only comparable between models scored on this
same table.

Usage:
    python train_goes_split.py --period 2024_01-12
    python train_goes_split.py --period 2024_01-12 --mode both --trials 20
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
N_LON = 29

# Derived from the outcome; never features.
LEAKY = {"spread_bearing", "spreads_downwind", "wind_spread_angle"}

CONTEXT = ["lat", "lon", "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

SPREAD_SETS = {
    "context": CONTEXT,
    "context_fire": CONTEXT + [
        "frp", "area", "temp", "nb_burning_now", "patch_px", "region_fire_px",
        "fire_hours_last6", "fire_hours_last24"],
    "context_fire_met": CONTEXT + [
        "frp", "area", "temp", "nb_burning_now", "patch_px", "region_fire_px",
        "fire_hours_last6", "fire_hours_last24",
        "t2m", "rh", "vpd", "wind_speed", "wind_u", "wind_v"],
}

IGNITION_SETS = {
    "context": CONTEXT,
    "context_activity": CONTEXT + [
        "dist_to_fire_km", "cell_fire_hours_last24", "region_fire_cells"],
    "context_activity_met": CONTEXT + [
        "dist_to_fire_km", "cell_fire_hours_last24", "region_fire_cells",
        "t2m", "rh", "vpd", "wind_speed"],
}

FIXED = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, objective="binary:logistic",
             tree_method="hist", n_jobs=-1)


def _ap_scorer(est, X, y):
    return average_precision_score(y, est.predict_proba(X)[:, 1])


def tune_xgb(X, y, trials, cv, scoring, spw):
    cv_scoring = _ap_scorer if scoring == "average_precision" else scoring

    def objective(t):
        m = XGBClassifier(
            n_estimators=t.suggest_int("n_estimators", 100, 700),
            max_depth=t.suggest_int("max_depth", 2, 8),
            learning_rate=t.suggest_float("learning_rate", 1e-3, 0.2, log=True),
            subsample=t.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=t.suggest_float("colsample_bytree", 0.6, 1.0),
            min_child_weight=t.suggest_float("min_child_weight", 1.0, 10.0),
            gamma=t.suggest_float("gamma", 0.0, 5.0),
            reg_alpha=t.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            reg_lambda=t.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            scale_pos_weight=spw, objective="binary:logistic",
            eval_metric="aucpr", tree_method="hist", random_state=SEED, n_jobs=-1)
        return np.mean(cross_val_score(m, X, y, cv=cv, scoring=cv_scoring, n_jobs=-1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = XGBClassifier(**study.best_params, scale_pos_weight=spw,
                         objective="binary:logistic", eval_metric="aucpr",
                         tree_method="hist", random_state=SEED, n_jobs=-1)
    best.fit(X, y)
    return "XGBoost", best, study.best_value


def tune_rf(X, y, trials, cv, scoring):
    def objective(t):
        m = RandomForestClassifier(
            n_estimators=t.suggest_int("n_estimators", 100, 350),
            max_depth=t.suggest_int("max_depth", 3, 25),
            min_samples_split=t.suggest_int("min_samples_split", 2, 20),
            min_samples_leaf=t.suggest_int("min_samples_leaf", 1, 10),
            max_features=t.suggest_categorical("max_features", ["sqrt", "log2", None]),
            class_weight=t.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample"]),
            random_state=SEED, n_jobs=-1)
        return np.mean(cross_val_score(m, X, y, cv=cv, scoring=scoring, n_jobs=1))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials)
    best = RandomForestClassifier(**study.best_params, random_state=SEED, n_jobs=-1)
    best.fit(X, y)
    return "RandomForest", best, study.best_value


def metrics(y, score, thr):
    pred = (score >= thr).astype(int)
    return {
        "n": len(y), "prevalence": float(np.mean(y)), "threshold": float(thr),
        "roc_auc": roc_auc_score(y, score),
        "pr_auc": average_precision_score(y, score),
        "lift_over_prevalence": average_precision_score(y, score) / float(np.mean(y)),
        "f1_macro": f1_score(y, pred, average="macro", zero_division=0),
        "f1_positive": f1_score(y, pred, pos_label=1, zero_division=0),
        "precision_positive": precision_score(y, pred, pos_label=1, zero_division=0),
        "recall_positive": recall_score(y, pred, pos_label=1, zero_division=0),
    }


def prepare(path: Path, target: str) -> pd.DataFrame:
    df = pd.read_parquet(path).sort_values("time").reset_index(drop=True)
    if "lat" not in df.columns:
        df["lat"] = -25.0 + (df["cell"] // N_LON) * 0.25
        df["lon"] = -60.0 + (df["cell"] % N_LON) * 0.25
    return df


def effect_size(tr, te, target, sets, seeds, headline):
    spw = (tr[target] == 0).sum() / max((tr[target] == 1).sum(), 1)
    rows = []
    for name, feats in sets.items():
        assert not (set(feats) & LEAKY), f"{name} contains outcome-derived features"
        scaler = StandardScaler().fit(tr[feats])
        Xtr, Xte = scaler.transform(tr[feats]), scaler.transform(te[feats])
        for seed in range(seeds):
            m = XGBClassifier(**FIXED, scale_pos_weight=spw, random_state=seed)
            m.fit(Xtr, tr[target])
            s = m.predict_proba(Xte)[:, 1]
            rows.append({"feature_set": name, "seed": seed,
                         "roc_auc": roc_auc_score(te[target], s),
                         "pr_auc": average_precision_score(te[target], s)})
    return pd.DataFrame(rows)


def report_effect(ef, sets, metric, prevalence, label):
    gmean = ef.groupby("feature_set")[metric].mean()
    gstd = ef.groupby("feature_set")[metric].std()
    base, noise = gmean["context"], gstd.max()
    print(f"\n  {label} ({metric}, prevalence {prevalence:.4f})")
    prev_set = None
    for fs in sets:
        d = gmean[fs] - base
        sig = "reference" if fs == "context" else f"{d / noise:+.1f} sigma"
        extra = ""
        if prev_set is not None:
            extra = f"   [{gmean[fs] - gmean[prev_set]:+.4f} over {prev_set}]"
        print(f"    {fs:22s} {gmean[fs]:.4f} +- {gstd[fs]:.4f}   "
              f"delta {d:+.4f}   {sig}{extra}")
        prev_set = fs


def run_one(df, target, sets, headline, label, args, tag):
    print("\n" + "=" * 74)
    print(f"{label}   target={target}   {len(df):,} rows   "
          f"prevalence {df[target].mean():.4f}")
    print("=" * 74)

    cut = pd.Timestamp(args.test_from)
    emb = cut - pd.Timedelta(hours=args.embargo_hours)
    tr, te = df[df.time < emb], df[df.time >= cut]
    print(f"train {len(tr):,} (< {emb.date()}) | test {len(te):,} (>= {cut.date()}) "
          f"| embargo {len(df) - len(tr) - len(te):,}")
    print(f"  prevalence: train {tr[target].mean():.4f} | test {te[target].mean():.4f}")

    rows, imps = [], []
    for name, feats in sets.items():
        assert not (set(feats) & LEAKY)
        scaler = StandardScaler().fit(tr[feats])
        Xtr, Xte = scaler.transform(tr[feats]), scaler.transform(te[feats])
        cv = TimeSeriesSplit(n_splits=4)
        spw = (tr[target] == 0).sum() / max((tr[target] == 1).sum(), 1)

        for tune in (tune_xgb, tune_rf):
            if tune is tune_xgb:
                mname, model, cvs = tune(Xtr, tr[target], args.trials, cv,
                                         "average_precision", spw)
            else:
                mname, model, cvs = tune(Xtr, tr[target], args.trials, cv,
                                         "average_precision")
            vs = model.predict_proba(Xtr[int(0.8 * len(tr)):])[:, 1]
            yv = tr[target].iloc[int(0.8 * len(tr)):]
            prec, rec, thr = precision_recall_curve(yv, vs)
            f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
            threshold = thr[int(np.argmax(f1))] if len(thr) else 0.5

            r = metrics(te[target], model.predict_proba(Xte)[:, 1], threshold)
            r.update(target=target, model=mname, feature_set=name,
                     n_features=len(feats), cv_score=cvs)
            rows.append(r)
            imps.append(pd.DataFrame({
                "target": target, "model": mname, "feature_set": name,
                "feature": feats, "importance": model.feature_importances_}))
            print(f"  {name:22s} {mname:13s} roc_auc={r['roc_auc']:.4f}  "
                  f"pr_auc={r['pr_auc']:.4f}  lift={r['lift_over_prevalence']:.2f}x")

    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"{tag}_metrics.csv", index=False)
    pd.concat(imps, ignore_index=True).to_csv(
        RESULTS_DIR / f"{tag}_importance.csv", index=False)

    print(f"\n  EFFECT SIZE -- fixed hyperparameters, {args.seeds} seeds")
    ef = effect_size(tr, te, target, sets, args.seeds, headline)
    report_effect(ef, sets, headline, te[target].mean(), label)
    ef.to_csv(RESULTS_DIR / f"{tag}_effect_size.csv", index=False)

    # rolling origin over the fire season
    if args.rolling:
        print(f"\n  ROLLING ORIGIN")
        rows = []
        for m in range(args.rolling_first, args.rolling_last + 1):
            start = pd.Timestamp(f"{args.year}-{m + 1:02d}-01")
            end = start + pd.DateOffset(months=1)
            e = start - pd.Timedelta(hours=args.embargo_hours)
            trf, tef = df[df.time < e], df[(df.time >= start) & (df.time < end)]
            if len(tef) < 300 or tef[target].nunique() < 2 or trf[target].nunique() < 2:
                continue
            ef = effect_size(trf, tef, target, sets, args.seeds, headline)
            g = ef.groupby("feature_set")[headline].mean()
            for fs in sets:
                rows.append({"test_month": m + 1, "feature_set": fs,
                             "score": g[fs], "delta_vs_context": g[fs] - g["context"],
                             "n_test": len(tef), "prevalence": tef[target].mean()})
        if rows:
            roll = pd.DataFrame(rows)
            roll.to_csv(RESULTS_DIR / f"{tag}_rolling.csv", index=False)
            piv = roll.pivot(index="test_month", columns="feature_set", values="score")
            print(piv[[c for c in sets if c in piv.columns]].round(4).to_string())
            names = list(sets)
            for a, b in zip(names[:-1], names[1:]):
                d = (roll[roll.feature_set == b].set_index("test_month").score
                     - roll[roll.feature_set == a].set_index("test_month").score)
                sem = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
                print(f"    {b} over {a}: mean {d.mean():+.4f} | "
                      f"t = {d.mean()/sem:+.2f} | {int((d > 0).sum())}/{len(d)} folds")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--period", default="2024_01-12")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--mode", choices=["spread", "ignition", "both"], default="both")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--test-from", default=None)
    ap.add_argument("--embargo-hours", type=int, default=24)
    ap.add_argument("--rolling", action="store_true", default=True)
    ap.add_argument("--rolling-first", type=int, default=5)
    ap.add_argument("--rolling-last", type=int, default=9)
    args = ap.parse_args()
    if args.test_from is None:
        args.test_from = f"{args.year}-09-01"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode in ("spread", "both"):
        p = DATA_DIR / f"goes_spread_2km_{args.period}.parquet"
        if not p.exists():
            raise SystemExit(f"{p.name} not found -- run build_goes_split_targets.py")
        run_one(prepare(p, "spreads_2km_t1h"), "spreads_2km_t1h", SPREAD_SETS,
                "pr_auc", "SPREAD (2 km)", args, "spread2km")

    if args.mode in ("ignition", "both"):
        p = DATA_DIR / f"goes_ignition_{args.period}.parquet"
        if not p.exists():
            raise SystemExit(f"{p.name} not found -- run build_goes_split_targets.py")
        print("\nNOTE: ignition negatives are subsampled, so its prevalence is "
              "artificial.\n      ROC-AUC transfers to the real rate; PR-AUC does not.")
        run_one(prepare(p, "ignites_t1h"), "ignites_t1h", IGNITION_SETS,
                "roc_auc", "IGNITION (0.25 deg)", args, "ignition")

    print(f"\nwrote results to {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
