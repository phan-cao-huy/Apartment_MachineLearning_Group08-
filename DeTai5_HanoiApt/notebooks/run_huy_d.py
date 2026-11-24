"""D.3 and D.4 on the GIS dataset. Run this file directly with Python."""

import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
FEATURES = (
    "area_m2", "bedrooms", "bathrooms", "has_red_book", "interior_level",
    "lat", "lon", "dist_center_km", "dist_metro_km", "dist_school_km", "dist_park_km",
)
SPACE = {
    "max_depth": [3, 6],
    "learning_rate": [0.05, 0.10, 0.20],
    "min_child_weight": [3, 12],
}
ROUNDS = 120
PATIENCE = 12


def metrics(y, pred):
    error = pred - y
    return {
        "rmse_log": float(np.sqrt(np.mean(error**2))),
        "mape_price_pct": float(100 * np.mean(np.abs(np.expm1(error)))),
    }


def load_data(path):
    frame = pd.read_csv(path, usecols=[*FEATURES, "collected_at", "price_per_m2",
                                     "ward_2025", "District", "source"])
    date = pd.to_datetime(frame["collected_at"].str.slice(0, 10), format="%Y-%m-%d")
    if frame.empty or date.isna().any():
        raise ValueError("Expected dated observations")
    price = frame["price_per_m2"].to_numpy(dtype=float)
    if not np.isfinite(price).all() or (price <= 0).any():
        raise ValueError("Expected positive finite prices")
    week_start = date.max() - pd.Timedelta(days=date.max().weekday())
    test = (date >= week_start).to_numpy()
    ward = frame["ward_2025"].astype("string").str.strip().replace("", None)
    district = frame["District"].astype("string").fillna("unknown").str.strip()
    # Incomplete ward labels share a block within their recorded district.
    groups = ("ward:" + ward.str.casefold()).fillna("missing-ward:" + district.str.casefold())
    X = frame[list(FEATURES)].to_numpy(dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    y = np.log(price)
    if test.all() or groups[~test].nunique() < 3:
        raise ValueError("Expected at least three training blocks before the last week")
    splits = list(GroupKFold(n_splits=3).split(X[~test], y[~test], groups[~test]))
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    audit = {
        "sha256": digest,
        "rows": len(frame), "train_rows": int((~test).sum()), "test_rows": int(test.sum()),
        "test_dates": [str(week_start.date()), str(date.max().date())],
        "rows_without_ward": int(ward.isna().sum()),
        "test_sources": frame.loc[test, "source"].value_counts().to_dict(),
        "sources": frame["source"].value_counts().to_dict(),
        "training_blocks": int(groups[~test].nunique()),
        "fold_validation_rows": [len(val) for _, val in splits],
    }
    return X[~test], y[~test], X[test], y[test], splits, audit


def xgb_params(params, seed):
    return {"objective": "reg:squarederror", "eval_metric": "rmse", "tree_method": "hist",
            "seed": seed, "nthread": 4, "subsample": 0.85, "colsample_bytree": 0.9,
            "eta": params["learning_rate"], "max_depth": params["max_depth"],
            "min_child_weight": params["min_child_weight"]}


def fit_xgb_folds(folds, params, seed):
    scores, rounds = [], []
    for train, val in folds:
        model = xgb.train(xgb_params(params, seed), train, ROUNDS, evals=[(val, "val")],
                          early_stopping_rounds=PATIENCE, verbose_eval=False)
        scores.append(float(model.best_score))
        rounds.append(model.best_iteration + 1)
    return {"params": params, "cv_rmse_log": float(np.mean(scores)),
            "fold_rmse_log": scores, "fold_rounds": rounds, "rounds": int(np.median(rounds))}


def run_search(folds, seed, method):
    grid = [dict(zip(SPACE, values)) for values in itertools.product(*SPACE.values())]
    trials = []
    start = time.perf_counter()

    def evaluate(params):
        result = fit_xgb_folds(folds, params, seed)
        trials.append(result)
        return result["cv_rmse_log"]

    if method == "tpe":
        study = optuna.create_study(
            direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=4)
        )
        study.optimize(lambda trial: evaluate({
            key: trial.suggest_categorical(key, choices) for key, choices in SPACE.items()
        }), n_trials=len(grid))
    else:
        # Random search draws independently from the same discrete space with replacement.
        candidates = grid if method == "grid" else [
            grid[i] for i in np.random.default_rng(seed).integers(len(grid), size=len(grid))
        ]
        for params in candidates:
            evaluate(params)
    return {"seed": seed, "method": method, "seconds": time.perf_counter() - start,
            "best": min(trials, key=lambda trial: trial["cv_rmse_log"]), "trials": trials}


def lightgbm_baseline(X, y, test, test_y, splits):
    params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.1,
              "max_depth": 6, "num_leaves": 31, "num_threads": 4, "verbosity": -1,
              "seed": SEEDS[0]}
    scores, rounds = [], []
    for tr, va in splits:
        train = lgb.Dataset(X[tr], label=y[tr])
        val = lgb.Dataset(X[va], label=y[va], reference=train)
        model = lgb.train(params, train, ROUNDS, valid_sets=[val],
                          callbacks=[lgb.early_stopping(PATIENCE, verbose=False)])
        scores.append(model.best_score["valid_0"]["rmse"])
        rounds.append(model.best_iteration)
    n = int(np.median(rounds))
    model = lgb.train(params, lgb.Dataset(X, label=y), n)
    return {"cv_rmse_log": float(np.mean(scores)), "rounds": n,
            **metrics(test_y, model.predict(test))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/hanoi_apt_final.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/tables/huy_d_results.json")
    args = parser.parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y, test, test_y, splits, audit = load_data(args.data)
    folds = [(xgb.DMatrix(X[tr], label=y[tr], nthread=4),
              xgb.DMatrix(X[va], label=y[va], nthread=4)) for tr, va in splits]
    train = xgb.DMatrix(X, label=y, nthread=4)
    holdout = xgb.DMatrix(test, nthread=4)
    runs = []
    for seed in SEEDS:
        for method in ("grid", "random", "tpe"):
            result = run_search(folds, seed, method)
            runs.append(result)
            print(f"{method} seed={seed}: CV RMSE={result['best']['cv_rmse_log']:.5f}", flush=True)

    def test_xgb(best, seed):
        model = xgb.train(xgb_params(best["params"], seed), train, best["rounds"])
        return metrics(test_y, model.predict(holdout))

    fixed = fit_xgb_folds(folds, {"max_depth": 6, "learning_rate": 0.1, "min_child_weight": 3},
                          SEEDS[0])
    baseline = {"mean_log_price": metrics(test_y, y.mean()),
                "xgboost": {**fixed, **test_xgb(fixed, SEEDS[0])},
                "lightgbm": lightgbm_baseline(X, y, test, test_y, splits)}
    summary = {}
    for method in ("grid", "random", "tpe"):
        selected = [run for run in runs if run["method"] == method]
        scores = [run["best"]["cv_rmse_log"] for run in selected]
        for run in selected:
            run["test"] = test_xgb(run["best"], run["seed"])
        summary[method] = {
            "mean_cv_rmse_log": float(np.mean(scores)), "sd_cv_rmse_log": float(np.std(scores, ddof=1)),
            "mean_search_seconds": float(np.mean([run["seconds"] for run in selected])),
            **{f"mean_test_{key}": float(np.mean([run["test"][key] for run in selected]))
               for key in ("rmse_log", "mape_price_pct")},
        }
    winner = min(summary, key=lambda method: summary[method]["mean_cv_rmse_log"])
    result = {"dataset": args.data.name, "audit": audit, "features": FEATURES,
              "space": SPACE, "trials_per_method_seed": 12, "seeds": SEEDS,
              "max_rounds": ROUNDS, "patience": PATIENCE, "baseline": baseline,
              "search_summary": summary, "method_selected_by_cv": winner, "runs": runs,
              "versions": {lib.__name__: lib.__version__ for lib in
                           (pd, np, sklearn, xgb, lgb, optuna)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
