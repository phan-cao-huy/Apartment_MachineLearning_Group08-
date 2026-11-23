"""Compare the A.4 implementation against XGBoost and LightGBM.

Run this file directly or use python -m notebooks.compare_xgb_a4
from the project directory.
The loss, optimal leaf weight and split score follow Chen and Guestrin
(2016), equations 5–7: https://arxiv.org/abs/1603.02754.
The CSV is the merged snapshot produced by 01_data_audit_and_clean.ipynb.
"""

import argparse
import csv
import importlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(PROJECT_ROOT))

from src.xgb_custom import XGBRegressorCustom  # noqa: E402

FEATURES = ("area_m2", "bedrooms", "bathrooms", "has_red_book", "interior_level")
DEFAULT_DATA = PROJECT_ROOT / "data/processed/hanoi_apt_clean_5k.csv"


def verify_formula():
    X = np.arange(4).reshape(-1, 1)
    y = np.array([0, 0, 4, 4])
    model = XGBRegressorCustom(n_estimators=1, max_depth=1, learning_rate=1,
                               reg_lambda=1, gamma=1).fit(X, y)
    root = model.trees_[0]
    np.testing.assert_allclose(
        [root.threshold, root.left.weight, root.right.weight, root.gain],
        [1.5, -4 / 3, 4 / 3, 16 / 3 - 1],
    )
    expected = np.array([2 / 3, 2 / 3, 10 / 3, 10 / 3])
    np.testing.assert_allclose(model.predict(X), expected)
    for settings in ({"gamma": 6}, {"min_child_weight": 3}, {"max_depth": 0}):
        constant = XGBRegressorCustom(n_estimators=1, **settings).fit(X, y)
        np.testing.assert_allclose(constant.predict(X), y.mean())
    return X, y, expected, {"threshold": root.threshold, "gain": root.gain,
                            "left_weight": root.left.weight, "right_weight": root.right.weight}


def load_data(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    dates = np.array([row["collected_at"][:10] for row in rows], dtype="datetime64[D]")
    if not len(rows) or np.isnat(dates).any():
        raise ValueError("Expected a nonempty dataset with valid collection dates")
    last_date = dates.max()
    test = dates >= last_date - np.timedelta64(6, "D")
    if test.all():
        raise ValueError("Expected observations before the last seven calendar days")
    X = np.array([[float(row[name] or "nan") for name in FEATURES] for row in rows])
    y = np.array([float(row["log_price_m2"]) for row in rows])
    if not np.isfinite(y).all():
        raise ValueError("Expected finite log prices")
    X[~np.isfinite(X)] = np.nan
    median = np.nanmedian(X[~test], axis=0)
    median = np.where(np.isfinite(median), median, 0)
    X = np.where(np.isnan(X), median, X)
    return X[~test], y[~test], X[test], y[test], str(last_date - np.timedelta64(6, "D")), str(last_date)


def compare_libraries(formula):
    lgb = importlib.import_module("lightgbm")
    xgb = importlib.import_module("xgboost")

    X, y, expected, _ = formula
    # The libraries score the split without the paper’s factor of 1/2.
    # Thus their minimum split gain is 2 for the hand-check with gamma=1.
    small = xgb.DMatrix(X, label=y, nthread=1)
    exact = xgb.train({"objective": "reg:squarederror", "tree_method": "exact",
                       "base_score": 2.0, "max_depth": 1, "eta": 1.0,
                       "lambda": 1.0, "gamma": 2.0, "nthread": 1}, small, 1)
    light = lgb.train({"objective": "regression", "max_depth": 1, "num_leaves": 2,
                       "learning_rate": 1.0, "lambda_l2": 1.0, "min_gain_to_split": 2.0,
                       "min_data_in_leaf": 1, "min_data_in_bin": 1,
                       "num_threads": 1, "verbosity": -1}, lgb.Dataset(X, label=y), 1)
    small_errors = {}
    for name, prediction in (("XGBoost", exact.predict(small)), ("LightGBM", light.predict(X))):
        np.testing.assert_allclose(prediction, expected, atol=1e-6, rtol=0)
        small_errors[name] = float(np.max(np.abs(prediction - expected)))

    return {"library_versions": {"xgboost": xgb.__version__, "lightgbm": lgb.__version__},
            "synthetic_max_abs_error": small_errors}


def benchmark(data, include_libraries):
    train_X, train_y, test_X, test_y, start, end = load_data(data)
    rounds, depth, rate, penalty = 30, 3, 0.1, 1.0
    custom = XGBRegressorCustom(rounds, depth, rate, penalty).fit(train_X, train_y)

    def metrics(prediction):
        error = prediction - test_y
        return {"rmse_log": float(np.sqrt(np.mean(error**2))),
                "mape_price_pct": float(100 * np.mean(np.abs(np.expm1(error))))}

    scores = {"Mean log-price": metrics(train_y.mean()),
              "Custom": metrics(custom.predict(test_X))}
    if include_libraries:
        lgb = importlib.import_module("lightgbm")
        xgb = importlib.import_module("xgboost")

        train = xgb.DMatrix(train_X, label=train_y, nthread=1)
        exact = xgb.train({"objective": "reg:squarederror", "tree_method": "exact",
                           "base_score": float(train_y.mean()), "max_depth": depth,
                           "eta": rate, "lambda": penalty, "nthread": 1}, train, rounds)
        light = lgb.train({"objective": "regression", "max_depth": depth,
                           "num_leaves": 2**depth, "learning_rate": rate,
                           "lambda_l2": penalty, "min_data_in_leaf": 1, "min_data_in_bin": 1,
                           "num_threads": 1, "verbosity": -1},
                          lgb.Dataset(train_X, label=train_y), rounds)
        scores["XGBoost"] = metrics(exact.predict(xgb.DMatrix(test_X, nthread=1)))
        scores["LightGBM"] = metrics(light.predict(test_X))
    return {
        "dataset": data.as_posix(), "train_rows": len(train_y), "test_rows": len(test_y),
        "test_dates": [start, end], "features": list(FEATURES),
        "parameters": {"rounds": rounds, "max_depth": depth, "learning_rate": rate,
                       "lambda_l2": penalty},
        "test_metrics": scores,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-formula", action="store_true", help="Run the hand-check only")
    mode.add_argument("--custom-only", action="store_true", help="Benchmark using NumPy only")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    formula = verify_formula()
    result = {"numpy_version": np.__version__, "formula_check": formula[-1]}
    if not args.check_formula:
        missing = [] if args.custom_only else [
            name for name in ("xgboost", "lightgbm")
            if importlib.util.find_spec(name) is None
        ]
        include_libraries = not args.custom_only and not missing
        if missing:
            result["missing_comparison_libraries"] = missing
            print("Running the custom benchmark only. For the full comparison, run: "
                  "python -m pip install xgboost lightgbm", file=sys.stderr)
        if include_libraries:
            result.update(compare_libraries(formula))
        result.update(benchmark(args.data, include_libraries=include_libraries))
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
