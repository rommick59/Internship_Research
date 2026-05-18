"""AI6 — Train/eval models on fixed split with TRAIN-only normalization.

Consumes the normalized split CSVs produced by:
- Internship_Research/AI6/build_splits_and_normalize_train_only.py

Then:
- trains on TRAIN (70%)
- tests on TEST (20%)
- validates on VAL (10%)

Outputs per-model results CSVs in Internship_Research/AI6, with the same metric
column names as the existing AI scripts (train_*, val_*, test_*).

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI6/train_eval_models_fixed_split.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r": r,
        "r2": r2_score(y_true, y_pred),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _load_split(csv_path: Path, target: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Missing target {target!r} in {csv_path}")
    X = df.drop(columns=[target]).to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    return X, y


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI6: train/eval models on fixed split")
    p.add_argument("--train-csv", type=Path, default=Path("Internship_Research/AI6/split_train_norm.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("Internship_Research/AI6/split_test_norm.csv"))
    p.add_argument("--val-csv", type=Path, default=Path("Internship_Research/AI6/split_val_norm.csv"))
    p.add_argument("--target", type=str, default="PR(mm/r)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--split", type=str, default="0.7,0.1,0.2")
    p.add_argument("--out-dir", type=Path, default=Path("Internship_Research/AI6"))
    return p.parse_args()


def _write_results(out_csv: Path, row: dict[str, object]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(out_csv, index=False)


def main() -> int:
    args = parse_args()

    X_train, y_train = _load_split(args.train_csv, args.target)
    X_test, y_test = _load_split(args.test_csv, args.target)
    X_val, y_val = _load_split(args.val_csv, args.target)

    train, val, test = (float(p) for p in args.split.split(","))

    models: list[tuple[str, object, dict[str, object], str]] = []

    models.append((
        "Linear Regression",
        LinearRegression(),
        {"estimator": "linear"},
        "linear_results.csv",
    ))

    models.append((
        "Random Forest",
        RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            random_state=args.random_state,
            n_jobs=-1,
        ),
        {
            "estimator": "random_forest",
            "n_estimators": 500,
            "max_depth": "None",
            "min_samples_leaf": 1,
        },
        "random_forest_results.csv",
    ))

    models.append((
        "SVR (RBF)",
        SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale"),
        {"estimator": "svr", "kernel": "rbf", "C": 10.0, "epsilon": 0.01, "gamma": "scale"},
        "svr_results.csv",
    ))

    # RVM (optional dependency)
    try:
        from sklearn_rvm import EMRVR

        models.append((
            "RVM",
            EMRVR(
                kernel="rbf",
                degree=3,
                gamma="scale",
                coef0=0.0,
                tol=0.001,
                max_iter=5000,
            ),
            {
                "estimator": "rvm",
                "kernel": "rbf",
                "degree": 3,
                "gamma": "scale",
                "coef0": 0.0,
                "tol": 0.001,
                "max_iter": 5000,
            },
            "rvm_results.csv",
        ))
    except ModuleNotFoundError:
        print("NOTE: sklearn-rvm not installed, skipping RVM")

    # XGBoost (optional dependency)
    try:
        from xgboost import XGBRegressor

        models.append((
            "XGBoost",
            XGBRegressor(
                objective="reg:squarederror",
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=args.random_state,
                n_jobs=-1,
            ),
            {
                "estimator": "xgboost",
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 4,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 1.0,
            },
            "xgboost_results.csv",
        ))
    except ModuleNotFoundError:
        print("NOTE: xgboost not installed, skipping XGBoost")

    models.append((
        "Gradient Boosting",
        GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=1.0,
            min_samples_leaf=1,
            random_state=args.random_state,
        ),
        {
            "estimator": "gradient_boosting",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 3,
            "subsample": 1.0,
            "min_samples_leaf": 1,
        },
        "gradient_boosting_results.csv",
    ))

    base = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1, random_state=args.random_state)
    models.append((
        "AdaBoost",
        AdaBoostRegressor(
            estimator=base,
            n_estimators=500,
            learning_rate=0.05,
            random_state=args.random_state,
            loss="linear",
        ),
        {
            "estimator": "adaboost",
            "n_estimators": 500,
            "learning_rate": 0.05,
            "base_max_depth": 3,
            "base_min_samples_leaf": 1,
        },
        "adaboost_results.csv",
    ))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model, meta, out_name in models:
        print(f"\n=== {model_name} ===")
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_val_pred = model.predict(X_val)

        row: dict[str, object] = {
            **meta,
            "train": train,
            "val": val,
            "test": test,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        }

        m = eval_metrics(y_train, y_train_pred)
        row.update({f"train_{k}": float(v) for k, v in m.items()})

        m = eval_metrics(y_val, y_val_pred)
        row.update({f"val_{k}": float(v) for k, v in m.items()})

        m = eval_metrics(y_test, y_test_pred)
        row.update({f"test_{k}": float(v) for k, v in m.items()})

        out_csv = args.out_dir / out_name
        _write_results(out_csv, row)
        print("Saved:", out_csv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
