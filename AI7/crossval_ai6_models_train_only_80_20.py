"""AI7 — Cross-validation on TRAIN (80%) using AI6 model set (NO CRS).

Important: To avoid leakage, the preprocessing (median imputer + MinMaxScaler)
MUST be fit inside each CV fold. This script therefore uses an sklearn Pipeline
(preprocessor + model) and refits it per fold.

Consumes:
- Internship_Research/AI7/split_train_raw_80_20.csv  (raw numeric, unscaled)

Outputs:
- Internship_Research/AI7/cv_folds_80_20.csv     (one row per model per fold)
- Internship_Research/AI7/cv_summary_80_20.csv   (mean/std per model)
- Internship_Research/AI7/cv_summary_80_20.md

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI7/crossval_ai6_models_train_only_80_20.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from Internship_Research.normalize_tbm_data_cleaned import build_preprocessor


def vaf_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.var(y_true))
    if denom == 0.0:
        return float("nan")
    resid = y_true - y_pred
    return 1.0 - float(np.var(resid)) / denom


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])

    resid = y_true - y_pred
    mse = float(np.mean(resid**2))

    return {
        "r": float(r),
        "r2": float(1.0 - np.sum(resid**2) / np.sum((y_true - np.mean(y_true)) ** 2))
        if float(np.var(y_true)) != 0.0
        else float("nan"),
        "vaf": float(vaf_score(y_true, y_pred)),
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": float(np.mean(np.abs(resid))),
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid)),
    }


@dataclass(frozen=True)
class ModelJob:
    name: str
    estimator: object
    meta: dict[str, object]


def make_models(random_state: int) -> list[ModelJob]:
    jobs: list[ModelJob] = []

    jobs.append(ModelJob("Linear Regression", LinearRegression(), {"estimator": "linear"}))

    jobs.append(
        ModelJob(
            "Random Forest",
            RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=5,
                max_features=0.5,
                random_state=random_state,
                n_jobs=-1,
            ),
            {"estimator": "random_forest", "n_estimators": 50, "max_depth": 3, "min_samples_leaf": 5, "max_features": 0.5},
        )
    )

    jobs.append(
        ModelJob(
            "SVR (RBF)",
            SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"),
            {"estimator": "svr", "kernel": "rbf", "C": 1.0, "epsilon": 0.1, "gamma": "scale"},
        )
    )

    try:
        from sklearn_rvm import EMRVR

        jobs.append(
            ModelJob(
                "RVM",
                EMRVR(kernel="rbf", degree=3, gamma=0.01, coef0=0.0, tol=0.01, max_iter=1000),
                {
                    "estimator": "rvm",
                    "kernel": "rbf",
                    "degree": 3,
                    "gamma": 0.01,
                    "coef0": 0.0,
                    "tol": 0.01,
                    "max_iter": 1000,
                },
            )
        )
    except ModuleNotFoundError:
        print("NOTE: sklearn-rvm not installed, skipping RVM")

    try:
        from xgboost import XGBRegressor

        jobs.append(
            ModelJob(
                "XGBoost",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=200,
                    learning_rate=0.01,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=5.0,
                    random_state=random_state,
                    n_jobs=-1,
                ),
                {
                    "estimator": "xgboost",
                    "n_estimators": 200,
                    "learning_rate": 0.01,
                    "max_depth": 3,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "reg_lambda": 5.0,
                },
            )
        )
    except ModuleNotFoundError:
        print("NOTE: xgboost not installed, skipping XGBoost")

    jobs.append(
        ModelJob(
            "Gradient Boosting",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=3,
                subsample=1.0,
                min_samples_leaf=2,
                random_state=random_state,
            ),
            {
                "estimator": "gradient_boosting",
                "n_estimators": 200,
                "learning_rate": 0.01,
                "max_depth": 3,
                "subsample": 1.0,
                "min_samples_leaf": 2,
            },
        )
    )

    base = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=random_state)
    jobs.append(
        ModelJob(
            "AdaBoost",
            AdaBoostRegressor(
                estimator=base,
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_state,
                loss="linear",
            ),
            {
                "estimator": "adaboost",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "base_max_depth": 2,
                "base_min_samples_leaf": 2,
            },
        )
    )

    return jobs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI7: cross-validation on TRAIN only (80/20 split)")
    p.add_argument(
        "--train-raw-csv",
        type=Path,
        default=Path("Internship_Research/AI7/split_train_raw_80_20.csv"),
    )
    p.add_argument("--target", type=str, default="PR(mm/r)")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--out-folds", type=Path, default=Path("Internship_Research/AI7/cv_folds_80_20.csv"))
    p.add_argument("--out-summary", type=Path, default=Path("Internship_Research/AI7/cv_summary_80_20.csv"))
    p.add_argument("--out-md", type=Path, default=Path("Internship_Research/AI7/cv_summary_80_20.md"))
    return p.parse_args()


def write_md(summary: pd.DataFrame, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("# Cross-validation summary (TRAIN only) — 80/20")
    lines.append("")
    lines.append("CV is performed on the TRAIN set only (80%), with preprocessing refit inside each fold.")
    lines.append("")
    lines.append("Sorted by `mean_r2` (descending).")
    lines.append("")

    show_cols = [
        "model",
        "mean_r2",
        "std_r2",
        "mean_vaf",
        "std_vaf",
        "mean_rmse",
        "std_rmse",
        "mean_mae",
        "std_mae",
    ]

    sub = summary[show_cols].sort_values("mean_r2", ascending=False)

    lines.append("| Model | mean R² | std R² | mean VAF | std VAF | mean RMSE | std RMSE | mean MAE | std MAE |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in sub.iterrows():
        lines.append(
            "| {model} | {mean_r2:.4f} | {std_r2:.4f} | {mean_vaf:.4f} | {std_vaf:.4f} | {mean_rmse:.4g} | {std_rmse:.4g} | {mean_mae:.4g} | {std_mae:.4g} |".format(
                **r.to_dict()
            )
        )

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.train_raw_csv)
    if args.target not in df.columns:
        raise ValueError(f"Missing target {args.target!r} in {args.train_raw_csv}")

    X = df.drop(columns=[args.target]).to_numpy(dtype=float)
    y = df[args.target].to_numpy(dtype=float)

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)

    folds_rows: list[dict[str, object]] = []

    for job in make_models(args.random_state):
        print(f"\n=== {job.name} (CV {args.n_splits}-fold) ===")

        for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            pipe = Pipeline(
                steps=[
                    ("preprocessor", build_preprocessor("minmax")),
                    ("model", clone(job.estimator)),
                ]
            )

            pipe.fit(X_tr, y_tr)
            y_va_pred = pipe.predict(X_va)

            m = eval_metrics(y_va, y_va_pred)
            folds_rows.append(
                {
                    **job.meta,
                    "model": job.name,
                    "fold": fold_idx,
                    "n_train_fold": int(len(y_tr)),
                    "n_val_fold": int(len(y_va)),
                    **{k: float(v) for k, v in m.items()},
                }
            )

    folds = pd.DataFrame(folds_rows)

    summary = (
        folds.groupby("model")
        .agg(
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            mean_vaf=("vaf", "mean"),
            std_vaf=("vaf", "std"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
        )
        .reset_index()
    )

    args.out_folds.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(args.out_folds, index=False)
    summary.to_csv(args.out_summary, index=False)
    write_md(summary, args.out_md)

    print("\nSaved CV outputs:")
    print("-", args.out_folds)
    print("-", args.out_summary)
    print("-", args.out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
