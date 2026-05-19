"""AI8 — Cross-validation on the TRAIN subset (80%) with fold-wise train-only normalization.

This performs CV ONLY on the TRAIN subset used in the 80/20 holdout split.
For each fold:
- fit preprocessor on fold-train
- transform fold-train and fold-val
- train model on fold-train
- evaluate on fold-val

Outputs:
- Internship_Research/AI8/cv_results_80_20.csv
- Internship_Research/AI8/cv_results_80_20.md

Run (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI8/cross_validation_train_only_80_20.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

try:
    from Internship_Research.normalize_tbm_data_cleaned import build_preprocessor, load_numeric_dataframe
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from Internship_Research.normalize_tbm_data_cleaned import build_preprocessor, load_numeric_dataframe


def vaf(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if np.var(y_true) == 0:
        return float("nan")
    res = y_true - y_pred
    return float(1.0 - (np.var(res) / np.var(y_true)))


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r": r,
        "r2": float(r2_score(y_true, y_pred)),
        "mse": float(mse),
        "rmse": float(math.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "vaf": float(vaf(y_true, y_pred)),
    }


def make_models(random_state: int) -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = []

    models.append(("Linear Regression", LinearRegression()))

    models.append(
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=5,
                max_features=0.5,
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    )

    # RVM (optional)
    try:
        from sklearn_rvm import EMRVR

        models.append(
            (
                "RVM",
                EMRVR(
                    kernel="rbf",
                    degree=3,
                    gamma=0.01,
                    coef0=0.0,
                    tol=0.01,
                    max_iter=1000,
                ),
            )
        )
    except ModuleNotFoundError:
        pass

    # XGBoost (optional)
    try:
        from xgboost import XGBRegressor

        models.append(
            (
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
            )
        )
    except ModuleNotFoundError:
        pass

    models.append(
        (
            "Gradient Boosting",
            GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.01,
                max_depth=3,
                subsample=1.0,
                min_samples_leaf=2,
                random_state=random_state,
            ),
        )
    )

    return models


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI8: CV on train subset with fold-wise train-only normalization")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned.csv"),
    )
    p.add_argument(
        "--indices",
        type=Path,
        default=Path("Internship_Research/AI8/split_indices_80_20.joblib"),
        help="Indices saved by build_split_and_normalize_train_only_80_20.py",
    )
    p.add_argument("--target", type=str, default="PR(mm/r)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Internship_Research/AI8/cv_results_80_20.csv"),
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=Path("Internship_Research/AI8/cv_results_80_20.md"),
    )
    return p.parse_args()


def save_md(df: pd.DataFrame, out_md: Path, folds: int) -> None:
    lines: list[str] = []
    lines.append(f"# Cross-validation results (TRAIN only) — {folds}-fold")
    lines.append("")
    lines.append("Metrics are computed on fold-validation sets. Normalization is fit on fold-train only.")
    lines.append("")
    # Avoid optional dependency on `tabulate` (required by pandas.DataFrame.to_markdown).
    cols = list(df.columns)
    lines.append("| " + " | ".join(map(str, cols)) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.iterrows():
        values: list[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, (int, float, np.floating)):
                # Keep numbers compact and stable.
                values.append(f"{float(v):.6g}")
            else:
                values.append(str(v))
        lines.append("| " + " | ".join(values) + " |")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    info = load(args.indices)
    train_index = info["train_index"]
    selected_columns = info["selected_columns"]

    df = load_numeric_dataframe(args.input_csv, strict=False)
    df = df.loc[train_index, selected_columns].copy()

    if args.target not in df.columns:
        raise ValueError(f"Target {args.target!r} not in selected columns")

    y_all = df[args.target].to_numpy(dtype=float)
    X_all = df.drop(columns=[args.target]).to_numpy(dtype=float)

    models = make_models(args.random_state)

    kf = KFold(n_splits=int(args.folds), shuffle=True, random_state=args.random_state)

    records: list[dict[str, object]] = []

    for model_name, model in models:
        fold_metrics: dict[str, list[float]] = {k: [] for k in ("r", "r2", "rmse", "mae", "vaf")}

        for train_idx, val_idx in kf.split(X_all):
            X_tr, y_tr = X_all[train_idx], y_all[train_idx]
            X_va, y_va = X_all[val_idx], y_all[val_idx]

            pre = build_preprocessor("minmax")
            X_tr_n = pre.fit_transform(X_tr)
            X_va_n = pre.transform(X_va)

            model.fit(X_tr_n, y_tr)
            y_va_pred = model.predict(X_va_n)

            m = eval_metrics(y_va, y_va_pred)
            fold_metrics["r"].append(m["r"])
            fold_metrics["r2"].append(m["r2"])
            fold_metrics["rmse"].append(m["rmse"])
            fold_metrics["mae"].append(m["mae"])
            fold_metrics["vaf"].append(m["vaf"])

        rec: dict[str, object] = {"model": model_name}
        for k, values in fold_metrics.items():
            arr = np.asarray(values, dtype=float)
            rec[f"cv_{k}_mean"] = float(np.nanmean(arr))
            rec[f"cv_{k}_std"] = float(np.nanstd(arr))
        records.append(rec)

    out = pd.DataFrame(records)

    # Present a friendly ordering: best cv_r2_mean first
    out = out.sort_values("cv_r2_mean", ascending=False)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    save_md(out, args.out_md, folds=int(args.folds))

    print("Saved CV results:")
    print("-", args.out_csv)
    print("-", args.out_md)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
