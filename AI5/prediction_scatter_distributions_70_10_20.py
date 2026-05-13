"""Prediction scatter distributions for AI5 (0.70/0.10/0.20).

Creates scatter plots of y_true vs y_pred for each model (TRAIN and TEST),
so you can visually compare the distribution of points across IA models.

Inputs (default):
- Internship_Research/AI5/TBM_data_cleaned_ml_ready_selected.csv

Outputs (default):
- Internship_Research/AI5/images/pred_scatter_train_70_10_20.png
- Internship_Research/AI5/images/pred_scatter_test_70_10_20.png

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI5/prediction_scatter_distributions_70_10_20.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class Split:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not (abs(total - 1.0) <= 1e-9):
            raise ValueError(f"Split must sum to 1.0, got {total:.6f}")


def parse_split(value: str) -> Split:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError('Split must be formatted like "0.7,0.1,0.2"')
    return Split(train=float(parts[0]), val=float(parts[1]), test=float(parts[2]))


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    split: Split,
    random_state: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=split.test,
        random_state=random_state,
        shuffle=True,
    )

    val_fraction_of_trainval = split.val / (split.train + split.val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=random_state,
        shuffle=True,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred)))


def make_models(random_state: int) -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = []

    models.append(("Linear Regression", LinearRegression()))

    models.append(
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
            ),
        )
    )

    models.append(("SVR (RBF)", SVR(kernel="rbf", C=10.0, epsilon=0.01, gamma="scale")))

    # RVM (optional dependency)
    try:
        from sklearn_rvm import EMRVR

        models.append(
            (
                "RVM",
                EMRVR(
                    kernel="rbf",
                    degree=3,
                    gamma="scale",
                    coef0=0.0,
                    tol=0.001,
                    max_iter=5000,
                ),
            )
        )
    except ModuleNotFoundError:
        pass

    # XGBoost (optional dependency)
    try:
        from xgboost import XGBRegressor

        models.append(
            (
                "XGBoost",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=1000,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
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
                n_estimators=500,
                learning_rate=0.05,
                max_depth=3,
                subsample=1.0,
                min_samples_leaf=1,
                random_state=random_state,
            ),
        )
    )

    base = DecisionTreeRegressor(max_depth=3, min_samples_leaf=1, random_state=random_state)
    models.append(
        (
            "AdaBoost",
            AdaBoostRegressor(
                estimator=base,
                n_estimators=500,
                learning_rate=0.05,
                random_state=random_state,
                loss="linear",
            ),
        )
    )

    return models


def plot_scatter_grid(
    title: str,
    y_true: np.ndarray,
    preds: list[tuple[str, np.ndarray]],
    out_path: Path,
    dpi: int,
) -> None:
    n = len(preds)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.3 * ncols, 4.7 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    # shared axis limits
    y_min = float(np.min(y_true))
    y_max = float(np.max(y_true))
    pad = 0.03 * (y_max - y_min if y_max > y_min else 1.0)
    lo, hi = y_min - pad, y_max + pad

    for idx, (name, y_pred) in enumerate(preds):
        ax = axes[idx // ncols, idx % ncols]

        r2 = float(r2_score(y_true, y_pred))
        e = rmse(y_true, y_pred)

        ax.scatter(y_true, y_pred, s=9, alpha=0.35)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2)
        ax.set_title(f"{name}\nR²={r2:.4f} | RMSE={e:.4f}")
        ax.set_xlabel("y_true")
        ax.set_ylabel("y_pred")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(linestyle=":", alpha=0.35)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(title)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Scatter distribution of predictions per model (AI5)")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("Internship_Research/AI5/TBM_data_cleaned_ml_ready_selected.csv"),
        help="Input ML-ready selected dataset",
    )
    p.add_argument(
        "--target",
        type=str,
        default="PR(mm/r)",
        help="Target column",
    )
    p.add_argument(
        "--split",
        type=str,
        default="0.7,0.1,0.2",
        help="train,val,test (default: 0.7,0.1,0.2)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG DPI",
    )
    p.add_argument(
        "--out-train",
        type=Path,
        default=Path("Internship_Research/AI5/images/pred_scatter_train_70_10_20.png"),
        help="Output train scatter PNG",
    )
    p.add_argument(
        "--out-test",
        type=Path,
        default=Path("Internship_Research/AI5/images/pred_scatter_test_70_10_20.png"),
        help="Output test scatter PNG",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    split = parse_split(args.split)

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        raise ValueError(f"Unknown target column: {args.target!r}. Available: {list(df.columns)!r}")

    # numeric-only (the dataset is already numeric, but keep it robust)
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")

    y = num[args.target].to_numpy(dtype=float)
    X = num.drop(columns=[args.target]).to_numpy(dtype=float)

    (X_train, y_train), (_X_val, _y_val), (X_test, y_test) = split_data(X, y, split, random_state=args.random_state)

    models = make_models(args.random_state)

    train_preds: list[tuple[str, np.ndarray]] = []
    test_preds: list[tuple[str, np.ndarray]] = []

    for name, model in models:
        model.fit(X_train, y_train)
        train_preds.append((name, model.predict(X_train)))
        test_preds.append((name, model.predict(X_test)))

    plot_scatter_grid(
        title=f"AI5 — Prediction scatter (TRAIN) split {split.train:.2f}/{split.val:.2f}/{split.test:.2f}",
        y_true=y_train,
        preds=train_preds,
        out_path=args.out_train,
        dpi=args.dpi,
    )

    plot_scatter_grid(
        title=f"AI5 — Prediction scatter (TEST) split {split.train:.2f}/{split.val:.2f}/{split.test:.2f}",
        y_true=y_test,
        preds=test_preds,
        out_path=args.out_test,
        dpi=args.dpi,
    )

    print("Saved:")
    print("-", args.out_train)
    print("-", args.out_test)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
