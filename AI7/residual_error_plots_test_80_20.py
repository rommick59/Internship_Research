"""AI7 — Residual error plots on TEST (80/20 split, train-only normalization).

For each IA model (same as AI6), trains on TRAIN and evaluates residuals on TEST:
residual = y_true - y_pred.

Outputs:
- Internship_Research/AI7/images/residuals_scatter_test_80_20.png
- Internship_Research/AI7/images/residuals_hist_test_80_20.png

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI7/residual_error_plots_test_80_20.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


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

    models.append(("SVR (RBF)", SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")))

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

    base = DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=random_state)
    models.append(
        (
            "AdaBoost",
            AdaBoostRegressor(
                estimator=base,
                n_estimators=100,
                learning_rate=0.05,
                random_state=random_state,
                loss="linear",
            ),
        )
    )

    return models


def plot_residual_scatter(preds: list[tuple[str, np.ndarray, np.ndarray]], out: Path, dpi: int) -> None:
    n = len(preds)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.3 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, (name, y_true, y_pred) in enumerate(preds):
        ax = axes[idx // ncols, idx % ncols]
        res = y_true - y_pred
        ax.scatter(y_pred, res, s=9, alpha=0.35)
        ax.axhline(0.0, linestyle="--", linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel("y_pred")
        ax.set_ylabel("residual (y_true - y_pred)")
        ax.grid(linestyle=":", alpha=0.35)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle("AI7 — Residual scatter on TEST (80/20)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_residual_hist(preds: list[tuple[str, np.ndarray, np.ndarray]], out: Path, dpi: int) -> None:
    n = len(preds)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.3 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, (name, y_true, y_pred) in enumerate(preds):
        ax = axes[idx // ncols, idx % ncols]
        res = y_true - y_pred
        ax.hist(res, bins=30)
        ax.axvline(0.0, linestyle="--", linewidth=1.2)
        ax.set_title(name)
        ax.set_xlabel("residual")
        ax.set_ylabel("count")
        ax.grid(linestyle=":", alpha=0.35)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle("AI7 — Residual histogram on TEST (80/20)")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI7 residual error plots (test)")
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("Internship_Research/AI7/split_train_norm_80_20.csv"),
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("Internship_Research/AI7/split_test_norm_80_20.csv"),
    )
    p.add_argument("--target", type=str, default="PR(mm/r)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--out-scatter",
        type=Path,
        default=Path("Internship_Research/AI7/images/residuals_scatter_test_80_20.png"),
    )
    p.add_argument(
        "--out-hist",
        type=Path,
        default=Path("Internship_Research/AI7/images/residuals_hist_test_80_20.png"),
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    X_train = train_df.drop(columns=[args.target]).to_numpy(dtype=float)
    y_train = train_df[args.target].to_numpy(dtype=float)

    X_test = test_df.drop(columns=[args.target]).to_numpy(dtype=float)
    y_test = test_df[args.target].to_numpy(dtype=float)

    preds: list[tuple[str, np.ndarray, np.ndarray]] = []

    for name, model in make_models(args.random_state):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds.append((name, y_test, y_pred))

    plot_residual_scatter(preds, args.out_scatter, dpi=args.dpi)
    plot_residual_hist(preds, args.out_hist, dpi=args.dpi)

    print("Saved residual plots:")
    print("-", args.out_scatter)
    print("-", args.out_hist)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
