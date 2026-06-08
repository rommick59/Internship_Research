"""AI8 — Predicted vs Measured scatter grid on TEST (80/20, train-only normalization).

For each IA model (same set as AI6/AI8), trains on TRAIN and plots predicted vs
measured target on TEST as a grid of scatter subplots. Each subplot includes:
- the 1:1 line,
- +/-20% reference bands,
- a title with R^2 and RMSE.

Outputs:
- Internship_Research/AI8/images/predicted_vs_measured_test_80_20.png

Run (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI8/predicted_vs_measured_test_80_20.py
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Distinct color per model, in the order returned by make_models.
MODEL_COLORS = [
    "#4C72B0",  # Linear Regression
    "#F2A900",  # Random Forest
    "#55A868",  # RVM
    "#6FB7E8",  # XGBoost
    "#E07B39",  # Gradient Boosting
    "#D98AB5",  # extra (fallback)
]


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
        print("NOTE: sklearn-rvm not installed, skipping RVM")

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
        print("NOTE: xgboost not installed, skipping XGBoost")

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


def plot_predicted_vs_measured(
    preds: list[tuple[str, np.ndarray, np.ndarray]],
    out: Path,
    target: str,
    dpi: int,
) -> None:
    n = len(preds)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(4.8 * ncols, 4.4 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    for idx, (name, y_true, y_pred) in enumerate(preds):
        ax = axes[idx // ncols, idx % ncols]
        color = MODEL_COLORS[idx % len(MODEL_COLORS)]

        r2 = float(r2_score(y_true, y_pred))
        rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))

        ax.scatter(y_true, y_pred, s=22, alpha=0.75, color=color, edgecolors="none")

        lo = float(min(np.min(y_true), np.min(y_pred)))
        hi = float(max(np.max(y_true), np.max(y_pred)))
        pad = 0.05 * (hi - lo if hi > lo else 1.0)
        lo -= pad
        hi += pad
        line = np.array([lo, hi])

        ax.plot(line, line, color="black", linewidth=1.6, label="1:1")
        ax.plot(line, 1.2 * line, color="gray", linestyle="--", linewidth=1.0, label="\u00b120%")
        ax.plot(line, 0.8 * line, color="gray", linestyle="--", linewidth=1.0)

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{name}\n$R^2$={r2:.3f}, RMSE={rmse:.2f}")
        ax.set_xlabel(f"Measured {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.legend(loc="upper left", frameon=False, fontsize=9)
        ax.grid(linestyle=":", alpha=0.35)

    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI8 predicted vs measured plots (test)")
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("Internship_Research/AI8/split_train_norm_80_20.csv"),
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("Internship_Research/AI8/split_test_norm_80_20.csv"),
    )
    p.add_argument("--target", type=str, default="PR(mm/r)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Internship_Research/AI8/images/predicted_vs_measured_test_80_20.png"),
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
        y_pred = np.asarray(model.predict(X_test), dtype=float).reshape(-1)
        preds.append((name, y_test, y_pred))

    plot_predicted_vs_measured(preds, args.out, target=args.target, dpi=args.dpi)

    print("Saved predicted vs measured plot:")
    print("-", args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
