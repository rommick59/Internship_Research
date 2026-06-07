"""AI8 — Predicted vs Measured scatter plots (style 'regression diagnostic').

For each IA model trained on the 80% TRAIN split, generates a subplot
showing Measured (X) vs Predicted (Y) PR values on the 20% TEST split.
Includes 1:1 line, ±20% tolerance band, and metrics in the title.
Axes start at 0 for a clean engineering-style plot.

Outputs:
- Internship_Research/AI8/images/predicted_vs_measured_80_20.png

Run (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI8/predicted_vs_measured_80_20.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ----------------------------------------------------------------------
# Global matplotlib styling: large, readable fonts
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "figure.titlesize": 20,
})


def make_models(random_state: int) -> list[tuple[str, object]]:
    """Build the same 5 models used in AI8 (matching hyperparameters)."""
    models: list[tuple[str, object]] = []

    models.append(("Linear Regression", LinearRegression()))

    models.append((
        "Random Forest",
        RandomForestRegressor(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=5,
            max_features=0.5,
            random_state=random_state,
            n_jobs=-1,
        ),
    ))

    try:
        from sklearn_rvm import EMRVR
        models.append((
            "RVM",
            EMRVR(kernel="rbf", degree=3, gamma=0.01, coef0=0.0, tol=0.01, max_iter=1000),
        ))
    except ModuleNotFoundError:
        print("NOTE: sklearn-rvm not installed, skipping RVM")

    try:
        from xgboost import XGBRegressor
        models.append((
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
        ))
    except ModuleNotFoundError:
        print("NOTE: xgboost not installed, skipping XGBoost")

    models.append((
        "Gradient Boosting",
        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            subsample=1.0,
            min_samples_leaf=2,
            random_state=random_state,
        ),
    ))

    return models


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def main() -> int:
    # --- Paths ---
    ai8_dir = Path("Internship_Research/AI8")
    train_csv = ai8_dir / "split_train_norm_80_20.csv"
    test_csv = ai8_dir / "split_test_norm_80_20.csv"
    out_path = ai8_dir / "images" / "predicted_vs_measured_80_20.png"

    # --- Load splits ---
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    target = "PR(mm/r)"

    X_train = train_df.drop(columns=[target]).to_numpy(dtype=float)
    y_train = train_df[target].to_numpy(dtype=float)
    X_test = test_df.drop(columns=[target]).to_numpy(dtype=float)
    y_test = test_df[target].to_numpy(dtype=float)

    # --- Un-normalize PR back to original units (mm/r) for display ---
    try:
        raw_df = pd.read_csv(Path("Internship_Research/TBM_data_cleaned.csv"))
        pr_min = float(raw_df[target].min())
        pr_max = float(raw_df[target].max())
    except Exception:
        pr_min, pr_max = 0.0, 1.0

    def to_orig(y_norm: np.ndarray) -> np.ndarray:
        return y_norm * (pr_max - pr_min) + pr_min

    # --- Train models and collect predictions + metrics ---
    models = make_models(random_state=42)
    results: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred_n = model.predict(X_test)
        y_true = to_orig(y_test)
        y_pred = to_orig(y_pred_n)
        r2 = r2_score(y_true, y_pred)
        rmse = float(math.sqrt(np.mean((y_true - y_pred) ** 2)))
        results.append((name, y_true, y_pred, r2, rmse))

    # --- Plot (2 columns x N rows layout, like the reference image) ---
    n = len(results)
    ncols = 3
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols,
        figsize=(6.0 * ncols, 5.5 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    # Global axis upper bound (round up nicely)
    upper = 1

    for idx, (name, y_true, y_pred, r2, rmse) in enumerate(results):
        ax = axes[idx // ncols, idx % ncols]

        # Scatter points
        ax.scatter(y_true, y_pred, s=22, alpha=0.6, edgecolors="none", color="C1")

        # 1:1 line and ±20% band
        lims = [0, upper]
        ax.plot(lims, lims, "k-", linewidth=2.0, label="1:1")
        ax.plot(lims, [v * 0.8 for v in lims], "k--", linewidth=1.2, label="±20%")
        ax.plot(lims, [v * 1.2 for v in lims], "k--", linewidth=1.2)

        # Title with model name + metrics (large and bold)
        ax.set_title(
            f"{name}\n$R^2$={r2:.3f}, RMSE={rmse:.2f}",
            fontsize=18,
            fontweight="bold",
        )
        ax.set_xlabel("Measured PR (mm/r)", fontsize=16, fontweight="bold")
        ax.set_ylabel("Predicted PR (mm/r)", fontsize=16, fontweight="bold")
        ax.tick_params(axis="both", labelsize=14)

        # FORCE axes to start at 0
        ax.set_xlim(0, upper)
        ax.set_ylim(0, upper)
        ax.set_aspect("equal", adjustable="box")

        ax.grid(linestyle=":", alpha=0.4)
        ax.legend(loc="upper left", frameon=True)

    # Hide empty subplots (in case n < nrows * ncols)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(
        "AI8 — Predicted vs Measured PR (TEST 80/20, train-only normalization)",
        fontsize=20,
        fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
