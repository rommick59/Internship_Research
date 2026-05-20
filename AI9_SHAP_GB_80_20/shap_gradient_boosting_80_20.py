"""AI9 — SHAP analysis for Gradient Boosting only (80/20 split).

This script trains a GradientBoostingRegressor (same hyperparameters as AI8)
using the normalized TRAIN split, then computes SHAP values on the normalized
TEST split.

Default inputs reuse AI8 80/20 normalized splits:
- Internship_Research/AI8/split_train_norm_80_20.csv
- Internship_Research/AI8/split_test_norm_80_20.csv

Outputs (in this folder):
- images/shap_summary_beeswarm_test.png
- images/shap_summary_bar_test.png
- images/shap_decision_test.png
- images/shap_heatmap_test.png
- images/shap_waterfall_median_abs_error_test.png
- shap_importance_mean_abs_test.csv
- shap_local_explanation_median_abs_error_test.csv
- shap_directionality_test.csv

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI9_SHAP_GB_80_20/shap_gradient_boosting_80_20.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI9: SHAP for Gradient Boosting (80/20)")
    p.add_argument(
        "--train-csv",
        type=Path,
        default=Path("Internship_Research/AI8/split_train_norm_80_20.csv"),
        help="Normalized TRAIN split (features + target)",
    )
    p.add_argument(
        "--test-csv",
        type=Path,
        default=Path("Internship_Research/AI8/split_test_norm_80_20.csv"),
        help="Normalized TEST split (features + target)",
    )
    p.add_argument("--target", type=str, default="PR(mm/r)")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("Internship_Research/AI9_SHAP_GB_80_20"),
        help="Output directory",
    )
    p.add_argument(
        "--max-test-samples",
        type=int,
        default=0,
        help="If >0, subsample the TEST set to this many rows for faster SHAP.",
    )
    return p.parse_args()


def _load_xy(csv_path: Path, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(csv_path)
    if target not in df.columns:
        raise ValueError(f"Missing target {target!r} in {csv_path}")
    X = df.drop(columns=[target]).copy()
    y = df[target].to_numpy(dtype=float)
    return X, y


def main() -> int:
    args = parse_args()

    X_train, y_train = _load_xy(args.train_csv, args.target)
    X_test, y_test = _load_xy(args.test_csv, args.target)

    if args.max_test_samples and args.max_test_samples > 0 and len(X_test) > args.max_test_samples:
        # deterministic subsample
        X_test = X_test.iloc[: args.max_test_samples].copy()
        y_test = y_test[: args.max_test_samples]

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        subsample=1.0,
        min_samples_leaf=2,
        random_state=args.random_state,
    )

    print("Fitting Gradient Boosting...")
    model.fit(X_train.to_numpy(dtype=float), y_train)

    y_test_pred = model.predict(X_test.to_numpy(dtype=float))

    # SHAP
    try:
        import shap  # type: ignore
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Missing dependency 'shap'. Install with: pip install shap"
        ) from e

    print("Computing SHAP values on TEST...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test.to_numpy(dtype=float))

    # Convert to a stable 2D numpy array
    if isinstance(shap_values, list):
        # Some explainers return a list (e.g., multioutput). For regression it should be a single array.
        shap_values_arr = np.asarray(shap_values[0])
    else:
        shap_values_arr = np.asarray(shap_values)

    if shap_values_arr.ndim != 2 or shap_values_arr.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Unexpected SHAP shape {shap_values_arr.shape}; expected (n_samples, n_features={X_test.shape[1]})"
        )

    out_dir = args.out_dir
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values_arr, X_test, show=False)
    out_bee = img_dir / "shap_summary_beeswarm_test.png"
    plt.savefig(out_bee, dpi=220, bbox_inches="tight")
    plt.close()

    # Summary plot (bar)
    plt.figure()
    shap.summary_plot(shap_values_arr, X_test, plot_type="bar", show=False)
    out_bar = img_dir / "shap_summary_bar_test.png"
    plt.savefig(out_bar, dpi=220, bbox_inches="tight")
    plt.close()

    # Mean(|SHAP|) importance
    mean_abs = np.mean(np.abs(shap_values_arr), axis=0)
    imp = pd.DataFrame({"feature": list(X_test.columns), "mean_abs_shap_test": mean_abs}).sort_values(
        "mean_abs_shap_test", ascending=False
    )
    out_csv = out_dir / "shap_importance_mean_abs_test.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_csv, index=False)

    # Additional ultra-relevant plots (to have 5 total figures)
    # Build a SHAP Explanation object for the plotting API
    abs_err = np.abs(y_test - y_test_pred)
    order = np.argsort(abs_err)
    median_i = int(order[len(order) // 2])
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_value = expected_value[0]
    expected_value = float(expected_value)

    explanation = shap.Explanation(
        values=shap_values_arr,
        base_values=np.full(shape=(len(X_test),), fill_value=expected_value, dtype=float),
        data=X_test.to_numpy(),
        feature_names=list(X_test.columns),
    )

    # Decision plot (global view across multiple observations)
    n_decision = min(200, len(X_test))
    plt.figure(figsize=(11.5, 6.5))
    shap.decision_plot(
        expected_value,
        shap_values_arr[:n_decision, :],
        X_test.iloc[:n_decision, :],
        show=False,
    )
    out_decision = img_dir / "shap_decision_test.png"
    plt.savefig(out_decision, dpi=220, bbox_inches="tight")
    plt.close()

    # Heatmap (pattern view across observations)
    plt.figure(figsize=(12.0, 7.0))
    shap.plots.heatmap(explanation[:n_decision], show=False)
    out_heatmap = img_dir / "shap_heatmap_test.png"
    plt.savefig(out_heatmap, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.plots.waterfall(explanation[median_i], show=False)
    out_waterfall = img_dir / "shap_waterfall_median_abs_error_test.png"
    plt.savefig(out_waterfall, dpi=220, bbox_inches="tight")
    plt.close()

    # Detailed per-feature breakdown for the representative sample (median absolute error)
    x_row = X_test.iloc[median_i]
    base = expected_value
    pred = float(y_test_pred[median_i])
    true = float(y_test[median_i])
    residual = true - pred

    local = pd.DataFrame(
        {
            "feature": list(X_test.columns),
            "feature_value": [float(x_row[c]) for c in X_test.columns],
            "shap_value": shap_values_arr[median_i, :].astype(float),
        }
    )
    local["abs_shap_value"] = local["shap_value"].abs()
    local["direction"] = np.where(local["shap_value"] >= 0, "increase", "decrease")
    local = local.sort_values("abs_shap_value", ascending=False).reset_index(drop=True)

    running = float(base)
    cumulative: list[float] = []
    for v in local["shap_value"].to_list():
        running += float(v)
        cumulative.append(float(running))
    local["cumulative_output"] = cumulative

    # Repeat context columns on each row to keep the CSV self-contained.
    local.insert(0, "sample_index", int(median_i))
    local.insert(1, "base_value", float(base))
    local.insert(2, "model_output_pred", float(pred))
    local.insert(3, "true_target", float(true))
    local.insert(4, "residual_true_minus_pred", float(residual))

    out_local_csv = out_dir / "shap_local_explanation_median_abs_error_test.csv"
    local.to_csv(out_local_csv, index=False)

    # Global directionality: compare mean SHAP between low and high quantiles of each feature.
    # This helps answer: "when the feature value is high, does it tend to increase the prediction?"
    q_low, q_high = 0.2, 0.8
    rows: list[dict[str, float | int | str]] = []
    x_test_arr = X_test.to_numpy(dtype=float)
    for j, name in enumerate(X_test.columns):
        xj = x_test_arr[:, j]
        sj = shap_values_arr[:, j]
        lo = float(np.quantile(xj, q_low))
        hi = float(np.quantile(xj, q_high))

        mask_low = xj <= lo
        mask_high = xj >= hi
        n_low = int(np.sum(mask_low))
        n_high = int(np.sum(mask_high))

        # Guard against degenerate features (constant / too few unique values)
        mean_low = float(np.mean(sj[mask_low])) if n_low > 0 else float("nan")
        mean_high = float(np.mean(sj[mask_high])) if n_high > 0 else float("nan")
        delta = mean_high - mean_low
        trend = "higher_value_increases" if delta > 0 else "higher_value_decreases"
        if not np.isfinite(delta):
            trend = "insufficient_variation"

        rows.append(
            {
                "feature": str(name),
                "q_low": float(q_low),
                "q_high": float(q_high),
                "value_at_q_low": float(lo),
                "value_at_q_high": float(hi),
                "n_low": n_low,
                "n_high": n_high,
                "mean_shap_low": float(mean_low),
                "mean_shap_high": float(mean_high),
                "high_minus_low": float(delta),
                "trend": str(trend),
            }
        )

    directionality = pd.DataFrame(rows).sort_values("high_minus_low", ascending=False)
    out_dir_csv = out_dir / "shap_directionality_test.csv"
    directionality.to_csv(out_dir_csv, index=False)

    print("Saved:")
    print("-", out_bee)
    print("-", out_bar)
    print("-", out_decision)
    print("-", out_heatmap)
    print("-", out_waterfall)
    print("-", out_csv)
    print("-", out_local_csv)
    print("-", out_dir_csv)
    print("Note: SHAP computed on normalized features (same space as model training).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
