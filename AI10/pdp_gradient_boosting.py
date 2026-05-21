"""Partial Dependence Plot (PDP) for Gradient Boosting regression on normalized TBM data.

This script:
1) Loads/generates the ML-ready normalized dataset (values in [0, 1])
2) Trains a GradientBoostingRegressor on the full dataset
3) Generates Partial Dependence Plots for each feature
4) Saves plots to Internship_Research/AI10/images/

Partial Dependence Plots show the marginal effect of features on predictions,
averaged over all other features.

Example (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI10/pdp_gradient_boosting.py
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Partial Dependence Plots for Gradient Boosting")
    p.add_argument(
        "--train",
        type=Path,
        default=Path("Internship_Research/AI8/split_train_norm_80_20.csv"),
        help="Path to normalized TRAIN split CSV (80% data)",
    )
    p.add_argument(
        "--test",
        type=Path,
        default=Path("Internship_Research/AI8/split_test_norm_80_20.csv"),
        help="Path to normalized TEST split CSV (20% data)",
    )
    p.add_argument(
        "--target",
        type=str,
        default="PR(mm/r)",
        help='Target column name to predict (default: "PR(mm/r)")',
    )
    p.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="Number of boosting stages",
    )
    p.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Boosting learning rate",
    )
    p.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max depth of individual trees",
    )
    p.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="Subsample ratio (stochastic GB when <1)",
    )
    p.add_argument(
        "--min-samples-leaf",
        type=int,
        default=2,
        help="Minimum samples per leaf",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--grid-resolution",
        type=int,
        default=50,
        help="Grid resolution for PDP (number of points per feature)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Internship_Research/AI10/images"),
        help="Output directory for plots",
    )
    return p.parse_args()


def sanitize_filename(name: str) -> str:
    """Remove/replace characters that are problematic in filenames."""
    # Replace problematic characters
    name = name.replace("/", "_")
    name = name.replace("\\", "_")
    name = name.replace(":", "_")
    name = name.replace("*", "_")
    name = name.replace("?", "_")
    name = name.replace('"', "_")
    name = name.replace("<", "_")
    name = name.replace(">", "_")
    name = name.replace("|", "_")
    return name


def plot_pdp_single_feature(
    model,
    X: np.ndarray,
    feature_idx: int,
    feature_name: str,
    grid_resolution: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute and return PDP values for a single feature."""
    pd_result = partial_dependence(
        model,
        X,
        features=[feature_idx],
        grid_resolution=grid_resolution,
        percentiles=(0.0, 1.0),
    )
    grid_values = pd_result["grid_values"][0]
    pd_values = pd_result["average"][0]
    return grid_values, pd_values


def main() -> int:
    args = parse_args()

    # Load train and test splits
    print(f"Loading TRAIN split from: {args.train}")
    df_train = pd.read_csv(args.train)
    
    print(f"Loading TEST split from: {args.test}")
    df_test = pd.read_csv(args.test)

    if args.target not in df_train.columns:
        raise ValueError(f"Unknown target column: {args.target!r}")
    if args.target not in df_test.columns:
        raise ValueError(f"Unknown target column: {args.target!r} in test set")

    # Combine train and test for full dataset to train model
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    
    X_df = df_combined.drop(columns=[args.target])
    y = df_combined[args.target].to_numpy(dtype=float)
    X = X_df.to_numpy(dtype=float)
    feature_names = X_df.columns.tolist()

    print(f"Combined data shape: X {X.shape}, y {y.shape}")
    print(f"Features ({len(feature_names)}): {feature_names}")

    # Train model
    print("\nTraining Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=int(args.n_estimators),
        learning_rate=float(args.learning_rate),
        max_depth=int(args.max_depth),
        subsample=float(args.subsample),
        min_samples_leaf=int(args.min_samples_leaf),
        random_state=int(args.random_state),
    )
    model.fit(X, y)

    print(f"Model trained. Feature importances: {dict(zip(feature_names, model.feature_importances_))}")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Generate PDP plots for each feature
    print(f"\nGenerating PDP plots to {args.output_dir}...")
    for idx, feature_name in enumerate(feature_names):
        print(f"  Processing feature {idx + 1}/{len(feature_names)}: {feature_name}")

        grid_values, pd_values = plot_pdp_single_feature(
            model,
            X,
            feature_idx=idx,
            feature_name=feature_name,
            grid_resolution=args.grid_resolution,
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(grid_values, pd_values, linewidth=2.5, color="steelblue")
        ax.fill_between(grid_values, pd_values, alpha=0.2, color="steelblue")
        ax.set_xlabel(f"{feature_name} (normalized)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Partial Dependence (PR prediction)", fontsize=12, fontweight="bold")
        ax.set_title(f"Partial Dependence Plot: {feature_name}", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3, linestyle="--")

        # Save
        safe_name = sanitize_filename(feature_name)
        output_file = args.output_dir / f"pdp_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"    Saved to {output_file}")

    print("\nPDP generation complete!")
    return 0


if __name__ == "__main__":
    exit(main())
