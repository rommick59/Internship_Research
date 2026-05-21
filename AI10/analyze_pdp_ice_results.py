"""Analyze PDP and ICE results and generate a detailed report with metrics.

This script:
1) Loads the train/test splits (6 features)
2) Trains a GradientBoostingRegressor
3) Computes PDP, ICE, and feature importance
4) Calculates dependency metrics for each feature:
   - PDP Range: spread of partial dependence values
   - ICE Variance: heterogeneity of individual effects
   - Feature Importance: importance from the model
   - PDP Slope: average rate of change
5) Generates a detailed markdown report

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI10/analyze_pdp_ice_results.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import partial_dependence


def compute_ice_curves(model, X: np.ndarray, feature_idx: int, grid_resolution: int = 50) -> np.ndarray:
    """Compute ICE curves for all samples. Returns shape (n_samples, grid_resolution)."""
    feature_min = X[:, feature_idx].min()
    feature_max = X[:, feature_idx].max()
    grid_values = np.linspace(feature_min, feature_max, grid_resolution)
    
    ice_curves = np.zeros((len(X), grid_resolution))
    for i in range(len(X)):
        X_modified = np.repeat(X[i:i+1], len(grid_values), axis=0)
        X_modified[:, feature_idx] = grid_values
        ice_curves[i, :] = model.predict(X_modified)
    
    return ice_curves


def compute_dependency_metrics(
    model,
    X: np.ndarray,
    feature_names: list[str],
    grid_resolution: int = 50,
) -> dict[str, dict[str, float]]:
    """Compute detailed dependency metrics for each feature."""
    metrics = {}
    
    for feat_idx, feat_name in enumerate(feature_names):
        feat_min = X[:, feat_idx].min()
        feat_max = X[:, feat_idx].max()
        
        # 1. PDP metrics
        pd_result = partial_dependence(
            model,
            X,
            features=[feat_idx],
            grid_resolution=grid_resolution,
            percentiles=(0.0, 1.0),
        )
        pdp_values = pd_result["average"][0]
        pdp_range = float(np.max(pdp_values) - np.min(pdp_values))
        pdp_mean = float(np.mean(pdp_values))
        pdp_std = float(np.std(pdp_values))
        
        # Slope: average absolute change between adjacent points
        pdp_diffs = np.abs(np.diff(pdp_values))
        pdp_slope = float(np.mean(pdp_diffs))
        
        # 2. ICE metrics (sample a subset for speed)
        sample_size = min(500, len(X))
        sample_idx = np.random.RandomState(42).choice(len(X), size=sample_size, replace=False)
        X_sample = X[sample_idx]
        
        ice_curves = compute_ice_curves(model, X_sample, feat_idx, grid_resolution)
        
        # ICE std (heterogeneity across samples)
        ice_std_per_point = np.std(ice_curves, axis=0)
        ice_heterogeneity = float(np.mean(ice_std_per_point))
        
        # ICE range per sample (variability within each sample)
        ice_ranges = np.max(ice_curves, axis=1) - np.min(ice_curves, axis=1)
        ice_range_mean = float(np.mean(ice_ranges))
        ice_range_std = float(np.std(ice_ranges))
        
        # 3. Feature importance
        feat_importance = float(model.feature_importances_[feat_idx])
        
        # 4. Normalized metrics (0-100 scale)
        metrics[feat_name] = {
            "feature_importance": feat_importance * 100,
            "pdp_range": pdp_range,
            "pdp_mean": pdp_mean,
            "pdp_std": pdp_std,
            "pdp_slope": pdp_slope,
            "ice_heterogeneity": ice_heterogeneity,
            "ice_range_mean": ice_range_mean,
            "ice_range_std": ice_range_std,
            "feature_min": float(feat_min),
            "feature_max": float(feat_max),
        }
    
    return metrics


def generate_report(
    metrics: dict[str, dict[str, float]],
    feature_names: list[str],
    output_file: Path,
) -> None:
    """Generate a detailed markdown report."""
    
    # Sort by feature importance
    sorted_features = sorted(metrics.keys(), 
                           key=lambda x: metrics[x]["feature_importance"], 
                           reverse=True)
    
    report_lines = [
        "# AI10: PDP & ICE Analysis Report",
        "## Gradient Boosting Model - 6 Features (80/20 Split)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report provides a detailed analysis of feature dependence in the Gradient Boosting model",
        "using Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots.",
        "",
        "### Dependency Metrics Overview",
        "",
    ]
    
    # Create summary table
    report_lines.append("| Feature | Importance (%) | PDP Range | PDP Slope | ICE Heterogeneity |")
    report_lines.append("|---------|---|---|---|---|")
    
    for feat in sorted_features:
        m = metrics[feat]
        report_lines.append(
            f"| {feat} | {m['feature_importance']:.2f}% | "
            f"{m['pdp_range']:.6f} | {m['pdp_slope']:.6f} | {m['ice_heterogeneity']:.6f} |"
        )
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Detailed Feature Analysis",
        "",
    ])
    
    for rank, feat in enumerate(sorted_features, 1):
        m = metrics[feat]
        
        report_lines.extend([
            f"### {rank}. {feat}",
            "",
            "#### Feature Statistics",
            f"- **Range**: [{m['feature_min']:.6f}, {m['feature_max']:.6f}]",
            f"- **Importance**: {m['feature_importance']:.2f}%",
            "",
            "#### Partial Dependence Plot (PDP) Metrics",
            f"- **PDP Range**: {m['pdp_range']:.6f}",
            f"  - Interpretation: How much the predicted value varies across the feature range",
            f"  - Higher values indicate stronger marginal effect",
            f"- **PDP Mean**: {m['pdp_mean']:.6f}",
            f"  - Average prediction when averaging over all other features",
            f"- **PDP Std Dev**: {m['pdp_std']:.6f}",
            f"  - Variability of predictions within the PDP",
            f"- **PDP Slope**: {m['pdp_slope']:.6f}",
            f"  - Average rate of change per unit increase in feature",
            f"  - Higher values indicate non-linearity or strong interactions",
            "",
            "#### Individual Conditional Expectation (ICE) Metrics",
            f"- **ICE Heterogeneity**: {m['ice_heterogeneity']:.6f}",
            f"  - Standard deviation of predictions across individual curves at each point",
            f"  - High values indicate heterogeneous effects (different impact per sample)",
            f"- **ICE Range (Mean)**: {m['ice_range_mean']:.6f}",
            f"  - Average spread of predictions for individual samples",
            f"  - Shows instance-specific variability",
            f"- **ICE Range (Std)**: {m['ice_range_std']:.6f}",
            f"  - Consistency of individual effect sizes across samples",
            f"  - Low values = consistent effects; High values = heterogeneous effects",
            "",
        ])
    
    # Summary section
    report_lines.extend([
        "---",
        "",
        "## Dependency Interpretation Guide",
        "",
        "### What do these metrics mean?",
        "",
        "**Feature Importance**",
        "- Percentage contribution to model predictions",
        "- Sum of all features = 100%",
        "- Higher = more influential on model decisions",
        "",
        "**PDP Range**",
        "- Spread of average predictions across the feature range",
        "- Larger range = stronger marginal effect",
        "- Shows 'average effect' after averaging over other features",
        "",
        "**PDP Slope**",
        "- Average rate of prediction change",
        "- Indicates non-linearity and average steepness",
        "- High slope = predictions change rapidly with feature",
        "",
        "**ICE Heterogeneity**",
        "- How much individual samples deviate from the PDP average",
        "- High heterogeneity = different samples affected differently (interactions present)",
        "- Low heterogeneity = uniform effect across all samples",
        "",
        "**ICE Range**",
        "- Prediction spread for individual samples",
        "- Shows how sensitive each instance is to feature changes",
        "- High variance = some samples very sensitive, others less so",
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ])
    
    # Add key findings
    top_feature = sorted_features[0]
    top_importance = metrics[top_feature]["feature_importance"]
    
    report_lines.extend([
        f"1. **{top_feature}** is the dominant feature with {top_importance:.2f}% importance",
        f"   - PDP Range: {metrics[top_feature]['pdp_range']:.6f}",
        f"   - ICE Heterogeneity: {metrics[top_feature]['ice_heterogeneity']:.6f}",
        "",
    ])
    
    # Heterogeneity analysis
    max_heterogeneity_feat = max(sorted_features, key=lambda x: metrics[x]["ice_heterogeneity"])
    report_lines.append(
        f"2. **{max_heterogeneity_feat}** shows the highest heterogeneity "
        f"({metrics[max_heterogeneity_feat]['ice_heterogeneity']:.6f})"
    )
    report_lines.append("   - Indicates strong instance-specific effects or interactions")
    report_lines.append("")
    
    # Linear vs non-linear
    max_slope_feat = max(sorted_features, key=lambda x: metrics[x]["pdp_slope"])
    report_lines.append(
        f"3. **{max_slope_feat}** shows the steepest average change "
        f"({metrics[max_slope_feat]['pdp_slope']:.6f})"
    )
    report_lines.append("   - Non-linear relationships present")
    report_lines.append("")
    
    report_lines.extend([
        "---",
        "",
        "## Generated Visualizations",
        "",
        "The following visualizations have been generated in `AI10/images/`:",
        "",
        "### Partial Dependence Plots (PDP)",
        "- Show the marginal effect of each feature",
        "- X-axis: Feature value (normalized)",
        "- Y-axis: Predicted PR (penetration rate)",
        "- One line per feature showing average effect",
        "",
        "### Individual Conditional Expectation (ICE) Plots",
        "- Show individual sample effects",
        "- Thin lines: individual sample curves",
        "- Thick red line: PDP average",
        "- Dense/sparse spread indicates homogeneous/heterogeneous effects",
        "",
    ])
    
    # Write report
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"✅ Report saved to: {output_file}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze PDP/ICE results and generate report")
    p.add_argument(
        "--train",
        type=Path,
        default=Path("Internship_Research/AI8/split_train_norm_80_20.csv"),
        help="Normalized TRAIN split",
    )
    p.add_argument(
        "--test",
        type=Path,
        default=Path("Internship_Research/AI8/split_test_norm_80_20.csv"),
        help="Normalized TEST split",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Internship_Research/AI10/ANALYSIS_REPORT.md"),
        help="Output markdown report",
    )
    p.add_argument(
        "--target",
        type=str,
        default="PR(mm/r)",
        help="Target column",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    
    print("Loading data...")
    df_train = pd.read_csv(args.train)
    df_test = pd.read_csv(args.test)
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    
    X_df = df_combined.drop(columns=[args.target])
    y = df_combined[args.target].to_numpy(dtype=float)
    X = X_df.to_numpy(dtype=float)
    feature_names = X_df.columns.tolist()
    
    print(f"Data shape: {X.shape}")
    print(f"Features: {feature_names}")
    
    print("\nTraining Gradient Boosting model...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.01,
        max_depth=3,
        subsample=1.0,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X, y)
    
    print("\nComputing dependency metrics...")
    metrics = compute_dependency_metrics(model, X, feature_names, grid_resolution=50)
    
    print("\nGenerating report...")
    generate_report(metrics, feature_names, args.output)
    
    print("\n✅ Analysis complete!")
    return 0


if __name__ == "__main__":
    exit(main())
