"""AI8 — Ranking + diagrams for the split 0.80/0.00/0.20.

Reads results from Internship_Research/AI8 and writes rankings/figures there.
Uses TRAIN and TEST metrics only (no validation for 80/20).

Outputs:
- Internship_Research/AI8/rankings_80_20.csv
- Internship_Research/AI8/rankings_80_20.md
- Internship_Research/AI8/images/ranking_80_20_train.png
- Internship_Research/AI8/images/ranking_80_20_test.png
- Internship_Research/AI8/images/ranking_80_20_gap.png
- Internship_Research/AI8/images/ranking_80_20_train_<metric>.png
- Internship_Research/AI8/images/ranking_80_20_test_<metric>.png
- Internship_Research/AI8/images/ranking_80_20_gap_<metric>.png

Run (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI8/ranking_diagrams_80_20.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def metric_direction(metric: str) -> str:
    if metric.endswith(("_r", "_r2", "_vaf")):
        return "higher"
    if metric.endswith(("_mse", "_rmse", "_mae")):
        return "lower"
    raise ValueError(f"Unknown metric direction for: {metric}")


def make_model_registry(ai_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("Linear Regression", ai_dir / "linear_results.csv"),
        ("Random Forest", ai_dir / "random_forest_results.csv"),
        ("RVM", ai_dir / "rvm_results.csv"),
        ("XGBoost", ai_dir / "xgboost_results.csv"),
        ("Gradient Boosting", ai_dir / "gradient_boosting_results.csv"),
    ]


def load_first_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if len(df) != 1:
        raise ValueError(f"Expected 1 row in {csv_path}, got {len(df)}")
    return df.iloc[0]


def build_metrics_table(models: list[tuple[str, Path]]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name, path in models:
        if not path.exists():
            continue
        row = load_first_row(path)
        # Only train_* and test_* (ignore val_)
        metric_cols = [c for c in row.index if c.startswith(("train_", "test_"))]
        d: dict[str, float | str] = {"model": model_name}
        for c in metric_cols:
            d[c] = float(row[c])
        rows.append(d)

    out = pd.DataFrame(rows).set_index("model")
    cols = sorted(out.columns, key=lambda x: (x.split("_")[0], x.split("_")[1]))
    return out[cols]


def build_rankings(metrics_table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for metric in metrics_table.columns:
        direction = metric_direction(metric)
        series = metrics_table[metric]
        ascending = direction == "lower"
        ordered = series.sort_values(ascending=ascending)
        for rank, (model, value) in enumerate(ordered.items(), start=1):
            records.append(
                {
                    "metric": metric,
                    "direction": direction,
                    "rank": rank,
                    "model": model,
                    "value": float(value),
                }
            )
    return pd.DataFrame(records)


def save_rankings_md(rankings: pd.DataFrame, out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Rankings — split 0.80/0.20")
    lines.append("")
    lines.append(
        "Each metric is ranked separately. For `*_r`, `*_r2`, `*_vaf`: higher = better. "
        "For error metrics `*_mse`, `*_rmse`, `*_mae`: lower = better."
    )
    lines.append("")

    for metric in sorted(rankings["metric"].unique()):
        sub = rankings[rankings["metric"] == metric].sort_values("rank")
        direction = sub["direction"].iloc[0]
        lines.append(f"## {metric} ({direction})")
        lines.append("")
        lines.append("| Rank | Model | Value |")
        lines.append("|---:|---|---:|")
        for _, r in sub.iterrows():
            lines.append(f"| {int(r['rank'])} | {r['model']} | {float(r['value']):.6g} |")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def plot_barh(ax, series: pd.Series, title: str, better: str) -> None:
    if better == "higher":
        ordered = series.sort_values(ascending=True)
    else:
        ordered = series.sort_values(ascending=False)

    ax.barh(ordered.index, ordered.values)
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.4)


def save_barh(series: pd.Series, title: str, better: str, out: Path, dpi: int = 200) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
    plot_barh(ax, series, title, better=better)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="AI8 ranking + diagrams for split 80/20")
    p.add_argument("--ai-dir", type=Path, default=Path("Internship_Research/AI8"))
    p.add_argument("--out-rankings-csv", type=Path, default=Path("Internship_Research/AI8/rankings_80_20.csv"))
    p.add_argument("--out-rankings-md", type=Path, default=Path("Internship_Research/AI8/rankings_80_20.md"))
    p.add_argument("--out-dir-images", type=Path, default=Path("Internship_Research/AI8/images"))
    args = p.parse_args()

    models = make_model_registry(args.ai_dir)

    metrics_table = build_metrics_table(models)
    rankings = build_rankings(metrics_table)

    args.out_rankings_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_rankings_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir_images.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(args.out_rankings_csv, index=False)
    save_rankings_md(rankings, args.out_rankings_md)
    saved_figures: list[Path] = []

    train_key = pd.DataFrame(
        {
            "train_r2": metrics_table["train_r2"],
            "train_rmse": metrics_table["train_rmse"],
            "train_mae": metrics_table["train_mae"],
        }
    )

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), constrained_layout=True)
    plot_barh(axes[0], train_key["train_r2"], "TRAIN R² (higher is better)", better="higher")
    plot_barh(axes[1], train_key["train_rmse"], "TRAIN RMSE (lower is better)", better="lower")
    plot_barh(axes[2], train_key["train_mae"], "TRAIN MAE (lower is better)", better="lower")
    fig.suptitle("Model ranking — split 0.80/0.20 (TRAIN)")
    out_train = args.out_dir_images / "ranking_80_20_train.png"
    fig.savefig(out_train, dpi=200)
    plt.close(fig)
    saved_figures.append(out_train)
    saved_figures.extend(
        [
            save_barh(
                train_key["train_r2"],
                "TRAIN R² (higher is better)",
                "higher",
                args.out_dir_images / "ranking_80_20_train_r2.png",
            ),
            save_barh(
                train_key["train_rmse"],
                "TRAIN RMSE (lower is better)",
                "lower",
                args.out_dir_images / "ranking_80_20_train_rmse.png",
            ),
            save_barh(
                train_key["train_mae"],
                "TRAIN MAE (lower is better)",
                "lower",
                args.out_dir_images / "ranking_80_20_train_mae.png",
            ),
        ]
    )

    test_key = pd.DataFrame(
        {
            "test_r2": metrics_table["test_r2"],
            "test_rmse": metrics_table["test_rmse"],
            "test_mae": metrics_table["test_mae"],
        }
    )

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), constrained_layout=True)
    plot_barh(axes[0], test_key["test_r2"], "TEST R² (higher is better)", better="higher")
    plot_barh(axes[1], test_key["test_rmse"], "TEST RMSE (lower is better)", better="lower")
    plot_barh(axes[2], test_key["test_mae"], "TEST MAE (lower is better)", better="lower")
    fig.suptitle("Model ranking — split 0.80/0.20 (TEST)")
    out_test = args.out_dir_images / "ranking_80_20_test.png"
    fig.savefig(out_test, dpi=200)
    plt.close(fig)
    saved_figures.append(out_test)
    saved_figures.extend(
        [
            save_barh(
                test_key["test_r2"],
                "TEST R² (higher is better)",
                "higher",
                args.out_dir_images / "ranking_80_20_test_r2.png",
            ),
            save_barh(
                test_key["test_rmse"],
                "TEST RMSE (lower is better)",
                "lower",
                args.out_dir_images / "ranking_80_20_test_rmse.png",
            ),
            save_barh(
                test_key["test_mae"],
                "TEST MAE (lower is better)",
                "lower",
                args.out_dir_images / "ranking_80_20_test_mae.png",
            ),
        ]
    )

    gap = pd.DataFrame(
        {
            "rmse_gap": (metrics_table["test_rmse"] - metrics_table["train_rmse"]),
            "mae_gap": (metrics_table["test_mae"] - metrics_table["train_mae"]),
        }
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
    plot_barh(axes[0], gap["rmse_gap"], "RMSE gap (TEST - TRAIN) (lower is better)", better="lower")
    plot_barh(axes[1], gap["mae_gap"], "MAE gap (TEST - TRAIN) (lower is better)", better="lower")
    fig.suptitle("Overfitting view (TRAIN→TEST gap) — split 0.80/0.20")
    out_gap = args.out_dir_images / "ranking_80_20_gap.png"
    fig.savefig(out_gap, dpi=200)
    plt.close(fig)
    saved_figures.append(out_gap)
    saved_figures.extend(
        [
            save_barh(
                gap["rmse_gap"],
                "RMSE gap (TEST - TRAIN) (lower is better)",
                "lower",
                args.out_dir_images / "ranking_80_20_gap_rmse.png",
            ),
            save_barh(
                gap["mae_gap"],
                "MAE gap (TEST - TRAIN) (lower is better)",
                "lower",
                args.out_dir_images / "ranking_80_20_gap_mae.png",
            ),
        ]
    )

    print("Saved rankings:")
    print("-", args.out_rankings_csv)
    print("-", args.out_rankings_md)
    print("Saved figures:")
    for path in saved_figures:
        print("-", path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
