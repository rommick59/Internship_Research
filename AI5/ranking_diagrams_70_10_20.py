"""Ranking + diagrams for the split 0.70/0.10/0.20 (AI5).

Reads results from Internship_Research/AI5 and writes rankings/figures there.

Outputs:
- Internship_Research/AI5/rankings_70_10_20.csv
- Internship_Research/AI5/rankings_70_10_20.md
- Internship_Research/AI5/images/ranking_70_10_20_train.png
- Internship_Research/AI5/images/ranking_70_10_20_test.png
- Internship_Research/AI5/images/ranking_70_10_20_gap.png

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI5/ranking_diagrams_70_10_20.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class Split:
    train: float
    val: float
    test: float


def parse_split(value: str) -> Split:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError('Split must be formatted like "0.7,0.1,0.2"')
    return Split(train=float(parts[0]), val=float(parts[1]), test=float(parts[2]))


def metric_direction(metric: str) -> str:
    if metric.endswith("_r") or metric.endswith("_r2"):
        return "higher"
    if metric.endswith("_mse") or metric.endswith("_rmse") or metric.endswith("_mae"):
        return "lower"
    raise ValueError(f"Unknown metric direction for: {metric}")


def make_model_registry(ai_dir: Path) -> list[tuple[str, Path]]:
    return [
        ("Linear Regression", ai_dir / "linear_results.csv"),
        ("Random Forest", ai_dir / "random_forest_results.csv"),
        ("SVR (RBF)", ai_dir / "svr_results.csv"),
        ("RVM", ai_dir / "rvm_results.csv"),
        ("XGBoost", ai_dir / "xgboost_results.csv"),
        ("Gradient Boosting", ai_dir / "gradient_boosting_results.csv"),
        ("AdaBoost", ai_dir / "adaboost_results.csv"),
    ]


def load_split_row(csv_path: Path, split: Split) -> pd.Series:
    df = pd.read_csv(csv_path)
    mask = (df["train"] == split.train) & (df["val"] == split.val) & (df["test"] == split.test)
    sub = df.loc[mask]
    if len(sub) != 1:
        raise ValueError(f"Expected 1 row for split {split} in {csv_path}, got {len(sub)}")
    return sub.iloc[0]


def build_metrics_table(models: list[tuple[str, Path]], split: Split) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for model_name, path in models:
        row = load_split_row(path, split)
        metric_cols = [c for c in row.index if c.startswith(("train_", "val_", "test_"))]
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


def save_rankings_md(rankings: pd.DataFrame, out_path: Path, split: Split) -> None:
    lines: list[str] = []
    lines.append(f"# Rankings — split {split.train:.2f}/{split.val:.2f}/{split.test:.2f}")
    lines.append("")
    lines.append(
        "Each metric is ranked separately. For `*_r` and `*_r2`: higher = better. "
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


def main() -> int:
    p = argparse.ArgumentParser(description="Ranking + diagrams for split 0.70/0.10/0.20 (AI5)")
    p.add_argument(
        "--split",
        type=str,
        default="0.7,0.1,0.2",
        help="train,val,test (default: 0.7,0.1,0.2)",
    )
    p.add_argument(
        "--ai-dir",
        type=Path,
        default=Path("Internship_Research/AI5"),
        help="Folder containing *_results.csv",
    )
    p.add_argument(
        "--out-rankings-csv",
        type=Path,
        default=Path("Internship_Research/AI5/rankings_70_10_20.csv"),
    )
    p.add_argument(
        "--out-rankings-md",
        type=Path,
        default=Path("Internship_Research/AI5/rankings_70_10_20.md"),
    )
    p.add_argument(
        "--out-dir-images",
        type=Path,
        default=Path("Internship_Research/AI5/images"),
    )

    args = p.parse_args()

    split = parse_split(args.split)
    models = make_model_registry(args.ai_dir)

    metrics_table = build_metrics_table(models, split)
    rankings = build_rankings(metrics_table)

    args.out_rankings_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_rankings_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_dir_images.mkdir(parents=True, exist_ok=True)

    rankings.to_csv(args.out_rankings_csv, index=False)
    save_rankings_md(rankings, args.out_rankings_md, split)

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
    fig.suptitle(f"Model ranking — split {split.train:.2f}/{split.val:.2f}/{split.test:.2f} (TRAIN)")
    out_train = args.out_dir_images / "ranking_70_10_20_train.png"
    fig.savefig(out_train, dpi=200)
    plt.close(fig)

    key = pd.DataFrame(
        {
            "test_r2": metrics_table["test_r2"],
            "test_rmse": metrics_table["test_rmse"],
            "test_mae": metrics_table["test_mae"],
        }
    )

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 5), constrained_layout=True)
    plot_barh(axes[0], key["test_r2"], "TEST R² (higher is better)", better="higher")
    plot_barh(axes[1], key["test_rmse"], "TEST RMSE (lower is better)", better="lower")
    plot_barh(axes[2], key["test_mae"], "TEST MAE (lower is better)", better="lower")
    fig.suptitle(f"Model ranking — split {split.train:.2f}/{split.val:.2f}/{split.test:.2f} (TEST)")
    out_test = args.out_dir_images / "ranking_70_10_20_test.png"
    fig.savefig(out_test, dpi=200)
    plt.close(fig)

    gap = pd.DataFrame(
        {
            "rmse_gap": (metrics_table["test_rmse"] - metrics_table["train_rmse"]),
            "mae_gap": (metrics_table["test_mae"] - metrics_table["train_mae"]),
        }
    )

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
    plot_barh(axes[0], gap["rmse_gap"], "RMSE gap (TEST - TRAIN) (lower is better)", better="lower")
    plot_barh(axes[1], gap["mae_gap"], "MAE gap (TEST - TRAIN) (lower is better)", better="lower")
    fig.suptitle(
        f"Overfitting view (TRAIN→TEST gap) — split {split.train:.2f}/{split.val:.2f}/{split.test:.2f}"
    )
    out_gap = args.out_dir_images / "ranking_70_10_20_gap.png"
    fig.savefig(out_gap, dpi=200)
    plt.close(fig)

    print("\nSaved rankings:")
    print("-", args.out_rankings_csv)
    print("-", args.out_rankings_md)
    print("Saved figures:")
    print("-", out_train)
    print("-", out_test)
    print("-", out_gap)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
