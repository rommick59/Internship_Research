"""Distance correlation heatmap for AI8.

Adapted from AI5 version to use the AI8 normalised split by default.

Input (default):
- Internship_Research/AI8/split_train_norm_80_20.csv

Output:
- Internship_Research/AI8/images/heatmap_distance_correlation_ai8.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Distance correlation heatmap (AI8)")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("Internship_Research/AI8/split_train_norm_80_20.csv"),
        help="Input CSV",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Internship_Research/AI8/images/heatmap_distance_correlation_ai8.png"),
        help="Output PNG",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Figure DPI",
    )
    return p.parse_args()


def _distance_matrix_1d(x: np.ndarray) -> np.ndarray:
    x = x.reshape(-1)
    return np.abs(x[:, None] - x[None, :])


def _double_center(a: np.ndarray) -> np.ndarray:
    row_mean = a.mean(axis=1, keepdims=True)
    col_mean = a.mean(axis=0, keepdims=True)
    grand_mean = a.mean()
    return a - row_mean - col_mean + grand_mean


def distance_correlation_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0

    a = _double_center(_distance_matrix_1d(x))
    b = _double_center(_distance_matrix_1d(y))

    dcov2 = (a * b).mean()
    dvarx2 = (a * a).mean()
    dvary2 = (b * b).mean()

    dcov2 = max(dcov2, 0.0)
    dvarx2 = max(dvarx2, 0.0)
    dvary2 = max(dvary2, 0.0)

    denom = np.sqrt(dvarx2 * dvary2)
    if denom == 0:
        return 0.0

    dcor = np.sqrt(dcov2) / np.sqrt(denom)
    return float(min(max(dcor, 0.0), 1.0))


def compute_dcor_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    arr = df.to_numpy(dtype=float)

    n = len(cols)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        out[i, i] = 1.0
        for j in range(i + 1, n):
            v = distance_correlation_1d(arr[:, i], arr[:, j])
            out[i, j] = v
            out[j, i] = v

    return pd.DataFrame(out, index=cols, columns=cols)


def plot_heatmap(mat: pd.DataFrame, out_path: Path, dpi: int) -> None:
    fig_w = max(8, 0.75 * len(mat.columns))
    fig_h = max(6, 0.75 * len(mat.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)

    im = ax.imshow(mat.to_numpy(), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Distance Correlation Heatmap (AI8)")

    ax.set_xticks(range(len(mat.columns)))
    ax.set_yticks(range(len(mat.index)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right")
    ax.set_yticklabels(mat.index)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat.iat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("dCor")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.csv)

    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=0, how="any")

    mat = compute_dcor_matrix(num)
    plot_heatmap(mat, args.out, dpi=args.dpi)

    print(f"Saved heatmap: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
