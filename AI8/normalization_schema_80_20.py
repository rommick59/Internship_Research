"""AI8 — Draw the requested normalization+IA schema (80% train, 20% test).

Output:
- Internship_Research/AI8/images/schema_normalization_train_test_80_20.png

Run (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI8/normalization_schema_80_20.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI8 schema diagram")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Internship_Research/AI8/images/schema_normalization_train_test_80_20.png"),
        help="Output PNG",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def box(ax, xy, text, w=3.0, h=0.9):
    x, y = xy
    rect = plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8)
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)
    return (x, y, w, h)


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.6),
    )


def main() -> int:
    args = parse_args()

    fig, ax = plt.subplots(figsize=(13.0, 4.2), constrained_layout=True)
    ax.set_xlim(0, 13.0)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    b0 = box(ax, (0.4, 1.65), "Dataset brut\n(TBM_data_cleaned.csv)")
    b1 = box(ax, (3.9, 2.7), "Split 80%\nTRAIN")
    b2 = box(ax, (3.9, 0.6), "Split 20%\nTEST")

    b3 = box(ax, (7.4, 2.7), "Fit normalisation\nSUR TRAIN")
    b4 = box(ax, (10.3, 2.7), "Normaliser TRAIN\n(minmax + imputation)")
    b5 = box(ax, (10.3, 0.6), "Normaliser TEST\n(avec scaler TRAIN)")

    b6 = box(ax, (10.3, 3.75), "Entraîner IA\nSUR TRAIN")
    b7 = box(ax, (10.3, 1.65), "Tester IA\nSUR TEST")

    arrow(ax, (b0[0] + b0[2], b0[1] + b0[3] / 2), (b1[0], b1[1] + b1[3] / 2))
    arrow(ax, (b0[0] + b0[2], b0[1] + b0[3] / 2), (b2[0], b2[1] + b2[3] / 2))

    arrow(ax, (b1[0] + b1[2], b1[1] + b1[3] / 2), (b3[0], b3[1] + b3[3] / 2))
    arrow(ax, (b3[0] + b3[2], b3[1] + b3[3] / 2), (b4[0], b4[1] + b4[3] / 2))

    arrow(ax, (b2[0] + b2[2], b2[1] + b2[3] / 2), (b5[0], b5[1] + b5[3] / 2))

    arrow(ax, (b4[0] + b4[2] / 2, b4[1] + b4[3]), (b6[0] + b6[2] / 2, b6[1]))
    arrow(ax, (b5[0] + b5[2] / 2, b5[1] + b5[3]), (b7[0] + b7[2] / 2, b7[1]))

    ax.text(
        7.55,
        1.75,
        "Même normalisation (paramètres appris sur TRAIN)\n→ appliquée à TEST",
        fontsize=9,
        ha="left",
        va="center",
    )

    ax.set_title("AI8 — Schéma normalisation + IA (80% TRAIN, 20% TEST)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)

    print("Saved schema:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
