"""AI6 — Draw the requested normalization+IA schema (70% train, 20% test, 10% val).

Output:
- Internship_Research/AI6/images/schema_normalization_train_test_val.png

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI6/normalization_schema_70_20_10.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI6 schema diagram")
    p.add_argument(
        "--out",
        type=Path,
        default=Path("Internship_Research/AI6/images/schema_normalization_train_test_val.png"),
        help="Output PNG",
    )
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def box(ax, xy, text, w=2.9, h=0.9):
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

    fig, ax = plt.subplots(figsize=(13.5, 5.2), constrained_layout=True)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")

    b0 = box(ax, (0.4, 2.15), "Dataset brut\n(TBM_data_cleaned.csv)")
    b1 = box(ax, (3.9, 3.5), "Split 70%\nTRAIN")
    b2 = box(ax, (3.9, 1.8), "Split 20%\nTEST")
    b3 = box(ax, (3.9, 0.1), "Split 10%\nVALIDATION")

    b4 = box(ax, (7.4, 3.5), "Fit normalisation\nSUR TRAIN")
    b5 = box(ax, (10.4, 3.5), "Normaliser TRAIN\n(minmax + imputation)")
    b6 = box(ax, (10.4, 1.8), "Normaliser TEST\n(avec scaler TRAIN)")
    b7 = box(ax, (10.4, 0.1), "Normaliser VAL\n(avec scaler TRAIN)")

    # Arrows from dataset to splits
    arrow(ax, (b0[0] + b0[2], b0[1] + b0[3] / 2), (b1[0], b1[1] + b1[3] / 2))
    arrow(ax, (b0[0] + b0[2], b0[1] + b0[3] / 2), (b2[0], b2[1] + b2[3] / 2))
    arrow(ax, (b0[0] + b0[2], b0[1] + b0[3] / 2), (b3[0], b3[1] + b3[3] / 2))

    # Train path: split -> fit scaler -> normalize train -> train model
    arrow(ax, (b1[0] + b1[2], b1[1] + b1[3] / 2), (b4[0], b4[1] + b4[3] / 2))
    arrow(ax, (b4[0] + b4[2], b4[1] + b4[3] / 2), (b5[0], b5[1] + b5[3] / 2))

    b8 = box(ax, (10.4, 4.6), "Entraîner IA\nSUR TRAIN")
    arrow(ax, (b5[0] + b5[2] / 2, b5[1] + b5[3]), (b8[0] + b8[2] / 2, b8[1]))

    # Test then Val evaluation (as requested order)
    b9 = box(ax, (10.4, 2.95), "Tester IA\nSUR TEST")
    arrow(ax, (b6[0] + b6[2] / 2, b6[1] + b6[3]), (b9[0] + b9[2] / 2, b9[1]))

    b10 = box(ax, (10.4, 1.25), "Valider IA\nSUR VAL")
    arrow(ax, (b7[0] + b7[2] / 2, b7[1] + b7[3]), (b10[0] + b10[2] / 2, b10[1]))

    # Connect split test/val to their normalization boxes
    arrow(ax, (b2[0] + b2[2], b2[1] + b2[3] / 2), (b6[0], b6[1] + b6[3] / 2))
    arrow(ax, (b3[0] + b3[2], b3[1] + b3[3] / 2), (b7[0], b7[1] + b7[3] / 2))

    # Note: scaler fitted on train applied to test/val
    ax.text(
        7.55,
        2.65,
        "Même normalisation (paramètres appris sur TRAIN)\n→ appliquée à TEST puis à VAL",
        fontsize=9,
        ha="left",
        va="center",
    )

    ax.set_title("AI6 — Schéma normalisation + IA (70% TRAIN, 20% TEST, 10% VAL)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi)
    plt.close(fig)

    print("Saved schema:", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
