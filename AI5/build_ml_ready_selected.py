"""Build an ML-ready dataset with only selected IA features (AI5).

Selected feature columns (as requested):
- CRS (RPM)
- F/A(MF)
- T/D3(MT)
- UEP (MPa)
- LEP (MPa)
- TPI

Target column:
- PR(mm/r)

Input:
- Internship_Research/TBM_data_cleaned_ml_ready.csv

Output:
- Internship_Research/AI5/TBM_data_cleaned_ml_ready_selected.csv

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI5/build_ml_ready_selected.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SELECTED_FEATURES = [
    "CRS (RPM)",
    "F/A(MF)",
    "T/D3(MT)",
    "UEP (MPa)",
    "LEP (MPa)",
    "TPI",
]
TARGET = "PR(mm/r)"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an ML-ready CSV with selected features only (AI5)")
    p.add_argument(
        "--in-csv",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Input ML-ready CSV",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Internship_Research/AI5/TBM_data_cleaned_ml_ready_selected.csv"),
        help="Output CSV",
    )
    p.add_argument(
        "--target",
        type=str,
        default=TARGET,
        help="Target column name",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    required = SELECTED_FEATURES + [args.target]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {args.in_csv}: {missing}. Available: {df.columns.tolist()}")

    out = df[required].copy()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")
    print("Columns:", out.columns.tolist())
    print("Shape:", out.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
