"""Build an ML-ready dataset with leakage columns removed.

We already have a normalized ML-ready dataset in this repo:
- Internship_Research/TBM_data_cleaned_ml_ready.csv

This script writes a copy with selected columns removed (default: the direct
leakage columns used to compute PR):
- AR (mm/min)
- CRS (RPM)

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI3/build_ml_ready_no_leak.py

Optional:
  --in-csv  Internship_Research/TBM_data_cleaned_ml_ready.csv
  --out-csv Internship_Research/AI3/TBM_data_cleaned_ml_ready_no_leak.csv
  --drop-cols "AR (mm/min),CRS (RPM)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an ML-ready CSV with selected columns removed")
    p.add_argument(
        "--in-csv",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Input normalized CSV",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Internship_Research/AI3/TBM_data_cleaned_ml_ready_no_leak.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--drop-cols",
        type=str,
        default="AR (mm/min),CRS (RPM)",
        help="Comma-separated column names to drop",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    to_drop = [c for c in drop_cols if c in df.columns]
    missing = [c for c in drop_cols if c not in df.columns]

    if to_drop:
        df = df.drop(columns=to_drop)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print(f"Saved: {args.out_csv}")
    if to_drop:
        print("Dropped:", ", ".join(to_drop))
    if missing:
        print("Not found (ignored):", ", ".join(missing))
    print("Columns:", df.columns.tolist())
    print("Shape:", df.shape)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
