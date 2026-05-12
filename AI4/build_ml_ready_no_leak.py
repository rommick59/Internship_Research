"""Build an ML-ready dataset with selected columns removed (AI4).

AI4 goal: remove the strongly leakage-prone / redundant indices SE/FPI/TPI,
while keeping AR and CRS.

Default dropped columns:
- SE (any column starting with "SE(")
- FPI
- TPI

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI4/build_ml_ready_no_leak.py

Optional:
  --in-csv  Internship_Research/TBM_data_cleaned_ml_ready.csv
  --out-csv Internship_Research/AI4/TBM_data_cleaned_ml_ready_no_leak.csv
    --drop-cols "SE,FPI,TPI"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create an ML-ready CSV with selected columns removed (AI4)")
    p.add_argument(
        "--in-csv",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Input normalized CSV",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("Internship_Research/AI4/TBM_data_cleaned_ml_ready_no_leak.csv"),
        help="Output CSV path",
    )
    p.add_argument(
        "--drop-cols",
        type=str,
        default="SE,FPI,TPI",
        help="Comma-separated column names to drop",
    )
    return p.parse_args()


def resolve_drop_columns(df: pd.DataFrame, requested: list[str]) -> tuple[list[str], list[str]]:
    """Return (to_drop, missing) columns.

    Supports a robust token `SE` meaning: drop any column starting with `SE(`,
    which avoids encoding differences in the unit string.
    """

    cols = list(df.columns)
    lower_cols = {c.lower(): c for c in cols}

    to_drop: list[str] = []
    missing: list[str] = []

    for spec in requested:
        if not spec:
            continue

        key = spec.strip()
        key_lower = key.lower()

        # Robust SE handling: allow `SE` or any `SE(...)` spec to drop by prefix.
        if key_lower == "se" or key_lower.startswith("se("):
            se_cols = [c for c in cols if c.lower().startswith("se(")]
            if se_cols:
                to_drop.extend(se_cols)
            else:
                missing.append(key)
            continue

        # Exact match, case-insensitive.
        if key in cols:
            to_drop.append(key)
        elif key_lower in lower_cols:
            to_drop.append(lower_cols[key_lower])
        else:
            missing.append(key)

    # Deduplicate while preserving order.
    seen: set[str] = set()
    unique_to_drop: list[str] = []
    for c in to_drop:
        if c not in seen:
            unique_to_drop.append(c)
            seen.add(c)

    return unique_to_drop, missing


def main() -> int:
    args = parse_args()

    df = pd.read_csv(args.in_csv)

    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()]
    to_drop, missing = resolve_drop_columns(df, drop_cols)

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
