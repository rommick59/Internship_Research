"""AI6 — Split then normalize (TRAIN-only fit), without normalizing the whole file at once.

Schema (as requested):
1) Take 70% TRAIN
2) Fit normalizer on TRAIN only, normalize TRAIN
3) Normalize 20% TEST using the TRAIN-fitted normalizer
4) Normalize 10% VALIDATION using the TRAIN-fitted normalizer

This avoids leakage from test/val into normalization parameters.

Input (raw cleaned):
- Internship_Research/TBM_data_cleaned.csv

Selected columns (same as AI5 selected features):
- CRS (RPM)
- F/A(MF)
- T/D3(MT)
- UEP (MPa)
- LEP (MPa)
- TPI
- PR(mm/r)  (target)

Outputs (normalized splits):
- Internship_Research/AI6/split_train_norm.csv
- Internship_Research/AI6/split_test_norm.csv
- Internship_Research/AI6/split_val_norm.csv
- Internship_Research/AI6/preprocessor_fit_on_train.joblib

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI6/build_splits_and_normalize_train_only.py
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from joblib import dump

try:
    from Internship_Research.normalize_tbm_data_cleaned import build_preprocessor, load_numeric_dataframe
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from Internship_Research.normalize_tbm_data_cleaned import build_preprocessor, load_numeric_dataframe

from sklearn.model_selection import train_test_split


SELECTED_COLUMNS = [
    "CRS (RPM)",
    "F/A(MF)",
    "T/D3(MT)",
    "UEP (MPa)",
    "LEP (MPa)",
    "TPI",
    "PR(mm/r)",
]
TARGET = "PR(mm/r)"


@dataclass(frozen=True)
class Split:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not (abs(total - 1.0) <= 1e-9):
            raise ValueError(f"Split must sum to 1.0, got {total:.6f}")


def parse_split(value: str) -> Split:
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise ValueError('Split must be formatted like "0.7,0.1,0.2"')
    return Split(train=float(parts[0]), val=float(parts[1]), test=float(parts[2]))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI6: split then normalize (fit on train only)")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned.csv"),
        help="Raw cleaned input CSV (comma decimals allowed)",
    )
    p.add_argument(
        "--split",
        type=str,
        default="0.7,0.1,0.2",
        help="train,val,test (default: 0.7,0.1,0.2)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--out-train",
        type=Path,
        default=Path("Internship_Research/AI6/split_train_norm.csv"),
    )
    p.add_argument(
        "--out-test",
        type=Path,
        default=Path("Internship_Research/AI6/split_test_norm.csv"),
    )
    p.add_argument(
        "--out-val",
        type=Path,
        default=Path("Internship_Research/AI6/split_val_norm.csv"),
    )
    p.add_argument(
        "--save-preprocessor",
        type=Path,
        default=Path("Internship_Research/AI6/preprocessor_fit_on_train.joblib"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    split = parse_split(args.split)

    df = load_numeric_dataframe(args.input_csv, strict=False)

    missing = [c for c in SELECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {args.input_csv}: {missing}. Available: {list(df.columns)}"
        )

    selected = df[SELECTED_COLUMNS].copy()

    # Split order to match requested proportions: 70% train, then (20% test + 10% val)
    train_df, temp_df = train_test_split(
        selected,
        test_size=(split.val + split.test),
        random_state=args.random_state,
        shuffle=True,
    )

    # From remaining 30%: take 20% test and 10% val.
    # test fraction of temp = 0.2 / (0.1+0.2) = 2/3
    test_frac_of_temp = split.test / (split.val + split.test)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=test_frac_of_temp,
        random_state=args.random_state,
        shuffle=True,
    )

    preprocessor = build_preprocessor("minmax")
    train_norm = pd.DataFrame(preprocessor.fit_transform(train_df), columns=train_df.columns)
    test_norm = pd.DataFrame(preprocessor.transform(test_df), columns=test_df.columns)
    val_norm = pd.DataFrame(preprocessor.transform(val_df), columns=val_df.columns)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_test.parent.mkdir(parents=True, exist_ok=True)
    args.out_val.parent.mkdir(parents=True, exist_ok=True)

    train_norm.to_csv(args.out_train, index=False, float_format="%.6f")
    test_norm.to_csv(args.out_test, index=False, float_format="%.6f")
    val_norm.to_csv(args.out_val, index=False, float_format="%.6f")

    args.save_preprocessor.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "preprocessor": preprocessor,
            "fit_on": "train_only",
            "selected_columns": SELECTED_COLUMNS,
            "target": TARGET,
            "split": {"train": split.train, "val": split.val, "test": split.test},
            "random_state": int(args.random_state),
            "source": str(args.input_csv),
        },
        args.save_preprocessor,
    )

    print("Saved normalized splits:")
    print("-", args.out_train, "shape=", train_norm.shape)
    print("-", args.out_test, "shape=", test_norm.shape)
    print("-", args.out_val, "shape=", val_norm.shape)
    print("Saved preprocessor:")
    print("-", args.save_preprocessor)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
