"""AI8 — Split 80/20 then normalize (fit on TRAIN only).

Requested schema:
- 80% TRAIN: fit normalizer on TRAIN only, then normalize TRAIN
- 20% TEST: normalize TEST using the TRAIN-fitted normalizer

Input (raw cleaned):
- Internship_Research/TBM_data_cleaned.csv

Selected columns (same as AI6 / AI5-selected):
- CRS (RPM)
- F/A(MF)
- T/D3(MT)
- UEP (MPa)
- LEP (MPa)
- TPI
- PR(mm/r)  (target)

Outputs:
- Internship_Research/AI8/split_train_norm_80_20.csv
- Internship_Research/AI8/split_test_norm_80_20.csv
- Internship_Research/AI8/preprocessor_fit_on_train_80_20.joblib
- Internship_Research/AI8/split_indices_80_20.joblib

Run (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI8/build_split_and_normalize_train_only_80_20.py
"""

from __future__ import annotations

import argparse
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI8: split 80/20 then normalize (fit on train only)")
    p.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned.csv"),
        help="Raw cleaned input CSV (comma decimals allowed)",
    )
    p.add_argument("--train", type=float, default=0.8)
    p.add_argument("--test", type=float, default=0.2)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument(
        "--out-train",
        type=Path,
        default=Path("Internship_Research/AI8/split_train_norm_80_20.csv"),
    )
    p.add_argument(
        "--out-test",
        type=Path,
        default=Path("Internship_Research/AI8/split_test_norm_80_20.csv"),
    )
    p.add_argument(
        "--save-preprocessor",
        type=Path,
        default=Path("Internship_Research/AI8/preprocessor_fit_on_train_80_20.joblib"),
    )
    p.add_argument(
        "--save-indices",
        type=Path,
        default=Path("Internship_Research/AI8/split_indices_80_20.joblib"),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if abs((args.train + args.test) - 1.0) > 1e-9:
        raise ValueError("train+test must sum to 1.0")

    df = load_numeric_dataframe(args.input_csv, strict=False)

    missing = [c for c in SELECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {args.input_csv}: {missing}. Available: {list(df.columns)}"
        )

    selected = df[SELECTED_COLUMNS].copy()

    train_df, test_df = train_test_split(
        selected,
        test_size=args.test,
        random_state=args.random_state,
        shuffle=True,
    )

    preprocessor = build_preprocessor("minmax")
    train_norm = pd.DataFrame(preprocessor.fit_transform(train_df), columns=train_df.columns)
    test_norm = pd.DataFrame(preprocessor.transform(test_df), columns=test_df.columns)

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_test.parent.mkdir(parents=True, exist_ok=True)

    train_norm.to_csv(args.out_train, index=False, float_format="%.6f")
    test_norm.to_csv(args.out_test, index=False, float_format="%.6f")

    args.save_preprocessor.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "preprocessor": preprocessor,
            "fit_on": "train_only",
            "selected_columns": SELECTED_COLUMNS,
            "target": TARGET,
            "split": {"train": float(args.train), "test": float(args.test)},
            "random_state": int(args.random_state),
            "source": str(args.input_csv),
        },
        args.save_preprocessor,
    )

    # Save indices to reproduce CV on the same TRAIN subset
    args.save_indices.parent.mkdir(parents=True, exist_ok=True)
    dump(
        {
            "train_index": train_df.index.to_numpy(),
            "test_index": test_df.index.to_numpy(),
            "selected_columns": SELECTED_COLUMNS,
            "target": TARGET,
            "split": {"train": float(args.train), "test": float(args.test)},
            "random_state": int(args.random_state),
            "source": str(args.input_csv),
        },
        args.save_indices,
    )

    print("Saved normalized splits:")
    print("-", args.out_train, "shape=", train_norm.shape)
    print("-", args.out_test, "shape=", test_norm.shape)
    print("Saved preprocessor:")
    print("-", args.save_preprocessor)
    print("Saved indices:")
    print("-", args.save_indices)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
