"""Normalize TBM_data_cleaned.csv for Machine Learning.

This script is tailored to the repository's `TBM_data_cleaned.csv` format:
- separator: comma (`,`) ;
- many numeric values are stored as strings with comma decimals (e.g. "1,3").

It:
1) loads the CSV as strings;
2) cleans column names (removes newlines, extra spaces);
3) converts all columns to float (`,` -> `.`);
4) applies imputation (median) + scaling (Min-Max [0,1] by default);
5) writes an ML-ready CSV and (optionally) saves the fitted preprocessor.

Usage (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/normalize_tbm_data_cleaned.py 
        --input Internship_Research/TBM_data_cleaned.csv 
        --output Internship_Research/TBM_data_cleaned_ml_ready.csv 
        --scaler minmax 
        --save-preprocessor Internship_Research/tbm_preprocessor.joblib
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def _clean_column_name(name: str) -> str:
    cleaned = name.replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Some characters may be mis-decoded; keep it simple.
    cleaned = cleaned.replace("\ufffd", "")  # replacement character (�), often due to encoding issues
    return cleaned


def _to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace("\u00a0", "", regex=False)  # non-breaking spaces
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def load_numeric_dataframe(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, sep=",", quotechar='"', dtype=str, engine="python")
    raw.columns = [_clean_column_name(c) for c in raw.columns]

    numeric = raw.apply(_to_float_series)

    nan_ratio = numeric.isna().mean()
    if (nan_ratio > 0).any():
        bad_cols = nan_ratio[nan_ratio > 0].sort_values(ascending=False)
        preview = ", ".join([f"{c}={bad_cols[c]:.1%}" for c in bad_cols.index[:5]])
        raise ValueError(
            "Some columns could not be converted to numeric (NaN after conversion). "
            f"Top: {preview}.\n"
            "Check the CSV format (commas, quotes, encoding) or adapt the conversion logic."
        )

    return numeric


def build_preprocessor(scaler_name: str) -> Pipeline:
    scaler_name = scaler_name.lower().strip()

    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    elif scaler_name == "robust":
        scaler = RobustScaler()
    elif scaler_name == "none":
        scaler = "passthrough"
    else:
        raise ValueError("--scaler must be one of {standard, minmax, robust, none}.")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalize TBM_data_cleaned.csv for ML")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned.csv"),
        help="Path to input CSV",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Path to output (scaled) CSV",
    )
    p.add_argument(
        "--scaler",
        choices=["standard", "minmax", "robust", "none"],
        default="minmax",
        help="Scaling type (minmax recommended for [0,1] normalization)",
    )
    p.add_argument(
        "--save-preprocessor",
        type=Path,
        default=None,
        help="Path to save the fitted sklearn Pipeline (.joblib)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"File not found: {args.input}")

    df = load_numeric_dataframe(args.input)

    preprocessor = build_preprocessor(args.scaler)
    transformed = preprocessor.fit_transform(df)

    out_df = pd.DataFrame(transformed, columns=df.columns)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, float_format="%.6f")

    if args.save_preprocessor is not None:
        args.save_preprocessor.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "preprocessor": preprocessor,
                "feature_names": list(df.columns),
                "source": str(args.input),
            },
            args.save_preprocessor,
        )

    print(f"OK: {args.input} -> {args.output} | shape={out_df.shape} | scaler={args.scaler}")
    if args.save_preprocessor is not None:
        print(f"Preprocessor saved: {args.save_preprocessor}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
