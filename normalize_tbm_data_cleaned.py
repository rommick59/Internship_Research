"""Normalize TBM_data_cleaned.csv for Machine Learning.

This script is tailored to the repository's `TBM_data_cleaned.csv` format:
- separator: comma (`,`) ;
- many numeric values are stored as strings with comma decimals (e.g. "1,3").

It:
1) loads the CSV as strings;
2) cleans column names (removes newlines, extra spaces);
3) converts all columns to float (`,` -> `.`);
4) applies imputation (median) + scaling (Min-Max [0,1]);
5) writes an ML-ready CSV and (optionally) saves the fitted preprocessor.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def _clean_column_name(name: str) -> str:
    cleaned = name.replace("\n", " ").replace("\r", " ")
    cleaned = cleaned.strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace("\ufffd", "")
    return cleaned


def _to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace("\u00a0", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    s = s.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA})
    return pd.to_numeric(s, errors="coerce")


def load_numeric_dataframe(csv_path: Path, *, strict: bool = True) -> pd.DataFrame:
    raw = pd.read_csv(csv_path, sep=",", quotechar='"', dtype=str, engine="python")
    raw.columns = [_clean_column_name(c) for c in raw.columns]

    numeric = raw.apply(_to_float_series)

    nan_ratio = numeric.isna().mean()
    if (nan_ratio > 0).any():
        bad_cols = nan_ratio[nan_ratio > 0].sort_values(ascending=False)
        preview = ", ".join([f"{c}={bad_cols[c]:.1%}" for c in bad_cols.index[:5]])
        if strict:
            raise ValueError(
                "Some columns could not be converted to numeric (NaN after conversion). "
                f"Top: {preview}."
            )
        print(
            "WARNING: Some values could not be converted to numeric and will be imputed. "
            f"Top NaN ratios: {preview}."
        )

    return numeric


def normalize_to_ml_ready(
    *,
    input_csv: Path,
    output_csv: Path,
    save_preprocessor: Path | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Normalize an input CSV to an ML-ready scaled CSV.

    - Converts comma-decimal strings to floats
    - Imputes missing values with the median
    - Scales all columns to [0,1] using MinMaxScaler
    - Writes the transformed dataframe to `output_csv`
    """

    if not input_csv.exists():
        raise FileNotFoundError(f"File not found: {input_csv}")

    df = load_numeric_dataframe(input_csv, strict=strict)

    preprocessor = build_preprocessor("minmax")
    transformed = preprocessor.fit_transform(df)

    out_df = pd.DataFrame(transformed, columns=df.columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False, float_format="%.6f")

    if save_preprocessor is not None:
        save_preprocessor.parent.mkdir(parents=True, exist_ok=True)
        dump(
            {
                "preprocessor": preprocessor,
                "feature_names": list(df.columns),
                "source": str(input_csv),
            },
            save_preprocessor,
        )

    return out_df


def build_preprocessor(_: str | None = None) -> Pipeline:
    # Always use Min-Max scaling to map features to [0, 1].
    scaler = MinMaxScaler()
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
        choices=["minmax"],
        default="minmax",
        help="Scaling is fixed to minmax ([0,1]); kept for backward compatibility",
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

    out_df = normalize_to_ml_ready(
        input_csv=args.input,
        output_csv=args.output,
        save_preprocessor=args.save_preprocessor,
        strict=True,
    )

    print(f"OK: {args.input} -> {args.output} | shape={out_df.shape} | scaler={args.scaler}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
