"""Colab-ready normalization script for a TBM CSV file.

Default paths are set for Google Colab:
- input:  /content/PPV dataset.xlsx
- output: /content/PPV dataset_ml_ready.xlsx
- preprocessor: /content/tbm_preprocessor.joblib

The script:
1) loads the CSV as strings;
2) cleans column names;
3) converts numeric-looking values to floats;
4) imputes missing values with the median;
5) scales features to [0, 1];
6) saves the normalized CSV and the fitted preprocessing pipeline.
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


DEFAULT_INPUT = Path("/content/PPV dataset.xlsx")
DEFAULT_OUTPUT = Path("/content/PPV dataset_ml_ready.xlsx")
DEFAULT_PREPROCESSOR = Path("/content/tbm_preprocessor.joblib")
DEFAULT_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")


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


def _read_csv_with_fallback(csv_path: Path, *, encodings: tuple[str, ...]) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(
                csv_path,
                sep=",",
                quotechar='"',
                dtype=str,
                engine="python",
                encoding=encoding,
                skip_blank_lines=True,
                on_bad_lines="skip",
                keep_default_na=True,
                na_values=["", " ", "NA", "N/A", "null", "None"],
            )
        except UnicodeDecodeError as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode CSV file")


def _read_tabular_file(csv_path: Path, *, encodings: tuple[str, ...]) -> pd.DataFrame:
    suffix = csv_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(csv_path, dtype=str)
    return _read_csv_with_fallback(csv_path, encodings=encodings)


def load_numeric_dataframe(
    csv_path: Path,
    *,
    strict: bool = True,
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS,
) -> pd.DataFrame:
    raw = _read_tabular_file(csv_path, encodings=encodings)
    raw.columns = [_clean_column_name(col) for col in raw.columns]

    numeric = raw.apply(_to_float_series)

    fully_non_numeric = numeric.columns[numeric.isna().all()].tolist()
    if fully_non_numeric:
        preview = ", ".join(fully_non_numeric[:5])
        print(f"WARNING: Dropping non-numeric columns: {preview}")
        numeric = numeric.drop(columns=fully_non_numeric)

    if numeric.shape[1] == 0:
        raise ValueError(
            "No numeric columns could be loaded from the input file. "
            "Check that the file contains numeric data and that the correct sheet/path was used."
        )

    nan_ratio = numeric.isna().mean()
    if (nan_ratio > 0).any():
        bad_cols = nan_ratio[nan_ratio > 0].sort_values(ascending=False)
        preview = ", ".join([f"{col}={bad_cols[col]:.1%}" for col in bad_cols.index[:5]])
        print(
            "WARNING: Some values could not be converted to numeric and will be imputed. "
            f"Top NaN ratios: {preview}."
        )

    return numeric


def build_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", MinMaxScaler()),
        ]
    )


def normalize_to_ml_ready(
    *,
    input_csv: Path,
    output_csv: Path,
    save_preprocessor: Path | None = None,
    strict: bool = True,
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"File not found: {input_csv}")

    df = load_numeric_dataframe(input_csv, strict=strict, encodings=encodings)
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(df)

    out_df = pd.DataFrame(transformed, columns=df.columns)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.suffix.lower() in {".xlsx", ".xls"}:
        out_df.to_excel(output_csv, index=False)
    else:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize a TBM CSV file for Colab")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to input file (.csv, .xlsx, .xls)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output file (.csv, .xlsx, .xls)")
    parser.add_argument(
        "--save-preprocessor",
        type=Path,
        default=DEFAULT_PREPROCESSOR,
        help="Path to save the fitted sklearn pipeline (.joblib)",
    )
    parser.add_argument(
        "--no-save-preprocessor",
        action="store_true",
        help="Disable saving the fitted sklearn pipeline",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Allow non-numeric conversions and impute them instead of failing",
    )
    parser.add_argument(
        "--encoding",
        action="append",
        default=None,
        help="CSV encoding to try. Repeat the flag to provide multiple encodings.",
    )
    args, _unknown = parser.parse_known_args()
    return args


def main() -> int:
    args = parse_args()
    preprocessor_path = None if args.no_save_preprocessor else args.save_preprocessor
    encodings = tuple(args.encoding) if args.encoding else DEFAULT_ENCODINGS

    out_df = normalize_to_ml_ready(
        input_csv=args.input,
        output_csv=args.output,
        save_preprocessor=preprocessor_path,
        strict=not args.lenient,
        encodings=encodings,
    )

    print(f"OK: {args.input} -> {args.output} | shape={out_df.shape}")
    if preprocessor_path is not None:
        print(f"Preprocessor saved to: {preprocessor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())