"""Normalise TBM_data_cleaned.csv pour le Machine Learning.

Ce script est adapté au format du fichier `TBM_data_cleaned.csv` du repo :
- séparateur : virgule (`,`) ;
- beaucoup de valeurs numériques sont encodées en texte avec des décimales en virgule (ex: "1,3").

Il :
1) charge le CSV en chaînes ;
2) nettoie les noms de colonnes (suppression des retours à la ligne, espaces) ;
3) convertit toutes les colonnes en float (`,` -> `.`) ;
4) applique une imputation (médiane) + un scaler (standard par défaut) ;
5) exporte un CSV "ML-ready" et (optionnel) sauvegarde le préprocesseur.

Usage (PowerShell) :
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/normalize_tbm_data_cleaned.py 
    --input Internship_Research/TBM_data_cleaned.csv 
    --output Internship_Research/TBM_data_cleaned_ml_ready.csv 
    --scaler standard 
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
    # Certains caractères peuvent être mal décodés (ex: ). On garde le plus simple.
    cleaned = cleaned.replace("\ufffd", "")  # caractère de remplacement "�"
    return cleaned


def _to_float_series(series: pd.Series) -> pd.Series:
    s = series.astype("string").str.strip()
    s = s.str.replace("\u00a0", "", regex=False)  # espaces insécables
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
            "Certaines colonnes n'ont pas pu être converties en numérique (NaN après conversion). "
            f"Top: {preview}.\n"
            "Vérifie le format du CSV (virgules, guillemets, encodage) ou adapte la conversion."
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
        raise ValueError("--scaler doit être dans {standard, minmax, robust, none}.")

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", scaler),
        ]
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normalise TBM_data_cleaned.csv pour ML")
    p.add_argument(
        "--input",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned.csv"),
        help="Chemin du CSV source",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Chemin du CSV de sortie normalisé",
    )
    p.add_argument(
        "--scaler",
        choices=["standard", "minmax", "robust", "none"],
        default="standard",
        help="Type de normalisation/standardisation",
    )
    p.add_argument(
        "--save-preprocessor",
        type=Path,
        default=None,
        help="Chemin pour sauvegarder le Pipeline sklearn (.joblib)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Fichier introuvable: {args.input}")

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
        print(f"Préprocesseur sauvegardé: {args.save_preprocessor}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
