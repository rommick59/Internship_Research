"""Run all models on an AI4 dataset without SE/FPI/TPI (and also AR/CRS).

AI4 purpose: make the modeling harder/cleaner by removing indices that are
highly correlated with PR and likely derived/duplicated.

Default dropped columns:
- SE (any column starting with "SE(")
- FPI
- TPI

This script:
1) Builds Internship_Research/AI4/TBM_data_cleaned_ml_ready_no_leak.csv
2) Runs all model scripts from Internship_Research/AI but writes results into AI4
3) Generates rankings + diagrams for split 0.70/0.10/0.20

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI4/run_all_no_leak.py

Optional:
    --drop-cols "SE,FPI,TPI"  # default
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_DIR = REPO_ROOT / "Internship_Research" / "AI"
AI4_DIR = REPO_ROOT / "Internship_Research" / "AI4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all AI models on the AI4 no-leak dataset")
    p.add_argument(
        "--base-ml-ready",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Input ML-ready CSV",
    )
    p.add_argument(
        "--target",
        type=str,
        default="PR(mm/r)",
        help="Target column",
    )
    p.add_argument(
        "--splits",
        type=str,
        default="0.7,0.15,0.15;0.6,0.2,0.2;0.7,0.1,0.2",
        help="Split configs (train,val,test;...)",
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )
    p.add_argument(
        "--drop-cols",
        type=str,
        default="SE,FPI,TPI",
        help="Comma-separated columns to drop from features",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    no_leak_csv = Path("Internship_Research/AI4/TBM_data_cleaned_ml_ready_no_leak.csv")

    # 1) Build the AI4 dataset
    run(
        [
            sys.executable,
            str(AI4_DIR / "build_ml_ready_no_leak.py"),
            "--in-csv",
            str(args.base_ml_ready),
            "--out-csv",
            str(no_leak_csv),
            "--drop-cols",
            args.drop_cols,
        ]
    )

    # 2) Train/evaluate models (write results to AI4)
    jobs: list[tuple[str, Path, Path]] = [
        ("linear", AI_DIR / "linear_ml.py", AI4_DIR / "linear_results.csv"),
        ("random_forest", AI_DIR / "random_forest_ml.py", AI4_DIR / "random_forest_results.csv"),
        ("svr", AI_DIR / "svr_ml.py", AI4_DIR / "svr_results.csv"),
        ("rvm", AI_DIR / "rvm_ml.py", AI4_DIR / "rvm_results.csv"),
        ("xgboost", AI_DIR / "xgboost_ml.py", AI4_DIR / "xgboost_results.csv"),
        ("gradient_boosting", AI_DIR / "gradient_boosting_ml.py", AI4_DIR / "gradient_boosting_results.csv"),
        ("adaboost", AI_DIR / "adaboost_ml.py", AI4_DIR / "adaboost_results.csv"),
    ]

    for name, script_path, out_csv in jobs:
        print(f"\n=== {name} ===")
        run(
            [
                sys.executable,
                str(script_path),
                "--data",
                str(no_leak_csv),
                "--target",
                args.target,
                "--splits",
                args.splits,
                "--random-state",
                str(args.random_state),
                "--out-results",
                str(out_csv),
            ]
        )

    # 3) Rankings/diagrams (0.70/0.10/0.20)
    run([sys.executable, str(AI4_DIR / "ranking_diagrams_70_10_20.py")])

    print("\nDone. Outputs:")
    print("-", AI4_DIR)
    print("-", AI4_DIR / "images")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
