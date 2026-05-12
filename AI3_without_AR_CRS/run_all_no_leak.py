"""Run all models on a leakage-reduced dataset (AI3).

This script:
1) Builds a "no leak" ML-ready CSV by dropping selected columns (default: AR + CRS)
2) Runs all model training scripts (same as AI) using that dataset
3) Generates rankings + diagrams for split 0.70/0.10/0.20

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI3/run_all_no_leak.py

Optional:
  --drop-cols "AR (mm/min),CRS (RPM)"              # default
  --drop-cols "AR (mm/min),CRS (RPM),SE(kW� h/m3),FPI,TPI"  # stricter
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_DIR = REPO_ROOT / "Internship_Research" / "AI"
AI3_DIR = REPO_ROOT / "Internship_Research" / "AI3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run all AI models on a no-leak dataset (AI3)")
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
        default="AR (mm/min),CRS (RPM)",
        help="Comma-separated columns to drop from features",
    )
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    no_leak_csv = Path("Internship_Research/AI3/TBM_data_cleaned_ml_ready_no_leak.csv")

    # 1) Build the AI3 dataset
    run(
        [
            sys.executable,
            str(AI3_DIR / "build_ml_ready_no_leak.py"),
            "--in-csv",
            str(args.base_ml_ready),
            "--out-csv",
            str(no_leak_csv),
            "--drop-cols",
            args.drop_cols,
        ]
    )

    # 2) Train/evaluate models (write results to AI3)
    jobs: list[tuple[str, Path, Path]] = [
        ("linear", AI_DIR / "linear_ml.py", AI3_DIR / "linear_results.csv"),
        ("random_forest", AI_DIR / "random_forest_ml.py", AI3_DIR / "random_forest_results.csv"),
        ("svr", AI_DIR / "svr_ml.py", AI3_DIR / "svr_results.csv"),
        ("rvm", AI_DIR / "rvm_ml.py", AI3_DIR / "rvm_results.csv"),
        ("xgboost", AI_DIR / "xgboost_ml.py", AI3_DIR / "xgboost_results.csv"),
        ("gradient_boosting", AI_DIR / "gradient_boosting_ml.py", AI3_DIR / "gradient_boosting_results.csv"),
        ("adaboost", AI_DIR / "adaboost_ml.py", AI3_DIR / "adaboost_results.csv"),
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
    run([sys.executable, str(AI3_DIR / "ranking_diagrams_70_10_20.py")])

    print("\nDone. Outputs:")
    print("-", AI3_DIR)
    print("-", AI3_DIR / "images")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
