"""Run all models on selected IA features + distance correlation heatmap (AI5).

Requested IA inputs (features):
- CRS (RPM)
- F/A(MF)
- T/D3(MT)
- UEP (MPa)
- LEP (MPa)
- TPI

This script:
1) Builds Internship_Research/AI5/TBM_data_cleaned_ml_ready_selected.csv
2) Trains/evaluates all models (same scripts as AI) and saves results into AI5
3) Builds rankings/diagrams for split 0.70/0.10/0.20
4) Generates a distance correlation heatmap (AI5/images)

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI5/run_all_selected.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AI_DIR = REPO_ROOT / "Internship_Research" / "AI"
AI5_DIR = REPO_ROOT / "Internship_Research" / "AI5"


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    selected_csv = Path("Internship_Research/AI5/TBM_data_cleaned_ml_ready_selected.csv")

    # 1) Build selected dataset
    run([sys.executable, str(AI5_DIR / "build_ml_ready_selected.py")])

    # 2) Train/evaluate models (write results to AI5)
    jobs: list[tuple[str, Path, Path]] = [
        ("linear", AI_DIR / "linear_ml.py", AI5_DIR / "linear_results.csv"),
        ("random_forest", AI_DIR / "random_forest_ml.py", AI5_DIR / "random_forest_results.csv"),
        ("svr", AI_DIR / "svr_ml.py", AI5_DIR / "svr_results.csv"),
        ("rvm", AI_DIR / "rvm_ml.py", AI5_DIR / "rvm_results.csv"),
        ("xgboost", AI_DIR / "xgboost_ml.py", AI5_DIR / "xgboost_results.csv"),
        ("gradient_boosting", AI_DIR / "gradient_boosting_ml.py", AI5_DIR / "gradient_boosting_results.csv"),
        ("adaboost", AI_DIR / "adaboost_ml.py", AI5_DIR / "adaboost_results.csv"),
    ]

    common_args = [
        "--data",
        str(selected_csv),
        "--target",
        "PR(mm/r)",
        "--splits",
        "0.7,0.15,0.15;0.6,0.2,0.2;0.7,0.1,0.2",
        "--random-state",
        "42",
    ]

    for name, script_path, out_csv in jobs:
        print(f"\n=== {name} ===")
        run([sys.executable, str(script_path), *common_args, "--out-results", str(out_csv)])

    # 3) Rankings/diagrams (0.70/0.10/0.20)
    run([sys.executable, str(AI5_DIR / "ranking_diagrams_70_10_20.py")])

    # 4) Distance correlation heatmap
    run([sys.executable, str(AI5_DIR / "distance_correlation_heatmap.py")])

    print("\nDone. Outputs:")
    print("-", AI5_DIR)
    print("-", AI5_DIR / "images")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
