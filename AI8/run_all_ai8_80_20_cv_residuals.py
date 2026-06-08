"""AI8 — Full pipeline: 80/20 split → train-only normalization → IA + VAF → CV → residual plots → schema → rankings.

Same as AI7, but scoped to AI8 outputs.

Run (PowerShell):
    c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI8/run_all_ai8_80_20_cv_residuals.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AI8_DIR = REPO_ROOT / "Internship_Research" / "AI8"


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    run([sys.executable, str(AI8_DIR / "build_split_and_normalize_train_only_80_20.py")])
    run([sys.executable, str(AI8_DIR / "train_eval_models_fixed_split_80_20.py")])
    run([sys.executable, str(AI8_DIR / "plot_vaf_ranking.py")])
    run([sys.executable, str(AI8_DIR / "cross_validation_train_only_80_20.py")])
    run([sys.executable, str(AI8_DIR / "residual_error_plots_test_80_20.py")])
    run([sys.executable, str(AI8_DIR / "predicted_vs_measured_test_80_20.py")])
    run([sys.executable, str(AI8_DIR / "normalization_schema_80_20.py")])
    run([sys.executable, str(AI8_DIR / "ranking_diagrams_80_20.py")])

    print("\nDone. Outputs in:")
    print("-", AI8_DIR)
    print("-", AI8_DIR / "images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
