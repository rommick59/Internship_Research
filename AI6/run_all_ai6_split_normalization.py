"""AI6 — Full pipeline: split → normalize train only → train/test/val → rankings + schema.

Run (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI6/run_all_ai6_split_normalization.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
AI6_DIR = REPO_ROOT / "Internship_Research" / "AI6"


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    run([sys.executable, str(AI6_DIR / "build_splits_and_normalize_train_only.py")])
    run([sys.executable, str(AI6_DIR / "train_eval_models_fixed_split.py")])
    run([sys.executable, str(AI6_DIR / "ranking_diagrams_70_10_20.py")])
    run([sys.executable, str(AI6_DIR / "normalization_schema_70_20_10.py")])

    print("\nDone. Outputs in:")
    print("-", AI6_DIR)
    print("-", AI6_DIR / "images")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
