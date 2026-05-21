"""Run both PDP and ICE analyses for Gradient Boosting (AI10).

This script:
1) Runs pdp_gradient_boosting.py to generate Partial Dependence Plots
2) Runs ice_gradient_boosting.py to generate Individual Conditional Expectation plots

Example (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI10/run_pdp_ice.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ai10_dir = repo_root / "Internship_Research" / "AI10"

    scripts = [
        ("PDP (Partial Dependence Plots)", ai10_dir / "pdp_gradient_boosting.py"),
        ("ICE (Individual Conditional Expectation)", ai10_dir / "ice_gradient_boosting.py"),
    ]

    for name, script in scripts:
        print(f"\n{'='*70}")
        print(f"Running {name}...")
        print(f"{'='*70}")
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=repo_root,
        )
        if result.returncode != 0:
            print(f"ERROR: {name} failed with code {result.returncode}")
            return result.returncode

    print(f"\n{'='*70}")
    print("All analyses complete! Check AI10/images/ for results.")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    exit(main())
