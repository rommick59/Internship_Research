"""Relevance Vector Machine (RVM) regression training/evaluation on the TBM dataset.

This script uses the normalized dataset produced in this repo (values in [0, 1]).
It trains an RVM regressor (EMRVR from sklearn-rvm) and evaluates it on multiple
train/validation/test split configurations.

Metrics reported (on validation and test):
- R (Pearson correlation)
- R2
- MSE
- RMSE
- MAE

Example (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI/rvm_ml.py

You can customize split configurations:
  --splits "0.7,0.15,0.15;0.6,0.2,0.2;0.7,0.1,0.2"

You can customize kernel hyperparameters:
  --kernel rbf --gamma scale --tol 0.001 --max-iter 5000
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from Internship_Research.normalize_tbm_data_cleaned import normalize_to_ml_ready
except ModuleNotFoundError:
    # Allow running as a plain script: `python Internship_Research/AI/rvm_ml.py`
    # by injecting the repository root into sys.path.
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from Internship_Research.normalize_tbm_data_cleaned import normalize_to_ml_ready

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

try:
    from sklearn_rvm import EMRVR
except ModuleNotFoundError as e:
    raise SystemExit(
        "Missing dependency 'sklearn-rvm'. Install it with: pip install sklearn-rvm\n"
        f"Original error: {e}"
    )


@dataclass(frozen=True)
class SplitConfig:
    train: float
    val: float
    test: float

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not (abs(total - 1.0) <= 1e-9):
            raise ValueError(f"Split must sum to 1.0, got {total:.6f}")
        for name, value in (("train", self.train), ("val", self.val), ("test", self.test)):
            if value < 0:
                raise ValueError(f"Split '{name}' must be >= 0")
        if self.train <= 0:
            raise ValueError("Train split must be > 0")
        if self.val <= 0:
            raise ValueError("Validation split must be > 0")
        if self.test <= 0:
            raise ValueError("Test split must be > 0")


def parse_splits(value: str) -> list[SplitConfig]:
    configs: list[SplitConfig] = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        pieces = [p.strip() for p in part.split(",")]
        if len(pieces) != 3:
            raise ValueError(
                "Each split must have 3 comma-separated values: train,val,test. "
                "Example: 0.7,0.15,0.15"
            )
        train, val, test = (float(p) for p in pieces)
        configs.append(SplitConfig(train=train, val=val, test=test))
    if not configs:
        raise ValueError("No valid splits provided")
    return configs


def make_default_splits() -> list[SplitConfig]:
    return [
        SplitConfig(0.7, 0.15, 0.15),
        SplitConfig(0.6, 0.2, 0.2),
        SplitConfig(0.7, 0.1, 0.2),
    ]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        r = float("nan")
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r": r,
        "r2": r2_score(y_true, y_pred),
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    config: SplitConfig,
    random_state: int,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=config.test,
        random_state=random_state,
        shuffle=True,
    )

    val_fraction_of_trainval = config.val / (config.train + config.val)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_fraction_of_trainval,
        random_state=random_state,
        shuffle=True,
    )

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RVM regression (EMRVR) with multiple train/val/test splits")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Path to normalized CSV (values in [0,1])",
    )
    p.add_argument(
        "--raw-data",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned.csv"),
        help="Raw cleaned CSV to normalize if --data does not exist",
    )
    p.add_argument(
        "--target",
        type=str,
        default="PR(mm/r)",
        help='Target column name to predict (default: "PR(mm/r)")',
    )
    p.add_argument(
        "--splits",
        type=str,
        default=None,
        help='Split configs like "0.7,0.15,0.15;0.6,0.2,0.2" (train,val,test;...)',
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shuffling/splitting",
    )

    p.add_argument("--kernel", type=str, default="rbf", help="Kernel: linear|poly|rbf|sigmoid")
    p.add_argument("--degree", type=int, default=3, help="Polynomial degree (poly kernel)")
    p.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help="Kernel coefficient: scale|auto|<float> (rbf/poly/sigmoid)",
    )
    p.add_argument("--coef0", type=float, default=0.0, help="Independent term in poly/sigmoid kernels")
    p.add_argument("--tol", type=float, default=0.001, help="Stopping tolerance")
    p.add_argument("--max-iter", type=int, default=5000, help="Maximum number of EM iterations")

    p.add_argument(
        "--out-results",
        type=Path,
        default=Path("Internship_Research/AI/rvm_results.csv"),
        help="Where to write the evaluation table (CSV)",
    )

    return p.parse_args()


def _parse_gamma(value: str) -> str | float:
    v = value.strip().lower()
    if v in {"scale", "auto"}:
        return v
    return float(value)


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        print(f"ML-ready CSV not found: {args.data}. Generating it from: {args.raw_data}")
        normalize_to_ml_ready(
            input_csv=args.raw_data,
            output_csv=args.data,
            save_preprocessor=None,
            strict=False,
        )

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(f"Unknown target column: {args.target!r}. Available: {list(df.columns)!r}")

    X_df = df.drop(columns=[args.target])
    y = df[args.target].to_numpy(dtype=float)
    X = X_df.to_numpy(dtype=float)

    split_configs = parse_splits(args.splits) if args.splits else make_default_splits()

    gamma = _parse_gamma(args.gamma)

    rows: list[dict[str, object]] = []

    for cfg in split_configs:
        train, val, test = split_data(X, y, cfg, random_state=args.random_state)
        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        model = EMRVR(
            kernel=args.kernel,
            degree=int(args.degree),
            gamma=gamma,
            coef0=float(args.coef0),
            tol=float(args.tol),
            max_iter=int(args.max_iter),
        )
        model.fit(X_train, y_train)

        row: dict[str, object] = {
            "estimator": "rvm",
            "kernel": args.kernel,
            "degree": int(args.degree),
            "gamma": (gamma if isinstance(gamma, str) else float(gamma)),
            "coef0": float(args.coef0),
            "tol": float(args.tol),
            "max_iter": int(args.max_iter),
            "train": cfg.train,
            "val": cfg.val,
            "test": cfg.test,
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_test": int(len(y_test)),
        }

        y_val_pred = model.predict(X_val)
        m = eval_metrics(y_val, y_val_pred)
        row.update({f"val_{k}": float(v) for k, v in m.items()})

        y_test_pred = model.predict(X_test)
        m = eval_metrics(y_test, y_test_pred)
        row.update({f"test_{k}": float(v) for k, v in m.items()})

        rows.append(row)

    results = pd.DataFrame(rows)
    args.out_results.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_results, index=False)

    display_cols = [
        "train",
        "val",
        "test",
        "val_r",
        "val_r2",
        "val_mse",
        "val_rmse",
        "val_mae",
        "test_r",
        "test_r2",
        "test_mse",
        "test_rmse",
        "test_mae",
    ]
    print("\nResults (higher R2 is better; lower errors are better):")
    print(results[display_cols].to_string(index=False))
    print(f"\nSaved: {args.out_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
