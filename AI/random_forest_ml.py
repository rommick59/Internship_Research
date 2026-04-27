"""Random Forest regression training/evaluation on the TBM normalized dataset.

This script uses the normalized dataset produced in this repo (values in [0, 1]).
It trains a RandomForestRegressor and evaluates it on multiple train/validation/test
split configurations.

Default splits intentionally exclude any configuration with test=0.

Metrics reported (on validation and test):
- R
- R2
- MSE
- RMSE
- MAE

Example (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI/random_forest_ml.py \
    --data Internship_Research/TBM_data_cleaned_ml_ready.csv \
    --target "PR(mm/r)"
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
    # Allow running as a plain script: `python Internship_Research/AI/random_forest_ml.py`
    # by injecting the repository root into sys.path.
    import sys

    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from Internship_Research.normalize_tbm_data_cleaned import normalize_to_ml_ready
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


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
    # Deliberately excludes 0.8/0.2/0.0
    return [
        SplitConfig(0.7, 0.15, 0.15),
        SplitConfig(0.6, 0.2, 0.2),
        SplitConfig(0.7, 0.1, 0.2),
    ]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    # Pearson correlation coefficient (R). Can be NaN if variance is 0.
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
    # 1) split off test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=config.test,
        random_state=random_state,
        shuffle=True,
    )

    # 2) split validation from trainval
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
    p = argparse.ArgumentParser(description="Random Forest regression with multiple train/val/test splits")
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

    p.add_argument("--n-estimators", type=int, default=500, help="Number of trees")
    p.add_argument("--max-depth", type=int, default=None, help="Max tree depth (default: unlimited)")
    p.add_argument("--min-samples-leaf", type=int, default=1, help="Minimum samples per leaf")

    p.add_argument(
        "--out-results",
        type=Path,
        default=Path("Internship_Research/AI/random_forest_results.csv"),
        help="Where to write the evaluation table (CSV)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        # Make the script runnable without a separate normalization command.
        print(f"ML-ready CSV not found: {args.data}. Generating it from: {args.raw_data}")
        normalize_to_ml_ready(
            input_csv=args.raw_data,
            output_csv=args.data,
            save_preprocessor=None,
            strict=False,
        )

    df = pd.read_csv(args.data)

    if args.target not in df.columns:
        raise ValueError(
            f"Unknown target column: {args.target!r}. Available: {list(df.columns)!r}"
        )

    X_df = df.drop(columns=[args.target])
    y = df[args.target].to_numpy(dtype=float)
    X = X_df.to_numpy(dtype=float)

    split_configs = parse_splits(args.splits) if args.splits else make_default_splits()

    rows: list[dict[str, object]] = []

    for cfg in split_configs:
        train, val, test = split_data(X, y, cfg, random_state=args.random_state)
        X_train, y_train = train
        X_val, y_val = val
        X_test, y_test = test

        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            random_state=args.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        row: dict[str, object] = {
            "estimator": "random_forest",
            "n_estimators": int(args.n_estimators),
            "max_depth": (None if args.max_depth is None else int(args.max_depth)),
            "min_samples_leaf": int(args.min_samples_leaf),
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
