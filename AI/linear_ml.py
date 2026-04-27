"""Linear regression training/evaluation on the TBM normalized dataset.

This script uses the normalized dataset produced in this repo (values in [0, 1]).
It trains a Linear Regression model (ordinary least squares) and evaluates it on
multiple train/validation/test split configurations (test size can be 0).

Metrics reported (on validation and test when available):
- R2
- MSE
- RMSE
- MAE

Example (PowerShell):
  c:/Users/siame/Desktop/Stage/.venv/Scripts/python.exe Internship_Research/AI/linear_ml.py \
    --data Internship_Research/TBM_data_cleaned_ml_ready.csv \
    --target "PR(mm/r)"

You can customize split configurations:
  --splits "0.7,0.15,0.15;0.8,0.2,0;0.6,0.2,0.2"
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
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
        SplitConfig(0.8, 0.2, 0.0),
        SplitConfig(0.6, 0.2, 0.2),
    ]


def build_estimator() -> LinearRegression:
    return LinearRegression()


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
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
    tuple[np.ndarray, np.ndarray] | None,
    tuple[np.ndarray, np.ndarray] | None,
]:
    if config.test > 0:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X,
            y,
            test_size=config.test,
            random_state=random_state,
            shuffle=True,
        )
    else:
        X_trainval, y_trainval = X, y
        X_test, y_test = None, None

    if config.val > 0:
        # Validation is taken from trainval.
        val_fraction_of_trainval = config.val / (config.train + config.val)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_fraction_of_trainval,
            random_state=random_state,
            shuffle=True,
        )
    else:
        X_train, y_train = X_trainval, y_trainval
        X_val, y_val = None, None

    train = (X_train, y_train)
    val = (X_val, y_val) if X_val is not None else None
    test = (X_test, y_test) if X_test is not None else None
    return train, val, test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear regression with multiple train/val/test splits")
    p.add_argument(
        "--data",
        type=Path,
        default=Path("Internship_Research/TBM_data_cleaned_ml_ready.csv"),
        help="Path to normalized CSV (values in [0,1])",
    )
    p.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target column name to predict (required)",
    )
    p.add_argument(
        "--splits",
        type=str,
        default=None,
        help='Split configs like "0.7,0.15,0.15;0.8,0.2,0" (train,val,test;...)',
    )
    p.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for shuffling/splitting",
    )
    p.add_argument(
        "--out-results",
        type=Path,
        default=Path("Internship_Research/AI/linear_results.csv"),
        help="Where to write the evaluation table (CSV)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    df = pd.read_csv(args.data)

    if args.target is None:
        cols = "\n".join([f"- {c}" for c in df.columns])
        raise SystemExit(
            "--target is required. Available columns are:\n" + cols + "\n\n" +
            'Example: --target "PR(mm/r)"'
        )

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

        estimator = build_estimator()
        estimator.fit(X_train, y_train)

        row: dict[str, object] = {
            "estimator": "linear_regression",
            "train": cfg.train,
            "val": cfg.val,
            "test": cfg.test,
            "n_train": int(len(y_train)),
            "n_val": int(len(val[1])) if val is not None else 0,
            "n_test": int(len(test[1])) if test is not None else 0,
        }

        if val is not None:
            X_val, y_val = val
            y_val_pred = estimator.predict(X_val)
            m = eval_metrics(y_val, y_val_pred)
            row.update({f"val_{k}": float(v) for k, v in m.items()})
        else:
            row.update({"val_r2": np.nan, "val_mse": np.nan, "val_rmse": np.nan, "val_mae": np.nan})

        if test is not None:
            X_test, y_test = test
            y_test_pred = estimator.predict(X_test)
            m = eval_metrics(y_test, y_test_pred)
            row.update({f"test_{k}": float(v) for k, v in m.items()})
        else:
            row.update({"test_r2": np.nan, "test_mse": np.nan, "test_rmse": np.nan, "test_mae": np.nan})

        rows.append(row)

    results = pd.DataFrame(rows)

    args.out_results.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_results, index=False)

    display_cols = [
        "train",
        "val",
        "test",
        "val_r2",
        "val_rmse",
        "val_mae",
        "test_r2",
        "test_rmse",
        "test_mae",
    ]
    print("\nResults (higher R2 is better; lower errors are better):")
    print(results[display_cols].to_string(index=False))
    print(f"\nSaved: {args.out_results}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
