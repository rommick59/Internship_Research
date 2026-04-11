import os
import math
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = "Internship_Research/TBM_data_cleaned.csv"
OUTPUT_DIR = "Internship_Research/outlier_study"
RANDOM_STATE = 42
TARGET_CANDIDATES = [
    "PR(mm/r)",
    "PR (mm/r)",
    "PR",
]


os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_col_name(name: str) -> str:
    cleaned = str(name).strip().replace("\n", " ").replace("\r", " ")
    while "  " in cleaned:
        cleaned = cleaned.replace("  ", " ")
    return cleaned


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [normalize_col_name(c) for c in out.columns]
    for col in out.columns:
        series = out[col].astype(str).str.replace(",", ".", regex=False).str.replace(" ", "", regex=False)
        out[col] = pd.to_numeric(series, errors="coerce")
    return out


def find_target(df: pd.DataFrame) -> str:
    norm_map = {c.replace(" ", "").lower(): c for c in df.columns}
    for cand in TARGET_CANDIDATES:
        key = cand.replace(" ", "").lower()
        if key in norm_map:
            return norm_map[key]
    raise ValueError("Target column not found. Expected one of: {}".format(TARGET_CANDIDATES))


def remove_iqr_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    q1 = df[cols].quantile(0.25)
    q3 = df[cols].quantile(0.75)
    iqr = q3 - q1
    mask = ~((df[cols] < (q1 - 1.5 * iqr)) | (df[cols] > (q3 + 1.5 * iqr))).any(axis=1)
    return df[mask].copy()


def winsorize_clip(df: pd.DataFrame, cols: list[str], lower_q: float = 0.01, upper_q: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    lower = out[cols].quantile(lower_q)
    upper = out[cols].quantile(upper_q)
    for col in cols:
        out[col] = out[col].clip(lower=lower[col], upper=upper[col])
    return out


def build_datasets(df_numeric: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    modeling_cols = [c for c in df_numeric.columns if c != target]
    d0 = df_numeric.dropna().copy()
    d1 = remove_iqr_outliers(d0, modeling_cols)
    d2 = winsorize_clip(d0, modeling_cols, lower_q=0.01, upper_q=0.99)
    return {
        "D0_raw_no_drop_outliers": d0,
        "D1_iqr_removed": d1,
        "D2_winsorized_1_99": d2,
    }


def get_models() -> dict[str, object]:
    return {
        "LinearRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "HuberRegressor": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor()),
        ]),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate_dataset(df: pd.DataFrame, target: str, dataset_name: str) -> list[dict]:
    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    results = []
    for model_name, model in get_models().items():
        cv_scores = cross_validate(model, x_train, y_train, scoring=scoring, cv=cv, n_jobs=-1)

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        row = {
            "dataset": dataset_name,
            "model": model_name,
            "n_rows": int(df.shape[0]),
            "n_features": int(x.shape[1]),
            "cv_mae_mean": float(-np.mean(cv_scores["test_mae"])),
            "cv_mae_std": float(np.std(-cv_scores["test_mae"])),
            "cv_rmse_mean": float(-np.mean(cv_scores["test_rmse"])),
            "cv_rmse_std": float(np.std(-cv_scores["test_rmse"])),
            "cv_r2_mean": float(np.mean(cv_scores["test_r2"])),
            "cv_r2_std": float(np.std(cv_scores["test_r2"])),
            "test_mae": float(mean_absolute_error(y_test, y_pred)),
            "test_rmse": float(math.sqrt(mean_squared_error(y_test, y_pred))),
            "test_r2": float(r2_score(y_test, y_pred)),
        }
        results.append(row)

    return results


def main() -> None:
    print("Loading data from", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    df_numeric = convert_to_numeric(df)

    target = find_target(df_numeric)
    print("Detected target:", target)

    datasets = build_datasets(df_numeric, target)

    dataset_overview = []
    all_results = []

    for name, dset in datasets.items():
        dataset_overview.append(
            {
                "dataset": name,
                "rows": int(dset.shape[0]),
                "columns": int(dset.shape[1]),
                "rows_removed_vs_d0": int(datasets["D0_raw_no_drop_outliers"].shape[0] - dset.shape[0]),
            }
        )

        print("Evaluating", name, "with", dset.shape[0], "rows")
        all_results.extend(evaluate_dataset(dset, target, name))

    overview_df = pd.DataFrame(dataset_overview)
    results_df = pd.DataFrame(all_results)

    overview_path = os.path.join(OUTPUT_DIR, "dataset_overview.csv")
    results_path = os.path.join(OUTPUT_DIR, "model_comparison_cv_test.csv")

    overview_df.to_csv(overview_path, index=False)
    results_df.to_csv(results_path, index=False)

    best_rows = results_df.sort_values(by=["cv_rmse_mean", "cv_mae_mean"]).groupby("dataset", as_index=False).first()
    best_path = os.path.join(OUTPUT_DIR, "best_model_per_dataset.csv")
    best_rows.to_csv(best_path, index=False)

    print("Saved:")
    print(" -", overview_path)
    print(" -", results_path)
    print(" -", best_path)


if __name__ == "__main__":
    main()
