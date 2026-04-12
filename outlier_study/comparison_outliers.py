import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def save_table_png(df: pd.DataFrame, title: str, out_path: str, round_decimals: int = 4) -> None:
    view = df.copy()
    num_cols = view.select_dtypes(include=[np.number]).columns
    view[num_cols] = view[num_cols].round(round_decimals)

    n_rows, n_cols = view.shape
    fig_w = max(12, n_cols * 1.8)
    fig_h = max(2.8, n_rows * 0.55 + 1.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, pad=12)

    table = ax.table(
        cellText=view.values,
        colLabels=view.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e9edf5")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_bar_chart(
    df: pd.DataFrame,
    metric_col: str,
    title: str,
    out_path: str,
    higher_is_better: bool,
) -> None:
    # Grouped bars: x=Dataset, one bar per model, with value labels.
    order_dataset = ["D0_raw_no_drop_outliers", "D1_iqr_removed", "D2_winsorized_1_99"]
    order_model = ["LinearRegression", "HuberRegressor", "RandomForest"]
    model_colors = {
        "LinearRegression": "#4c78a8",
        "HuberRegressor": "#f58518",
        "RandomForest": "#54a24b",
    }

    plot_df = df.copy()
    plot_df = plot_df[plot_df["dataset"].isin(order_dataset) & plot_df["model"].isin(order_model)]

    x = np.arange(len(order_dataset))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6.5))

    for i, model in enumerate(order_model):
        vals = []
        for ds in order_dataset:
            row = plot_df[(plot_df["dataset"] == ds) & (plot_df["model"] == model)]
            vals.append(float(row[metric_col].iloc[0]) if not row.empty else np.nan)

        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=model,
            color=model_colors[model],
            alpha=0.9,
        )

        for bar, v in zip(bars, vals):
            if np.isnan(v):
                continue
            y_offset = 0.01 if higher_is_better else 0.005
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_offset,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(metric_col)
    ax.set_xticks(x)
    ax.set_xticklabels(order_dataset, rotation=10)
    ax.legend(title="Model")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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

    rename_map = {
        "dataset": "Dataset",
        "model": "Model",
        "n_rows": "Rows",
        "n_features": "Features",
        "rows": "Rows",
        "columns": "Columns",
        "rows_removed_vs_d0": "Rows Removed vs D0",
        "cv_mae_mean": "CV MAE (mean)",
        "cv_mae_std": "CV MAE (std)",
        "cv_rmse_mean": "CV RMSE (mean)",
        "cv_rmse_std": "CV RMSE (std)",
        "cv_r2_mean": "CV R2 (mean)",
        "cv_r2_std": "CV R2 (std)",
        "test_mae": "Test MAE",
        "test_rmse": "Test RMSE",
        "test_r2": "Test R2",
    }

    overview_path = os.path.join(OUTPUT_DIR, "dataset_overview.csv")
    results_path = os.path.join(OUTPUT_DIR, "model_comparison_cv_test.csv")

    overview_export = overview_df.rename(columns=rename_map)
    results_export = results_df.rename(columns=rename_map)

    overview_export.to_csv(overview_path, index=False)
    results_export.to_csv(results_path, index=False)

    best_rows = results_df.sort_values(by=["cv_rmse_mean", "cv_mae_mean"]).groupby("dataset", as_index=False).first()
    best_path = os.path.join(OUTPUT_DIR, "best_model_per_dataset.csv")
    best_export = best_rows.rename(columns=rename_map)
    best_export.to_csv(best_path, index=False)

    # Save matplotlib table images for reporting.
    overview_png = os.path.join(OUTPUT_DIR, "dataset_overview_table.png")
    comparison_png = os.path.join(OUTPUT_DIR, "model_comparison_table.png")
    best_png = os.path.join(OUTPUT_DIR, "best_model_per_dataset_table.png")

    save_table_png(overview_export, "Dataset Overview", overview_png, round_decimals=0)

    display_cols = [
        "Dataset",
        "Model",
        "Rows",
        "CV MAE (mean)",
        "CV RMSE (mean)",
        "CV R2 (mean)",
        "Test MAE",
        "Test RMSE",
        "Test R2",
    ]
    comparison_display = results_export[display_cols].sort_values(by=["Dataset", "Model"]).reset_index(drop=True)
    save_table_png(comparison_display, "Model Comparison (CV + Test)", comparison_png)

    best_display = best_export[display_cols].sort_values(by=["Dataset", "Model"]).reset_index(drop=True)
    save_table_png(best_display, "Best Model per Dataset", best_png)

    # Save score charts with values displayed on bars.
    rmse_cv_png = os.path.join(OUTPUT_DIR, "score_cv_rmse_bar.png")
    rmse_test_png = os.path.join(OUTPUT_DIR, "score_test_rmse_bar.png")
    mae_cv_png = os.path.join(OUTPUT_DIR, "score_cv_mae_bar.png")
    mae_test_png = os.path.join(OUTPUT_DIR, "score_test_mae_bar.png")
    r2_cv_png = os.path.join(OUTPUT_DIR, "score_cv_r2_bar.png")
    r2_test_png = os.path.join(OUTPUT_DIR, "score_test_r2_bar.png")

    save_metric_bar_chart(results_df, "cv_rmse_mean", "CV RMSE by Dataset and Model", rmse_cv_png, higher_is_better=False)
    save_metric_bar_chart(results_df, "test_rmse", "Test RMSE by Dataset and Model", rmse_test_png, higher_is_better=False)
    save_metric_bar_chart(results_df, "cv_mae_mean", "CV MAE by Dataset and Model", mae_cv_png, higher_is_better=False)
    save_metric_bar_chart(results_df, "test_mae", "Test MAE by Dataset and Model", mae_test_png, higher_is_better=False)
    save_metric_bar_chart(results_df, "cv_r2_mean", "CV R2 by Dataset and Model", r2_cv_png, higher_is_better=True)
    save_metric_bar_chart(results_df, "test_r2", "Test R2 by Dataset and Model", r2_test_png, higher_is_better=True)

    print("Saved:")
    print(" -", overview_path)
    print(" -", results_path)
    print(" -", best_path)
    print(" -", overview_png)
    print(" -", comparison_png)
    print(" -", best_png)
    print(" -", rmse_cv_png)
    print(" -", rmse_test_png)
    print(" -", mae_cv_png)
    print(" -", mae_test_png)
    print(" -", r2_cv_png)
    print(" -", r2_test_png)


if __name__ == "__main__":
    main()
