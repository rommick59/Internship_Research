from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_single_row(csv_path: Path) -> pd.Series:
    df = pd.read_csv(csv_path)
    if len(df) != 1:
        raise ValueError(f"Expected 1 row in {csv_path}, got {len(df)}")
    return df.iloc[0]


p = Path(__file__).resolve().parent

# Build ranking from current per-model results (authoritative source)
registry: list[tuple[str, Path]] = [
    ("linear", p / "linear_results.csv"),
    ("random_forest", p / "random_forest_results.csv"),
    ("rvm", p / "rvm_results.csv"),
    ("xgboost", p / "xgboost_results.csv"),
    ("gradient_boosting", p / "gradient_boosting_results.csv"),
]

rows: list[dict[str, object]] = []
for estimator_key, path in registry:
    if not path.exists():
        continue
    row = load_single_row(path)
    test_vaf = pd.to_numeric(row.get("test_vaf", float("nan")), errors="coerce")
    rows.append({"estimator": str(row.get("estimator", estimator_key)), "test_vaf": float(test_vaf)})

df = pd.DataFrame(rows)
if df.empty:
    raise FileNotFoundError("No *_results.csv found to compute VAF ranking")

df = df.sort_values("test_vaf", ascending=False)

csv_path = p / "vaf_ranking.csv"
df.to_csv(csv_path, index=False)

plt.figure(figsize=(8, 4))
plt.bar(df["estimator"], df["test_vaf"], color="C0")
plt.ylabel("Test VAF")
plt.ylim(0, 1.05)
plt.title("VAF ranking (test)")
for i, v in enumerate(df["test_vaf"].to_list()):
    plt.text(i, v + 0.01, f"{float(v):.3f}", ha="center", va="bottom")
plt.tight_layout()

out = p / "vaf_ranking.png"
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
print(f"Saved: {csv_path}")
