import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

p = Path(__file__).resolve().parent
csv_path = p / 'vaf_ranking.csv'
if not csv_path.exists():
    raise FileNotFoundError(f"Missing {csv_path}")

df = pd.read_csv(csv_path)
df['test_vaf'] = pd.to_numeric(df['test_vaf'], errors='coerce')
df = df.sort_values('test_vaf', ascending=False)

plt.figure(figsize=(8, 4))
plt.bar(df['estimator'], df['test_vaf'], color='C0')
plt.ylabel('Test VAF')
plt.ylim(0, 1.05)
plt.title('VAF ranking (test)')
for i, v in enumerate(df['test_vaf']):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
plt.tight_layout()

out = p / 'vaf_ranking.png'
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
