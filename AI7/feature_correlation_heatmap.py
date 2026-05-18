"""Trace une heatmap de corrélation des features à partir du split normalisé.

Usage:
  python feature_correlation_heatmap.py

Sortie:
  Internship_Research/AI7/images/feature_correlation_heatmap.png
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

AI7 = Path(__file__).resolve().parent
img_dir = AI7 / 'images'
img_dir.mkdir(parents=True, exist_ok=True)

# Préférer le split normalisé si présent, sinon le fichier raw nettoyé
train_csv = AI7 / 'split_train_norm_80_20.csv'
raw_csv = Path(__file__).resolve().parents[1] / 'TBM_data_cleaned.csv'

if train_csv.exists():
    df = pd.read_csv(train_csv)
else:
    if not raw_csv.exists():
        raise FileNotFoundError(f"Aucun fichier de données trouvé ({train_csv} ou {raw_csv})")
    df = pd.read_csv(raw_csv)

# Garder uniquement colonnes numériques (exclut la cible si présente)
numeric = df.select_dtypes(include=[np.number]).copy()
if numeric.shape[1] == 0:
    raise ValueError("Aucune colonne numérique trouvée pour calculer la corrélation")

corr = numeric.corr()

plt.figure(figsize=(8, 6))
try:
    import seaborn as sns
    sns.set(font_scale=0.9)
    ax = sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True, cbar_kws={'shrink': .8})
    ax.set_title('Correlation heatmap (features)')
    plt.tight_layout()
except Exception:
    # Fallback: matplotlib imshow
    im = plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = range(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=90)
    plt.yticks(ticks, corr.columns)
    plt.title('Correlation heatmap (features)')
    plt.tight_layout()

out = img_dir / 'feature_correlation_heatmap.png'
plt.savefig(out, dpi=150)
print('Saved:', out)
