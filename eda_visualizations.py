import math
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set seaborn style for better visuals
sns.set(style='whitegrid')

# Load the cleaned dataset (handle comma as decimal separator and strip spaces)
file_path = 'Internship_Research/TBM_data_cleaned.csv'
df = pd.read_csv(file_path, decimal=',')

# Remove possible leading/trailing spaces in column names
df.columns = df.columns.str.strip()
df.columns = df.columns.str.replace('\n', ' ', regex=False).str.replace('\r', ' ', regex=False)
df.columns = df.columns.str.replace('  ', ' ', regex=False)

# Convert all columns to numeric where possible (force non-numeric values to NaN)
for col in df.columns:
    cleaned_col = df[col].astype(str).str.replace(',', '.', regex=False).str.replace(' ', '', regex=False)
    df[col] = pd.to_numeric(cleaned_col, errors='coerce')

# Numeric-only dataframe for EDA
df_numeric = df.select_dtypes(include=['float64', 'int64'])
numeric_cols = list(df_numeric.columns)


def pretty_label(label: str) -> str:
    # Shorten long engineering units for cleaner subplot titles.
    return label.replace('kW· h/m3', 'kW.h/m3')

if len(numeric_cols) == 0:
    raise ValueError('No numeric columns were detected in TBM_data_cleaned.csv.')

# 1. All histograms in a single figure (subplots)
num_cols = len(numeric_cols)
cols = 3
rows = math.ceil(num_cols / cols)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.6, rows * 5.0))
axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

for i, col in enumerate(numeric_cols):
    sns.histplot(df_numeric[col].dropna(), kde=True, bins=30, ax=axes[i])
    axes[i].set_title(pretty_label(col), fontsize=11, pad=10)
    axes[i].set_xlabel(pretty_label(col), fontsize=10)
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].xaxis.label.set_visible(True)
    axes[i].yaxis.label.set_visible(True)
    axes[i].tick_params(axis='x', labelbottom=True, labelrotation=25, labelsize=8)
    axes[i].tick_params(axis='y', labelsize=9)

for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])

fig.subplots_adjust(hspace=0.95, wspace=0.28, bottom=0.12, top=0.95)
plt.show()

# 2. Boxplots for outlier detection (vertical, one per subplot, aligned horizontally, independent y-axis)
fig, axes = plt.subplots(1, len(numeric_cols), figsize=(2.5 * len(numeric_cols), 6))
if len(numeric_cols) == 1:
    axes = [axes]

for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df_numeric[col], ax=axes[i], color='skyblue')
    axes[i].set_title(pretty_label(col), fontsize=11, pad=10)
    axes[i].set_xlabel('')
    axes[i].set_ylabel(pretty_label(col), fontsize=10)
    axes[i].tick_params(axis='y', labelsize=9)
    axes[i].set_xticks([])

fig.suptitle("Boxplots of Variables (Each Variable Independent)", fontsize=14, y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()

# 3. Scatter plots: all in one figure (target vs all other numeric variables)
target = 'PR(mm/r)'
if target not in df_numeric.columns:
    normalized_target = target.replace(' ', '').lower()
    for col in df_numeric.columns:
        if col.replace(' ', '').lower() == normalized_target:
            target = col
            break

if target in df_numeric.columns:
    scatter_features = [col for col in numeric_cols if col != target]
    if len(scatter_features) > 0:
        scat_cols = 3
        scat_rows = math.ceil(len(scatter_features) / scat_cols)
        fig, axes = plt.subplots(scat_rows, scat_cols, figsize=(scat_cols * 8, scat_rows * 6))
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, col in enumerate(scatter_features):
            sns.scatterplot(x=df_numeric[target], y=df_numeric[col], ax=axes[i], s=18)
            axes[i].set_title(f'{pretty_label(target)} vs {pretty_label(col)}', fontsize=12, pad=12)
            axes[i].set_xlabel(pretty_label(target), fontsize=11)
            axes[i].set_ylabel(pretty_label(col), fontsize=11)
            axes[i].xaxis.label.set_visible(True)
            axes[i].yaxis.label.set_visible(True)
            axes[i].tick_params(axis='x', labelrotation=25, labelsize=9)
            axes[i].tick_params(axis='y', labelsize=9)

        for j in range(len(scatter_features), len(axes)):
            fig.delaxes(axes[j])

        fig.subplots_adjust(hspace=0.95, wspace=0.30, bottom=0.12, top=0.95)
        plt.show()
else:
    print(f"Target column '{target}' not found in numeric columns.")

# 4. Correlation matrix (heatmap) - Spearman for non-linear monotonic relationships
corr = df_numeric.corr(method='spearman')
if corr.size > 0:
    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    short_labels = [
        pretty_label(c)
        for c in corr.columns
    ]
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 8}, ax=ax)
    ax.set_xticklabels(short_labels, rotation=45, ha='right', rotation_mode='anchor', fontsize=9)
    ax.set_yticklabels(short_labels, rotation=0, fontsize=9)
    ax.set_title('Correlation Matrix (Spearman)', pad=12)
    fig.subplots_adjust(bottom=0.24, left=0.20, right=0.98, top=0.93)
    plt.show()
else:
    print('No numeric columns available for correlation heatmap.')

# 5. 2D Kernel Density Estimation (KDE) in one grouped figure (target vs all features)
print('Affichage des 2D Kernel Density plots regroupes...')
if target in df_numeric.columns:
    kde_features = [col for col in numeric_cols if col != target]
    if len(kde_features) > 0:
        kde_cols = 3
        kde_rows = math.ceil(len(kde_features) / kde_cols)
        fig, axes = plt.subplots(
            kde_rows,
            kde_cols,
            figsize=(kde_cols * 6.8, kde_rows * 5.2),
            squeeze=False,
        )
        flat_axes = axes.ravel()

        for i, col in enumerate(kde_features):
            ax = flat_axes[i]
            subset = df_numeric[[col, target]].dropna()
            if subset.shape[0] < 5:
                ax.set_title(f'Not enough data: {col} vs {target}', fontsize=10)
                ax.set_axis_off()
                continue

            sns.kdeplot(
                data=subset,
                x=col,
                y=target,
                fill=True,
                cmap='viridis',
                thresh=0.05,
                levels=40,
                alpha=0.7,
                ax=ax,
            )
            ax.set_title(f'{pretty_label(col)} vs {pretty_label(target)}', fontsize=10, pad=10)
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.xaxis.label.set_visible(False)
            ax.yaxis.label.set_visible(False)
            row_idx = i // kde_cols
            if row_idx < kde_rows - 1:
                ax.tick_params(axis='x', labelbottom=False)
            else:
                ax.tick_params(axis='x', labelrotation=25, labelsize=8)
            ax.tick_params(axis='y', labelsize=8)

        for j in range(len(kde_features), len(flat_axes)):
            flat_axes[j].set_axis_off()

        fig.subplots_adjust(hspace=0.95, wspace=0.30, bottom=0.12, top=0.95)
        plt.show()
    else:
        print('No feature available for 2D KDE plots.')
else:
    print(f"Target column '{target}' not found for 2D KDE plots.")
