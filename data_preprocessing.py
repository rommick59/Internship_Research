def check_raw_data(df):
	print("\n--- Vérification du fichier brut ---")
	print(f"Forme du DataFrame : {df.shape}")
	print(f"Colonnes : {list(df.columns)}")
	missing = df.isnull().sum()
	print("Valeurs manquantes par colonne :")
	print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante.")
	dups = df.duplicated().sum()
	print(f"Nombre de lignes dupliquées : {dups}")
	print("Résumé statistique :")
	print(df.describe(include='all'))
import pandas as pd
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
from sklearn.preprocessing import StandardScaler


# Lecture prioritaire du fichier Excel (.xlsx), sinon fallback sur le CSV
import os
df = None
csv_path = 'Internship_Research/TBM data.csv'

if os.path.exists(csv_path):
	# Utilise la deuxième ligne comme en-tête (header=1), saute la première ligne
	df = pd.read_csv(csv_path, encoding='latin-1', sep=',', header=1)
	print("Lecture du fichier CSV réussie (séparateur virgule, header=1).")
	print("Colonnes CSV :", list(df.columns))
	print(df.head())
	check_raw_data(df)
	# Suppression des lignes avec au moins une valeur manquante
	df = df.dropna()
	print(f"Après suppression des lignes avec valeurs manquantes : {df.shape[0]} lignes restantes.")

# 1. Gérer les valeurs manquantes (remplacement par la moyenne)
df = df.fillna(df.mean(numeric_only=True))

# 2. Supprimer les outliers (méthode IQR) uniquement sur les colonnes numériques
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) > 0:
	Q1 = df[numeric_cols].quantile(0.25)
	Q3 = df[numeric_cols].quantile(0.75)
	IQR = Q3 - Q1
	condition = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
	df = df[condition]
else:
	print("Aucune colonne numérique pour la détection des outliers.")

# 3. Normaliser les données (Standardisation)
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
if len(numeric_cols) > 0:
	scaler = StandardScaler()
	df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
else:
	print("Aucune donnée ou colonne numérique à normaliser après suppression des outliers.")


# Arrondi des colonnes numériques à 3 décimales juste avant la sauvegarde
df = df.round(3)
df.to_csv('Internship_Research/TBM_data_cleaned.csv', index=False)

print("Prétraitement terminé. Jeu de données nettoyé sauvegardé dans 'Internship_Research/TBM_data_cleaned.csv'.")
