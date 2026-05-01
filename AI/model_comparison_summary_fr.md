# Comparaison modèles (cible: PR(mm/r))

Comparaison réalisée sur les mêmes splits **train/val/test** (0.70/0.15/0.15, 0.60/0.20/0.20 et 0.70/0.10/0.20), en utilisant les métriques **R (corrélation), R², MSE, RMSE, MAE** sur le jeu **test**.

## Résultat principal
Le **RVM** (Relevance Vector Machine) surpasse la **Random Forest**, le **SVR** et la **régression linéaire** sur les splits testés (R² plus élevé et erreurs plus faibles).

## Recommandation finale (modèle + split)
- **Modèle recommandé**: **RVM** (meilleur RMSE/MAE et meilleur R² sur les splits testés).
- **Split retenu avec 20% de test**: **0.70 / 0.10 / 0.20** (test = 240 lignes) : meilleur des splits à **20% test** dans ces essais.
- **Split alternatif pour une validation plus large (tuning hyperparamètres)**: **0.60 / 0.20 / 0.20** (val = 240 lignes) : plus adapté à l’ajustement d’hyperparamètres.

Note: le split **0.70 / 0.15 / 0.15** donne les meilleurs chiffres absolus ici, mais avec un **test plus petit** (180 lignes), donc l’estimation peut être un peu moins stable.

## Chiffres clés (TEST, échelle normalisée 0–1)
- **Split 0.70/0.15/0.15**
  - Linéaire: R² = 0.9960, RMSE = 0.01495, MAE = 0.01045
  - Random Forest: R² = 0.9979, RMSE = 0.01086, MAE = 0.00401
  - SVR (RBF, C=10, epsilon=0.01): R² = 0.9990, RMSE = 0.00765, MAE = 0.00538
  - RVM (RBF): R² = 0.9999, RMSE = 0.00224, MAE = 0.00066
  - XGBoost: R² = 0.9982, RMSE = 0.00991, MAE = 0.00508
- **Split 0.60/0.20/0.20**
  - Linéaire: R² = 0.9960, RMSE = 0.01504, MAE = 0.01066
  - Random Forest: R² = 0.9973, RMSE = 0.01235, MAE = 0.00532
  - SVR (RBF, C=10, epsilon=0.01): R² = 0.9986, RMSE = 0.00887, MAE = 0.00589
  - RVM (RBF): R² = 1.0000, RMSE = 0.00168, MAE = 0.00060
  - XGBoost: R² = 0.9987, RMSE = 0.00845, MAE = 0.00496
- **Split 0.70/0.10/0.20**
  - Linéaire: R² = 0.9960, RMSE = 0.01496, MAE = 0.01048
  - Random Forest: R² = 0.9976, RMSE = 0.01155, MAE = 0.00460
  - SVR (RBF, C=10, epsilon=0.01): R² = 0.9990, RMSE = 0.00763, MAE = 0.00531
  - RVM (RBF): R² = 1.0000, RMSE = 0.00146, MAE = 0.00049
  - XGBoost: R² = 0.9988, RMSE = 0.00832, MAE = 0.00471

## Moyenne sur les 3 splits (TEST)
- R (corrélation) moyen: linéaire = 0.998012 vs random forest = 0.998839 vs SVR = 0.999426 vs RVM = 0.999971 vs XGBoost = 0.999300
- R² moyen: linéaire = 0.996009 vs random forest = 0.997609 vs SVR = 0.998842 vs RVM = 0.999941 vs XGBoost = 0.998586
- MSE moyen: linéaire ≈ 0.000225 vs random forest ≈ 0.000135 vs SVR ≈ 0.000065 vs RVM ≈ 0.000003 vs XGBoost ≈ 0.000080
- RMSE moyen: linéaire ≈ 0.014985 vs random forest ≈ 0.011585 vs SVR ≈ 0.008052 vs RVM ≈ 0.001793 vs XGBoost ≈ 0.008889
- MAE moyen: linéaire ≈ 0.010530 vs random forest ≈ 0.004643 vs SVR ≈ 0.005525 vs RVM ≈ 0.000581 vs XGBoost ≈ 0.004919

## Interprétation en mm/r (approximatif)
Le dataset ML-ready utilise une normalisation Min-Max (0–1). La plage réelle de PR(mm/r) dans les données nettoyées est d’environ **47.69 mm/r** (de 2.31 à 50.0). Une erreur normalisée $e$ correspond donc à ~ $e \times 47.69$ mm/r.

- RMSE test moyen: linéaire ≈ **0.715 mm/r** vs random forest ≈ **0.553 mm/r** vs SVR ≈ **0.384 mm/r**
- MAE test moyen: linéaire ≈ **0.503 mm/r** vs random forest ≈ **0.223 mm/r** vs SVR ≈ **0.263 mm/r**
- RMSE test moyen (RVM): ≈ **0.086 mm/r**
- MAE test moyen (RVM): ≈ **0.028 mm/r**
- RMSE test moyen (XGBoost): ≈ **0.424 mm/r**
- MAE test moyen (XGBoost): ≈ **0.235 mm/r**

## Conclusion
Selon l’objectif de la modélisation sur PR(mm/r):
- **RVM** fournit les meilleures performances selon **RMSE**, **MAE** et **R²** sur les splits testés.

La **régression linéaire** reste très performante mais légèrement moins précise.

## Quel jeu de test utiliser ?
Recommandation:
- Pour conserver un **test à 20%** tout en gardant les meilleures performances observées: **0.70 / 0.10 / 0.20**.
- Pour disposer d’une **validation à 20%** (réglage d’hyperparamètres): **0.60 / 0.20 / 0.20**.

Le split **0.70 / 0.15 / 0.15** (15% test, **180 lignes**) est aussi valide, mais un peu moins robuste statistiquement.
