# Comparaison modèles (cible: PR(mm/r))

Comparaison réalisée sur les mêmes splits **train/val/test** (0.70/0.15/0.15, 0.60/0.20/0.20 et 0.70/0.10/0.20), en utilisant les métriques **R (corrélation), R², MSE, RMSE, MAE** sur le jeu **test**.

## Résultat principal
La **Random Forest** surpasse la **régression linéaire** sur les splits testés (R² plus élevé et erreurs plus faibles).

## Recommandation finale (modèle + split)
- **Modèle à prendre**: **Random Forest** (meilleure performance sur tous les splits testés).
- **Split à privilégier si tu veux garder 20% de test**: **0.70 / 0.10 / 0.20** (test = 240 lignes) : c’est le meilleur des splits à **20% test** dans nos essais.
- **Split à privilégier si tu veux une validation plus large (tuning hyperparamètres)**: **0.60 / 0.20 / 0.20** (val = 240 lignes) : plus confortable si tu ajustes le modèle.

Note: le split **0.70 / 0.15 / 0.15** donne les meilleurs chiffres absolus ici, mais avec un **test plus petit** (180 lignes), donc l’estimation peut être un peu moins stable.

## Chiffres clés (TEST, échelle normalisée 0–1)
- **Split 0.70/0.15/0.15**
  - Linéaire: R² = 0.9960, RMSE = 0.01495, MAE = 0.01045
  - Random Forest: R² = 0.9979, RMSE = 0.01086, MAE = 0.00401
- **Split 0.60/0.20/0.20**
  - Linéaire: R² = 0.9960, RMSE = 0.01504, MAE = 0.01066
  - Random Forest: R² = 0.9973, RMSE = 0.01235, MAE = 0.00532
- **Split 0.70/0.10/0.20**
  - Linéaire: R² = 0.9960, RMSE = 0.01496, MAE = 0.01048
  - Random Forest: R² = 0.9976, RMSE = 0.01155, MAE = 0.00460

## Moyenne sur les 3 splits (TEST)
- R (corrélation) moyen: linéaire = 0.998012 vs random forest = 0.998839
- R² moyen: linéaire = 0.996009 vs random forest = 0.997609
- MSE moyen: linéaire ≈ 0.000225 vs random forest ≈ 0.000135
- RMSE moyen: linéaire ≈ 0.014985 vs random forest ≈ 0.011585
- MAE moyen: linéaire ≈ 0.010530 vs random forest ≈ 0.004643

## Interprétation en mm/r (approximatif)
Le dataset ML-ready utilise une normalisation Min-Max (0–1). La plage réelle de PR(mm/r) dans les données nettoyées est d’environ **47.69 mm/r** (de 2.31 à 50.0). Une erreur normalisée $e$ correspond donc à ~ $e \times 47.69$ mm/r.

- RMSE test moyen: linéaire ≈ **0.715 mm/r** vs random forest ≈ **0.553 mm/r**
- MAE test moyen: linéaire ≈ **0.503 mm/r** vs random forest ≈ **0.223 mm/r**

## Conclusion
Si l’objectif est la meilleure performance prédictive sur PR(mm/r), la **Random Forest** est le meilleur choix sur ces splits. La **régression linéaire** reste très performante mais légèrement moins précise.

## Quel jeu de test utiliser ?
Recommandation pratique:
- Si tu veux un **test plus grand (20%)** tout en gardant les meilleures performances observées: utilise le split **0.70 / 0.10 / 0.20**.
- Si tu veux une **validation plus grande (20%)** pour régler des hyperparamètres: utilise le split **0.60 / 0.20 / 0.20**.

Le split **0.70 / 0.15 / 0.15** (15% test, **180 lignes**) est aussi valide, mais un peu moins robuste statistiquement.
