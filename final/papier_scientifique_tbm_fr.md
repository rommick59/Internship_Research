# Introduction

L'excavation mécanisée par tunnelier (Tunnel Boring Machine, TBM) est une solution standard pour les projets d'infrastructures souterraines (transports, réseaux d'eau, services). En fonctionnement, la tête de coupe interagit en continu avec des conditions géologiques variables, ce qui rend nécessaire un pilotage précis des paramètres opératoires.

Dans ce contexte, le taux de progression (PR, mm/r) est un indicateur clé : il mesure l'avancement par révolution de la tête de coupe et reflète la productivité, l'usure des outils, la consommation d'énergie et, indirectement, les coûts et la durée du projet. Améliorer la prédiction du PR permet d'affiner les réglages machine et d'identifier plus tôt des conditions défavorables.

![TBM en fonctionnement dans le tunnel](TBM.jpg)

Figure 1. Photographie du TBM en fonctionnement dans le tunnel.

![Schéma TBM et paramètres opératoires](TBM_density.jpg)

Figure 2. Schéma du TBM montrant les paramètres opératoires et l'interaction avec le massif.

![Schéma TBM et évacuation des déblais](téléchargé.jpg)

Figure 3. Schéma du TBM illustrant la zone d'excavation, l'évacuation des déblais et le contexte géologique.

La présente étude se situe à l'interface de la géomécanique appliquée et de l'apprentissage automatique. Les signaux disponibles — vitesse de rotation, poussée, couple, pressions au front et indices dérivés tels que l'énergie spécifique, FPI, TPI, etc. — caractérisent à la fois le comportement de la machine et la réponse du terrain. Les méthodes d'apprentissage automatique, en particulier les modèles basés sur les arbres et le boosting (Random Forest, Gradient Boosting, XGBoost) ainsi que les approches à noyau (SVR, RVM), sont mobilisées pour analyser ces signaux et saisir des relations non linéaires et des interactions complexes difficiles à décrire par des règles empiriques simples (Khatti & Mishra, 2025 ; Yagiz, 2008 ; Breiman, 2001 ; Friedman, 2001). Les jeux de données utilisés dans ce travail proviennent de l'étude en accès libre de Khatti et Mishra (2025), qui porte sur l'estimation du taux de pénétration des tunnelier en conditions de front mixte et examine la sélection de variables et les effets de multicolinéarité sur les modèles de machine learning et deep learning.

L'objectif principal de cet article est d'évaluer la capacité des modèles d'apprentissage automatique à prédire le taux de pénétration d'un TBM à partir de variables opérationnelles et dérivées, et d'identifier les prédicteurs les plus influents. Au-delà de la précision prédictive, l'étude examine également la cohérence des résultats entre différentes familles de modèles et met en évidence le rôle des indices dérivés dans l'interprétation des performances du TBM. Le reste de l'article est structuré comme suit : la Section 2 présente la préparation des données et la méthodologie, la Section 3 rapporte les résultats et la discussion, et la Section 4 conclut l'article.

Références citées dans cette introduction :
- Khatti, J., & Mishra, S. (2025). Estimating shield tunnel boring machine penetration rate in mixed face conditions: feature selection and multicollinearity effects on machine and deep learning models. Frontiers in Built Environment, 11, 1699466.
- Yagiz, S. (2008). Utilizing rock mass properties for predicting TBM performance in hard rock tunneling. Tunnelling and Underground Space Technology, 23(3), 326–337.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.

## 2 Méthodologie

### 2.1 Source des données et variables étudiées
L’analyse repose sur le jeu de données TBM nettoyé et réduit à un sous-ensemble de modélisation comprenant six variables explicatives et une variable cible. La cible est le taux de pénétration PR (mm/r). Le jeu final utilisé dans le pipeline fixe 80/20 retient : CRS (RPM), F/A(MF), T/D3(MT), UEP (MPa), LEP (MPa) et TPI. Ces variables ont été retenues car elles résument les principales conditions de fonctionnement de la tête de coupe, l’équilibre des pressions au front et l’effort d’excavation.

Le protocole de recherche conserve la même définition de la cible tout au long du prétraitement, de l’entraînement des modèles et de l’explicabilité, afin que les performances et les attributions SHAP soient calculées dans le même espace de modélisation.

### 2.2 Analyse exploratoire des données
Avant l’entraînement des modèles, le jeu nettoyé est exploré visuellement afin de vérifier les formes de distribution, détecter les valeurs extrêmes et inspecter les relations deux à deux. Le protocole EDA combine des histogrammes avec courbe KDE, des boxplots, des nuages de points cible-vs-variable, une carte de corrélation et des graphes KDE 2D. Cette étape exploratoire sert à identifier les variables asymétriques, les éventuels outliers, ainsi que les relations monotones ou non linéaires avec PR.

Dans le manuscrit, l’étape EDA est illustrée par une carte de corrélation résumant la structure de dépendance entre les variables sélectionnées.

![Carte de corrélation EDA](../AI8/images/heatmap_distance_correlation_ai7.png)

Figure 4. Carte de corrélation exploratoire utilisée pour inspecter la structure de dépendance entre PR et les principales variables opératoires.

### 2.3 Prétraitement et normalisation des données
Le prétraitement est organisé en deux étapes successives. Premièrement, le fichier TBM brut est nettoyé : le jeu de données est vérifié pour les valeurs manquantes et les doublons, les valeurs numériques manquantes sont imputées, les outliers sont retirés par la règle de l’écart interquartile (IQR), puis les colonnes numériques sont standardisées dans la passe de nettoyage qui produit `TBM_data_cleaned.csv`. Deuxièmement, le sous-ensemble destiné à la modélisation est construit à partir des six prédicteurs retenus plus PR, puis les données sont séparées en 80% d’entraînement et 20% de test avec `random_state=42`.

Pour le pipeline final 80/20, la normalisation est ajustée uniquement sur TRAIN puis appliquée à TEST afin d’éviter toute fuite d’information. Un scaler MinMax est utilisé, et le préprocesseur ajusté est enregistré avec les indices de découpage pour garantir la reproductibilité.

![Normalisation sur TRAIN uniquement](../AI8/images/schema_normalization_train_test_80_20.png)

Figure 5. Schéma de normalisation sur le split fixe 80/20. Le scaler est ajusté sur TRAIN puis réutilisé tel quel sur TEST.

### 2.4 Protocole d’apprentissage automatique
Plusieurs familles de régression sont évaluées sur le split fixe : régression linéaire, Random Forest, RVM, XGBoost et Gradient Boosting. La comparaison permet d’opposer un modèle linéaire de base à des apprentissages non linéaires à base d’arbres et de noyaux. L’évaluation est réalisée sur le même split 80/20 normalisé.

Les métriques principales de comparaison sont le coefficient de corrélation $r$, le coefficient de détermination $R^2$, l’erreur quadratique moyenne (MSE), la racine de l’erreur quadratique moyenne (RMSE), l’erreur absolue moyenne (MAE) et la variance expliquée (VAF). Le meilleur modèle sur l’ensemble de test est ensuite utilisé comme modèle de référence pour l’analyse SHAP.

### 2.5 Explicabilité basée sur SHAP
Pour interpréter le modèle Gradient Boosting retenu, SHAP (Shapley Additive Explanations) est calculé sur l’ensemble TEST normalisé via `shap.TreeExplainer`. SHAP fournit une décomposition additive de la prédiction en un terme de base et des contributions par variable. L’analyse globale utilise les valeurs absolues moyennes de SHAP pour classer les prédicteurs, tandis que les graphes beeswarm, decision, heatmap et waterfall sont présentés dans la section d’explicabilité dédiée pour examiner le signe, la dispersion et les contributions locales.

L’analyse SHAP est réalisée dans le même espace normalisé que le modèle, ce qui garantit la cohérence entre la fonction apprise et les attributions reportées. Comme TPI est un indice dérivé opérationnel potentiellement proche conceptuellement de la cible, son importance doit être interprétée avec prudence et discutée comme une source possible de circularité.

### 2.6 Reproductibilité et implémentation
Toutes les étapes de prétraitement, de modélisation et d’explicabilité sont implémentées sous forme de scripts autonomes et d’artefacts sauvegardés dans le dépôt. Les indices de split, les objets de prétraitement ajustés, les sorties des modèles et les tableaux SHAP sont enregistrés afin de rendre le protocole reproductible. Cette organisation permet de rejouer le même pipeline sur le jeu nettoyé, le split fixe 80/20 et l’étape SHAP sans modifier les règles de sélection des données.

