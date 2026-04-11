# Rapport 2 : Prétraitement des données et Analyse exploratoire

## 1. Prétraitement des données

Avant toute analyse exploratoire ou modélisation, le jeu de données brut a fait l’objet d’un prétraitement rigoureux afin de garantir la qualité et la fiabilité des données. Les étapes suivantes ont été appliquées :

### 1.1 Gestion des valeurs manquantes
Toutes les lignes contenant au moins une valeur manquante ont été supprimées. Cette approche, bien qu’elle puisse réduire la taille du jeu de données, permet d’éviter l’introduction de biais liés à l’imputation et garantit que les analyses ultérieures reposent uniquement sur des cas complets. Cela est particulièrement important pour les modèles de machine learning sensibles aux données manquantes.

### 1.2 Suppression des outliers
Les valeurs aberrantes ont été détectées et éliminées à l’aide de la méthode de l’écart interquartile (IQR), appliquée à toutes les colonnes numériques. Pour chaque variable, les valeurs situées en dehors de 1,5 fois l’IQR à partir des premier et troisième quartiles ont été considérées comme des outliers et supprimées. Cette étape vise à réduire l’influence des valeurs extrêmes, qui peuvent fausser les analyses statistiques et dégrader la performance des modèles.

### 1.3 Normalisation des données
Les variables numériques restantes ont été standardisées à l’aide de la méthode StandardScaler, de sorte que chaque variable ait une moyenne nulle et un écart-type égal à un. Cette normalisation est essentielle pour les algorithmes sensibles à l’échelle des données, notamment ceux basés sur des mesures de distance ou la régularisation.

### 1.4 Export du jeu de données nettoyé
Les données prétraitées ont été sauvegardées dans un nouveau fichier, garantissant la reproductibilité et la traçabilité pour les étapes d’analyse suivantes. Ce jeu de données propre constitue la base de tout le travail exploratoire et de modélisation à venir.

Grâce à ce pipeline de prétraitement, le jeu de données est désormais propre, cohérent et prêt pour une analyse exploratoire et un développement de modèles robustes.

### 1.5 Etude d'impact des outliers sur la prediction de PR
Afin de verifier si les valeurs extremes sont nuisibles ou informatives pour la prediction de PR, une etude comparative a ete realisee sur trois variantes du jeu de donnees :

- D0 : donnees nettoyees sans suppression des outliers (1197 lignes)
- D1 : suppression des outliers par methode IQR (1071 lignes, 126 lignes retirees)
- D2 : winsorization 1%-99% (1197 lignes)

Trois modeles ont ete evalues de facon identique sur D0, D1 et D2 : Linear Regression, Huber Regressor et Random Forest.
Les metriques retenues sont MAE, RMSE et R2 en validation croisee ainsi qu'en test.

Les resultats montrent que Random Forest est le meilleur modele sur les trois variantes. En validation croisee, D0 presente le meilleur compromis global (RMSE moyen le plus bas : 0.596). En test, D1 obtient le RMSE le plus faible (0.410), ce qui suggere que la suppression IQR peut ameliorer la prediction sur des echantillons moins extremes.

Ces observations indiquent que les outliers n'ont pas un effet uniquement negatif :

- ils degradent certains modeles lineaires,
- mais peuvent aussi contenir une information utile,
- et leur suppression peut modifier la distribution du probleme.

Sur le plan methodologique, la decision finale retenue pour la suite du projet est d'utiliser D0 (donnees avec outliers conserves) comme jeu de donnees unique de reference. Ce choix est justifie par sa meilleure performance globale en validation croisee, sa taille d'echantillon plus elevee et la preservation d'informations potentiellement pertinentes pour la prediction de PR.

Les tableaux de resultats utilises pour cette analyse sont disponibles dans le dossier outlier_study : dataset_overview.csv, model_comparison_cv_test.csv et best_model_per_dataset.csv.

## 2. Analyse exploratoire des donnees (EDA)

### 2.1 Objectif et demarche
Conformement au planning de la semaine 4, l'EDA a ete conduite pour caracteriser la structure statistique du jeu de donnees retenu (D0), identifier les relations entre PR et les variables explicatives, et orienter le choix des modeles de regression. Les analyses suivantes ont ete realisees :

- visualisation univariee (histogrammes),
- inspection des valeurs extremes (boxplots),
- visualisation bivariée PR vs parametres (nuages de points),
- analyse de correlation (matrice de Spearman),
- preparation du protocole de separation train/test.

### 2.2 Distributions univariees
Les histogrammes montrent des distributions heterogenes selon les variables :

- CRS et AR presentent des concentrations sur certains intervalles operationnels,
- F/A, T/D3, UEP et LEP montrent des distributions plus etalees,
- SE, FPI et TPI sont asymetriques a droite, avec des queues longues,
- PR presente une distribution non gaussienne, avec une concentration sur des valeurs faibles a intermediaires et quelques valeurs elevees.

Ces observations confirment une structure non lineaire du probleme et justifient l'usage d'indicateurs robustes en complement des statistiques classiques.

### 2.3 Boxplots et interpretation des valeurs extremes
Les boxplots montrent la presence de points extremes, en particulier sur SE, FPI et TPI. Du point de vue geotechnique et operationnel, ces points peuvent correspondre a des conditions de creusement difficiles (energie specifique elevee, penetration plus faible), et non a des erreurs systematiques.

En coherence avec l'etude d'impact (Section 1.5), ces observations ont conduit a conserver D0 comme jeu de reference, afin de ne pas eliminer des regimes operationnels potentiellement informatifs pour la prediction de PR.

### 2.4 Relations PR vs variables explicatives
Les nuages de points PR vs parametres mettent en evidence :

- une relation positive forte entre AR et PR,
- des relations negatives marquees entre PR et SE/FPI/TPI,
- des relations plus complexes et non lineaires pour F/A, T/D3, UEP et LEP,
- une tendance globale negative entre CRS et PR sur la plage observee.

La structure en nuage et les motifs non lineaires indiquent qu'un modele purement lineaire risque de ne pas capturer toute la dynamique du systeme.

### 2.5 Analyse de correlation (Spearman)
La matrice de Spearman confirme les tendances visuelles et quantifie les dependances monotones :

- corr(AR, PR) = 0.98 (positive tres forte),
- corr(CRS, PR) = -0.57 (negative moderee a forte),
- corr(SE, PR) = -0.85, corr(FPI, PR) = -0.86, corr(TPI, PR) = -0.85,
- corr(F/A, PR) = -0.26, corr(T/D3, PR) = -0.36,
- corr(UEP, PR) = -0.04, corr(LEP, PR) = -0.13.

Des correlations elevees entre variables explicatives sont egalement observees (par exemple SE-FPI-TPI et UEP-LEP), suggerant la presence de multicolinearite partielle. Cette propriete a ete prise en compte dans la selection des modeles (Section 1.5).

### 2.6 Separation train/test
Pour preparer la phase de modelisation, les donnees ont ete separees en ensembles d'entrainement et de test selon un schema hold-out (80/20), complete par une validation croisee a 5 plis sur l'ensemble d'entrainement. Cette strategie permet :

- une estimation robuste de la generalisation,
- une comparaison equitable des approches de regression,
- une limitation du risque de surajustement.

### 2.7 Synthese EDA
L'EDA confirme que la prediction de PR releve d'un probleme de regression non lineaire avec interactions entre variables et presence de regimes extremes informatifs. Les analyses de semaine 4 soutiennent donc les choix suivants pour la suite :

- conservation de D0 comme dataset de reference,
- utilisation prioritaire de modeles robustes/non lineaires,
- interpretation des performances a la fois par metriques globales et par plausibilite physique.

## 3. Figures EDA

### Figure 1. Distributions univariees (histogrammes)
![Figure 1 - Histogrammes des variables](Histo.png)

### Figure 2. Detection visuelle des valeurs extremes (boxplots)
![Figure 2 - Boxplots](Box_plots.png)

### Figure 3. Relations bivariées PR vs variables explicatives (scatter plots)
![Figure 3 - Scatter plots PR vs parametres](scatter_plot.png)

### Figure 4. Matrice de correlation de Spearman
![Figure 4 - Heatmap de correlation Spearman](Heatmap.png)

### Figure 5. Densite jointe 2D (Kernel Density Estimation)
![Figure 5 - 2D KDE](2DKDE.png)
