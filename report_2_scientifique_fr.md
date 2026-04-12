# Rapport 2 : Prétraitement des données et Analyse exploratoire

## 1. Prétraitement des données

Avant toute analyse exploratoire ou modélisation, le jeu de données brut a fait l’objet d’un prétraitement rigoureux afin de garantir la qualité et la fiabilité des données. Les étapes suivantes ont été appliquées :

### 1.1 Gestion des valeurs manquantes
Toutes les lignes contenant au moins une valeur manquante ont été supprimées. Cette approche, bien qu’elle puisse réduire la taille du jeu de données, permet d’éviter l’introduction de biais liés à l’imputation et garantit que les analyses ultérieures reposent uniquement sur des cas complets. Cela est particulièrement important pour les modèles de machine learning sensibles aux données manquantes.

### 1.2 Traitement des outliers (strategie comparee)
Le traitement des valeurs aberrantes n'a pas ete impose de maniere unique au depart. Une strategie comparative a ete adoptee pour evaluer l'impact reel des outliers sur la prediction de PR :

- D0 : conservation des outliers,
- D1 : suppression des outliers via la methode IQR,
- D2 : capping des extremes par winsorization (1%-99%).

Cette approche permet de decider sur des bases quantitatives (performance predictive) et non sur une hypothese a priori.

### 1.3 Normalisation des données
Les variables numériques restantes ont été standardisées à l’aide de la méthode StandardScaler, de sorte que chaque variable ait une moyenne nulle et un écart-type égal à un. Cette normalisation est essentielle pour les algorithmes sensibles à l’échelle des données, notamment ceux basés sur des mesures de distance ou la régularisation.

### 1.4 Export des jeux de donnees preprocesses
Les donnees preprocesses ont ete sauvegardees afin de garantir la reproductibilite et la tracabilite des analyses ulterieures :

- jeu de reference pour EDA et modelisation: D0,
- jeux comparatifs pour l'etude d'impact: D1 et D2.

Les tableaux de comparaison associes sont stockes dans le dossier outlier_study.

Grace a ce pipeline, le jeu de donnees de reference (D0) est pret pour l'analyse exploratoire et la modelisation, tandis que D1 et D2 servent de bases de comparaison methodologique.

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

### 1.6 Resultats visuels de l'etude d'impact des outliers

#### 1.6.1 Structure des jeux de donnees apres traitement des outliers
![Apercu des jeux de donnees](outlier_study/dataset_overview_table.png)

Interpretation des donnees du tableau :

- `Rows` represente le nombre d'observations disponibles pour l'apprentissage.
- `Rows Removed vs D0` quantifie la perte d'information par rapport au jeu complet de reference.

Le tableau montre que D1 retire 126 observations (environ 10,5% de D0), alors que D0 et D2 conservent les 1197 lignes. Cette reduction peut diminuer la robustesse des modeles et la representation des regimes rares.

#### 1.6.2 Tableaux de comparaison des modeles
![Comparaison des modeles (CV + Test)](outlier_study/model_comparison_table.png)

![Meilleur modele par jeu de donnees](outlier_study/best_model_per_dataset_table.png)

Comment lire ces tableaux :

- `CV MAE (mean)` et `CV RMSE (mean)` mesurent l'erreur moyenne en validation croisee (plus faible = meilleur).
- `CV R2 (mean)` mesure la variance expliquee en validation croisee (plus eleve = meilleur).
- `Test MAE`, `Test RMSE` et `Test R2` sont calcules sur le split test hold-out.

Resultat principal : RandomForest est le meilleur modele sur les trois jeux (D0, D1, D2), ce qui confirme la pertinence d'une approche non lineaire pour la prediction de PR.

#### 1.6.3 Graphiques de scores avec valeurs affichees
![CV RMSE par jeu de donnees et modele](outlier_study/score_cv_rmse_bar.png)

Interpretation detaillee :

- Pour RandomForest, le CV RMSE est minimal sur D0 (0.596), puis D1 (0.601), puis D2 (0.622).
- Comme la validation croisee moyenne plusieurs partitions, ce critere est prioritaire pour la decision finale.
- Cela positionne D0 comme meilleur compromis precision/stabilite.

![Test RMSE par jeu de donnees et modele](outlier_study/score_test_rmse_bar.png)

Sur le test hold-out unique, D1 obtient le meilleur RMSE avec RandomForest (0.410 contre 0.468 pour D0 et 0.488 pour D2). Ce gain ponctuel montre que le filtrage IQR peut aider sur certains splits, sans inverser la tendance globale en validation croisee.

![CV MAE par jeu de donnees et modele](outlier_study/score_cv_mae_bar.png)

![Test MAE par jeu de donnees et modele](outlier_study/score_test_mae_bar.png)

Les tendances MAE confirment l'analyse RMSE :

- D0 est le meilleur en CV MAE pour RandomForest (0.263),
- D1 peut etre avantagé sur certaines metriques test pour des modeles lineaires/robustes,
- mais RandomForest demeure le meilleur en valeur absolue.

![CV R2 par jeu de donnees et modele](outlier_study/score_cv_r2_bar.png)

![Test R2 par jeu de donnees et modele](outlier_study/score_test_r2_bar.png)

Les scores R2 sont eleves dans tous les cas (environ 0.995-0.999). La decision doit donc s'appuyer prioritairement sur les erreurs (MAE/RMSE) et la conservation des donnees.

Conclusion interpretee :

- D0 est le meilleur globalement en validation croisee,
- D0 preserve toute la diversite des observations,
- D1 apporte un gain local sur le split test mais au prix d'une perte d'echantillons,
- D2 ne depasse pas D0.

Ainsi, D0 est retenu comme jeu de donnees de reference le plus robuste pour la suite du projet.

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

## 3. Interpretation detaillee des 4 graphiques EDA

### 3.1 Graphique 1 - Histogrammes des variables
![Graphique 1 - Histogrammes des variables](Histo.png)

**A quoi sert ce graphique**
Les histogrammes permettent de decrire la distribution de chaque variable (forme, asymetrie, concentration, dispersion), ce qui est essentiel pour choisir des modeles et verifier les hypotheses statistiques.

**Ce que l'on observe**
- Les variables SE, FPI et TPI sont fortement asymetriques a droite, avec une longue queue de valeurs elevees.
- PR n'est pas distribuee normalement; la masse principale se situe sur des valeurs faibles a moyennes.
- CRS et AR presentent des zones de concentration operationnelles (regimes de fonctionnement preferentiels).

**Conclusion scientifique**
La structure des distributions est non gaussienne et heterogene. Cela confirme qu'une modelisation strictement lineaire et basee sur des hypothese de normalite forte est insuffisante pour capturer la dynamique de PR.

### 3.2 Graphique 2 - Boxplots (dispersion et valeurs extremes)
![Graphique 2 - Boxplots](Box_plots.png)

**A quoi sert ce graphique**
Le boxplot resume mediane, quartiles, dispersion interquartile et valeurs extremes. Il sert a evaluer la variabilite des variables et a identifier les regimes potentiellement atypiques.

**Ce que l'on observe**
- SE, FPI et TPI montrent une dispersion importante et des observations extremes nombreuses.
- Les variables UEP et LEP restent plus concentrees, avec des plages plus bornees.
- PR presente une variabilite significative entre regimes faibles et eleves de penetration.

**Conclusion scientifique**
Les valeurs extremes observees ne doivent pas etre considerees automatiquement comme du bruit. Dans un contexte TBM, elles peuvent correspondre a des conditions geologiques difficiles ou a des etats operationnels critiques. Cette lecture est coherente avec la decision de conserver D0 comme reference.

### 3.3 Graphique 3 - Nuages de points PR vs predicteurs
![Graphique 3 - Scatter plots PR vs parametres](scatter_plot.png)

**A quoi sert ce graphique**
Les nuages de points permettent d'analyser la forme des relations entre PR (cible) et chaque variable explicative: linearite, non-linearite, saturation, heteroscedasticite et clusters operationnels.

**Ce que l'on observe**
- AR et PR montrent une relation croissante forte.
- SE, FPI et TPI sont associees a une baisse de PR (relation inverse marquee).
- Les relations F/A, T/D3, UEP et LEP avec PR sont non lineaires et presentent des regimes multiples.
- CRS montre une tendance globale negative vis-a-vis de PR dans le domaine observe.

**Conclusion scientifique**
Le comportement de PR depend de relations mixtes (lineaires et non lineaires) avec interactions implicites entre variables. Les modeles non lineaires (par exemple Random Forest) sont donc mieux adaptes que les modeles lineaires simples.

### 3.4 Graphique 4 - Matrice de correlation de Spearman
![Graphique 4 - Heatmap de correlation Spearman](Heatmap.png)

**A quoi sert ce graphique**
La heatmap de Spearman quantifie les dependances monotones entre variables, y compris lorsque la relation n'est pas strictement lineaire.

**Resultats principaux**
- corr(AR, PR) = 0.98: relation positive tres forte.
- corr(CRS, PR) = -0.57: relation negative moderee a forte.
- corr(SE, PR) = -0.85, corr(FPI, PR) = -0.86, corr(TPI, PR) = -0.85: relations negatives fortes.
- corr(F/A, PR) = -0.26 et corr(T/D3, PR) = -0.36: effets negatifs plus modérés.
- corr(UEP, PR) = -0.04 et corr(LEP, PR) = -0.13: effet direct faible sur PR.

**Conclusion scientifique**
La prediction de PR est principalement soutenue par AR, SE, FPI et TPI. La presence de fortes correlations entre variables explicatives (SE-FPI-TPI notamment) indique une multicolinearite partielle, ce qui renforce le choix de modeles robustes et de protocoles de validation stricts.

### 3.5 Conclusion generale de l'EDA (semaine 4)
L'analyse conjointe des 4 graphiques montre que PR est gouvernee par des mecanismes non lineaires, avec regimes operationnels contrastes et valeurs extremes informatives. Les implications directes pour la suite sont:

- conservation de D0 comme jeu de reference,
- priorite aux modeles robustes/non lineaires,
- validation par train/test et validation croisee pour garantir la generalisation.
