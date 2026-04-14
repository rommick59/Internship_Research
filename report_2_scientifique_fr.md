# Rapport 2 : Prétraitement des données et Analyse exploratoire

## Introduction
Ce rapport présente en détail les étapes de prétraitement des données et l’analyse exploratoire réalisées dans le cadre de l’étude sur la prédiction du taux de progression (PR) des tunneliers. L’objectif est de préparer un jeu de données fiable et d’explorer les relations entre le PR et les paramètres influents, afin de mieux comprendre les facteurs déterminants pour la modélisation future.

## Prétraitement des données
La première étape a consisté à examiner le jeu de données afin d’identifier et de traiter les valeurs manquantes. Les valeurs absentes ont été soit supprimées lorsqu’elles étaient peu nombreuses, soit imputées à l’aide de la moyenne ou de la médiane pour préserver la cohérence du jeu de données. Les valeurs aberrantes ont été détectées à l’aide de boîtes à moustaches (boxplots) et de méthodes statistiques comme l’écart interquartile. Ces valeurs extrêmes ont été retirées pour éviter qu’elles n’influencent négativement les analyses ultérieures. La normalisation des variables a ensuite permis d’harmoniser les échelles des différents paramètres, facilitant ainsi la comparaison et l’interprétation des résultats.

### Boîtes à moustaches des variables
Les boxplots ci-dessous illustrent la distribution et la présence de valeurs extrêmes pour chaque variable :
![Boîtes à moustaches des variables](Box_plots.png)
*Figure 1 : Boîtes à moustaches montrant la dispersion et les valeurs extrêmes de chaque paramètre.*
Chaque boîte à moustaches représente la répartition des valeurs pour une variable donnée. On observe que certaines variables, comme SE (énergie spécifique), FPI et TPI, présentent de nombreux points au-delà des moustaches, indiquant la présence de valeurs extrêmes (outliers). Cela justifie leur traitement lors du prétraitement. D’autres variables, comme CRS (RPM) ou AR (mm/min), montrent une distribution plus resserrée, traduisant une variabilité plus faible. L’analyse de ces boxplots permet d’identifier rapidement les variables nécessitant une attention particulière pour garantir la robustesse des analyses ultérieures.

Par exemple, les variables SE (énergie spécifique), FPI et TPI présentent de nombreux points au-delà des moustaches, ce qui indique la présence de valeurs extrêmes (outliers) susceptibles de biaiser les analyses statistiques. Ces variables requièrent donc un traitement spécifique, tel que la suppression ou l’imputation des valeurs aberrantes, afin d’assurer la fiabilité des résultats. À l’inverse, des variables comme CRS (RPM) ou AR (mm/min) montrent une distribution plus homogène et nécessitent moins d’ajustements. Ainsi, l’analyse des boxplots permet de cibler précisément les variables à surveiller et à corriger lors du prétraitement.

## Analyse exploratoire des données

### Histogrammes des variables
Les histogrammes suivants montrent la distribution de chaque variable, mettant en évidence l’asymétrie et la présence de valeurs extrêmes :
![Histogrammes des variables](Histo.png)
*Figure 2 : Histogrammes montrant la distribution de chaque paramètre.*
Les histogrammes révèlent que la plupart des variables, comme PR, SE, FPI et TPI, présentent une distribution asymétrique avec une majorité de valeurs faibles et quelques valeurs élevées. Par exemple, la variable AR (avance) montre une concentration de valeurs autour de 10 à 20 mm/min, tandis que SE et FPI affichent une longue traîne vers la droite, signe de quelques valeurs très élevées. Cette asymétrie indique que le comportement de la machine est dominé par des conditions « normales » mais que des situations extrêmes existent et peuvent influencer la performance globale.

### Nuages de points : PR vs paramètres
Les nuages de points ci-dessous illustrent les relations entre le taux de progression (PR) et les principaux paramètres :
![Nuages de points : PR vs paramètres](scatter_plot.png)
*Figure 3 : Nuages de points montrant la relation entre le PR et chaque paramètre.*
Chaque graphique met en relation le PR avec un paramètre spécifique. On remarque une forte corrélation positive entre AR (avance) et PR : plus l’avance est élevée, plus le taux de progression augmente, ce qui est logique mécaniquement. À l’inverse, pour SE, FPI et TPI, la relation est négative : lorsque ces indices augmentent, le PR diminue, traduisant une difficulté accrue d’avancement. Les autres paramètres, comme CRS (RPM) ou F/A (MF), montrent des relations plus diffuses, suggérant une influence moins directe sur le PR.

### Cartes de densité : PR vs paramètres
Les cartes de densité apportent un éclairage supplémentaire sur la concentration des données et la nature des relations :
![Cartes de densité : PR vs paramètres](2DKDE.png)
*Figure 4 : Cartes de densité pour le PR et chaque paramètre.*
Les cartes de densité permettent de visualiser les zones où les observations sont les plus fréquentes. Pour AR vs PR, la densité maximale se situe dans la zone de forte avance et de haut PR, confirmant la relation positive. Pour SE, FPI et TPI, la densité se concentre dans les zones de faible PR et de valeurs élevées de ces indices, ce qui confirme la difficulté d’avancement dans ces conditions. Ces visualisations renforcent les conclusions tirées des nuages de points.

### Matrice de corrélation (Spearman)
La matrice de corrélation de Spearman ci-dessous synthétise la force et le sens des relations entre toutes les variables :
![Matrice de corrélation de Spearman](Heatmap.png)
*Figure 5 : Matrice de corrélation de Spearman pour l’ensemble des variables.*
La matrice de corrélation met en évidence les liens entre toutes les variables. On observe que AR (mm/min) est très fortement corrélé positivement au PR (0,98), ce qui en fait le paramètre le plus déterminant. À l’inverse, SE, FPI et TPI présentent des corrélations négatives très marquées (environ -0,85), confirmant leur rôle d’indicateurs de difficulté. CRS (RPM) a une corrélation négative modérée avec le PR (-0,57). Les autres paramètres montrent des corrélations plus faibles, ce qui suggère qu’ils jouent un rôle secondaire dans la prédiction du PR.

## Interprétation et discussion
Ces résultats confirment l’importance de certains paramètres mécaniques et énergétiques dans la progression du tunnelier. L’avance (AR) apparaît comme le facteur principal, tandis que l’augmentation de l’énergie spécifique ou des indices de performance traduit une difficulté accrue d’avancement. Les visualisations et l’analyse statistique permettent ainsi de sélectionner les variables les plus pertinentes pour la modélisation prédictive à venir.

## Conclusion
Le prétraitement et l’analyse exploratoire ont permis de fiabiliser le jeu de données et de mieux comprendre la structure des relations entre les variables. Ces étapes constituent une base solide pour le développement de modèles de prédiction du taux de progression cependant le fait qu'il y a des outliners dans le jeu de données nécessite une attention particulière lors de la modélisation et donc on devra regarde les impacts de ces valeurs extrêmes sur les résultats du modèle (pour les garder ou non).
