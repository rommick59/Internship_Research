# Predire le taux de progression d'un tunnelier a partir de donnees operationnelles et geotechniques : comparaison de modeles de machine learning et explicabilite

## Resume
La prediction du taux de progression (PR, mm/r) des tunneliers (TBM) est un probleme de genie civil a fort impact pratique, car elle influence directement la planification, le cout et la securite des chantiers souterrains. Ce travail propose une etude complete allant du pretraitement des donnees a l'analyse exploratoire, puis a la comparaison de plusieurs modeles supervises de machine learning et a leur explicabilite. Le jeu de donnees regroupe des variables operationnelles et derivees telles que CRS (RPM), F/A(MF), T/D3(MT), UEP (MPa), LEP (MPa), SE, FPI, TPI et la cible PR. Les resultats exploratoires montrent une relation positive tres forte entre AR et PR, ainsi que des relations negatives marquees entre PR et les indices de difficulte tels que SE, FPI et TPI. Sur les essais de modelisation, les familles RVM, Gradient Boosting, XGBoost, Random Forest, SVR et Linear Regression ont ete comparees via des decoupes train/validation/test multiples. Le RVM fournit les meilleures performances globales sur plusieurs splits, tandis que le Gradient Boosting est le meilleur sur le split 80/20 utilise pour l'analyse d'explicabilite. Les analyses SHAP, PDP et ICE montrent que TPI domine largement les decisions du Gradient Boosting, avec un effet moyen negatif sur la prediction, alors que T/D3(MT) joue un role secondaire positif. L'ensemble du travail montre qu'un pipeline base sur des donnees operationnelles bien pretraitees peut fournir une prediction fiable du PR, tout en restant interpretable.

## Mots-cles
TBM, taux de progression, machine learning, Gradient Boosting, RVM, SHAP, PDP, ICE, analyse exploratoire.

## 1. Introduction
L'excavation mecanisee par tunnelier (Tunnel Boring Machine, TBM) constitue une solution majeure pour la realisation de tunnels de transport, de reseaux d'assainissement et d'ouvrages hydrauliques. Dans ces chantiers, le taux de progression (PR) est une variable cle car il conditionne le rendement journalier, la consommation energetique, l'usure des outils et, au final, le cout global du projet. La capacite a prevoir ce taux avant ou pendant l'excavation represente donc un avantage decisif pour l'ingenierie des travaux souterrains.

La litterature montre que la performance d'un TBM depend d'un ensemble complexe de variables geotechniques et operationnelles : resistance de la roche, qualite du massif, poussée, couple, vitesse de rotation, pressions au front et indicateurs derives de difficulte. Les travaux classiques sur la performance des TBM ont etabli tres tot que la prediction du comportement d'excavation ne peut pas se reduire a une seule variable geologique [1]. Plus recemment, les methodes de machine learning se sont imposees comme des alternatives pertinentes aux approches empiriques, car elles peuvent capturer des relations non lineaires et des interactions entre variables, tout en ameliorant la precision predictive [2-5].

Le plan de ce papier suit la logique du schema de travail fourni : resume, introduction, methode, resultats et discussion, conclusion, references. L'objectif scientifique est double. D'une part, il s'agit d'evaluer la capacite de plusieurs modeles supervises a predire le PR a partir des donnees disponibles. D'autre part, il s'agit de comprendre quelles variables gouvernent les predictions du meilleur modele a l'aide d'outils d'explicabilite tels que SHAP, PDP et ICE.

### 1.1. Contexte physique et variables pertinentes
Les variables explicatives du jeu de donnees reflètent les conditions operationnelles du tunnelier. CRS correspond a la vitesse de rotation de la tete de coupe, F/A(MF) a la poussee moyenne, T/D3(MT) au couple normalise, UEP et LEP aux pressions au front, SE a l'energie specifique, FPI et TPI a des indices de penetration ou d'effort. Sur le plan mecanique, un PR eleve est attendu lorsque l'interaction entre poussée, couple et rotation est efficace, alors qu'une augmentation de SE, FPI ou TPI traduit generalement une plus grande difficulte d'excavation.

### 1.2. Appui sur la litterature scientifique
Les approches de prediction du PR s'inscrivent dans une ligne de recherche plus large en geomecanique et en apprentissage automatique. Les modeles d'arbres et de boosting ont gagne en popularite grace a leur performance sur donnees tabulaires [2,3]. Les modeles a noyau, tels que SVR ou RVM, restent attractifs pour des jeux de donnees compacts et fortement structurees [4,5]. Enfin, les methodes d'explicabilite comme SHAP rendent les predictions plus auditables et facilitent la lecture scientifique des resultats [6].

## 2. Materiels et methodes

### 2.1. Jeu de donnees et cible
L'etude s'appuie sur le jeu de donnees TBM nettoye et pretraite dans le dossier Internship_Research. La variable cible est le PR (mm/r), c'est-a-dire le taux de progression par revolution. Les variables explicatives principales sont : CRS (RPM), AR (mm/min), F/A(MF), T/D3(MT), UEP (MPa), LEP (MPa), SE, FPI et TPI.

Les rapports precedents du projet ont permis d'etablir une base robuste pour la suite de l'analyse : le premier rapport a synthese la litterature et identifie les variables les plus pertinentes, tandis que le second a detaille le pretraitement et l'analyse exploratoire [voir rapports internes du projet].

### 2.2. Pretraitement
Le pretraitement a comporte plusieurs etapes :
- traitement des valeurs manquantes par suppression ou imputation selon leur frequence ;
- detection et gestion des valeurs aberrantes via boxplots et ecart interquartile ;
- normalisation des variables pour homogeniser les echelles et faciliter la comparaison inter-modeles.

Cette etape est importante car les modeles de regression peuvent etre sensibles aux echelles des variables, surtout pour les methodes a noyau et les approches lineaires regularisees. Elle permet aussi de limiter l'impact de valeurs extrêmes sur les metriques d'erreur.

### 2.3. Analyse exploratoire des donnees
L'analyse exploratoire a ete conduite a l'aide de boxplots, histogrammes, nuages de points, cartes de densite 2D et matrice de correlation de Spearman. Les figures suivantes illustrent les principaux motifs observes.

![Boites a moustaches des variables](../Internship_Research/images/Box_plots_p1.png)
![Boites a moustaches des variables](../Internship_Research/images/Box_plots_p2.png)

Figure 1. Boites a moustaches des variables du jeu de donnees.

![Histogrammes des variables](../Internship_Research/images/Histo_p1.png)
![Histogrammes des variables](../Internship_Research/images/Histo_p2.png)
![Histogrammes des variables](../Internship_Research/images/Histo_p3.png)

Figure 2. Histogrammes des variables operationnelles et derivees.

![Nuages de points PR vs variables](../Internship_Research/images/scatter_plot_p1.png)
![Nuages de points PR vs variables](../Internship_Research/images/scatter_plot_p2.png)
![Nuages de points PR vs variables](../Internship_Research/images/scatter_plot_p3.png)

Figure 3. Nuages de points entre PR et les variables explicatives.

![Carte de correlation de Spearman](../Internship_Research/images/Heatmap2.png)

Figure 4. Matrice de correlation de Spearman.

Les cartes de densite 2D confirment les tendances observees dans les nuages de points, notamment la relation positive forte entre AR et PR ainsi que les relations negatives entre PR et les indices de difficulte. Les variables SE, FPI et TPI presentent des distributions asymetriques et des queues droites importantes, ce qui suggere la presence de situations operationnelles plus difficiles mais plus rares.

### 2.4. Modeles compares
Plusieurs familles de modeles supervises ont ete evaluees dans le projet :
- Regression lineaire comme baseline interpretable ;
- Random Forest comme ensemble d'arbres robuste ;
- SVR comme modele a noyau ;
- RVM comme approche bayesienne parcimonieuse ;
- Gradient Boosting comme modele d'arbres sequentiel ;
- XGBoost comme version optimisee du boosting ;
- AdaBoost comme autre methode d'ensemble.

Les performances ont ete mesurees a l'aide de R, R2, MSE, RMSE et MAE sur differents splits train/validation/test. Les principaux resultats de synthese du dossier interne montrent que le RVM est globalement le meilleur modele sur plusieurs decoupes, tandis que le Gradient Boosting devient la meilleure option sur le split 80/20 retenu pour l'analyse explicable [voir synthese interne de comparaison].

### 2.5. Explicabilite
L'explicabilite du meilleur modele sur le split 80/20 a ete analysee via SHAP, PDP et ICE. SHAP decompose la prediction sous la forme :

$$f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j$$

ou $\phi_0$ est la valeur de base et $\phi_j$ la contribution de la variable $j$. Les PDP donnent l'effet moyen d'une variable sur la prediction, tandis que les courbes ICE montrent la variabilite de cet effet d'un individu a l'autre.

## 3. Resultats

### 3.1. Resultats exploratoires
L'analyse exploratoire met en evidence plusieurs faits robustes :
- AR et PR sont fortement et positivement correles ;
- SE, FPI et TPI sont fortement et negativement lies a PR ;
- CRS, F/A(MF), UEP et LEP apportent un signal plus faible lorsqu'elles sont considerees seules ;
- certaines variables sont fortement collineaires entre elles, ce qui justifie une modelisation prudente et une interpretation orientee vers la robustesse.

Ces motifs sont coherents avec l'interpretation physique du processus d'excavation : plus le terrain est difficile ou plus l'effort specifique augmente, plus le PR diminue.

### 3.2. Comparaison des modeles
La synthese des essais de modelisation montre que les performances varient selon le split, mais plusieurs tendances sont nettes. Sur les splits 70/10/20 et 60/20/20, le RVM produit les meilleurs scores de test avec des erreurs tres faibles et des coefficients de determination proches de 1. Sur le split 80/20, le Gradient Boosting devient le meilleur modele selon les metriques de test, avec par exemple un R2 de 0.9317, un RMSE de 0.0621 et un MAE de 0.0486 en echelle normalisee.

Le tableau ci-dessous resume la logique generale issue des fichiers de ranking et de comparaison internes :

| Famille de modele | Comportement global |
|---|---|
| Regression lineaire | Baseline solide, mais moins precise que les modeles non lineaires |
| Random Forest | Bonne robustesse, bonnes performances globales |
| SVR | Performant sur certains splits, surtout avec parametrage adapte |
| RVM | Meilleur compromis global sur plusieurs tests |
| Gradient Boosting / XGBoost | Tres competitifs sur donnees tabulaires |
| AdaBoost | Inferieur aux autres methodes dans ces essais |

Dans le detail, la comparaison du split 0.70/0.10/0.20 montre que le RVM atteint R2 = 0.999962 sur le test, avec RMSE = 0.001458 et MAE = 0.000486 en echelle normalisee. Cette precision tres elevee suggere que le probleme est fortement structure par les variables disponibles. Toutefois, elle impose aussi de rester vigilant quant au risque de fuite d'information ou de redondance entre variables derivees et cible.

### 3.3. Explicabilite du Gradient Boosting
Le modele Gradient Boosting sur le split 80/20 a ete choisi pour l'analyse SHAP, PDP et ICE. Les resultats internes montrent que TPI domine largement l'importance moyenne absolue, suivi de T/D3(MT), tandis que les autres variables contribuent tres peu dans ce cadre.

![Importance globale SHAP](../Internship_Research/AI9_SHAP_GB_80_20/images/shap_summary_bar_test.png)

Figure 5. Importance globale SHAP sur l'ensemble de test.

Le resume quantitatif du dossier d'explicabilite indique que TPI porte environ 89.56 % de l'importance, contre 10.34 % pour T/D3(MT), et des contributions quasi nulles pour UEP, CRS, F/A(MF) et LEP. Le beeswarm SHAP confirme la directionnalite : des valeurs elevees de TPI tendent a diminuer la prediction, alors que des valeurs elevees de T/D3(MT) tendent a l'augmenter.

Les PDP et ICE apportent la meme lecture a l'echelle marginale et individuelle. Pour TPI, le PDP montre un effet moyen dominant, tandis que les courbes ICE restent relativement homogenes pour cette variable, ce qui signifie que son effet est stable sur la plupart des observations. A l'inverse, T/D3(MT) presente une heterogeneite plus forte entre individus, suggeree par une dispersion ICE plus importante.

![PDP TPI](../Internship_Research/AI10/images/pdp_TPI.png)
![ICE TPI](../Internship_Research/AI10/images/ice_TPI.png)

Figure 6. Effet moyen et effet individuel de TPI dans l'analyse PDP/ICE.

Ces resultats sont importants d'un point de vue metier. Ils signifient que le modele apprend principalement une relation monotone entre un indice de difficulte et le PR, avec un effet secondaire de couple normalise. Ils montrent aussi que certaines variables, bien qu'utiles physiquement pour la stabilite du front, ont peu d'influence sur la prediction du PR dans ce modele particulier.

## 4. Discussion
Les resultats experimentaux confirment que la prediction du PR depend d'un petit nombre de variables dominantes. La relation positive entre AR et PR et la relation negative entre les indices de difficulte et PR sont cohérentes avec la physique du tunnelier. D'un point de vue algorithmique, cela explique pourquoi les modeles non lineaires, en particulier les methodes a noyau et le boosting, surpassent une regression lineaire simple.

Un point critique concerne TPI. Comme cet indice peut etre defini a partir de quantites deja proches de la penetration, il existe un risque de circularite ou de quasi-fuite d'information si sa construction incorpore implicitement la cible. Cela ne remet pas en cause sa valeur predictive, mais impose une interpretation prudente : une forte importance SHAP ne signifie pas automatiquement causalite physique. Dans une version future du travail, il serait utile de refaire les essais sans TPI pour mesurer la robustesse du signal portee par les autres variables.

Par ailleurs, les variables UEP et LEP ont une influence faible sur PR dans les modeles testes, mais elles restent indispensables pour la securite et la stabilite du front. Il s'agit donc de variables a forte importance operationnelle, meme si leur apport predictif direct est limite.

Enfin, le fait que plusieurs modeles atteignent des performances tres elevees montre que la structure du jeu de donnees est favorable a l'apprentissage, mais invite a verifier soigneusement l'independance des splits, la definition exacte des variables derivees et l'absence de contamination entre train et test.

## 5. Conclusion
Cette etude montre qu'il est possible de predire le taux de progression d'un tunnelier avec une grande precision a partir de donnees operationnelles et derivees, a condition de disposer d'un pretraitement rigoureux et d'un choix approprie du modele. Sur le plan compare, le RVM donne les meilleures performances globales sur plusieurs splits, tandis que le Gradient Boosting est le meilleur modele sur le split 80/20 retenu pour l'explicabilite. Les analyses SHAP, PDP et ICE montrent que TPI est la variable dominante du modele explicable, suivie par T/D3(MT).

Au niveau scientifique, le travail confirme trois points essentiels : la forte structure du probleme, l'interet des modeles non lineaires pour les donnees tabulaires et l'importance de l'explicabilite pour interpreter des performances predictives tres elevees. Au niveau pratique, il fournit une base solide pour developper un outil d'aide a la decision pour le pilotage des TBM.

## References
[1] Yagiz, S. (2008). Utilizing rock mass properties for predicting TBM performance in hard rock tunneling. Tunnelling and Underground Space Technology, 23(3), 326-337.

[2] Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.

[3] Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189-1232.

[4] Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector regression. Statistics and Computing, 14, 199-222.

[5] Tipping, M. E. (2001). Sparse Bayesian learning and the relevance vector machine. Journal of Machine Learning Research, 1, 211-244.

[6] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.

[7] Bieniawski, Z. T. (1989). Engineering rock mass classifications. Wiley.

---

Note interne : les figures referencees ci-dessus proviennent des analyses deja generees dans le dossier Internship_Research. Si tu veux, je peux ensuite convertir ce manuscrit en version Word ou PDF, ou encore remplacer les figures par une maquette plus academique dans le dossier final.
