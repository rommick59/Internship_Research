# 1 Introduction

L'excavation mécanisée par tunnelier (Tunnel Boring Machine, TBM) est une technique essentielle pour la réalisation d'infrastructures souterraines (transports, eaux, services). Le taux de progression (PR, mm/r), qui mesure l'avancement par révolution de la tête de coupe, est un indicateur opérationnel central : il conditionne la productivité, l'usure des outils, la consommation d'énergie et, indirectement, les coûts et délais du projet. Prédire précisément le PR à partir de signaux machine et de variables dérivées permet d'anticiper les réglages opératoires, d'alerter sur des conditions défavorables et d'améliorer la planification des chantiers.

Le présent travail se situe à l'intersection de la géomécanique appliquée et de l'apprentissage automatique. Les signaux disponibles (vitesse de rotation, poussée, couple, pressions au front, indices dérivés tels que l'énergie spécifique, FPI, TPI, etc.) incarnent à la fois des informations physiques et des résumés statistiques des interactions outil–milieu. Les méthodes de machine learning — en particulier les modèles d'arbres et de boosting (Random Forest, Gradient Boosting, XGBoost) et les approches à noyau (SVR, RVM) — offrent la capacité de capturer des relations non linéaires et des interactions complexes entre ces variables, souvent difficiles à modéliser par des lois empiriques simples [1–4].


Le manuscrit s'appuie sur les rapports internes du projet et intègre les photos et figures générées (EDA, SHAP, PDP/ICE) pour illustrer et étayer l'analyse.

Outre la performance prédictive, l'explicabilité des modèles est fondamentale dans ce contexte industriel : comprendre quelles variables influencent les prédictions aide à détecter d'éventuelles fuites d'information (features dérivées proches de la cible), à valider la cohérence physique des résultats et à rendre les modèles utilisables par les décideurs. Pour cela, nous utilisons des méthodes d'interprétation locale et globale telles que SHAP (Shapley Additive exPlanations), PDP (Partial Dependence Plots) et ICE (Individual Conditional Expectation) afin d'expliquer les contributions des variables au niveau global et par observation [5].

L'objectif de l'étude est double : (i) comparer rigoureusement plusieurs familles de modèles supervisés pour la prédiction du PR sur un jeu de données TBM prétraité, et (ii) fournir une analyse d'explicabilité détaillée du modèle retenu pour en tirer des enseignements physiques et opérationnels. Plus précisément, nous évaluons la robustesse des modèles sur plusieurs protocoles de découpe train/validation/test, quantifions les performances à l'aide d'indicateurs standards (R, R2, RMSE, MAE) et analysons l'importance et l'effet des variables via SHAP, PDP et ICE.

Les contributions principales de cet article sont donc :
- une méthodologie de prétraitement adaptée aux données TBM et une sélection d'indices opérationnels pertinents ;
- une comparaison empirique de modèles de référence et de modèles avancés sur plusieurs protocoles de validation ;
- une étude d'explicabilité approfondie du modèle retenu, mettant en évidence les variables dominantes et discutant la robustesse des conclusions face à des indices dérivés potentiellement proches de la cible.

Organisation du manuscrit : la section 2 décrit les données, le prétraitement et les modèles testés ; la section 3 présente les résultats quantitatifs et l'analyse d'explicabilité ; la section 4 discute des limites (notamment le risque de circularité lié aux features dérivées) et des perspectives pratiques ; la section 5 conclut.

Références citées (exemples) :
[1] études sur l'application du boosting aux données industrielles ;
[2] travaux comparatifs sur Random Forest et XGBoost ;
[3] articles sur SVR/RVM pour la régression en géomécanique ;
[4] revues méthodologiques sur le prétraitement des données opérationnelles ;
[5] Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. NeurIPS.

# Prédire le taux de progression d'un tunnelier à partir de données opérationnelles et géotechniques : comparaison de modèles de machine learning et explicabilité

## Résumé
La prédiction du taux de progression (PR, mm/r) des tunneliers (TBM) est un problème de génie civil à fort impact pratique, car elle influence directement la planification, le coût et la sécurité des chantiers souterrains. Ce travail propose une étude complète allant du prétraitement des données à l'analyse exploratoire, puis à la comparaison de plusieurs modèles supervisés de machine learning et à leur explicabilité. Le jeu de données regroupe des variables opérationnelles et dérivées telles que CRS (RPM), F/A(MF), T/D3(MT), UEP (MPa), LEP (MPa), SE, FPI, TPI et la cible PR. Les résultats exploratoires montrent une relation positive très forte entre AR et PR, ainsi que des relations négatives marquées entre PR et les indices de difficulté tels que SE, FPI et TPI. Sur les essais de modélisation, les familles RVM, Gradient Boosting, XGBoost, Random Forest, SVR et régression linéaire ont été comparées via des découpes train/validation/test multiples. Le RVM fournit les meilleures performances globales sur plusieurs splits, tandis que le Gradient Boosting est le meilleur sur le split 80/20 utilisé pour l'analyse d'explicabilité. Les analyses SHAP, PDP et ICE montrent que TPI domine largement les décisions du Gradient Boosting, avec un effet moyen négatif sur la prédiction, alors que T/D3(MT) joue un rôle secondaire positif. L'ensemble du travail montre qu'un pipeline basé sur des données opérationnelles bien prétraitées peut fournir une prédiction fiable du PR, tout en restant interprétable.

## Mots-clés
TBM, taux de progression, machine learning, Gradient Boosting, RVM, explicabilité, SHAP, PDP, ICE, analyse exploratoire.

## 1. Introduction
L'excavation mécanisée par tunnelier (Tunnel Boring Machine, TBM) constitue une solution majeure pour la réalisation de tunnels de transport, de réseaux d'assainissement et d'ouvrages hydrauliques. Dans ces chantiers, le taux de progression (PR) est une variable clé car il conditionne le rendement journalier, la consommation énergétique, l'usure des outils et, au final, le coût global du projet. La capacité à prévoir ce taux avant ou pendant l'excavation représente donc un avantage décisif pour l'ingénierie des travaux souterrains.

La littérature montre que la performance d'un TBM dépend d'un ensemble complexe de variables géotechniques et opérationnelles : résistance de la roche, qualité du massif, poussée, couple, vitesse de rotation, pressions au front et indicateurs dérivés de difficulté. Les travaux classiques sur la performance des TBM ont établi très tôt que la prédiction du comportement d'excavation ne peut pas se réduire à une seule variable géologique [1]. Plus récemment, les méthodes de machine learning se sont imposées comme des alternatives pertinentes aux approches empiriques, car elles peuvent capturer des relations non-linéaires et des interactions entre variables, tout en améliorant la précision prédictive [2-5].

Le plan de ce papier suit la logique du schéma de travail fourni : résumé, introduction, méthode, résultats et discussion, conclusion, références. L'objectif scientifique est double. D'une part, il s'agit d'évaluer la capacité de plusieurs modèles supervisés à prédire le PR à partir des données disponibles. D'autre part, il s'agit de comprendre quelles variables gouvernent les prédictions du meilleur modèle à l'aide d'outils d'explicabilité tels que SHAP, PDP et ICE.

### 1.1. Contexte physique et variables pertinentes
Les variables explicatives du jeu de données reflètent les conditions opérationnelles du tunnelier. CRS correspond à la vitesse de rotation de la tête de coupe, F/A(MF) à la poussée moyenne, T/D3(MT) au couple normalisé, UEP et LEP aux pressions au front, SE à l'énergie spécifique, FPI et TPI à des indices de pénétration ou d'effort. Sur le plan mécanique, un PR élevé est attendu lorsque l'interaction entre poussée, couple et rotation est efficace, alors qu'une augmentation de SE, FPI ou TPI traduit généralement une plus grande difficulté d'excavation.

### 1.2. Appui sur la littérature scientifique
Les approches de prédiction du PR s'inscrivent dans une ligne de recherche plus large en géomécanique et en apprentissage automatique. Les modèles d'arbres et de boosting ont gagné en popularité grâce à leur performance sur données tabulaires [2,3]. Les modèles à noyau, tels que SVR ou RVM, restent attractifs pour des jeux de données compacts et fortement structurés [4,5]. Enfin, les méthodes d'explicabilité comme SHAP rendent les prédictions plus auditables et facilitent la lecture scientifique des résultats [6].

## 2. Matériels et méthodes

### 2.1. Jeu de données et cible
L'étude s'appuie sur le jeu de données TBM nettoyé et prétraité dans le dossier Internship_Research. La variable cible est le PR (mm/r), c'est-à-dire le taux de progression par révolution. Les variables explicatives principales sont : CRS (RPM), AR (mm/min), F/A(MF), T/D3(MT), UEP (MPa), LEP (MPa), SE, FPI et TPI.

Les rapports précédents du projet ont permis d'établir une base robuste pour la suite de l'analyse : le premier rapport a synthétisé la littérature et identifié les variables les plus pertinentes, tandis que le second a détaillé le prétraitement et l'analyse exploratoire [voir rapports internes du projet].

### 2.2. Prétraitement
Le prétraitement a comporté plusieurs étapes :
- traitement des valeurs manquantes par suppression ou imputation selon leur fréquence ;
- détection et gestion des valeurs aberrantes via boxplots et écart interquartile ;
- normalisation des variables pour homogénéiser les échelles et faciliter la comparaison inter-modèles.

Cette étape est importante car les modèles de régression peuvent être sensibles aux échelles des variables, surtout pour les méthodes à noyau et les approches linéaires régularisées. Elle permet aussi de limiter l'impact de valeurs extrêmes sur les métriques d'erreur.

### 2.3. Analyse exploratoire des données
L'analyse exploratoire a été conduite à l'aide de boxplots, histogrammes, nuages de points, cartes de densité 2D et matrice de corrélation de Spearman. Les figures suivantes illustrent les principaux motifs observés.

![Boites a moustaches des variables](../Internship_Research/images/Box_plots_p1.png)
![Boites a moustaches des variables](../Internship_Research/images/Box_plots_p2.png)

Figure 1. Boîtes à moustaches des variables du jeu de données.

![Histogrammes des variables](../Internship_Research/images/Histo_p1.png)
![Histogrammes des variables](../Internship_Research/images/Histo_p2.png)
![Histogrammes des variables](../Internship_Research/images/Histo_p3.png)

Figure 2. Histogrammes des variables opérationnelles et dérivées.

![Nuages de points PR vs variables](../Internship_Research/images/scatter_plot_p1.png)
![Nuages de points PR vs variables](../Internship_Research/images/scatter_plot_p2.png)
![Nuages de points PR vs variables](../Internship_Research/images/scatter_plot_p3.png)

Figure 3. Nuages de points entre PR et les variables explicatives.

![Carte de correlation de Spearman](../Internship_Research/images/Heatmap2.png)

Figure 4. Matrice de corrélation de Spearman.

Les cartes de densité 2D confirment les tendances observées dans les nuages de points, notamment la relation positive forte entre AR et PR ainsi que les relations négatives entre PR et les indices de difficulté. Les variables SE, FPI et TPI présentent des distributions asymétriques et des queues droites importantes, ce qui suggère la présence de situations opérationnelles plus difficiles mais plus rares.

### 2.4. Modèles comparés
Plusieurs familles de modèles supervisés ont été évaluées dans le projet :
- Régression linéaire comme baseline interprétable ;
- Random Forest comme ensemble d'arbres robuste ;
- SVR comme modèle à noyau ;
- RVM comme approche bayésienne parcimonieuse ;
- Gradient Boosting comme modèle d'arbres séquentiel ;
- XGBoost comme version optimisée du boosting ;
- AdaBoost comme autre méthode d'ensemble.

Les performances ont été mesurées à l'aide de R, R2, MSE, RMSE et MAE sur différents splits train/validation/test. Les principaux résultats de synthèse du dossier interne montrent que le RVM est globalement le meilleur modèle sur plusieurs découpes, tandis que le Gradient Boosting devient la meilleure option sur le split 80/20 retenu pour l'analyse explicable [voir synthèse interne de comparaison].

### 2.5. Explicabilité
L'explicabilité du meilleur modèle sur le split 80/20 a été analysée via SHAP, PDP et ICE. SHAP (SHapley Additive exPlanations) est une méthode basée sur la théorie des valeurs de Shapley (théorie des jeux) qui décompose toute prédiction en contributions additives :

$$f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j$$

ici $\phi_0$ est la valeur de base (espérance de la sortie) et $\phi_j$ représente la contribution attribuée à la variable $j$ pour l'observation $x$. Les propriétés des valeurs de Shapley (additivité, équité) assurent une décomposition cohérente ; `TreeExplainer` permet un calcul efficace pour les modèles d'arbres.

Interprétation pratique :
- $\phi_j>0$ indique que la variable $j$ augmente la prédiction pour cette observation ; $\phi_j<0$ indique qu'elle la diminue.
- L'importance globale d'une variable se mesure souvent par $\mathbb{E}[|\phi_j|]$ (moyenne des contributions absolues).
- Les représentations visuelles (beeswarm, heatmap, waterfall) aident à lire la direction et l'amplitude des effets, tandis que les PDP montrent l'effet marginal moyen et les ICE les variations individuelles.

Limites :
- SHAP traduit des relations statistiques, pas automatiquement de la causalité ; attention aux variables dérivées (ex. TPI) susceptibles d'encoder de l'information proche de la cible.
- La présence de fortes collinarités affecte la répartition des contributions entre variables ; l'interprétation doit tenir compte de la redondance.
- Les valeurs SHAP dans ce projet sont calculées sur des features normalisés (MinMax) ; pour interpréter en mm/r, dé-normaliser la sortie est nécessaire.

PDP/ICE complementent SHAP : PDP donne l'effet moyen attendu d'une variable, ICE montre comment cet effet varie d'un individu a l'autre.

## 3. Résultats

### 3.1. Résultats exploratoires
L'analyse exploratoire met en évidence plusieurs faits robustes :
- AR et PR sont fortement et positivement corrélés ;
- SE, FPI et TPI sont fortement et négativement liés à PR ;
- CRS, F/A(MF), UEP et LEP apportent un signal plus faible lorsqu'elles sont considérées seules ;
- certaines variables sont fortement collinéaires entre elles, ce qui justifie une modélisation prudente et une interprétation orientée vers la robustesse.

Ces motifs sont cohérents avec l'interprétation physique du processus d'excavation : plus le terrain est difficile ou plus l'effort spécifique augmente, plus le PR diminue.

### 3.2. Comparaison des modèles
La synthèse des essais de modélisation montre que les performances varient selon le split, mais plusieurs tendances sont nettes. Sur les splits 70/10/20 et 60/20/20, le RVM produit les meilleurs scores de test avec des erreurs très faibles et des coefficients de détermination proches de 1. Sur le split 80/20, le Gradient Boosting devient le meilleur modèle selon les métriques de test, avec par exemple un R2 de 0.9317, un RMSE de 0.0621 et un MAE de 0.0486 en échelle normalisée.

Le tableau ci-dessous resume la logique generale issue des fichiers de ranking et de comparaison internes :

| Famille de modèle | Comportement global |
|---|---|
| Régression linéaire | Baseline solide, mais moins précise que les modèles non-linéaires |
| Random Forest | Bonne robustesse, bonnes performances globales |
| SVR | Performant sur certains splits, surtout avec parametrage adapte |
| RVM | Meilleur compromis global sur plusieurs tests |
| Gradient Boosting / XGBoost | Très compétitifs sur données tabulaires |
| AdaBoost | Inférieur aux autres méthodes dans ces essais |

Dans le détail, la comparaison du split 0.70/0.10/0.20 montre que le RVM atteint R2 = 0.999962 sur le test, avec RMSE = 0.001458 et MAE = 0.000486 en échelle normalisée. Cette précision très élevée suggère que le problème est fortement structuré par les variables disponibles. Toutefois, elle impose aussi de rester vigilant quant au risque de fuite d'information ou de redondance entre variables dérivées et cible.

### 3.3. Explicabilité du Gradient Boosting
Le modèle Gradient Boosting sur le split 80/20 a été choisi pour l'analyse SHAP, PDP et ICE. Les résultats internes montrent que TPI domine largement l'importance moyenne absolue, suivi de T/D3(MT), tandis que les autres variables contribuent très peu dans ce cadre.

![Importance globale SHAP](../AI9_SHAP_GB_80_20/images/shap_summary_bar_test.png)

Figure 5. Importance globale SHAP sur l'ensemble de test (moyenne des valeurs absolues des contributions). TPI domine nettement l'importance globale.

![SHAP Beeswarm](../AI9_SHAP_GB_80_20/images/shap_summary_beeswarm_test.png)

Figure 6. Beeswarm SHAP : direction et distribution des contributions par variable. La couleur indique la valeur de la feature (bleu = faible, rouge = élevée) ; la position horizontale indique l'impact sur la prédiction.

![Decision plot SHAP](../AI9_SHAP_GB_80_20/images/shap_decision_test.png)

Figure 7. Decision plot SHAP : trajectoires cumulées des contributions par feature pour les observations test. L'ouverture importante au niveau de TPI montre son effet discriminant.

![SHAP Heatmap](../AI9_SHAP_GB_80_20/images/shap_heatmap_test.png)

Figure 8. Heatmap des contributions SHAP (observations × variables) : met en évidence les sous-populations et les motifs récurrents de contributions.

![Waterfall SHAP (mediane)](../AI9_SHAP_GB_80_20/images/shap_waterfall_median_abs_error_test.png)

Figure 9. Waterfall SHAP pour une observation représentative (erreur absolue médiane) : décomposition additive de la prédiction en contributions par variable.

Le résumé quantitatif du dossier d'explicabilité indique que TPI porte environ 89.56 % de l'importance, contre 10.34 % pour T/D3(MT), et des contributions quasi nulles pour UEP, CRS, F/A(MF) et LEP. Le beeswarm (Figure 6) confirme la directionnalité : des valeurs élevées de TPI tendent à diminuer la prédiction, tandis que des valeurs élevées de T/D3(MT) tendent à l'augmenter.

La Figure 7 (decision plot) montre comment l'application séquentielle des contributions sépare les observations ; TPI crée l'écartement le plus important entre trajectoires. La heatmap (Figure 8) illustre la répartition spatiale des contributions et met en évidence des groupes d'observations où certaines variables ont des effets opposés.

La Figure 9 (waterfall) permet d'expliquer localement une prédiction représentative : elle montre quelles variables poussent la prédiction vers le haut ou vers le bas pour cette observation.

Les PDP et ICE (Figures 6 + fichiers PDP/ICE) confirment la même lecture à l'échelle marginale et individuelle : pour TPI, le PDP montre un effet moyen dominant, tandis que les courbes ICE restent relativement homogènes pour cette variable (effet stable). À l'inverse, T/D3(MT) présente une hétérogénéité plus marquée entre individus, suggérée par la dispersion ICE.

Ces figures SHAP sont calculées sur les variables normalisées (MinMax) ; pour exprimer les contributions en mm/r, il faut appliquer l'inverse de la normalisation sur la sortie.

Ces résultats sont importants d'un point de vue métier. Ils signifient que le modèle apprend principalement une relation monotone entre un indice de difficulté (TPI) et le PR, avec un effet secondaire de couple normalisé. Ils montrent aussi que certaines variables, bien qu'utiles physiquement pour la stabilité du front, ont peu d'influence sur la prédiction du PR dans ce modèle particulier.

## 4. Discussion
Les résultats expérimentaux confirment que la prédiction du PR dépend d'un petit nombre de variables dominantes. La relation positive entre AR et PR et la relation négative entre les indices de difficulté et PR sont cohérentes avec la physique du tunnelier. D'un point de vue algorithmique, cela explique pourquoi les modèles non-linéaires, en particulier les méthodes à noyau et le boosting, surpassent une régression linéaire simple.

Un point critique concerne TPI. Comme cet indice peut être défini à partir de quantités déjà proches de la pénétration, il existe un risque de circularité ou de quasi-fuite d'information si sa construction incorpore implicitement la cible. Cela ne remet pas en cause sa valeur prédictive, mais impose une interprétation prudente : une forte importance SHAP ne signifie pas automatiquement causalité physique. Dans une version future du travail, il serait utile de refaire les essais sans TPI pour mesurer la robustesse du signal porté par les autres variables.

Par ailleurs, les variables UEP et LEP ont une influence faible sur PR dans les modèles testés, mais elles restent indispensables pour la sécurité et la stabilité du front. Il s'agit donc de variables à forte importance opérationnelle, même si leur apport prédictif direct est limité.

Enfin, le fait que plusieurs modèles atteignent des performances très élevées montre que la structure du jeu de données est favorable à l'apprentissage, mais invite à vérifier soigneusement l'indépendance des splits, la définition exacte des variables dérivées et l'absence de contamination entre train et test.

## 5. Conclusion
Cette étude montre qu'il est possible de prédire le taux de progression d'un tunnelier avec une grande précision à partir de données opérationnelles et dérivées, à condition de disposer d'un prétraitement rigoureux et d'un choix approprié du modèle. Sur le plan comparé, le RVM donne les meilleures performances globales sur plusieurs splits, tandis que le Gradient Boosting est le meilleur modèle sur le split 80/20 retenu pour l'explicabilité. Les analyses SHAP, PDP et ICE montrent que TPI est la variable dominante du modèle explicable, suivie par T/D3(MT).

Au niveau scientifique, le travail confirme trois points essentiels : la forte structure du problème, l'intérêt des modèles non-linéaires pour les données tabulaires et l'importance de l'explicabilité pour interpréter des performances prédictives très élevées. Au niveau pratique, il fournit une base solide pour développer un outil d'aide à la décision pour le pilotage des TBM.

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
