# 1 Introduction

L'excavation mécanisée par tunnelier (Tunnel Boring Machine, TBM) est une technique essentielle pour la réalisation d'infrastructures souterraines (transports, eaux, services). Le taux de progression (PR, mm/r), qui mesure l'avancement par révolution de la tête de coupe, est un indicateur opérationnel central : il conditionne la productivité, l'usure des outils, la consommation d'énergie et, indirectement, les coûts et délais du projet. Prédire précisément le PR à partir de signaux machine et de variables dérivées permet d'anticiper les réglages opératoires, d'alerter sur des conditions défavorables et d'améliorer la planification des chantiers.

Le présent travail se situe à l'intersection de la géomécanique appliquée et de l'apprentissage automatique. Les signaux disponibles (vitesse de rotation, poussée, couple, pressions au front, indices dérivés tels que l'énergie spécifique, FPI, TPI, etc.) incarnent à la fois des informations physiques et des résumés statistiques des interactions outil–milieu. Les méthodes de machine learning — en particulier les modèles d'arbres et de boosting (Random Forest, Gradient Boosting, XGBoost) et les approches à noyau (SVR, RVM) — offrent la capacité de capturer des relations non linéaires et des interactions complexes entre ces variables, souvent difficiles à modéliser par des lois empiriques simples [1–4]. Dans ce cadre, les jeux de données utilisés dans cette étude ont été repris à partir de l'article open access de Khatti et Mishra (2025), consacré à l'estimation du taux de pénétration d'un tunnelier en conditions de front mixte et à l'effet de la sélection de variables sur les modèles de machine learning et de deep learning.


Le manuscrit s'appuie sur les rapports internes du projet et intègre les photos et figures générées (EDA, SHAP, PDP/ICE) pour illustrer et étayer l'analyse.

Outre la performance prédictive, l'explicabilité des modèles est fondamentale dans ce contexte industriel : comprendre quelles variables influencent les prédictions aide à détecter d'éventuelles fuites d'information (features dérivées proches de la cible), à valider la cohérence physique des résultats et à rendre les modèles utilisables par les décideurs. Pour cela, nous utilisons des méthodes d'interprétation locale et globale telles que SHAP (Shapley Additive exPlanations), PDP (Partial Dependence Plots) et ICE (Individual Conditional Expectation) afin d'expliquer les contributions des variables au niveau global et par observation [5].

L'objectif de l'étude est double : (i) comparer rigoureusement plusieurs familles de modèles supervisés pour la prédiction du PR sur un jeu de données TBM prétraité, et (ii) fournir une analyse d'explicabilité détaillée du modèle retenu pour en tirer des enseignements physiques et opérationnels. Plus précisément, nous évaluons la robustesse des modèles sur plusieurs protocoles de découpe train/validation/test, quantifions les performances à l'aide d'indicateurs standards (R, R2, RMSE, MAE) et analysons l'importance et l'effet des variables via SHAP, PDP et ICE [5–7].

Les contributions principales de cet article sont donc :
- une méthodologie de prétraitement adaptée aux données TBM et une sélection d'indices opérationnels pertinents ;
- une comparaison empirique de modèles de référence et de modèles avancés sur plusieurs protocoles de validation ;
- une étude d'explicabilité approfondie du modèle retenu, mettant en évidence les variables dominantes et discutant la robustesse des conclusions face à des indices dérivés potentiellement proches de la cible.

Références citées dans cette introduction :
- Khatti, J., & Mishra, S. (2025). Estimating shield tunnel boring machine penetration rate in mixed face conditions: feature selection and multicollinearity effects on machine and deep learning models. Frontiers in Built Environment, 11, 1699466.
- Yagiz, S. (2008). Utilizing rock mass properties for predicting TBM performance in hard rock tunneling. Tunnelling and Underground Space Technology, 23(3), 326-337.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. The Annals of Statistics, 29(5), 1189-1232.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.