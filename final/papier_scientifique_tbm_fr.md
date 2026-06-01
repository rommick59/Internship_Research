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
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.# 1 Introduction

