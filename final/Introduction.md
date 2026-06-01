# 1 Introduction

L'excavation mécanisée par tunnelier (Tunnel Boring Machine, TBM) constitue une solution de référence pour la réalisation d'infrastructures souterraines (transports, eaux, services). Dans un chantier en cours d'avancement, la tête de coupe interagit en continu avec un milieu géologique variable, ce qui impose un pilotage précis des paramètres opératoires.

Dans ce contexte, le taux de progression (PR, mm/r) est un indicateur clé : il mesure l'avancement par révolution de la tête de coupe et résume à la fois la productivité, l'usure des outils, la consommation d'énergie et, indirectement, les coûts et délais du projet. Une meilleure prédiction du PR permet d'améliorer le réglage de la machine et d'identifier plus tôt les conditions défavorables.

![TBM en fonctionnement dans le tunnel](TBM.jpg)

Figure 1. Photographie du TBM en fonctionnement dans le tunnel.

![Schéma TBM et paramètres opératoires](TBM_density.jpg)

Figure 2. Schéma du TBM montrant les paramètres opératoires et l'interaction avec le massif.

![Schéma TBM et évacuation des déblais](téléchargé.jpg)

Figure 3. Schéma du TBM illustrant la zone d'excavation, l'évacuation des déblais et le contexte géologique.

Cette étude s'inscrit à l'interface de la géomécanique appliquée et de l'apprentissage automatique. Les signaux disponibles — vitesse de rotation, poussée, couple, pressions au front et indices dérivés tels que l'énergie spécifique, FPI, TPI, etc. — décrivent à la fois le comportement de la machine et la réponse du terrain. Les méthodes de machine learning, en particulier les modèles d'arbres et de boosting (Random Forest, Gradient Boosting, XGBoost) et les approches à noyau (SVR, RVM), permettent d'analyser ces signaux afin de capturer des relations non linéaires et des interactions complexes, difficiles à représenter par des règles empiriques simples (Khatti et Mishra, 2025 ; Yagiz, 2008 ; Breiman, 2001 ; Friedman, 2001). Les jeux de données utilisés dans cette étude proviennent de l'article de Khatti et Mishra (2025), consacré à l'estimation du taux de pénétration d'un tunnelier en conditions de front mixte et à l'effet de la sélection de variables sur les modèles de machine learning et de deep learning.

Références citées dans cette introduction :
- Khatti, J., & Mishra, S. (2025). Estimating shield tunnel boring machine penetration rate in mixed face conditions: feature selection and multicollinearity effects on machine and deep learning models. Frontiers in Built Environment, 11, 1699466.
- Yagiz, S. (2008). Utilizing rock mass properties for predicting TBM performance in hard rock tunneling. Tunnelling and Underground Space Technology, 23(3), 326–337.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.