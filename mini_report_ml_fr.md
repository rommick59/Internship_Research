## Mini-report – Modèles de Machine Learning (Supervised, Unsupervised, Reinforcement)

### Résumé
Le machine learning (ML) regroupe des méthodes qui apprennent à partir de données afin de prédire une quantité, attribuer une classe ou structurer l’information. Ce mini-report se concentre sur le **supervised machine learning** (apprentissage supervisé), puis présente brièvement le **unsupervised machine learning** (apprentissage non supervisé) et le **reinforcement learning** (apprentissage par renforcement).

### 1. Introduction : qu’est-ce qu’un « modèle » en ML ?
Un modèle de ML est une fonction paramétrée $f_\theta$ qui transforme des entrées (features) $\mathbf{x}$ en une sortie $\hat{y}$. En régression, $\hat{y}$ est une valeur réelle (par exemple un taux, un coût ou une durée) ; en classification, $\hat{y}$ est une étiquette ou une probabilité. L’apprentissage consiste à ajuster $\theta$ pour minimiser une perte $\mathcal{L}$ qui mesure l’écart entre $\hat{y}$ et la vérité terrain $y$, avec un objectif central : bien **généraliser** sur de nouvelles données.

### 2. Supervised Machine Learning
#### 2.1. Données, objectif et généralisation
En apprentissage supervisé, on dispose d’exemples étiquetés sous la forme $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$. Les entrées $\mathbf{x}$ regroupent des variables explicatives (capteurs, paramètres opérationnels, variables géologiques, etc.) et la cible $y$ correspond à la quantité à prédire. L’objectif est d’apprendre une règle qui fonctionne sur de nouvelles observations, ce qui implique de maîtriser à la fois la qualité des données, le choix du modèle et la méthodologie d’évaluation.

#### 2.2. Démarche de modélisation (pipeline)
Une démarche supervisée robuste est généralement la suivante : définir précisément la cible et les contraintes, préparer les données (nettoyage, valeurs manquantes, encodage), puis séparer les données pour mesurer la généralisation. On entraîne d’abord des baselines, puis des modèles plus expressifs, et l’on compare les résultats avec des métriques adaptées.

Une source d’erreur classique est la **fuite de données** (data leakage) : il s’agit d’utiliser, volontairement ou non, une information qui ne serait pas disponible au moment de la prédiction (par exemple une variable calculée avec des informations futures, ou une transformation qui « mélange » train et test). La fuite de données conduit à des scores artificiellement élevés et à des modèles décevants en conditions réelles.

#### 2.3. Découpage train/validation/test
La séparation des données structure l’évaluation. L’ensemble d’entraînement (train) sert à ajuster les paramètres $\theta$ ; l’ensemble de validation sert à choisir l’architecture et les hyperparamètres sans toucher au test ; l’ensemble de test fournit l’estimation finale la plus honnête possible. Pour des données temporelles, il est essentiel de respecter la chronologie afin d’éviter de mélanger passé et futur, ce qui reviendrait à donner indirectement de l’information sur ce qu’on cherche à prédire.

#### 2.4. Fonctions de perte et apprentissage
Apprendre un modèle revient à résoudre un problème d’optimisation du type $\min_\theta \frac{1}{n}\sum_{i=1}^n \mathcal{L}(f_\theta(\mathbf{x}_i), y_i)$. En régression, des pertes courantes sont la MSE (qui accentue les grandes erreurs) et la MAE (souvent plus robuste). L’algorithme d’optimisation dépend du modèle : descente de gradient pour les réseaux de neurones, procédures itératives pour les SVM, ou solutions analytiques/quasi-analytiques pour certains modèles linéaires.

#### 2.5. Principales familles de modèles supervisés
Les modèles les plus courants incluent les **modèles linéaires** (rapides et interprétables), les **arbres/ensembles** (Random Forest, Gradient Boosting, souvent très performants sur données tabulaires), les **SVM/SVR** (efficaces en haute dimension, parfois coûteux) et les **réseaux de neurones** (très flexibles, adaptés aux images, textes et séries temporelles, mais demandant plus de réglages). Le choix dépend surtout du volume de données, de la non-linéarité attendue et du niveau d’explicabilité requis.

#### 2.6. Évaluation : $R^2$, $R$, MSE, RMSE et MAE
En régression, on résume l’erreur avec la **MSE** (moyenne des erreurs au carré) et la **MAE** (moyenne des erreurs absolues), tandis que la **RMSE** ($\sqrt{\mathrm{MSE}}$) remet l’erreur dans l’unité de la cible. La qualité d’ajustement est souvent décrite par $R^2 = 1-\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}$ (comparaison à un prédicteur constant égal à la moyenne) et par le coefficient de corrélation $R$ entre $y$ et $\hat{y}$, qui indique à quel point les variations prédites suivent les variations réelles.

#### 2.7. Surapprentissage, biais-variance et régularisation
Un modèle trop simple sous-apprend, tandis qu’un modèle trop flexible sur-apprend (excellent sur train mais moins bon sur test). La régularisation (Ridge/Lasso, early stopping, contraintes sur les arbres, etc.) et la validation croisée aident à améliorer la généralisation.

#### 2.8. Features, normalisation et robustesse
La qualité des features est souvent déterminante : transformations (log, ratios), agrégations et interactions peuvent améliorer la prédiction. La standardisation est importante pour les modèles sensibles à l’échelle (kNN, SVM, régressions régularisées), et moins critique pour les arbres.

#### 2.9. Interprétabilité et incertitude
L’explicabilité peut être globale (variables importantes, effets moyens) ou locale (explication d’une prédiction). Des outils comme SHAP aident à interpréter les modèles ; selon le contexte, on peut aussi estimer l’incertitude (intervalles de prédiction, calibration).

### 3. Unsupervised Machine Learning
En non supervisé, il n’y a pas de cible $y$ : on cherche à structurer les données via le clustering (k-means, DBSCAN/HDBSCAN), la réduction de dimension (PCA/UMAP) et la détection d’anomalies (Isolation Forest, One-Class SVM). L’évaluation est souvent plus indirecte et s’appuie sur la stabilité des résultats et l’expertise métier.

### 4. Reinforcement Learning
Le RL apprend une politique de décision séquentielle : un agent observe un état $s_t$, choisit une action $a_t$ et reçoit une récompense, avec l’objectif de maximiser un retour cumulé $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k}$. Il est puissant pour des problèmes de contrôle, mais souvent plus coûteux à mettre en œuvre (exploration/exploitation, besoin de simulateur, contraintes de sécurité).

### 5. Conclusion
Le supervised ML est généralement le cadre le plus direct pour prédire une cible à partir de données étiquetées. Le non supervisé complète l’analyse en révélant des structures et des anomalies, tandis que le RL est adapté aux décisions séquentielles quand un objectif de long terme doit être optimisé sous contraintes.