# AI9 — Explicabilité (SHAP) du modèle Gradient Boosting (split 80/20)

## Résumé (Abstract)
Ce document présente une analyse d’explicabilité du modèle **GradientBoostingRegressor** entraîné sur un split **80/20** afin de prédire le **taux de pénétration** PR (mm/r) d’un tunnelier à partir de variables opérationnelles et dérivées. L’explicabilité est réalisée via **SHAP** (Shapley Additive Explanations) sur l’ensemble **TEST**. Les résultats montrent une domination très forte de **TPI** (Torque Penetration Index) dans la variabilité des prédictions, suivie de **T/D3(MT)** (couple normalisé). Les variables **UEP**, **CRS**, **F/A** et **LEP** ont une contribution moyenne proche de zéro dans ce modèle. Une analyse de directionnalité (quantiles 20% vs 80%) indique que des valeurs élevées de **TPI** tendent à **diminuer** la prédiction de PR, tandis que des valeurs élevées de **T/D3(MT)** tendent à **augmenter** la prédiction.

## 1. Contexte et objectif
Les modèles d’arbres boosting (Gradient Boosting) offrent de bonnes performances prédictives mais restent difficilement interprétables à l’échelle des non-spécialistes. L’objectif est d’apporter une lecture scientifique et reproductible de la relation entre entrées et sortie du modèle :

$$\hat{y}(x) = f(x)$$

où $\hat{y}$ est la valeur prédite (ici en échelle normalisée, cf. Méthodes), et $x$ le vecteur des variables d’entrée.

## 2. Données et variables
### 2.1. Variable cible
**PR (mm/r)** : taux de pénétration (pénétration par révolution), utilisé comme variable cible.

### 2.2. Variables explicatives
Les features utilisées par le modèle sont :

- **CRS (RPM)** : vitesse de rotation de la tête de coupe.
- **F/A(MF)** : poussée moyenne (thrust moyen).
- **T/D3(MT)** : couple moyen normalisé (normalisation par $D^3$).
- **UEP (MPa)** : pression supérieure au front.
- **LEP (MPa)** : pression inférieure au front.
- **TPI (Torque Penetration Index)** : indice d’effort basé sur le couple et la pénétration.

Interprétation physique attendue (au sens “indicateur de difficulté d’excavation”) :

- TPI élevé est généralement associé à une résistance élevée et donc à un PR plus faible.
- UEP/LEP sont critiques pour la stabilité/sécurité mais peuvent avoir un effet direct faible sur PR, selon le contexte.

Remarque méthodologique importante : TPI est classiquement défini comme un ratio impliquant la pénétration. Selon la manière dont TPI a été calculé dans la chaîne de données, il peut contenir une part d’information directement liée à la cible PR, ce qui peut amplifier artificiellement son importance explicative. Cette question est traitée en Discussion.

## 3. Méthodologie
### 3.1. Split et normalisation
Les entrées sont celles issues du pipeline AI8 :

- TRAIN : `Internship_Research/AI8/split_train_norm_80_20.csv`
- TEST : `Internship_Research/AI8/split_test_norm_80_20.csv`

Les variables sont normalisées via **MinMax** ajusté sur TRAIN puis appliqué à TEST. La cible PR est également normalisée dans ce pipeline. Par conséquent :

- Les figures SHAP et les contributions sont exprimées dans l’espace des variables normalisées.
- La sortie $f(x)$ correspond à la sortie du modèle sur cible normalisée, et non en mm/r.

### 3.2. Modèle
Le modèle est un **GradientBoostingRegressor** paramétré comme suit :

- `n_estimators=200`
- `learning_rate=0.01`
- `max_depth=3`
- `subsample=1.0`
- `min_samples_leaf=2`
- `random_state=42`

### 3.3. SHAP (définition et conventions)
SHAP fournit une décomposition additive des prédictions basée sur les valeurs de Shapley :

$$f(x) = \phi_0 + \sum_{j=1}^{p} \phi_j$$

- $\phi_0$ : valeur de base (espérance de la sortie du modèle sur la distribution de référence).
- $\phi_j$ : contribution de la variable $j$.

Signe des contributions :

- $\phi_j > 0$ augmente la prédiction.
- $\phi_j < 0$ diminue la prédiction.

Les valeurs SHAP sont calculées sur TEST via `shap.TreeExplainer`.

### 3.4. Figures produites
Cinq figures sont générées (toutes sur TEST) :

- Figure 1 : `images/shap_summary_bar_test.png` (importance globale).
- Figure 2 : `images/shap_summary_beeswarm_test.png` (importance + direction).
- Figure 3 : `images/shap_decision_test.png` (trajectoires, jusqu’à 200 observations).
- Figure 4 : `images/shap_heatmap_test.png` (carte de chaleur, jusqu’à 200 observations).
- Figure 5 : `images/shap_waterfall_median_abs_error_test.png` (explication locale).

Pour la Figure 5, l’observation choisie est celle dont l’erreur absolue $|y_{true} - y_{pred}|$ est médiane sur TEST (choix “représentatif”, évitant un cas extrême).

Les figures sont détaillées et commentées dans la section 5, où elles sont placées à l’endroit logique de leur interprétation.

### 3.5. Sorties tabulaires complémentaires
Deux CSV sont produits pour une analyse quantitative :

- `shap_importance_mean_abs_test.csv` : importance globale $\mathbb{E}(|\phi_j|)$.
- `shap_directionality_test.csv` : comparaison des contributions moyennes entre bas et haut quantiles (20% vs 80%) pour chaque variable.
- `shap_local_explanation_median_abs_error_test.csv` : décomposition locale détaillée (valeur de feature, $\phi_j$, cumul).

## 4. Résultats
### 4.1. Importance globale (Figure 1)
L’importance globale est mesurée par $\mathbb{E}(|\phi_j|)$ sur TEST. Les valeurs observées sont :

- TPI : 0.179827
- T/D3(MT) : 0.041554
- UEP (MPa) : 0.002870
- CRS (RPM) : 0.000146
- F/A(MF) : 0.000000
- LEP (MPa) : 0.000000

Conclusion quantitative : la contribution moyenne absolue de TPI est environ 4.3 fois celle de T/D3(MT), et plusieurs ordres de grandeur au-dessus de CRS. Dans ce modèle, la variabilité des prédictions est donc dominée par TPI.

### 4.2. Sens d’influence (Figure 2 + analyse quantile)
La Figure 2 (beeswarm) associe :

- Axe $x$ : valeur SHAP (impact sur $f(x)$).
- Couleur : valeur de la feature (faible à élevée).

Pour quantifier le sens, `shap_directionality_test.csv` compare la moyenne des SHAP entre :

- “bas quantile” : $x_j \le Q_{0.2}(x_j)$
- “haut quantile” : $x_j \ge Q_{0.8}(x_j)$

Résultats (différence haut-bas) :

- TPI : $\Delta= -0.493627$ (haut quantile diminue fortement la prédiction)
- T/D3(MT) : $\Delta= +0.092208$ (haut quantile augmente la prédiction)
- UEP (MPa) : $\Delta= +0.007920$ (effet positif faible)
- CRS (RPM) : $\Delta= -0.000314$ (effet négatif négligeable)
- F/A(MF) : $\Delta= 0.000000$ (aucun effet observé)
- LEP (MPa) : $\Delta= 0.000000$ (aucun effet observé)

Ces résultats confirment la lecture qualitative du beeswarm : TPI porte un effet directionnel dominant (valeurs élevées associées à une baisse de $f(x)$), tandis que T/D3(MT) apporte un effet secondaire en moyenne positif pour des valeurs élevées.

### 4.3. Décomposition locale (Figure 5)
La Figure 5 est une explication additive d’une observation “représentative” (erreur absolue médiane). D’après `shap_local_explanation_median_abs_error_test.csv` :

- Index TEST : 128
- $\phi_0$ (base) : 0.295543
- $f(x)$ (prédiction) : 0.550632
- $y_{true}$ : 0.596771
- Résidu $y_{true}-f(x)$ : 0.046139

Contributions principales :

- TPI (valeur normalisée 0.006836) : $\phi_{TPI}= +0.314838$
- T/D3(MT) (0.077279) : $\phi_{T/D3}= -0.060872$
- UEP (0.545455) : $\phi_{UEP}= +0.001055$
- CRS (0.363636) : $\phi_{CRS}= +0.000068$
- F/A(MF) et LEP : $\phi=0$

La somme $\phi_0 + \sum_j \phi_j$ reconstruit $f(x)$ (à l’arrondi près), ce qui garantit une interprétation directe : la prédiction est principalement “tirée vers le haut” par un TPI très faible, partiellement compensée par l’effet négatif de T/D3(MT) dans ce cas précis.

## 5. Interprétation des figures (détaillée)
### 5.1. Figure 1 : Summary bar (importance globale)
La Figure 1 classe les variables par $\mathbb{E}(|\phi_j|)$, ce qui mesure l’amplitude moyenne de l’influence d’une variable sur la sortie du modèle. Une variable peut apparaître importante même si elle augmente ou diminue selon les cas, car la mesure est en valeur absolue. La dominance de TPI indique que la structure décisionnelle du boosting “explique” l’essentiel de la variance prédite via TPI.

![Figure 1 - Importance globale SHAP](images/shap_summary_bar_test.png)

Figure 1. Importance globale des variables selon $\mathbb{E}(|\phi_j|)$ sur l’ensemble TEST. TPI domine largement, suivi par T/D3(MT).

#### Interprétation détaillée (Figure 1)
- TPI (0.179827) : sa moyenne de |SHAP| est l’ordre de grandeur dominant. Cela signifie que, en moyenne, changer TPI d’un point dans l’espace normalisé provoque la plus grande variation absolue de $f(x)$ parmi toutes les features.
- T/D3(MT) (0.041554) : importance significative mais bien inférieure à TPI (≈ 23% de l’importance de TPI). Cela indique qu’il s’agit d’un ajusteur secondaire sur lequel le modèle s’appuie.
- UEP (0.002870), CRS (0.000146) : contributions négligeables au regard de TPI/T/D3. Elles ne sont pas utilisées de façon systématique par les arbres appris.
- F/A, LEP (≈0) : absence d’effet observé dans la configuration d’entraînement.

Implication pour l’analyse : toute conclusion opérationnelle ou décisionnelle devrait d’abord considérer TPI, puis tester la stabilité du modèle en retirant TPI pour évaluer la robustesse des autres variables.

### 5.2. Figure 2 : Beeswarm (distribution + direction)
La Figure 2 combine :

- la dispersion des impacts (largeur horizontale)
- l’association valeur-feature / signe SHAP (couleurs)

Pour TPI, la séparation nette des couleurs (valeurs élevées alignées sur SHAP négatif, valeurs faibles sur SHAP positif) est typique d’un effet monotone dominant. Pour T/D3(MT), la structure est plus compacte et moins monotone, suggérant un effet secondaire, possiblement modulé par des interactions internes du boosting.

![Figure 2 - Beeswarm SHAP](images/shap_summary_beeswarm_test.png)

Figure 2. Distribution des contributions SHAP sur TEST. La position horizontale indique si la variable augmente ou diminue la prédiction, tandis que la couleur encode la valeur de la variable.

#### Interprétation détaillée (Figure 2) — par variable
- TPI : distribution fortement asymétrique des SHAP selon la valeur de la variable. Quantitativement, la comparaison quantile (20% vs 80%) donne une différence moyenne des SHAP de $\Delta=-0.4936$, ce qui signifie que les observations du 80e centile de TPI baissent en moyenne la prédiction de presque 0.5 unité normalisée par rapport au 20e centile. C’est un effet de grande ampleur.
- T/D3(MT) : $\Delta=+0.0922$ entre haut et bas quantiles — signe positif moyen pour valeurs élevées ; dans le beeswarm on observe des points rouges souvent à droite, indiquant que pour des valeurs élevées de T/D3(MT) le modèle ajoute en moyenne à la prédiction.
- UEP : $\Delta=+0.00792$ — effet faible, visible comme un léger nuage asymétrique dans le beeswarm.
- CRS : $\Delta\approx -0.000314$ — très faible, aucun effet pratique.
- F/A, LEP : distributions concentrées en zéro (aucun effet détectable).

Conséquence : le beeswarm confirme la hiérarchie d’importance et précise le sens moyen des effets, avec TPI et T/D3 comme variables apportant le signal principal.

### 5.3. Figure 3 : Decision plot (trajectoires)
Le decision plot ordonne l’application des contributions par feature (selon l’ordre du graphe) et visualise, pour chaque observation, comment $\phi_0$ est déplacé vers $f(x)$. Lorsque l’éventail s’ouvre fortement à une étape (ici TPI), cela signifie que cette feature est le principal facteur de séparation des observations en termes de sortie modèle.

![Figure 3 - Decision plot SHAP](images/shap_decision_test.png)

Figure 3. Trajectoires cumulées des prédictions individuelles. L’ouverture du faisceau à l’étape TPI montre que cette variable sépare fortement les observations en termes de prédiction.

#### Interprétation détaillée (Figure 3)
- Mécanique : chaque ligne représente l’addition séquentielle des contributions par feature pour une observation. L’écart horizontal entre lignes à un point particulier montre l’impact discriminant de la feature concernée.
- Observation clé : l’écartement maximal se produit lors de l’application de la contribution associée à TPI — visuellement et numériquement cohérent avec la valeur $\mathbb{E}(|\phi_{TPI}|)=0.1798$.
- Interprétation opérationnelle : TPI agit comme un critère de partition fort dans les arbres — il crée des groupes d’observations avec prédictions nettement différentes, ce qui signifie que les décisions (splits) des arbres s’appuient souvent sur des seuils de TPI.

### 5.4. Figure 4 : Heatmap (motifs)
La heatmap montre la matrice $[\phi_{i,j}]$ (observations $i$, variables $j$), généralement après un réordonnancement des observations. Une alternance structurée rouge/bleu sur une feature (TPI) indique que cette feature crée des sous-populations avec effets opposés et de grande amplitude. Les bandes quasi blanches sur d’autres variables indiquent que leurs contributions sont proches de zéro pour la majorité des observations.

![Figure 4 - Heatmap SHAP](images/shap_heatmap_test.png)

Figure 4. Carte de chaleur des contributions SHAP sur TEST. Les contrastes marqués sur TPI traduisent une variabilité forte de son effet selon les observations.

#### Interprétation détaillée (Figure 4)
- TPI : zones étendues de contributions positives et négatives (rouge/bleu) — elles matérialisent l’existence de sous-populations : certaines observations bénéficient d’une contribution positive forte, d’autres d’une contribution négative forte.
- T/D3(MT) : motifs plus discrets mais visibles en bandes fines — la variable intervient régulièrement mais avec une amplitude moindre.
- UEP/CRS/F/A/LEP : absence de structure marquée, confirmant le faible rôle heuristique.

Usage pratique : la heatmap sert à repérer si certaines combinaisons de variables (interaction) produisent motifs récurrents — utile pour prioriser investigations géotechniques ou opérationnelles sur sous-groupes.

### 5.5. Figure 5 : Waterfall (attribution additive locale)
Le waterfall expose une instance $x$ en listant la valeur observée de chaque feature et la contribution SHAP correspondante. La lecture est strictement additive :

$$f(x) = \phi_0 + \phi_{TPI} + \phi_{T/D3} + \dots$$

Ce graphique est approprié pour documenter, de manière vérifiable, les raisons d’une prédiction individuelle.

![Figure 5 - Waterfall SHAP](images/shap_waterfall_median_abs_error_test.png)

Figure 5. Décomposition locale d’une prédiction représentative sur TEST. La valeur de base est déplacée par chaque contribution SHAP jusqu’à la prédiction finale.

#### Interprétation détaillée (Figure 5)
- Contexte chiffré (sample_index=128) : base $\phi_0=0.295543$, prédiction $f(x)=0.550632$, vérité $y=0.596771$, résidu = 0.046139.
- Contributions : TPI fournit +0.31484 (valeur normalisée 0.006836), ce qui est supérieur en magnitude à toutes les autres contributions; T/D3 fournit -0.060872.
- Sens métier : pour cette observation, un TPI faible (0.0068 normalisé) correspond à une situation de faible effort par unité d’avancement, et le modèle traduit cela par une forte augmentation de la prédiction normalisée ; inversement T/D3 réduit la prédiction.
- Vérification d’addition : $\phi_0 + \sum_j \phi_j \approx f(x)$ (contrôle de cohérence mathématique).

Interprétation pour l’audit modèle : la décomposition locale permet d’identifier précisément quelles mesures devraient être vérifiées en cas d’écart majeur entre $y$ et $f(x)$ (ex. capteur de couple, calcul de TPI, valeurs de normalisation).

## 6. Discussion et limites
### 6.1. Dominance de TPI et risque de circularité
TPI est défini comme un indice lié à la pénétration (et donc potentiellement à PR). Si TPI a été calculé à partir de PR ou d’une grandeur fortement corrélée à PR, son importance SHAP peut refléter un mécanisme de “cible indirecte” (information de PR encodée dans un input), et non une relation purement causale. Dans un cadre scientifique, il est recommandé de compléter l’analyse par :

- une expérience “sans TPI” (entraînement + SHAP) pour vérifier la robustesse des conclusions.
- une vérification de la définition exacte de TPI dans le fichier source (formule, unités, utilisation éventuelle de PR).

### 6.2. Interprétation de T/D3(MT)
L’effet positif moyen de T/D3(MT) (quantiles) ne doit pas être interprété comme causalité directe sans prudence. Plusieurs scénarios de corrélation opérationnelle sont possibles : adaptation des réglages (torque) pour maintenir une pénétration élevée, conditions de terrain et stratégies d’exploitation, ou interactions avec d’autres variables non observées.

### 6.3. Variables à contribution nulle
Une contribution nulle moyenne (F/A, LEP) peut provenir de plusieurs causes :

- la variable ne fournit pas d’information prédictive supplémentaire compte tenu des autres features.
- colinéarité rendant la variable redondante.
- structure du boosting (seuils) qui n’utilise pas la variable dans les splits appris.

### 6.4. Échelle normalisée
Les contributions SHAP et les valeurs affichées dans les figures sont en échelle normalisée. Pour une interprétation directe en mm/r, il faut appliquer l’inverse de la normalisation de la cible (si disponible) et, idéalement, recalculer les explications en rapportant les effets à l’unité physique (au minimum pour $f(x)$ et $y$).

## 7. Reproductibilité
### 7.1. Script
`Internship_Research/AI9_SHAP_GB_80_20/shap_gradient_boosting_80_20.py`

### 7.2. Sorties
- Figures : `Internship_Research/AI9_SHAP_GB_80_20/images/*.png`
- Importance globale : `Internship_Research/AI9_SHAP_GB_80_20/shap_importance_mean_abs_test.csv`
- Directionnalité : `Internship_Research/AI9_SHAP_GB_80_20/shap_directionality_test.csv`
- Décomposition locale : `Internship_Research/AI9_SHAP_GB_80_20/shap_local_explanation_median_abs_error_test.csv`
