# Rapport 1 – Revue de littérature et aperçu du jeu de données

## Résumé
Ce rapport présente une synthèse de l’état de l’art sur la prédiction de la performance des tunneliers (TBM), centrée sur le taux de progression (TPR/PR), et propose un aperçu détaillé du jeu de données disponible. L’objectif est d’identifier les variables les plus pertinentes pour la prédiction du PR et de poser les bases pour la modélisation à venir.

## 1. Introduction
L’excavation de la roche est une étape fondamentale dans la réalisation de nombreux projets d’infrastructures souterraines, tels que les tunnels de transport, les réseaux d’assainissement ou les galeries hydrauliques. Elle permet de franchir des obstacles géologiques et d’aménager des espaces indispensables au développement urbain et industriel. Dans ce contexte, la performance des tunneliers (Tunnel Boring Machines, TBM) revêt une importance capitale, car elle conditionne la productivité, la sécurité et le coût global des chantiers. Les paramètres essentiels influençant la performance d’une TBM incluent la résistance de la roche, la qualité du massif rocheux, ainsi que les caractéristiques opérationnelles de la machine (poussée, couple, vitesse de rotation, pression au front, etc.). Comprendre et modéliser ces paramètres est donc crucial pour optimiser les opérations d’excavation et anticiper les difficultés rencontrées lors du creusement.

La prédiction du taux de progression (TPR/PR) des tunneliers est un enjeu majeur en génie civil, avec des impacts directs sur la productivité, la sécurité et les coûts. Les avancées récentes en machine learning permettent de modéliser le PR de façon plus précise en exploitant à la fois les données opérationnelles et géologiques.

## 2. Revue de littérature
### 2.1. Paramètres clés influençant le PR

**Paramètres clés et leur influence sur le PR (Taux de Progression)**

- **UCS (résistance à la compression uniaxiale)** : Il s’agit de la résistance maximale de la roche à la compression. Plus l’UCS est élevée, plus la roche est dure à pénétrer, ce qui fait baisser le PR. À l’inverse, une UCS faible facilite la pénétration et augmente le PR. Dans les approches modernes, l’influence directe de l’UCS peut être atténuée au profit des paramètres machine.

- **RQD (Rock Quality Designation)** : Cet indice mesure la qualité du massif rocheux (proportion de carottes intactes). Un RQD élevé indique une roche massive et peu fracturée, donc plus difficile à excaver : le PR diminue. Un RQD faible (roche fracturée) facilite l’excavation et augmente le PR.

- **Paramètres machine (Thrust, RPM, Torque)** :
    - **Thrust (force d’appui)** : Une poussée optimale augmente le PR. Si la poussée est trop faible, le PR diminue ; si elle est trop élevée, cela peut provoquer une usure excessive sans gain de PR.
    - **RPM (vitesse de rotation de la tête)** : Un RPM trop bas ralentit l’avancement (PR faible), un RPM trop élevé réduit la pénétration par tour (PR faible aussi). Un réglage optimal du RPM maximise le PR.
    - **Torque (couple appliqué)** : Un couple élevé peut indiquer un terrain dur ou des problèmes de coupe, ce qui fait baisser le PR. Un couple modéré, adapté au terrain, favorise un PR élevé.

- **UEP/LEP (Upper/Lower Earth Pressure)** : Ces pressions assurent la stabilité du front de taille. Leur impact direct sur le PR est limité : des pressions trop élevées ou trop faibles peuvent indirectement réduire le PR en générant des problèmes d’excavation.

- **Indices dérivés (TPI, FPI, SE)** :
    - **TPI (Torque Penetration Index)** : Un TPI élevé signifie que la machine rencontre une forte résistance : le PR diminue. Un TPI faible indique une coupe efficace et un PR élevé.
    - **FPI (Field Penetration Index)** : Un FPI élevé indique qu’il faut beaucoup de poussée pour peu d’avancement (PR faible). Un FPI faible reflète une excavation efficace (PR élevé).
    - **SE (Specific Energy)** : Plus le SE est bas, plus l’excavation est efficace et le PR élevé. Un SE élevé traduit une consommation d’énergie importante pour peu de volume excavé (PR faible).

### 2.2. Approches de modélisation

- **Modèles empiriques/statistiques** : Historiquement, la prédiction du PR reposait sur des modèles linéaires ou des formules empiriques reliant le PR à quelques paramètres géotechniques et machine. Ces modèles sont simples à mettre en œuvre et interpréter, mais peinent à capturer la complexité des interactions non linéaires et la multicolinéarité entre variables.
- **Machine learning et deep learning** : Les approches récentes exploitent des algorithmes avancés (SVM, SVR, LSTM, BiLSTM, GEP, etc.) capables de modéliser des relations complexes et non linéaires. Ces modèles tirent parti de grands volumes de données opérationnelles et géologiques, et surpassent généralement les méthodes classiques en précision prédictive, notamment en présence de multicolinéarité.
 **Indices composites et méthodes robustes** : L’utilisation d’indices comme le TPI, FPI ou SE permet de réduire la dimensionnalité et de limiter les effets de la multicolinéarité. Les méthodes robustes (régularisation, réseaux de neurones profonds) sont recommandées pour améliorer la stabilité et la généralisation des modèles.

## 3. Présentation du jeu de données
### 3.1. Source des données
Le jeu de données (TBM data.xlsx) contient des paramètres opérationnels et dérivés issus de chantiers TBM (source : TBM data.xlsx).

### 3.2. Variables
- **CRS (RPM):** Vitesse de rotation de la tête
- **AR (mm/min):** Taux d'avancement
- **F/A (Thrust moyen):** Force d'appui moyenne
- **T/D³ (Couple moyen):** Couple normalisé
- **UEP/LEP (MPa):** Pressions supérieure/inférieure
- **SE (kWh/m³):** Énergie spécifique
- **FPI, TPI:** Indices de pénétration
- **PR (mm/r):** Taux de progression (cible)

## 4. Discussion
- Les données confirment les tendances : nous disposons des paramètres pour toutes les variables clés.
- Les UEP/LEP ont un impact direct limité sur le PR, mais sont importants pour la sécurité ; nous avons donc l’obligation de les maintenir.
- La forte corrélation entre l’AR et le PR est due à leur relation mathématique.
- La multicolinéarité (par exemple, F/A et LEP) doit être prise en compte dans la modélisation.

## 5. Objectif du projet

L'objectif principal de ce projet est de développer le meilleur modèle d’intelligence artificielle (machine learning) possible pour prédire le taux de progression (PR) des tunneliers. Cela implique :
- L’exploitation conjointe des données opérationnelles et géologiques,
- L’application d’algorithmes avancés de machine learning,
- Un travail rigoureux sur l’ingénierie des variables et la gestion de la multicolinéarité,
- Une évaluation systématique des performances pour atteindre la meilleure précision prédictive.

Le but ultime est de fournir un outil robuste, basé sur les données, pour optimiser les opérations de creusement TBM.

---

*Ce rapport s’appuie sur le jeu de données fourni et une synthèse de la littérature scientifique récente.*