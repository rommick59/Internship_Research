# Introduction

Mechanized excavation using Tunnel Boring Machines (TBMs) is a standard solution for underground infrastructure projects (transportation, water, utilities). During operation, the cutterhead interacts continuously with variable geological conditions, which makes precise control of the operating parameters essential.

In this context, the penetration rate (PR, mm/rev) is a key indicator: it measures advancement per cutterhead revolution and reflects productivity, tool wear, energy consumption and, indirectly, project cost and schedule. Improving PR prediction helps refine machine settings and identify unfavorable ground conditions earlier.
Figure 1 shows the TBM working inside the tunnel.

![Figure 1. TBM working inside the tunnel](TBM.jpg)

Figure 2 shows the movement of the TBM and its interactions with the ground mass, highlighting the main operating parameters

![Figure 2. TBM schematic and operating parameters](TBM_density.jpg)

Figure 3 illustrates the excavation zone, muck removal and the geological context around the cutterhead.

![Figure 3. TBM schematic and muck removal](téléchargé.jpg)

This study is positioned at the interface of applied geomechanics and machine learning. The available signals — rotation speed, thrust, torque, face pressures, and derived indices such as specific energy, FPI, TPI, etc. — characterize both machine behavior and ground response. Machine learning methods, especially tree-based and boosting models (Random Forest, Gradient Boosting, XGBoost) and kernel-based approaches (SVR, RVM), are used to analyze these signals and capture nonlinear relationships and complex interactions that are difficult to describe with simple empirical rules (Khatti & Mishra, 2025; Yagiz, 2008; Breiman, 2001; Friedman, 2001). The datasets used in this work come from the open-access study by Khatti and Mishra (2025), which focuses on TBM penetration rate estimation in mixed-face conditions and examines feature selection and multicollinearity effects on machine and deep learning models.

This paper contributes in three practical ways: (1) it proposes a compact, reproducible modeling pipeline that links preprocessing, model selection and local explainability; (2) it assesses the relative importance of raw operating signals versus derived indices for PR prediction; and (3) it documents pitfalls related to circularity when using indices that are conceptually close to the target. The methodology emphasizes reproducibility: all splits, fitted preprocessors and trained models are archived so that the experiments can be replayed and independently inspected.

The main objective of this paper is to evaluate how well machine learning models can predict TBM penetration rate from operational and derived variables, and to identify the most influential predictors. Beyond prediction accuracy, the study also examines the consistency of the results across different model families and highlights the role of derived indices in the interpretation of TBM performance. The remainder of the paper is organized as follows: Section 2 presents the data preparation and methodology, Section 3 reports the results and discussion, and Section 4 concludes the paper.

## 2 Methodology

### 2.1 Data source and study variables
The analysis starts from the cleaned TBM dataset assembled in the repository and reduced to a modeling subset containing six explanatory variables and one target variable. The target is the penetration rate PR (mm/rev). The final feature set used in the fixed 80/20 pipeline is: CRS (RPM), F/A(MF), T/D3(MT), UEP (MPa), LEP (MPa), and TPI. These variables summarize the main operating conditions of the cutterhead, the pressure balance at the face, and the excavation effort.

The workflow keeps the target definition consistent across preprocessing, model training and explainability so that predictive performance and SHAP attributions are computed in the same modeling space.

More details on the variables and data provenance:
- CRS (cutter rotation speed, RPM): direct machine telemetry describing rotational velocity.
- F/A(MF) and T/D3(MT): operational ratios derived from thrust and torque measurements to better reflect interaction intensity at the cutting face.
- UEP and LEP (MPa): upper and lower face pressures measured at dedicated sensors; they inform about face support and potential overpressure or loss-of-support conditions.
- TPI (Torque Penetration Index): a composite index computed from torque, thrust and rotation to quantify energy expenditure per unit advance. Because TPI is derived from machine signals that are also linked to PR, we treat its interpretation with caution.

The raw dataset originates from Khatti & Mishra (2025) and was pre-cleaned as described in the accompanying repository scripts. No additional external measurements are used. Units are kept consistent with the original source; any derived ratio or index is computed using the same unit system to avoid scale inconsistencies.

### 2.2 Exploratory data analysis
Before training any model, the cleaned dataset is explored visually to verify distribution shapes, detect extreme values and inspect pairwise relationships. The EDA workflow combines histograms with KDE curves, boxplots, target-vs-feature scatter plots, a correlation heatmap, and 2D KDE plots. This exploratory step is used to identify skewed variables, potential outliers, and monotonic or nonlinear relationships with PR.

The manuscript uses a correlation heatmap as the compact visual summary of the EDA stage.

Figure 4. Exploratory correlation heatmap used to inspect the dependence structure between PR and the main operating variables.
A correlation heatmap visualizes the correlation coefficients (often Pearson or Spearman) between all pairs of variables in the dataset. Analysis of the heatmap shows that TPI and FPI are the most influential linear predictors ($r \approx -0.70$), whereas AR is excluded from the modeling pipeline due to its extreme correlation with the target ($r \approx 0.99$). The disparity between these strong correlations and the weak ones for face pressures (e.g., UEP $r \approx -0.05$) motivates the subsequent detailed visual diagnostics using KDE and LOWESS smoothing.

![Figure 4. Exploratory correlation heatmap](../AI8/images/heatmap_distance_correlation_ai7.png)

Figure 5 shows the univariate histograms for the main predictors and the target. These histograms display the marginal distribution of each variable and include skewness estimates: strongly right-skewed predictors are flagged as candidates for log or square-root transforms to stabilize variance. Inspecting the tails helps decide whether winsorizing or transformation is preferable to removing outliers, since extreme operational spikes can be physically meaningful. Showing all ten panels makes it possible to compare the full set of variables rather than only a reduced sample.

![Figure 5 Histo P1](Histo_1.png)

![Figure 5 Histo P2](Histo_2.png)

![Figure 5 Histo P3](Histo_3.png)

![Figure 5 Histo P4](Histo_4.png)

![Figure 5 Histo P5](Histo_5.png)

![Figure 5 Histo P6](Histo_6.png)

![Figure 5 Histo P7](Histo_7.png)

![Figure 5 Histo P8](Histo_8.png)

![Figure 5 Histo P9](Histo_9.png)

![Figure 5 Histo P10](Histo_10.png)


Figure 6 presents boxplots grouped by coarse geological facies to compare the central tendency and variability of each parameter across different ground conditions. Systematic shifts in median values are observed, particularly for energy indices; for instance, the median TPI shifts significantly between soft and hard facies, moving from a baseline of $\approx 450$ towards much higher values in denser ground. Similarly, the PR median fluctuates around $12\text{ mm/r}$ but shows a much wider dispersion (up to $50\text{ mm/r}$) in specific geological units. This heterogeneity suggests that while coarse facies are strong indicators of TBM behavior, they are best used as a qualitative validation tool to ensure the model's predictions are physically consistent with the known geology, rather than as direct input variables.

![Figure 6 Boxplots P1](box_plot_1.png)

![Figure 6 Boxplots P2](box_plot_2.png)

![Figure 6 Boxplots P3](box_plot_3.png)

![Figure 6 Boxplots P4](box_plot_4.png)

![Figure 6 Boxplots P5](box_plot_5.png)

![Figure 6 Boxplots P6](box_plot_6.png)

![Figure 6 Boxplots P7](box_plot_7.png)

![Figure 6 Boxplots P8](box_plot_8.png)

![Figure 6 Boxplots P9](box_plot_9.png)

![Figure 6 Boxplots P10](box_plot_10.png)


Figure 7 utilizes 2D KDE and pairwise density maps to visualize non-linear relationships and heteroscedasticity. The density contours reveal a strong, non-linear 'curved' decay for energy indices; for instance, the penetration rate drops sharply as TPI increases beyond $1,000$. Moreover, the plots exhibit clear conditional heteroscedasticity for face pressures (UEP and LEP): while a high-density cluster is concentrated around $0.20\text{ MPa}$, the vertical spread of PR at this specific value ranges from approximately $0$ to $40\text{ mm/r}$. This high variance at a constant pressure indicates that face support alone cannot linearly predict PR, justifying the use of non-linear learners capable of capturing such complex, regime-dependent behaviors.

![Figure 7 2D KDE](2D_KDE.png)

Figure 8 presents target-vs-feature scatter plots with LOWESS smoothing to highlight non-linear trends. For energy indices (SE, FPI, and TPI), the LOWESS curve reveals a sharp non-linear decay: for instance, the PR drops precipitously from $\approx 40\text{ mm/r}$ to below $10\text{ mm/r}$ as TPI increases from $0$ to $1,000$, before plateauing. In contrast, the curves for face pressures (UEP and LEP) remain nearly flat around a conditional mean of $15\text{--}20\text{ mm/r}$, regardless of the pressure value, confirming the lack of a strong linear relationship. Extreme outliers, such as those in FPI exceeding $10,000$, were retained as they represent physically genuine high-effort excavation events in very hard ground, and are thus handled via the robust loss functions of the boosting models.

![Figure 8 Scatter](Scatter_plot.png)

These diagnostics are used to identify skewed variables (candidate for log transforms), heteroscedastic relationships, and potential outliers that require winsorizing rather than removal in the modeling pass.

In addition to the heatmap, the EDA stage documents the following checks and visual diagnostics:
- Univariate distributions for each predictor and the target, annotated with skewness and suggested transformations when appropriate (e.g., log-transform for heavy right tails).
- Pairwise scatter plots with a lowess smoother to spot nonlinear trends and heteroscedasticity.
- Boxplots grouped by coarse geological facies when such labels are available in the raw metadata (used as qualitative cross-checks rather than model inputs).
- A simple missingness matrix to ensure that imputation strategies are well justified and do not hide systematic patterns.

### 2.3 Data preprocessing and normalization
The preprocessing stage is organized in two successive steps. First, the raw TBM file is cleaned: the input data are checked for missing values and duplicated rows, missing numeric entries are imputed, outliers are removed using the interquartile range (IQR) rule, and numeric columns are standardized in the cleaning pass that produces `TBM_data_cleaned.csv`. Second, the paper-ready modeling subset is built from the selected six predictors plus PR, and the data are split into 80% training and 20% test sets using `random_state=42`.

The diagrams in this subsection frame the research as a supervised regression problem within the broader AI landscape. They are useful because they explain why the paper uses regression models, how the data flow is organized, and why monitoring and retraining are part of the methodology. In short, they connect the TBM prediction problem to a full machine learning workflow rather than to a one-shot model fit.

Figure ML-1 shows how machine learning sits inside the broader AI hierarchy. This distinction matters because the present study does not rely on deep representation learning or generative models; instead, it uses structured operational variables to predict a continuous quantity. The figure therefore positions the work in the classical ML branch and helps the reader understand the level of abstraction used in the manuscript.

![Figure ML-1 AI hierarchy](IA.png)

Figure ML-2 presents the main machine learning families and highlights supervised learning as the relevant branch for this study. Because the TBM dataset contains labeled examples with a known penetration rate, the problem is naturally formulated as supervised regression rather than clustering or reinforcement learning. The figure therefore justifies the modeling choice made in the paper.

![Figure ML-2 ML techniques taxonomy](Schéma.png)

Figure ML-3 summarizes the standard machine learning life cycle from the business goal to monitoring. It is important because it shows that a useful model must be evaluated, deployed and monitored, not only trained. In this paper, that lifecycle perspective supports the reproducibility strategy and the storage of fitted artifacts.

![Figure ML-3 ML life cycle](Capture_d%27%C3%A9cran_2026-06-03_092359.png)

Figure ML-4 shows the iterative problem-framing workflow used to transform the business question into an ML problem. It highlights the feedback loop between data preparation, feature engineering, training and evaluation, which is essential when model quality depends on how the variables are built. The diagram also shows that a poor result should trigger a review of the inputs or the formulation, not only a change of algorithm.

![Figure ML-4 ML problem framing workflow](1.png)

Figure ML-5 makes the separation between data, code and model repositories explicit. This separation is central to reproducibility because the data can be versioned independently from the scripts and the fitted model can be archived for later inspection or reuse. The figure therefore matches the archival approach used throughout this manuscript.

![Figure ML-5 ML pipeline and repositories](2.png)

Figure ML-6 shows the operational workflow from identifying the business goal to monitoring the deployed solution. It reinforces the idea that machine learning is an end-to-end process, covering preprocessing, model building, evaluation, selection, deployment and monitoring. In the context of TBM prediction, it justifies a reusable workflow that can be updated when new operational data become available.

![Figure ML-6 Operational ML workflow](3.png)

For the final 80/20 modeling pipeline, normalization is fitted on TRAIN only and then applied to TEST to avoid information leakage. A MinMax scaler is used, and the fitted preprocessor is saved together with the split indices for reproducibility.

Implementation details and decisions made to preserve validity:
- Missing value imputation: numeric missing values are filled with the feature median when distributions are skewed, and with the mean for approximately symmetric variables. Categorical or label fields (when present) are imputed with a dedicated "missing" category; however, the modeling subset used here contains only numeric predictors.
- Outlier removal: outliers are flagged using the IQR rule (points below Q1 - 1.5*IQR or above Q3 + 1.5*IQR) evaluated on the TRAIN fold. When extreme values are likely to be genuine (operational spikes), these rows are clipped or winsorized rather than removed to retain physically meaningful events.
- Feature scaling: a `MinMaxScaler` is fitted on the TRAIN set and applied to TEST. Scaling parameters (min, max) are archived with the pipeline to ensure deterministic preprocessing when reproducing results.
- Splitting: the split is random without stratification because the target is continuous; `random_state=42` ensures reproducibility. Indices for TRAIN and TEST are saved as `train_idx.joblib` / `test_idx.joblib`.
- Artifact storage: fitted scalers and imputation rules are serialized with `joblib` in the repository `artifacts/` folder alongside model weights and evaluation tables.


### 2.4 Machine learning protocol
Several regression families are evaluated on the fixed split: Linear Regression, Random Forest, RVM, XGBoost, and Gradient Boosting. The comparison is designed to contrast a linear baseline with tree-based and kernel-based nonlinear learners. The evaluation is carried out on the same normalized 80/20 split.

The main metrics used to compare models are correlation coefficient $r$, coefficient of determination $R^2$, mean squared error (MSE), root mean squared error (RMSE), mean absolute error (MAE), and variance accounted for (VAF). The best-performing model on the test set is later used as the reference model for SHAP explainability.

Model selection and tuning details:
- Hyperparameter search is performed using 5-fold cross-validation on the TRAIN set only. A mix of grid search and randomized search is used depending on model complexity: randomized search for XGBoost/GradientBoosting to explore learning rate and tree-depth trade-offs, and grid search for Random Forest and SVR for a concise but targeted grid.
- Typical hyperparameter ranges considered (examples): Random Forest `n_estimators` in [100, 300, 500], `max_depth` in [None, 5, 10, 20]; Gradient Boosting / XGBoost `learning_rate` in [0.01, 0.05, 0.1], `n_estimators` in [100, 300, 500], `max_depth` in [3, 5, 8]; SVR `C` in [0.1, 1, 10], `gamma` in ['scale','auto', 0.01, 0.1]. The exact grids are provided in the `AI` script folder for reproducibility.
- For each candidate model, the CV-aggregated RMSE is used as the primary scoring metric to select hyperparameters; the selected model is then retrained on the full TRAIN set with the best hyperparameters and evaluated once on TEST.
- Linear Regression is used as an interpretable baseline (ordinary least squares) and is inspected for multicollinearity via variance inflation factors (VIF). Where multicollinearity is severe, coefficients are interpreted cautiously and comparison with regularized linear models is suggested as a follow-up.

### 2.5 SHAP-based explainability
To interpret the selected Gradient Boosting model, SHAP (Shapley Additive Explanations) is computed on the normalized TEST set with `shap.TreeExplainer`. SHAP provides an additive decomposition of the prediction into a baseline term and feature contributions. The global analysis uses mean absolute SHAP values to rank the predictors, while beeswarm, decision, heatmap and waterfall plots are used in the dedicated explainability section to inspect the sign, dispersion and local contribution patterns.

The SHAP analysis is performed in the same normalized feature space as the model, which guarantees consistency between the learned decision function and the reported attributions. Because TPI is a derived operational index that may be close to the target conceptually, its importance is interpreted cautiously and discussed as a possible source of circularity.

Practical SHAP steps used in this study:
- Compute SHAP values on the TEST set only, leaving TRAIN for model fitting and hyperparameter selection. This prevents information leakage into the explainability stage.
- Use `shap.summary_plot` (beeswarm) to display global importance and the sign of each feature's impact across observations.
- Produce `shap.dependence_plot` for the top 2–3 predictors to inspect interaction effects and non-monotonic relationships; where interactions appear strong, use `interaction_index` to quantify them.
- Use waterfall and force plots for selected individual cases to illustrate how combinations of features push the prediction above or below the baseline.
- Archive SHAP values and generated figures so that individual explanations can be revisited without recomputing heavy model artifacts.

Interpretation guidance: when a derived index (TPI) ranks highly, we check whether its contribution can be decomposed into the original low-level signals; when circularity is suspected, we rerun the explainability stage excluding TPI to evaluate stability of the variable ranking.

### 2.6 Reproducibility and implementation
All preprocessing, modeling and explainability steps are implemented as standalone scripts and saved artifacts in the repository. The split indices, fitted preprocessing objects, model outputs and SHAP tables are stored to make the workflow reproducible.

## References

- Khatti, J., & Mishra, S. (2025). Estimating shield tunnel boring machine penetration rate in mixed face conditions: feature selection and multicollinearity effects on machine and deep learning models. Frontiers in Built Environment, 11, 1699466.
- Yagiz, S. (2008). Utilizing rock mass properties for predicting TBM performance in hard rock tunneling. Tunnelling and Underground Space Technology, 23(3), 326–337.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.