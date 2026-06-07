# Introduction

Mechanized excavation using Tunnel Boring Machines (TBMs) is a standard solution for underground infrastructure projects (transportation, water, utilities). During operation, the cutterhead interacts continuously with variable geological conditions, which makes precise control of the operating parameters essential.

In this context, the penetration rate (PR, mm/rev) is a key indicator: it measures advancement per cutterhead revolution and reflects productivity, tool wear, energy consumption and, indirectly, project cost and schedule. Improving PR prediction helps refine machine settings and identify unfavorable ground conditions earlier.
Figure 1 shows the TBM working inside the tunnel.

![Figure 1. TBM working inside the tunnel](TBM.jpg)

Figure 2 shows the movement of the TBM and its interactions with the ground mass, highlighting the main operating parameters

![Figure 2. TBM schematic and operating parameters](TBM_density.jpg)

Figure 3 illustrates the excavation zone, muck removal and the geological context around the cutterhead.

![Figure 3. TBM schematic and muck removal](téléchargé.jpg)

This study is positioned at the interface of applied geomechanics and machine learning. The available signals â€” rotation speed, thrust, torque, face pressures, and derived indices such as specific energy, FPI, TPI, etc. â€” characterize both machine behavior and ground response. Machine learning methods, especially tree-based and boosting models (Random Forest, Gradient Boosting, XGBoost) and kernel-based approaches (SVR, RVM), are used to analyze these signals and capture nonlinear relationships and complex interactions that are difficult to describe with simple empirical rules (Khatti & Mishra, 2025; Yagiz, 2008; Breiman, 2001; Friedman, 2001). The datasets used in this work come from the open-access study by Khatti and Mishra (2025), which focuses on TBM penetration rate estimation in mixed-face conditions and examines feature selection and multicollinearity effects on machine and deep learning models.

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

![](Histo_1.png)

![](Histo_2.png)

![](Histo_3.png)

![](Histo_4.png)

![](Histo_5.png)

![](Histo_6.png)

![](Histo_7.png)

![](Histo_8.png)

![](Histo_9.png)

![Figure 5. Histograms of each variable](Histo_10.png)


Figure 6 presents boxplots grouped by coarse geological facies to compare the central tendency and variability of each parameter across different ground conditions. Systematic shifts in median values are observed, particularly for energy indices; for instance, the median TPI shifts significantly between soft and hard facies, moving from a baseline of $\approx 450$ towards much higher values in denser ground. Similarly, the PR median fluctuates around $12\text{ mm/r}$ but shows a much wider dispersion (up to $50\text{ mm/r}$) in specific geological units. This heterogeneity suggests that while coarse facies are strong indicators of TBM behavior, they are best used as a qualitative validation tool to ensure the model's predictions are physically consistent with the known geology, rather than as direct input variables.

![](box_plot_1.png)

![](box_plot_2.png)

![](box_plot_3.png)

![](box_plot_4.png)

![](box_plot_5.png)

![](box_plot_6.png)

![](box_plot_7.png)

![](box_plot_8.png)

![](box_plot_9.png)

![Figure 6. Boxplots of each variable grouped by geological facies](box_plot_10.png)


Figure 7 utilizes 2D KDE and pairwise density maps to visualize non-linear relationships and heteroscedasticity. The density contours reveal a strong, non-linear 'curved' decay for energy indices; for instance, the penetration rate drops sharply as TPI increases beyond $1,000$. Moreover, the plots exhibit clear conditional heteroscedasticity for face pressures (UEP and LEP): while a high-density cluster is concentrated around $0.20\text{ MPa}$, the vertical spread of PR at this specific value ranges from approximately $0$ to $40\text{ mm/r}$. This high variance at a constant pressure indicates that face support alone cannot linearly predict PR, justifying the use of non-linear learners capable of capturing such complex, regime-dependent behaviors.

![Figure 7. 2D KDE for each variables](2D_KDE.png)

Figure 8 presents target-vs-feature scatter plots with LOWESS smoothing to highlight non-linear trends. For energy indices (SE, FPI, and TPI), the LOWESS curve reveals a sharp non-linear decay: for instance, the PR drops precipitously from $\approx 40\text{ mm/r}$ to below $10\text{ mm/r}$ as TPI increases from $0$ to $1,000$, before plateauing. In contrast, the curves for face pressures (UEP and LEP) remain nearly flat around a conditional mean of $15\text{--}20\text{ mm/r}$, regardless of the pressure value, confirming the lack of a strong linear relationship. Extreme outliers, such as those in FPI exceeding $10,000$, were retained as they represent physically genuine high-effort excavation events in very hard ground, and are thus handled via the robust loss functions of the boosting models.

![Figure 8. Scatter plot for non-linear trends](Scatter_plot.png)

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
- Produce `shap.dependence_plot` for the top 2â€“3 predictors to inspect interaction effects and non-monotonic relationships; where interactions appear strong, use `interaction_index` to quantify them.
- Use waterfall and force plots for selected individual cases to illustrate how combinations of features push the prediction above or below the baseline.
- Archive SHAP values and generated figures so that individual explanations can be revisited without recomputing heavy model artifacts.

Interpretation guidance: when a derived index (TPI) ranks highly, we check whether its contribution can be decomposed into the original low-level signals; when circularity is suspected, we rerun the explainability stage excluding TPI to evaluate stability of the variable ranking.

### 2.6 Reproducibility and implementation
All preprocessing, modeling and explainability steps are implemented as standalone scripts and saved artifacts in the repository. The split indices, fitted preprocessing objects, model outputs and SHAP tables are stored to make the workflow reproducible.



## 3 Results and Discussion
### 3.1 Comparative Performance of Regression Models
The predictive performance of five regression families was evaluated using 5-fold cross-validation on the training set and a final evaluation on the independent test set. The results demonstrate a clear hierarchy in predictive accuracy, with tree-based boosting models significantly outperforming kernel-based and linear approaches.

As shown in the cross-validation results, Gradient Boosting emerged as the top-performing model across all metrics. It achieved a mean coefficient of determination ($R^2$) of $0.927$ and a mean RMSE of $3.01$ during the CV phase. This represents a substantial improvement over the Linear Regression baseline, which yielded an $R^2$ of only $0.61$ and a much higher RMSE of $6.98$. The Variance Accounted For (VAF) further confirms this gap, with Gradient Boosting capturing $92.8%$ of the target variance compared to only $61.5%$ for the linear model.

On the final test set, the rankings remain consistent, as illustrated in Figure 9.

![](../AI8/images/ranking_80_20_test_r2.png)

![](../AI8/images/ranking_80_20_test_rmse.png)

![Figure 9. Model ranking on TEST set for MAE RMSE and R²](../AI8/images/ranking_80_20_test_mae.png)

Figure 10. Predicted vs Measured PR scatter plots for all five models on the TEST set (80/20 split). The solid line represents the 1:1 perfect prediction line, while the dashed lines define the Â±20% tolerance band used in geotechnical engineering.To visually confirm these numerical rankings, Figure 10 presents the Predicted vs. Measured scatter plots for each model on the unseen test set. The diagonal solid line represents the ideal 1:1 relationship, while the dashed lines mark the Â±20% tolerance band commonly adopted in geotechnical practice. A model's accuracy is judged by how tightly its data points cluster around the 1:1 line and how many of them fall within the tolerance band. Overall, Figure 10 confirms that the transition from a linear baseline ($R^2 = 0.637$) to a tree-based ensemble ($R^2 = 0.932$) corresponds to a tangible reduction in prediction scatter, validating the use of Gradient Boosting as the reference model for the subsequent explainability analysis.

![Figure 10. Predicted vs Measured PR scatter plots for all models](../AI8/images/predicted_vs_measured_80_20.png)

Gradient Boosting maintained its dominance with a test $R^2$ of $0.932$ and a VAF of $0.932$ (Figure 11), indicating that the model generalizes exceptionally well to unseen operational data. XGBoost and Random Forest followed closely, with $R^2$ values around $0.86$, while the RVM (Relevance Vector Machine) performed moderately. The poor performance of Linear Regression confirms that the relationship between TBM operating parameters and the penetration rate is fundamentally non-linear.

![Figure 11. VAF ranking on the test set](../AI8/images/vaf_ranking.png)

### 3.2 Generalization and Overfitting Analysis
To ensure the reliability of the models, the gap between training and testing performance was analyzed. A significant gap would indicate overfitting, where the model memorizes noise rather than learning the underlying physical trend.

As shown in Figure 12, the RMSE and MAE gaps between the training and testing sets remain narrow for the top-performing models.

![](../AI8/images/ranking_80_20_gap_rmse.png)

![Figure 12. Overfitting view: RMSE and MAE gap between Train and Test](../AI8/images/ranking_80_20_gap_mae.png)

For Gradient Boosting, the $R^2$ transition from training ($0.940$) to testing ($0.932$) is minimal. This stability suggests that the model has successfully captured the underlying physical laws governing the excavation process. While XGBoost shows a slightly larger gap in MAE, its overall performance remains robust. This generalization capability is attributed to the rigorous hyperparameter tuning (controlling tree depth and learning rate) and the use of a normalized feature space, which prevents any single variable from dominating the loss function.

### 3.3 Residual Analysis and Error Distribution
The quality of the predictions was further inspected through residual analysis, which allows us to determine whether the model's errors are random (white noise) or systematic (bias).

The residual histograms for the test set (Figure 13) reveal a stark contrast between the ensemble learners and the linear baseline. The Gradient Boosting model produces a highly leptokurtic distribution, with the vast majority of normalized residuals tightly concentrated around zeroâ€”specifically within the range of $\pm 0.05$. The peak count at zero (exceeding 40 observations) indicates that the model is an unbiased estimator for a significant portion of the dataset. In contrast, the Linear Regression residuals are far more dispersed, spreading from approximately $-0.3$ to $+0.4$. This wide distribution confirms the linear model's inability to capture the inherent volatility and the non-linear "peaks" of the penetration rate, leading to systematic over- and under-estimations.

![](../AI8/images/residuals_hist_test_80_20_linear_regression.png)

![](../AI8/images/residuals_hist_test_80_20_random_forest.png)

![](../AI8/images/residuals_hist_test_80_20_rvm.png)

![](../AI8/images/residuals_hist_test_80_20_xgboost.png)

![Figure 13. Residual histogram for all the AI models on the TEST set](../AI8/images/residuals_hist_test_80_20_gradient_boosting.png)

The residual scatter plots (Figure 14) provide deeper insights into the error behavior relative to the predicted values ($y_{pred}$). For Gradient Boosting, the residuals remain remarkably stable and close to the zero-line for predicted values between $0.1$ and $0.4$. However, a slight "fan-out" effect is observed as $y_{pred}$ exceeds $0.5$, where the vertical dispersion of residuals increases.

![](../AI8/images/residuals_scatter_test_80_20_linear_regression.png)

![](../AI8/images/residuals_scatter_test_80_20_random_forest.png)

![](../AI8/images/residuals_scatter_test_80_20_rvm.png)

![](../AI8/images/residuals_scatter_test_80_20_xgboost.png)

![Figure 14e. Residual scatter plot for all the AI models on the TEST set](../AI8/images/residuals_scatter_test_80_20_gradient_boosting.png)

This widening of the error cloud is a clear manifestation of the conditional heteroscedasticity identified during the Exploratory Data Analysis (Figure 7). It indicates that while the model is extremely precise for typical penetration rates, the variance increases in high-performance regimesâ€”likely due to the higher sensitivity of the PR to subtle changes in geological facies when the ground is softer. Despite this, the maximum residuals for Gradient Boosting remain below $\pm 0.2$, whereas the Linear Regression model exhibits massive systematic errors, with residuals reaching $\pm 0.3$ and showing a distinct non-random pattern (curvature), which proves that the linear assumption is physically inappropriate for this problem.

### 3.4 SHAP-based Explainability of Model Predictions
To transition from raw performance metrics to physical understanding, a SHAP (Shapley Additive Explanations) analysis was performed on the normalized test set. This step is essential to verify whether the Gradient Boosting model's decisions are driven by physically meaningful relationships rather than spurious correlations, and to address the "black-box" criticism often associated with ensemble methods.

#### 3.4.1 Global Feature Importance
The bar plot of mean absolute SHAP values (Figure 14) ranks the operational variables by their overall contribution to the model output.

![Figure 15. Global feature importance based on mean |SHAP|](../AI9_SHAP_GB_80_20/images/shap_summary_bar_test.png)

The results reveal a strict hierarchy: TPI is by far the most influential predictor, with a mean absolute SHAP value of approximately $0.18$. It is followed by T/D3(MT) with an impact of roughly $0.05$. All other variables (UEP, CRS, LEP, F/A(MF)) have a negligible global impact (below $0.01$). This quantitatively confirms the hypothesis formulated during the EDA: the energy expenditure indices are the primary drivers of the penetration rate, while face pressures and raw rotational speed provide only marginal adjustments in the global view.

#### 3.4.2 Directional Impact and Variable Behavior
The SHAP beeswarm plot (Figure 15) illustrates not only the magnitude but also the direction of each feature's impact.

![Figure 16. SHAP beeswarm plot: distribution of feature impacts](../AI9_SHAP_GB_80_20/images/shap_summary_beeswarm_test.png)

The plot exposes a clear, physically coherent pattern:

TPI (Torque Penetration Index): High feature values (red points) are strongly associated with negative SHAP values, clustering around $-0.15$ to $-0.20$. This means that a high TPI pushes the PR prediction downwards. This perfectly aligns with operational geomechanics: high torque relative to penetration indicates a hard, resistant ground mass.
T/D3(MT): The relationship is non-linear. Both high and low extreme values tend to have a negative impact, while intermediate values push the PR upwards. This indicates that the model penalizes abnormal torque-to-thrust ratios, which typically correspond to cutterhead jamming events.

#### 3.4.3 Prediction Pathways and Local Instances
The SHAP decision plot (Figure 16) provides a macro-level view of how the model routes observations from the baseline expectation to their final predictions.

![Figure 17. SHAP decision plot showing prediction pathways](../AI9_SHAP_GB_80_20/images/shap_decision_test.png)

The curves start at the expected baseline value of approximately $0.30$ (the average PR). As they move to the right ($0.4$ to $0.6$), they represent high-performance excavation events, while curves to the left correspond to the "hard rock" regime. This visualization confirms that the model makes confident, decisive splits between the two main operational modes identified during the geological facies analysis.

At a local level, the waterfall plot (Figure 17) dissects a specific high-performance prediction.

![Figure 18. SHAP waterfall plot for a high-performance prediction](../AI9_SHAP_GB_80_20/images/shap_waterfall_median_abs_error_test.png)

For this specific test instance, the model predicted a normalized PR of $0.551$, far above the baseline expectation of $0.296$. The prediction is driven primarily by a very low normalized TPI value ($0.007$), which contributed a massive $+0.31$ to the final score. Other variables, such as T/D3(MT), contributed a minor negative adjustment ($-0.06$), while face pressures and rotational speed had a strictly zero contribution. This confirms that the model uses TPI as a direct "proxy" for ground hardness, isolating soft ground conditions as the key trigger for high penetration rates.

The clustering of these local decisions is summarized in the SHAP heatmap (Figure 18), which shows a clear gradient in the TPI row from red (high positive contribution) to blue (strong negative contribution), validating the model's ability to separate operational regimes.

![Figure 19. SHAP heatmap showing feature value clustering](../AI9_SHAP_GB_80_20/images/shap_heatmap_test.png)

#### 3.5 General Discussion and Operational Implications
The combined results of the regression metrics and the SHAP analysis provide a coherent picture of the TBM's behavior. The transition from a linear baseline ($R^2 \approx 0.63$) to a boosting approach ($R^2 \approx 0.93$) proves that the penetration rate is governed by complex, non-linear interactions between machine energy expenditure and geological response.

The failure of Linear Regression and the RVM (test $R^2 \approx 0.85$) is expected. As the 2D KDE plots revealed, the multimodal distributions of face pressures and the "banana-shaped" density contours imply that a single linear coefficient cannot describe the impact of these variables. Tree-based models succeed because they partition the feature space into hierarchical regions, effectively capturing the "threshold effects" where the PR drops precipitously once a certain ground hardness (TPI threshold $\approx 1,000$) is reached.

A critical concern that often arises in this type of study is the risk of circularity when using derived indices like TPI. However, the SHAP analysis effectively mitigates this concern. The non-linear behavior of T/D3(MT) and the non-zero local contributions of other variables prove that the Gradient Boosting model is not acting as a simple linear inverter. It uses the low-level signals to modulate the TPI-dominated prediction, effectively creating a robust physical proxy.

From an operational standpoint, this model architecture provides a transparent, case-by-case diagnostic tool. Engineers can now visualize why the machine is slowing down (e.g., a high TPI forcing the PR down) and correlate it with the geological facies. The TPI dominance also suggests that future work could focus on real-time TPI monitoring as a leading indicator of geological changes, potentially allowing for adaptive control of the thrust and rotation speed to maintain a target penetration rate.


## 4 Conclusion
This study proposed a reproducible machine learning workflow to predict the penetration rate (PR) of a Tunnel Boring Machine from operational and derived excavation variables. The workflow combined data cleaning, train-only normalization, a fixed 80/20 split, cross-validation on the training set, independent test evaluation, residual diagnostics, and SHAP-based explainability. This complete sequence makes the analysis traceable and reduces the risk of data leakage.

The results confirm that TBM penetration rate is governed by nonlinear relationships that a simple linear model cannot represent. Linear Regression provided a useful baseline, but its lower test performance and structured residual patterns confirmed its inability to capture the machine-ground interaction. In contrast, ensemble learning methods — especially Gradient Boosting — achieved the strongest generalization, reaching a test $R^2 \approx 0.93$ with low RMSE and MAE.

The residual analysis supports this conclusion: residuals of the best model are concentrated around zero, indicating limited global bias. A moderate widening of the residual cloud appears at higher predicted PR values, reflecting the conditional heteroscedasticity already observed during EDA, where high-performance regimes are more sensitive to small geological variations. The training-to-test gap remains limited for Gradient Boosting, confirming that the model does not suffer from severe overfitting and that the chosen preprocessing strategy effectively prevents information leakage.

The SHAP analysis provides the main physical interpretation. TPI emerges as the dominant predictor, with high TPI values associated with negative SHAP contributions — physically coherent, as a high torque demand relative to penetration reflects harder ground conditions. The secondary role of T/D3(MT) shows that the model does not rely on a single derived index but refines its predictions using additional torque-related information. Variables such as UEP, LEP, CRS, and F/A(MF) have weaker global effects but may still contribute locally.

From a methodological perspective, the study demonstrates the value of combining predictive models with diagnostic and explainability tools. Rankings identify the best algorithm, residual plots reveal whether errors are random or systematic, gap analysis evaluates generalization, and SHAP connects the output back to the physical meaning of the input variables. Together, these elements transform the model from a purely statistical predictor into an interpretable decision-support tool.

Operationally, the proposed approach could help engineers monitor excavation conditions in real time. A trained model can estimate expected PR from current operating parameters, while SHAP explanations can indicate which variables are pushing the prediction upward or downward, supporting early detection of unfavorable ground or inefficient operating regimes. Such a workflow could be integrated into a monitoring dashboard updated as new machine data become available.

Overall, this work shows that explainable machine learning can provide accurate and physically meaningful predictions of TBM penetration rate. Gradient Boosting offers the best balance between predictive performance and interpretability, and the strong influence of TPI confirms the importance of energy-based indicators. The proposed methodology represents a promising foundation for data-driven TBM performance assessment and for future decision-support systems in mechanized tunneling.


## References

- Khatti, J., & Mishra, S. (2025). Estimating shield tunnel boring machine penetration rate in mixed face conditions: feature selection and multicollinearity effects on machine and deep learning models. Frontiers in Built Environment, 11, 1699466.
- Yagiz, S. (2008). Utilizing rock mass properties for predicting TBM performance in hard rock tunneling. Tunnelling and Underground Space Technology, 23(3), 326â€“337.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5â€“32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. The Annals of Statistics, 29(5), 1189â€“1232.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.
