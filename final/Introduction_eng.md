# Introduction

Mechanized excavation using Tunnel Boring Machines (TBMs) is a cornerstone technique for building underground infrastructure (transportation, water, utilities). The penetration rate (PR, mm/rev), which measures advancement per cutterhead revolution, is a key operational indicator: it conditions productivity, tool wear, energy consumption and, ultimately, project cost and schedule. Accurately predicting PR from machine signals and derived indices enables better operational setpoints, early warning for adverse conditions and improved construction planning.

This study sits at the intersection of applied geomechanics and machine learning. The available signals (rotation speed, thrust, torque, face pressures, and derived indices such as specific energy, FPI, TPI, etc.) convey both physical measurements and compact summaries of the tool–ground interaction. Machine learning methods—particularly tree-based and boosting models (Random Forest, Gradient Boosting, XGBoost) and kernel-based approaches (SVR, RVM)—can capture nonlinear relationships and complex interactions among these variables, which are often hard to express with simple empirical rules (Khatti & Mishra, 2025; Yagiz, 2008; Breiman, 2001; Friedman, 2001). For this work, the datasets used are drawn from the open-access study by Khatti and Mishra (2025), which focuses on TBM penetration rate estimation in mixed-face conditions and examines feature selection and multicollinearity effects on machine and deep learning models.

The manuscript builds on internal project reports and incorporates previously generated visual materials (EDA plots and SHAP figures) to illustrate and support the analysis.

Beyond raw predictive performance, model interpretability is essential in this industrial context: identifying which variables drive predictions helps detect potential information leakage (derived features close to the target), verify the physical plausibility of results and increase model trust for practitioners. For interpretability, we rely primarily on SHAP (Shapley Additive exPlanations) to explain variable contributions at both the global and the observation level (Lundberg & Lee, 2017).

The study follows a concrete, reproducible protocol:

- Data collection and preparation: acquire TBM datasets (open-access sources and internal reports), clean the data, handle missing values and construct operational indices (specific energy, FPI, TPI, etc.).
- Model experimentation and comparison: train and validate several families of supervised models (linear regression, Random Forest, Gradient Boosting/XGBoost, SVR, RVM, AdaBoost) using multiple train/validation/test splits; evaluate using standard metrics (R, R2, RMSE, MAE) and assess performance robustness.
- Interpretability analysis: apply SHAP to identify dominant variables, examine the distribution of contributions across observations, and conduct sensitivity checks (for example, ablation of derived features such as TPI) to assess the stability of conclusions.
- Operational recommendations and perspectives: synthesize findings into practical guidance for TBM operation and propose future work (additional tests, external dataset validation).

These steps structure the experimental protocol described in the manuscript and support reproducibility of the reported results.

Main contributions of this paper:
- a preprocessing pipeline tailored to TBM data and a selection of relevant operational indices;
- an empirical, reproducible comparison of baseline and advanced models across multiple validation protocols;
- an in-depth SHAP-based interpretability study of the selected model that highlights dominant variables and assesses the robustness of conclusions with respect to derived indices.

Selected references mentioned in this introduction:
- Khatti, J., & Mishra, S. (2025). Estimating shield tunnel boring machine penetration rate in mixed face conditions: feature selection and multicollinearity effects on machine and deep learning models. Frontiers in Built Environment, 11, 1699466.
- Yagiz, S. (2008). Utilizing rock mass properties for predicting TBM performance in hard rock tunneling. Tunnelling and Underground Space Technology, 23(3), 326–337.
- Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.
- Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. The Annals of Statistics, 29(5), 1189–1232.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems.
