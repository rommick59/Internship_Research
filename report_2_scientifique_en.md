# Report 2: Data Preprocessing and Exploratory Data Analysis

## 1. Data Preprocessing

Prior to any exploratory or modeling steps, the raw dataset underwent a rigorous preprocessing phase to ensure data quality and reliability. The following steps were systematically applied:

### 1.1 Handling Missing Values
All rows containing at least one missing value were removed from the dataset. This approach, while potentially reducing the dataset size, ensures that no imputation bias is introduced and that subsequent analyses are based on complete cases only. This is particularly important for machine learning models that are sensitive to missing data.

### 1.2 Outlier Removal
Outliers were detected and eliminated using the Interquartile Range (IQR) method, applied to all numerical columns. For each feature, values lying outside 1.5 times the IQR from the first and third quartiles were considered outliers and removed. This step reduces the influence of extreme values, which can distort statistical analyses and degrade model performance.

### 1.3 Data Normalization
The remaining numerical features were standardized using the StandardScaler method, which transforms each feature to have a mean of zero and a standard deviation of one. This normalization is essential for algorithms that are sensitive to the scale of input data, such as those based on distance metrics or regularization.

### 1.4 Export of Cleaned Dataset
The preprocessed data was saved as a new file, ensuring reproducibility and traceability for subsequent analysis steps. This cleaned dataset serves as the foundation for all further exploratory and modeling work.

As a result of this preprocessing pipeline, the dataset is now clean, consistent, and ready for robust exploratory data analysis and model development.

### 1.5 Outlier Impact Study for PR Prediction
To determine whether extreme values are harmful or informative for PR prediction, a comparative study was conducted on three dataset variants:

- D0: cleaned data without outlier removal (1197 rows)
- D1: IQR-based outlier removal (1071 rows, 126 removed rows)
- D2: winsorization at 1%-99% (1197 rows)

Three models were evaluated under the same protocol on D0, D1, and D2: Linear Regression, Huber Regressor, and Random Forest.
The selected metrics were MAE, RMSE, and R2 in cross-validation and on a hold-out test set.

Results show that Random Forest is the best model across all dataset variants. In cross-validation, D0 provides the best overall trade-off (lowest mean RMSE: 0.596). On the test split, D1 yields the lowest RMSE (0.410), suggesting that IQR filtering can improve prediction for less extreme samples.

These findings indicate that outliers are not purely detrimental:

- they can degrade some linear models,
- they may still carry useful information,
- and removing them changes the problem distribution.

Methodologically, the final decision for the next stage is to use D0 (dataset with retained outliers) as the single reference dataset. This choice is justified by its best overall cross-validation performance, its larger sample size, and the preservation of potentially informative extreme observations for PR prediction.

The result tables used for this analysis are available in the outlier_study folder: dataset_overview.csv, model_comparison_cv_test.csv, and best_model_per_dataset.csv.

## 2. Exploratory Data Analysis (EDA)

### 2.1 Objective and workflow
In accordance with the Week 4 plan, EDA was conducted to characterize the statistical structure of the selected dataset (D0), identify relationships between PR and explanatory variables, and guide regression model selection. The following analyses were performed:

- univariate visualization (histograms),
- extreme-value inspection (boxplots),
- bivariate visualization of PR versus predictors (scatter plots),
- correlation analysis (Spearman matrix),
- preparation of a train/test split protocol.

### 2.2 Univariate distributions
Histogram analysis shows heterogeneous distributions across variables:

- CRS and AR are concentrated in specific operational ranges,
- F/A, T/D3, UEP, and LEP are more spread,
- SE, FPI, and TPI are right-skewed with long tails,
- PR is non-Gaussian, with concentration at low-to-intermediate values and a smaller number of high values.

These patterns confirm a non-linear data structure and motivate robust modeling choices beyond purely linear assumptions.

### 2.3 Boxplots and interpretation of extreme values
Boxplots reveal extreme observations, especially for SE, FPI, and TPI. From an operational and geotechnical perspective, these points can represent difficult excavation regimes (high specific energy and lower penetration) rather than systematic measurement errors.

Consistent with the outlier-impact study (Section 1.5), this supports retaining D0 as the reference dataset to preserve potentially informative operating conditions for PR prediction.

### 2.4 PR versus predictors
Scatter plots of PR against predictors highlight:

- a strong positive relationship between AR and PR,
- strong negative relationships between PR and SE/FPI/TPI,
- more complex non-linear relationships for F/A, T/D3, UEP, and LEP,
- an overall negative trend between CRS and PR over the observed range.

The observed cloud structures and non-linear patterns indicate that purely linear models may be insufficient to capture the full system behavior.

### 2.5 Correlation analysis (Spearman)
The Spearman correlation matrix confirms visual trends and quantifies monotonic dependencies:

- corr(AR, PR) = 0.98 (very strong positive),
- corr(CRS, PR) = -0.57 (moderate-to-strong negative),
- corr(SE, PR) = -0.85, corr(FPI, PR) = -0.86, corr(TPI, PR) = -0.85,
- corr(F/A, PR) = -0.26, corr(T/D3, PR) = -0.36,
- corr(UEP, PR) = -0.04, corr(LEP, PR) = -0.13.

High inter-feature correlations are also present (e.g., SE-FPI-TPI and UEP-LEP), indicating partial multicollinearity. This property was considered in the model-selection strategy (Section 1.5).

### 2.6 Train/test split strategy
To prepare the modeling phase, data were split into training and testing sets using an 80/20 hold-out design, combined with 5-fold cross-validation on the training set. This protocol provides:

- robust generalization assessment,
- fair comparison across regression methods,
- reduced overfitting risk.

### 2.7 EDA summary
EDA confirms that PR prediction is a non-linear regression problem with interacting variables and informative extreme regimes. Week 4 results therefore support the following decisions:

- retain D0 as the reference dataset,
- prioritize robust/non-linear models,
- interpret model quality through both predictive metrics and physical plausibility.

## 3. EDA Figures

### Figure 1. Univariate distributions (histograms)
![Figure 1 - Histograms of variables](Histo.png)

### Figure 2. Visual detection of extreme values (boxplots)
![Figure 2 - Boxplots](Box_plots.png)

### Figure 3. Bivariate relationships between PR and predictors (scatter plots)
![Figure 3 - Scatter plots PR vs predictors](scatter_plot.png)

### Figure 4. Spearman correlation matrix
![Figure 4 - Spearman correlation heatmap](Heatmap.png)

### Figure 5. Joint 2D density (Kernel Density Estimation)
![Figure 5 - 2D KDE](2DKDE.png)
