# Report 2: Data Preprocessing and Exploratory Data Analysis

## Introduction
This report presents in detail the data preprocessing and exploratory analysis steps carried out for the study on predicting the advance rate (PR) of tunnel boring machines. The objective is to prepare a reliable dataset and explore the relationships between PR and influential parameters, in order to better understand the key factors for future modeling.

## Data Preprocessing
The first step consisted of examining the dataset to identify and handle missing values. Missing entries were either removed when few in number or imputed using the mean or median to maintain dataset consistency. Outliers were detected using boxplots and statistical methods such as the interquartile range. These extreme values were removed to prevent them from negatively impacting subsequent analyses. Normalization was then applied to the variables to harmonize the scales of different parameters, making comparisons and interpretations easier.

### Boxplots of Variables
The following boxplots illustrate the distribution and presence of outliers for each variable:
![Boxplots of variables](Box_plots.png)
*Figure 1: Boxplots showing the spread and outliers for each parameter.*
Each boxplot represents the distribution of values for a given variable. Some variables, such as SE (specific energy), FPI, and TPI, show many points beyond the whiskers, indicating the presence of extreme values (outliers). This justifies their treatment during preprocessing. Other variables, such as CRS (RPM) or AR (mm/min), show a more compact distribution, indicating lower variability. The analysis of these boxplots quickly identifies variables that require special attention to ensure the robustness of subsequent analyses.

For example, the variables SE (specific energy), FPI, and TPI show many points beyond the whiskers, indicating the presence of extreme values (outliers) that could bias statistical analyses. These variables therefore require special treatment, such as removing or imputing outliers, to ensure the reliability of the results. In contrast, variables like CRS (RPM) or AR (mm/min) display a more homogeneous distribution and require fewer adjustments. Thus, the analysis of boxplots helps to precisely target which variables need to be monitored and corrected during preprocessing.

## Exploratory Data Analysis

### Histograms of Variables
The following histograms show the distribution of each variable, highlighting skewness and the presence of extreme values:
![Histograms of variables](Histo.png)
*Figure 2: Histograms showing the distribution of each parameter.*
The histograms reveal that most variables, such as PR, SE, FPI, and TPI, have a skewed distribution with a majority of low values and a few high values. For example, the AR (advance) variable shows a concentration of values around 10 to 20 mm/min, while SE and FPI display a long right tail, indicating a few very high values. This skewness indicates that the machine's behavior is dominated by "normal" conditions, but extreme situations exist and can influence overall performance.

A more detailed analysis of the histograms highlights the asymmetric nature of the distributions for most variables. This skewness, with a majority of low values and a few extreme ones, suggests that the machine’s behavior is generally stable, but rare and extreme events can occur and significantly impact overall performance. For instance, the long right tail seen for SE and FPI indicates the possibility of particularly challenging operating conditions, which should be considered in predictive modeling. This observation emphasizes the importance of using robust statistical methods or appropriate transformations to limit the influence of extreme values.

### Scatter Plots: PR vs Parameters
The following scatter plots illustrate the relationships between the advance rate (PR) and the main parameters:
![Scatter plots: PR vs parameters](scatter_plot.png)
*Figure 3: Scatter plots showing the relationship between PR and each parameter.*
Each plot relates PR to a specific parameter. There is a strong positive correlation between AR (advance) and PR: the higher the advance, the higher the advance rate, which is mechanically logical. Conversely, for SE, FPI, and TPI, the relationship is negative: as these indices increase, PR decreases, indicating increased difficulty in progress. Other parameters, such as CRS (RPM) or F/A (MF), show more diffuse relationships, suggesting a less direct influence on PR.

A deeper examination of the scatter plots reveals not only the strength of the correlations but also the presence of subgroups or nonlinear trends. For example, the positive relationship between AR and PR is very pronounced, but there is also increasing dispersion at higher values, which may indicate the influence of other unaccounted factors. For SE, FPI, and TPI, the negative trend is clear, but some points deviate from the general trend, suggesting special cases or atypical operating conditions. Visual analysis thus helps identify areas of interest for further study, particularly regarding extreme values or unusual groupings.

### Density Plots: PR vs Parameters
Density plots provide further insight into the concentration of data points and the nature of the relationships:
![Density plots: PR vs parameters](2DKDE.png)
*Figure 4: Density plots for PR and each parameter.*
Density plots allow visualization of the areas where observations are most frequent. For AR vs PR, the maximum density is in the area of high advance and high PR, confirming the positive relationship. For SE, FPI, and TPI, the density is concentrated in areas of low PR and high values of these indices, confirming the increased difficulty in these conditions. These visualizations reinforce the conclusions drawn from the scatter plots.

The density plots provide a complementary perspective by highlighting areas of high data concentration. For AR vs PR, the maximum density in the high-value region not only confirms the positive correlation but also suggests that most observations occur under optimal performance conditions. Conversely, for SE, FPI, and TPI, the density is concentrated in areas of low PR and high index values, indicating that difficult situations are less frequent but strongly associated with reduced performance. This analysis helps identify critical operating ranges and guides optimization or maintenance strategies.

### Correlation Matrix (Spearman)
The Spearman correlation matrix below summarizes the strength and direction of relationships between all variables:
![Spearman correlation matrix](Heatmap.png)
*Figure 5: Spearman correlation matrix for all variables.*
The correlation matrix highlights the links between all variables. AR (mm/min) is very strongly positively correlated with PR (0.98), making it the most determining parameter. Conversely, SE, FPI, and TPI show very strong negative correlations (around -0.85), confirming their role as difficulty indicators. CRS (RPM) has a moderate negative correlation with PR (-0.57). The other parameters show weaker correlations, suggesting they play a secondary role in predicting PR.

A detailed analysis of the correlation matrix highlights the structure of dependencies between variables. The very strong positive correlation between AR and PR confirms the central role of advance in machine performance. The marked negative correlations for SE, FPI, and TPI underline their impact as difficulty indicators. It is noteworthy that some variables show moderate or weak correlations, which may indicate indirect effects or complex interactions not captured by bivariate analysis. This matrix also helps detect possible redundancies between variables, thus guiding parameter selection for modeling.

## Interpretation and Discussion
These results confirm the importance of certain mechanical and energy parameters in the advance rate of the TBM. Advance (AR) appears as the main factor, while an increase in specific energy or performance indices indicates increased difficulty in progress. The visualizations and statistical analysis thus help select the most relevant variables for upcoming predictive modeling.

## Conclusion
The preprocessing and exploratory analysis have made the dataset more reliable and provided a better understanding of the structure and relationships between variables. These steps form a solid basis for developing advance rate prediction models.

However, the presence of outliers in the dataset requires special attention during modeling. It will be necessary to assess the impact of these extreme values on the model results to decide whether to keep them or not.
