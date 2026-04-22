# Report 2: Data Preprocessing and Exploratory Data Analysis

## Introduction

This report presents in detail the data preprocessing and exploratory analysis steps carried out for the study on predicting the advance rate (PR) of tunnel boring machines. The objective is to prepare a reliable dataset and explore the relationships between PR and influential parameters, in order to better understand the key factors for future modeling.

## Data Preprocessing

The first step consisted of examining the dataset to identify and handle missing values. Missing entries were either removed when few in number or imputed using the mean or median to maintain dataset consistency. Outliers were detected using boxplots and statistical methods such as the interquartile range. These extreme values were removed to prevent them from negatively impacting subsequent analyses. Normalization was then applied to the variables to harmonize the scales of different parameters, making comparisons and interpretations easier.

### Boxplots of Variables

The following boxplots illustrate the distribution and presence of outliers for each variable:
![Boxplots of variables](images/Box_plots_p1.png) 
![Boxplots of variables](images/Box_plots_p2.png)
_Figure 1 and 2: Boxplots showing the spread and outliers for each parameter._

### Legend for Boxplots

**Legend:**
- The bottom whisker indicates the minimum value.
- The top whisker indicates the maximum value.
- The line inside the box represents the median.
- The blue rectangle (box) represents the interquartile range (IQR), which contains the middle 50% of the data.
- The range (min–max) for each variable is displayed on its respective axis.

**Boxplot summary for each variable:**

**CRS (RPM)**
- The bottom whisker indicates the minimum value: 1.1
- The top whisker indicates the maximum value: 1.5
- The line inside the box represents the median: 1.3
- The blue box shows the interquartile range (IQR): from 1.2 (Q1) to 1.3 (Q3)
- The range for CRS (RPM) is 1.1 to 1.5

**AR (mm/min)**
- The bottom whisker indicates the minimum value: 7.0
- The top whisker indicates the maximum value: 51.0
- The line inside the box represents the median: 35.0
- The blue box shows the IQR: from 30.0 (Q1) to 40.0 (Q3)
- The range for AR (mm/min) is 7.0 to 51.0

**F/A(MF)**
- The bottom whisker indicates the minimum value: 196.17
- The top whisker indicates the maximum value: 488.14
- The line inside the box represents the median: 288.93
- The blue box shows the IQR: from 278.28 (Q1) to 349.76 (Q3)
- The range for F/A(MF) is 196.17 to 488.14

**T/D3(MT)**
- The bottom whisker indicates the minimum value: 1.92
- The top whisker indicates the maximum value: 9.27
- The line inside the box represents the median: 3.30
- The blue box shows the IQR: from 2.87 (Q1) to 4.31 (Q3)
- The range for T/D3(MT) is 1.92 to 9.27

**UEP (MPa)**
- The bottom whisker indicates the minimum value: 0.03
- The top whisker indicates the maximum value: 0.19
- The line inside the box represents the median: 0.13
- The blue box shows the IQR: from 0.12 (Q1) to 0.16 (Q3)
- The range for UEP (MPa) is 0.03 to 0.19

**LEP (MPa)**
- The bottom whisker indicates the minimum value: 0.04
- The top whisker indicates the maximum value: 0.21
- The line inside the box represents the median: 0.14
- The blue box shows the IQR: from 0.12 (Q1) to 0.16 (Q3)
- The range for LEP (MPa) is 0.04 to 0.21

**SE(kW.h/m3)**
- The bottom whisker indicates the minimum value: 1.58
- The top whisker indicates the maximum value: 31.96
- The line inside the box represents the median: 2.76
- The blue box shows the IQR: from 2.01 (Q1) to 3.24 (Q3)
- The range for SE(kW.h/m3) is 1.58 to 31.96

**FPI**
- The bottom whisker indicates the minimum value: 463.08
- The top whisker indicates the maximum value: 5142.86
- The line inside the box represents the median: 783.33
- The blue box shows the IQR: from 605.29 (Q1) to 1248.0 (Q3)
- The range for FPI is 463.08 to 5142.86

**TPI**
- The bottom whisker indicates the minimum value: 56.8
- The top whisker indicates the maximum value: 1200.0
- The line inside the box represents the median: 92.0
- The blue box shows the IQR: from 66.44 (Q1) to 197.6 (Q3)
- The range for TPI is 56.8 to 1200.0

**PR(mm/r)**
- The bottom whisker indicates the minimum value: 5.83
- The top whisker indicates the maximum value: 38.33
- The line inside the box represents the median: 26.92
- The blue box shows the IQR: from 15.38 (Q1) to 31.82 (Q3)
- The range for PR(mm/r) is 5.83 to 38.33


Each boxplot represents the distribution of values for a given variable. Some variables, such as SE (specific energy), FPI, and TPI, show many points beyond the whiskers, indicating the presence of extreme values (outliers). This justifies their treatment during preprocessing. Other variables, such as CRS (RPM) or AR (mm/min), show a more compact distribution, indicating lower variability. The analysis of these boxplots quickly identifies variables that require special attention to ensure the robustness of subsequent analyses.

For example, the variables SE (specific energy), FPI, and TPI show many points beyond the whiskers, indicating the presence of extreme values (outliers) that could bias statistical analyses. These variables therefore require special treatment, such as removing or imputing outliers, to ensure the reliability of the results. In contrast, variables like CRS (RPM) or AR (mm/min) display a more homogeneous distribution and require fewer adjustments. Thus, the analysis of boxplots helps to precisely target which variables need to be monitored and corrected during preprocessing.

## Exploratory Data Analysis

### Histograms of Variables

The following histograms show the distribution of each variable, highlighting skewness and the presence of extreme values:
![Histograms of variables](images/Histo_p1.png)
![Histograms of variables](images/Histo_p2.png)
![Histograms of variables](images/Histo_p3.png)

_Figure 3, 4 and 5 : Histograms showing the distribution of each parameter._

A more detailed analysis of the histograms highlights the asymmetric nature of the distributions for most variables. This skewness, with a majority of low values and a few extreme ones, suggests that the machine’s behavior is generally stable, but rare and extreme events can occur and significantly impact overall performance. For instance, the long right tail seen for SE and FPI indicates the possibility of particularly challenging operating conditions, which should be considered in predictive modeling. This observation emphasizes the importance of using robust statistical methods or appropriate transformations to limit the influence of extreme values.

## Histogram Analysis for Each Variable:

- **CRS (RPM):** The distribution of CRS (RPM) is left-skewed, with most values tightly concentrated between 1.2 and 1.3. The spread is very narrow, indicating low variability in this parameter. There are very few higher values, and no significant outliers are observed, suggesting stable machine rotation speed during operation.

- **AR (mm/min):** The AR (mm/min) variable shows a right-skewed distribution. Most values are clustered between 20 and 40 mm/min, but there is a noticeable tail towards higher values, indicating occasional periods of much higher advance rates. This suggests that while the machine usually operates at moderate speeds, it sometimes encounters conditions requiring faster advancement.

- **F/A(MF):** F/A(MF) is approximately symmetric, with the majority of data points falling between 250 and 350. The distribution is relatively balanced, with a few extreme values on both sides but no strong skewness. This indicates a consistent ratio for this parameter across the dataset.

- **T/D3(MT):** The T/D3(MT) distribution is slightly right-skewed, with most values between 2 and 4. There are a few higher values, creating a long right tail, which may correspond to rare events or specific operational conditions.

- **UEP (MPa):** UEP (MPa) is highly right-skewed, with the vast majority of values close to the minimum (around 0.1 MPa). A few higher outliers are present, indicating that high UEP values are rare but possible, likely reflecting occasional spikes in pressure.

- **LEP (MPa):** LEP (MPa) also displays a right-skewed distribution, with most values near the lower end (0.1–0.15 MPa) and some higher values extending the tail. This suggests that low LEP values are typical, but the system can experience higher pressures in certain cases.

- **SE(kW.h/m3):** SE(kW.h/m3) has a pronounced right-skew, with most data points at low values (around 2–3) and a long tail of higher values. This indicates that the machine usually operates efficiently, but there are rare instances of much higher specific energy consumption, possibly due to difficult ground conditions.

- **FPI:** FPI shows a strong right-skewed distribution, with most values concentrated at the lower end (around 500–800) and several high outliers. The presence of these outliers suggests that, while typical performance is consistent, there are occasional periods of much higher FPI.

- **TPI:** TPI is right-skewed, with the majority of values between 50 and 200. However, a few much higher values create a long tail, indicating that while most operations are within a standard range, some exceptional cases occur.

- **PR(mm/r):** PR(mm/r) is slightly right-skewed, with most values between 10 and 35. A few higher values extend the distribution, suggesting that while penetration rates are generally moderate, there are instances of significantly higher rates.

### Scatter Plots: PR vs Parameters

The following scatter plots illustrate the relationships between the advance rate (PR) and the main parameters:
![Scatter plots: PR vs parameters](images/scatter_plot_p1.png)
![Scatter plots: PR vs parameters](images/scatter_plot_p2.png)
![Scatter plots: PR vs parameters](images/scatter_plot_p3.png)

_Figure 4: Scatter plots showing the relationship between PR and each parameter._

A deeper examination of the scatter plots reveals not only the strength of the correlations but also the presence of subgroups or nonlinear trends. For example, the positive relationship between AR and PR is very pronounced, but there is also increasing dispersion at higher values, which may indicate the influence of other unaccounted factors. For SE, FPI, and TPI, the negative trend is clear, but some points deviate from the general trend, suggesting special cases or atypical operating conditions. Visual analysis thus helps identify areas of interest for further study, particularly regarding extreme values or unusual groupings.

**Detailed scatter plot analysis (PR on the x-axis):**

- **CRS (RPM) vs PR:** No clear trend is observed (relationship is weak/absent). Points are widely scattered across PR values, with no obvious subgroups and only limited atypical points.

- **AR (mm/min) vs PR:** A strong positive and mainly linear relationship is visible. Most points align along a clear trend, while dispersion increases at higher PR, suggesting additional factors may influence performance in that range.

- **F/A(MF) vs PR:** The relationship appears weak, with a diffuse cloud and no consistent increasing/decreasing pattern. This suggests F/A(MF) alone does not explain PR variability, and no distinct clusters dominate the plot.

- **T/D3(MT) vs PR:** A weak association is observed, possibly non-linear. The points are dispersed and include a few atypical observations, which may indicate special operating conditions rather than a stable monotonic trend.

- **UEP (MPa) vs PR:** A mild negative tendency is visible: higher UEP values occur more often at lower PR. The scatter remains broad, indicating a weak-to-moderate relationship with some exceptions/outliers.

- **LEP (MPa) vs PR:** The pattern is weak and slightly negative. Most points cluster at low LEP values while PR varies widely, with a small number of higher-LEP cases tending toward lower PR.

- **SE (kW.h/m3) vs PR:** A clear negative and non-linear relationship is present. Low SE corresponds to a wide range of PR, whereas high SE is mostly associated with low PR; extreme high-SE points reflect rare, difficult conditions.

- **FPI vs PR:** A negative association is observed, with higher FPI more frequent when PR is low. The trend is visible but not perfectly linear, and several points deviate from the main pattern.

- **TPI vs PR:** The scatter plot indicates a negative relationship, with higher TPI generally linked to lower PR. Dispersion increases at higher TPI values, and a few outliers suggest atypical or extreme cases.

### Density Plots: PR vs Parameters

Density plots provide further insight into the concentration of data points and the nature of the relationships:
![Density plots: PR vs parameters](images/2DKDE_p1.png)
![Density plots: PR vs parameters](images/2DKDE_p2.png)
![Density plots: PR vs parameters](images/2DKDE_p3.png)

_Figure 5: Density plots for PR and each parameter._

The density plots provide a complementary perspective by highlighting areas of high data concentration. For AR vs PR, the maximum density in the high-value region not only confirms the positive correlation but also suggests that most observations occur under optimal performance conditions. Conversely, for SE, FPI, and TPI, the density is concentrated in areas of low PR and high index values, indicating that difficult situations are less frequent but strongly associated with reduced performance. This analysis helps identify critical operating ranges and guides optimization or maintenance strategies.

**Detailed density plot analysis (PR on the x-axis):**

- **CRS (RPM) vs PR:** The highest-density region is concentrated around CRS values close to 1.3–1.4, with PR spanning a broad range. The pattern suggests a moderate negative tendency overall, but the relationship is not purely linear due to the discrete/clustered CRS levels (vertical bands) and dispersion across PR.

- **AR (mm/min) vs PR:** Density is strongly concentrated along a clear diagonal ridge, indicating a strong positive and mostly linear relationship. The highest-density zone occurs at lower-to-moderate PR and AR values, while the spread increases toward higher PR, suggesting additional variability under high-performance conditions.

- **F/A(MF) vs PR:** The density is relatively spread out, with a broad concentration band rather than a sharp ridge. Only a weak (slightly negative) tendency is visible, indicating a limited direct relationship with PR and no distinct subgroups dominating the distribution.

- **T/D3(MT) vs PR:** The highest-density region is centered around moderate T/D3(MT) values, while PR varies widely. A mild-to-moderate negative tendency can be observed, but the relationship appears weakly non-linear and dispersed, with some atypical zones that may reflect specific operating regimes.

- **UEP (MPa) vs PR:** Density is mainly concentrated at very low UEP values, with PR distributed across low to moderate ranges. No clear trend is visible (relationship is close to absent), and the plot mostly shows a compact high-density core with scattered low-density points.

- **LEP (MPa) vs PR:** The density peak occurs at low LEP values, and PR covers a wide range in that region. The tendency is weak and slightly negative, with a dispersed pattern rather than a distinct linear ridge, suggesting limited predictive value of LEP alone.

- **SE (kW.h/m3) vs PR:** A strong negative and clearly non-linear relationship is visible. The highest density occurs at low SE with high PR, while higher SE values concentrate at lower PR; a long tail toward extreme SE highlights rare but severe operating conditions (atypical cases).

- **FPI vs PR:** The density plot shows a strong negative association: high PR occurs mostly where FPI is low-to-moderate, whereas high FPI values are concentrated at low PR. The relationship is not perfectly linear and includes scattered low-density regions that may correspond to outliers or special operating events.

- **TPI vs PR:** A strong negative tendency is evident, with the highest density at low TPI and higher PR. As TPI increases, density shifts toward lower PR, and dispersion increases, indicating more variability and occasional atypical cases at high index values.

### Correlation Matrix (Spearman)

The Spearman correlation matrix below summarizes the strength and direction of relationships between all variables:
![Spearman correlation matrix](images/Heatmap2.png)
_Figure 6: Spearman correlation matrix for all variables._

A more detailed reading of the Spearman matrix reveals several important patterns in both direct and inverse monotonic relationships. First, the correlation between AR and PR is extremely strong and positive (0.98), confirming that higher advance is consistently associated with higher penetration rate. In contrast, the relationships between PR and SE, FPI, and TPI are strongly negative (-0.85 to -0.86), meaning that as these difficulty-related indicators increase, PR tends to decrease markedly; this is a clear inverse correlation pattern.

The matrix also distinguishes medium and weak effects: CRS has a moderate negative association with PR (0.57), while T/D3 and F/A(MF) are weaker-to-moderate negatives (-0.36$ and -0.26). UEP and LEP show weak links with PR (close to 0), suggesting limited direct monotonic predictive power when considered alone. In addition, very high positive correlations among explanatory variables (for example SE-TPI near 1.00 and strong FPI-SE/TPI links) indicate possible redundancy and multicollinearity risk. For modeling, this suggests prioritizing feature selection or regularization to avoid over-weighting strongly overlapping variables and to improve model robustness and interpretability.


## Interpretation and Discussion

These results confirm the importance of certain mechanical and energy parameters in the advance rate of the TBM. Advance (AR) appears as the main factor, while an increase in specific energy or performance indices indicates increased difficulty in progress. The visualizations and statistical analysis thus help select the most relevant variables for upcoming predictive modeling.

## Conclusion

The preprocessing and exploratory analysis have made the dataset more reliable and provided a better understanding of the structure and relationships between variables. These steps form a solid basis for developing advance rate prediction models.
However, the presence of outliers in the dataset requires special attention during modeling. It will be necessary to assess the impact of these extreme values on the model results to decide whether to keep them or not.
