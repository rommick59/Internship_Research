# Report 2: Data Preprocessing and Exploratory Data Analysis

## Introduction
This report details the data preprocessing and exploratory analysis steps carried out for the study on predicting the advance rate (PR) of tunnel boring machines. The goal is to prepare a reliable dataset and explore the relationships between PR and influential parameters, providing a solid foundation for future modeling.

## Data Preprocessing
The first step involved examining the dataset to identify and handle missing values. Missing entries were either removed when few in number or imputed using the mean or median to maintain dataset consistency. Outliers were detected using boxplots and statistical methods such as the interquartile range. These extreme values were removed to prevent them from negatively impacting subsequent analyses. Normalization was then applied to the variables to harmonize the scales of different parameters, making comparisons and interpretations easier.

### Boxplots of Variables
The following boxplots illustrate the distribution and presence of outliers for each variable:
![Boxplots of variables](Box_plots.png)
*Figure 1: Boxplots showing the spread and outliers for each parameter.*
#### Details and interpretation

## Exploratory Data Analysis

### Histograms of Variables
The histograms below show the distribution of each variable, highlighting skewness and the presence of high or low values:
![Histograms of variables](Histo.png)
*Figure 2: Histograms showing the distribution of each parameter.*
#### Details and interpretation

### Scatter Plots: PR vs Parameters
The following scatter plots illustrate the relationships between the advance rate (PR) and the main parameters:
![Scatter plots: PR vs parameters](scatter_plot.png)
*Figure 3: Scatter plots showing the relationship between PR and each parameter.*
#### Details and interpretation

### Density Plots: PR vs Parameters
Density plots provide further insight into the concentration of data points and the nature of the relationships:
![Density plots: PR vs parameters](2DKDE.png)
*Figure 4: Density plots for PR and each parameter.*
#### Details and interpretation

### Correlation Matrix (Spearman)
The Spearman correlation matrix below summarizes the strength and direction of relationships between all variables:
![Spearman correlation matrix](Heatmap.png)
*Figure 5: Spearman correlation matrix for all variables.*
#### Details and interpretation

## Interpretation and Discussion
These results confirm the importance of certain mechanical and energy parameters in the advance rate of the TBM. Advance rate (AR) appears as the main factor, while higher specific energy or performance indices indicate increased difficulty in progress. The visualizations and statistical analysis thus help select the most relevant variables for upcoming predictive modeling.

## Conclusion
The preprocessing and exploratory analysis have made the dataset more reliable and provided a better understanding of the structure and relationships between variables. These steps form a solid basis for developing advance rate prediction models.
