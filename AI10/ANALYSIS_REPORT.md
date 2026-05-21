# AI10: PDP & ICE Analysis Report
## Gradient Boosting Model - 6 Features (80/20 Split)

---

## Executive Summary

This report provides a detailed analysis of feature dependence in the Gradient Boosting model
using Partial Dependence Plots (PDP) and Individual Conditional Expectation (ICE) plots.

### Dependency Metrics Overview

| Feature | Importance (%) | PDP Range | PDP Slope | ICE Heterogeneity |
|---------|---|---|---|---|
| TPI | 89.56% | 0.538741 | 0.010995 | 0.032541 |
| T/D3(MT) | 10.34% | 0.135903 | 0.002774 | 0.235813 |
| UEP (MPa) | 0.10% | 0.002825 | 0.000086 | 0.189730 |
| CRS (RPM) | 0.00% | 0.000000 | 0.000000 | 0.189581 |
| F/A(MF) | 0.00% | 0.000000 | 0.000000 | 0.189581 |
| LEP (MPa) | 0.00% | 0.000000 | 0.000000 | 0.189581 |

---

## Detailed Feature Analysis

### 1. TPI

#### Feature Statistics
- **Range**: [-0.000977, 1.000000]
- **Importance**: 89.56%

#### Partial Dependence Plot (PDP) Metrics
- **PDP Range**: 0.538741
  - Interpretation: How much the predicted value varies across the feature range
  - Higher values indicate stronger marginal effect
- **PDP Mean**: -0.142344
  - Average prediction when averaging over all other features
- **PDP Std Dev**: 0.102709
  - Variability of predictions within the PDP
- **PDP Slope**: 0.010995
  - Average rate of change per unit increase in feature
  - Higher values indicate non-linearity or strong interactions

#### Individual Conditional Expectation (ICE) Metrics
- **ICE Heterogeneity**: 0.032541
  - Standard deviation of predictions across individual curves at each point
  - High values indicate heterogeneous effects (different impact per sample)
- **ICE Range (Mean)**: 0.662605
  - Average spread of predictions for individual samples
  - Shows instance-specific variability
- **ICE Range (Std)**: 0.072350
  - Consistency of individual effect sizes across samples
  - Low values = consistent effects; High values = heterogeneous effects

### 2. T/D3(MT)

#### Feature Statistics
- **Range**: [-0.017173, 1.000000]
- **Importance**: 10.34%

#### Partial Dependence Plot (PDP) Metrics
- **PDP Range**: 0.135903
  - Interpretation: How much the predicted value varies across the feature range
  - Higher values indicate stronger marginal effect
- **PDP Mean**: 0.038773
  - Average prediction when averaging over all other features
- **PDP Std Dev**: 0.047884
  - Variability of predictions within the PDP
- **PDP Slope**: 0.002774
  - Average rate of change per unit increase in feature
  - Higher values indicate non-linearity or strong interactions

#### Individual Conditional Expectation (ICE) Metrics
- **ICE Heterogeneity**: 0.235813
  - Standard deviation of predictions across individual curves at each point
  - High values indicate heterogeneous effects (different impact per sample)
- **ICE Range (Mean)**: 0.143627
  - Average spread of predictions for individual samples
  - Shows instance-specific variability
- **ICE Range (Std)**: 0.086704
  - Consistency of individual effect sizes across samples
  - Low values = consistent effects; High values = heterogeneous effects

### 3. UEP (MPa)

#### Feature Statistics
- **Range**: [0.000000, 1.000000]
- **Importance**: 0.10%

#### Partial Dependence Plot (PDP) Metrics
- **PDP Range**: 0.002825
  - Interpretation: How much the predicted value varies across the feature range
  - Higher values indicate stronger marginal effect
- **PDP Mean**: -0.000398
  - Average prediction when averaging over all other features
- **PDP Std Dev**: 0.001400
  - Variability of predictions within the PDP
- **PDP Slope**: 0.000086
  - Average rate of change per unit increase in feature
  - Higher values indicate non-linearity or strong interactions

#### Individual Conditional Expectation (ICE) Metrics
- **ICE Heterogeneity**: 0.189730
  - Standard deviation of predictions across individual curves at each point
  - High values indicate heterogeneous effects (different impact per sample)
- **ICE Range (Mean)**: 0.002854
  - Average spread of predictions for individual samples
  - Shows instance-specific variability
- **ICE Range (Std)**: 0.002398
  - Consistency of individual effect sizes across samples
  - Low values = consistent effects; High values = heterogeneous effects

### 4. CRS (RPM)

#### Feature Statistics
- **Range**: [0.000000, 1.000000]
- **Importance**: 0.00%

#### Partial Dependence Plot (PDP) Metrics
- **PDP Range**: 0.000000
  - Interpretation: How much the predicted value varies across the feature range
  - Higher values indicate stronger marginal effect
- **PDP Mean**: -0.000000
  - Average prediction when averaging over all other features
- **PDP Std Dev**: 0.000000
  - Variability of predictions within the PDP
- **PDP Slope**: 0.000000
  - Average rate of change per unit increase in feature
  - Higher values indicate non-linearity or strong interactions

#### Individual Conditional Expectation (ICE) Metrics
- **ICE Heterogeneity**: 0.189581
  - Standard deviation of predictions across individual curves at each point
  - High values indicate heterogeneous effects (different impact per sample)
- **ICE Range (Mean)**: 0.000000
  - Average spread of predictions for individual samples
  - Shows instance-specific variability
- **ICE Range (Std)**: 0.000000
  - Consistency of individual effect sizes across samples
  - Low values = consistent effects; High values = heterogeneous effects

### 5. F/A(MF)

#### Feature Statistics
- **Range**: [0.000000, 1.037309]
- **Importance**: 0.00%

#### Partial Dependence Plot (PDP) Metrics
- **PDP Range**: 0.000000
  - Interpretation: How much the predicted value varies across the feature range
  - Higher values indicate stronger marginal effect
- **PDP Mean**: -0.000000
  - Average prediction when averaging over all other features
- **PDP Std Dev**: 0.000000
  - Variability of predictions within the PDP
- **PDP Slope**: 0.000000
  - Average rate of change per unit increase in feature
  - Higher values indicate non-linearity or strong interactions

#### Individual Conditional Expectation (ICE) Metrics
- **ICE Heterogeneity**: 0.189581
  - Standard deviation of predictions across individual curves at each point
  - High values indicate heterogeneous effects (different impact per sample)
- **ICE Range (Mean)**: 0.000000
  - Average spread of predictions for individual samples
  - Shows instance-specific variability
- **ICE Range (Std)**: 0.000000
  - Consistency of individual effect sizes across samples
  - Low values = consistent effects; High values = heterogeneous effects

### 6. LEP (MPa)

#### Feature Statistics
- **Range**: [0.000000, 1.000000]
- **Importance**: 0.00%

#### Partial Dependence Plot (PDP) Metrics
- **PDP Range**: 0.000000
  - Interpretation: How much the predicted value varies across the feature range
  - Higher values indicate stronger marginal effect
- **PDP Mean**: -0.000000
  - Average prediction when averaging over all other features
- **PDP Std Dev**: 0.000000
  - Variability of predictions within the PDP
- **PDP Slope**: 0.000000
  - Average rate of change per unit increase in feature
  - Higher values indicate non-linearity or strong interactions

#### Individual Conditional Expectation (ICE) Metrics
- **ICE Heterogeneity**: 0.189581
  - Standard deviation of predictions across individual curves at each point
  - High values indicate heterogeneous effects (different impact per sample)
- **ICE Range (Mean)**: 0.000000
  - Average spread of predictions for individual samples
  - Shows instance-specific variability
- **ICE Range (Std)**: 0.000000
  - Consistency of individual effect sizes across samples
  - Low values = consistent effects; High values = heterogeneous effects

---

## Dependency Interpretation Guide

### What do these metrics mean?

**Feature Importance**
- Percentage contribution to model predictions
- Sum of all features = 100%
- Higher = more influential on model decisions

**PDP Range**
- Spread of average predictions across the feature range
- Larger range = stronger marginal effect
- Shows 'average effect' after averaging over other features

**PDP Slope**
- Average rate of prediction change
- Indicates non-linearity and average steepness
- High slope = predictions change rapidly with feature

**ICE Heterogeneity**
- How much individual samples deviate from the PDP average
- High heterogeneity = different samples affected differently (interactions present)
- Low heterogeneity = uniform effect across all samples

**ICE Range**
- Prediction spread for individual samples
- Shows how sensitive each instance is to feature changes
- High variance = some samples very sensitive, others less so

---

## Key Findings

1. **TPI** is the dominant feature with 89.56% importance
   - PDP Range: 0.538741
   - ICE Heterogeneity: 0.032541

2. **T/D3(MT)** shows the highest heterogeneity (0.235813)
   - Indicates strong instance-specific effects or interactions

3. **TPI** shows the steepest average change (0.010995)
   - Non-linear relationships present

---

## Generated Visualizations

The following visualizations have been generated in `AI10/images/`:

### Partial Dependence Plots (PDP)
- Show the marginal effect of each feature
- X-axis: Feature value (normalized)
- Y-axis: Predicted PR (penetration rate)
- One line per feature showing average effect

### Individual Conditional Expectation (ICE) Plots
- Show individual sample effects
- Thin lines: individual sample curves
- Thick red line: PDP average
- Dense/sparse spread indicates homogeneous/heterogeneous effects
