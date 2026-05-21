# AI10: PDP & ICE Analysis for Gradient Boosting

This directory contains tools to generate and visualize **Partial Dependence Plots (PDP)** and **Individual Conditional Expectation (ICE)** plots for a Gradient Boosting regression model trained on the TBM dataset.

## Overview

### Partial Dependence Plots (PDP)
- Show the **marginal effect** of each feature on the model's predictions
- Average the predictions across all other features
- Useful for understanding feature importance and non-linear relationships
- One plot per feature

### Individual Conditional Expectation (ICE) Plots
- Show how predictions change for **each individual sample** when varying a specific feature
- Individual curves show instance-specific effects
- Red line overlay shows the PDP (average across all samples)
- Reveals heterogeneous effects and interactions
- One plot per feature

## Files

- `pdp_gradient_boosting.py` - Generate Partial Dependence Plots
- `ice_gradient_boosting.py` - Generate Individual Conditional Expectation plots
- `run_pdp_ice.py` - Run both analyses automatically
- `images/` - Output directory for plots (PNG files)

## Usage

### Run Both Analyses (Recommended)
```powershell
cd C:\Users\siame\Desktop\Stage
.venv\Scripts\python.exe Internship_Research\AI10\run_pdp_ice.py
```

### Run PDP Only
```powershell
cd C:\Users\siame\Desktop\Stage
.venv\Scripts\python.exe Internship_Research\AI10\pdp_gradient_boosting.py
```

### Run ICE Only
```powershell
cd C:\Users\siame\Desktop\Stage
.venv\Scripts\python.exe Internship_Research\AI10\ice_gradient_boosting.py
```

## Customization

Both scripts support command-line arguments:

```powershell
# Custom data file
python Internship_Research\AI10\pdp_gradient_boosting.py --data path/to/data.csv

# Custom model hyperparameters
python Internship_Research\AI10\pdp_gradient_boosting.py `
  --n-estimators 500 `
  --learning-rate 0.05 `
  --max-depth 3 `
  --subsample 1.0

# Custom grid resolution (affects plot smoothness)
python Internship_Research\AI10\pdp_gradient_boosting.py --grid-resolution 100

# For ICE: limit number of individual curves for readability
python Internship_Research\AI10\ice_gradient_boosting.py --max-samples 200
```

## Output

All plots are saved to `AI10/images/`:
- `pdp_<feature_name>.png` - Partial Dependence Plot for each feature
- `ice_<feature_name>.png` - Individual Conditional Expectation plot for each feature

Each plot includes:
- Feature values on x-axis (normalized [0, 1])
- Model predictions on y-axis
- Titles and grid for readability
- 300 DPI resolution for high quality

## Example Results

For each feature, you will get insights such as:
- How penetration rate (PR) changes with feature values
- Non-linear relationships
- Interaction effects (visible as spread in ICE plots)
- Important feature ranges
