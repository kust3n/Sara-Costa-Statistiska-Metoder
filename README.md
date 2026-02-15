# Sara-Costa-Statistiska-Metoder

## Overview

This project implements Multiple Linear Regression using Ordinary Least Squares (OLS) without relying on external machine learning libraries. The implementation follows the mathematical formulation:

b = (XᵀX)⁻¹Xᵀy

Only `numpy` is used for linear algebra operations.

The dataset used is the California housing dataset (`housing.csv`).

---

## Files Included

- `linear_regression.py`  
  Contains the LinearRegression class implementation.

- `lab_linear_regression.ipynb`  
  Demonstrates the functionality of the class on the housing dataset.

- `housing.csv`  
  Dataset used for the statistical analysis.

---

## Model Implementation

The `LinearRegression` class performs:

- Ordinary Least Squares estimation
- Computation of:
  - Sample size (n)
  - Number of features (d)
  - Sample variance
  - Standard deviation
  - Root Mean Squared Error (RMSE)

The model automatically adds an intercept term.

---

## Data Preparation

Before fitting the model:

- Missing values are removed
- The categorical variable `ocean_proximity` is excluded
- The target variable is `median_house_value`

No normalization or scaling is performed, as this is a statistical analysis rather than a machine learning pipeline.

---

## How to Run

1. Ensure `housing.csv` is in the same directory.
2. Open `lab_linear_regression.ipynb`.
3. Run all cells in order.

The notebook will display:
- Sample size
- Number of features
- Variance
- Standard deviation
- RMSE

