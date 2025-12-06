# Wine Quality Prediction (WineQT) — CRISP-DM + Machine Learning

Predict red wine **quality scores (3–8)** from physicochemical properties using a **CRISP-DM** workflow and classic regression models (**Linear Regression** and **Random Forest**).

---

## Project Overview

This repository contains a small, modular ML pipeline that:
- loads the **WineQT** dataset,
- performs **EDA** (distribution + correlations),
- prepares data (drop `Id`, train/test split with stratification, scaling),
- trains and evaluates:
  - **Linear Regression** (baseline),
  - **Random Forest Regressor** (non-linear ensemble),
- visualizes:
  - **Actual vs Predicted**,
  - **Random Forest feature importances**,
- saves the best model as a reusable `.joblib` artifact.

---

## Methodology (CRISP-DM)

1. **Business Understanding** — predict quality for faster quality screening/decision support  
2. **Data Understanding** — EDA (distribution + correlations)  
3. **Data Preparation** — drop `Id`, check missingness, split & scale  
4. **Modeling** — Linear Regression vs Random Forest, 5-fold CV  
5. **Evaluation** — RMSE + R², plots  
6. **Deployment** — save model (`.joblib`) for reuse

---

## Dataset

- File: `WineQT.csv` (included in repo)
- Source (Kaggle):  
  `https://www.kaggle.com/datasets/yasserh/wine-quality-dataset`

Target distribution is imbalanced (mostly quality **5–6**), which can make extreme scores harder to predict.

---

## Results (example run)

From a typical run of `main.py`:

### Cross-validation (5-fold R²)
- Linear Regression: mean **~0.342**
- Random Forest: mean **~0.411**

### Test set
- Linear Regression: **RMSE ~0.628**, **R² ~0.388**
- Random Forest: **RMSE ~0.596**, **R² ~0.449**

### Top Random Forest features (importance)
- alcohol
- sulphates
- volatile acidity
- (then: total sulfur dioxide, pH, fixed acidity, …)

---

## Repository Structure

```text
.
├── WineQT.csv
├── main.py
├── data_loader.py
├── data_prep.py
├── model_trainer.py
├── random_forest_wine_quality_model.joblib
├── assignment5-DataScienceMehtodologyProject.pdf
└── Report/                      # report / figures / notebook(s) (if applicable)
