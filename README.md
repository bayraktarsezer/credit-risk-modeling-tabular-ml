# End-to-End Credit Risk Modeling with Tabular Machine Learning

## Project Overview

This project presents an end-to-end, methodology-driven study of credit risk modeling using real-world tabular data.
The objective is to estimate the probability of loan default for individual applicants by framing the task as a binary classification problem.

Rather than prioritizing aggressive hyperparameter tuning or leaderboard-oriented optimization, the project focuses on:
- Domain-driven feature engineering
- Clean and reproducible preprocessing pipelines
- Fair and consistent model comparison
- Global model explainability

> This project is not a prediction contest, but a disciplined credit risk modeling exercise.

---

## Problem Definition

**Goal**  
Predict whether a loan applicant will default on their loan.

**Problem Type**  
Binary classification  
- `1` → Default  
- `0` → Non-default  

**Motivation**  
Credit risk models are core components of decision-support systems in banking:
- False negatives result in direct financial loss  
- False positives lead to missed business opportunities  

---

## Dataset

- Source: Kaggle (Loan Default Dataset)
- Number of observations: ~255,000
- Number of features: 16 (numerical and categorical)
- Target variable: `Default`
- Missing values: None
- Class imbalance: Moderate (~11.6% default rate)

The dataset reflects a realistic consumer credit risk scenario.

> Due to licensing considerations, the raw dataset may not be included directly in this repository.

---

## Methodology

### 1. Minimal Exploratory Data Analysis (EDA)

EDA is intentionally concise and decision-oriented:
- Target distribution and imbalance assessment
- Sanity checks for invalid or unrealistic values
- Identification of ID and leakage-prone variables

Extensive visualization, correlation heatmaps, and exploratory plots are deliberately avoided.

> EDA is treated as a tool for decision-making, not presentation.

---

### 2. Feature Engineering

Feature engineering is guided by financial intuition and domain knowledge rather than automated feature generation.

Key signals include:
- Income and debt-related indicators
- Employment stability proxies
- Loan structure and creditworthiness signals

Each feature is introduced with a clear rationale grounded in credit risk logic.

---

### 3. Preprocessing Pipeline

A unified preprocessing pipeline is used consistently across all models:
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- Standard scaling for numerical variables
- One-hot encoding for categorical variables
- Stratified train–test split
- No data leakage

This ensures fairness, reproducibility, and comparability.

---

## Models

Three model families are evaluated under identical preprocessing and evaluation settings:

### Logistic Regression (Baseline)
- Widely used in financial risk modeling
- Highly interpretable
- Serves as a strong methodological reference

### Random Forest
- Captures non-linear relationships
- Automatically models feature interactions
- Provides global feature importance

### Gradient Boosting
- Represents a controlled increase in model complexity
- Explores the performance ceiling without excessive tuning
- Emphasizes stability and generalization

---

## Evaluation Metrics

Given the class imbalance, multiple metrics are reported:
- ROC-AUC
- PR-AUC
- 5-fold cross-validation (mean ± standard deviation)

The emphasis is on **model behavior, stability, and interpretability**, not marginal metric gains.

---

## Results Summary

| Model                | Test ROC-AUC | Test PR-AUC | CV ROC-AUC (mean) | CV ROC-AUC (std) |
|----------------------|--------------|-------------|-------------------|------------------|
| Logistic Regression  | ~0.75        | ~0.31       | ~0.75             | ~0.003           |
| Random Forest        | ~0.74        | ~0.31       | ~0.73             | ~0.003           |
| Gradient Boosting    | ~0.76        | ~0.33       | ~0.75             | ~0.002           |

---

## Model Explainability

Explainability is treated as a core requirement rather than an optional add-on.

- Logistic Regression coefficients are analyzed to understand the direction and magnitude of risk factors
- Tree-based models are examined using global feature importance
- Consistent risk signals emerge across all models:
  - Income, age, interest rate, loan amount, and employment stability dominate predictions

No individual-level case studies are included, as the focus is on **global model behavior**.

---

## Key Insights

- Strong feature engineering enables simple models to remain highly competitive
- Increased model complexity does not guarantee superior performance
- Gradient Boosting provides a modest but consistent improvement when built on a solid methodological foundation
- Interpretability and stability are essential in financial risk modeling

---

## Scope and Limitations

This project deliberately excludes:
- Deep learning models
- Extensive hyperparameter tuning
- Automated feature generation
- Leaderboard-driven optimization

These choices are intentional.

> The objective is to demonstrate methodological maturity, not to maximize benchmark scores.

---

## Conclusion

This project demonstrates that effective credit risk modeling is primarily driven by:
- Thoughtful feature engineering
- Clean and leakage-free preprocessing pipelines
- Fair and transparent model comparison
- Emphasis on explainability and stability

Model complexity is treated as a design decision, not a goal.

---

## Final Remark

This repository reflects a disciplined approach to tabular machine learning that prioritizes understanding, robustness, and real-world applicability over raw predictive performance.

---

**Author:** Sezer Bayraktar  
*Aspiring Data Scientist & AI Engineer* 🇩🇪
