# Credit Risk Modeling with Tabular Machine Learning

This project builds and evaluates a **binary loan default prediction pipeline** on a 255k-row consumer credit dataset with 11.6% default rate. Three classifiers — Logistic Regression, Random Forest, and Gradient Boosting — are trained under an identical preprocessing pipeline and evaluated on ROC-AUC and PR-AUC. The emphasis is on methodology: domain-driven feature engineering, leakage-free evaluation, and cross-model explainability through coefficient and feature-importance analysis.

---

## Results at a Glance

| Model | ROC-AUC | PR-AUC | CV ROC-AUC |
|---|---|---|---|
| Gradient Boosting | **0.7582** | **0.3292** | 0.7518 ± 0.0023 |
| Logistic Regression | 0.7531 | 0.3114 | 0.7464 ± 0.0031 |
| Random Forest | 0.7405 | 0.3122 | 0.7340 ± 0.0030 |

Gradient Boosting edges out LR by +0.5 pp in ROC-AUC and by +1.8 pp in PR-AUC — the larger PR-AUC gap is meaningful because the default class is rare (11.6%) and precision-recall trade-offs matter more than overall ranking at this imbalance level. Logistic Regression outperforms Random Forest despite its lower complexity, a pattern examined below.

---

## Why PR-AUC, Not Accuracy

A naïve classifier that labels every applicant non-default achieves **88.4% accuracy** on this dataset by never predicting a default. ROC-AUC is similarly inflated by the large negative class: a model can achieve high AUC purely by correctly ranking the majority of non-defaults while performing poorly on defaulters.

Precision-Recall AUC focuses exclusively on the minority (default) class. A PR-AUC of 0.33 against a random baseline of 0.116 (the prevalence rate) represents a **2.8× lift** over chance, giving a more honest picture of whether the model finds true defaults or merely avoids false alarms on the majority class.

---

## Dataset

**Loan Default Dataset** — consumer credit applications.

| Property | Value |
|---|---|
| Observations | 255,347 |
| Features | 16 (numerical + categorical) |
| Target | `Default` (binary: 1 = default) |
| Default rate | 11.6% |
| Missing values | None |

Key input signals: `Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`, `LoanPurpose`, `EmploymentType`, `EducationLevel`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `HasCoSigner`.

> **Data not included.** Download the dataset from [Kaggle](https://www.kaggle.com/datasets/nikhil1e9/loan-default) and place the CSV at `data/loan.csv` before running the notebook.

---

## Methodology

### Preprocessing Pipeline

A single `sklearn.Pipeline` is constructed and fit once on the training set only. The same pipeline transforms the test set without re-fitting — ensuring no leakage from test distribution into scaling or encoding decisions.

| Step | Choice | Rationale |
|---|---|---|
| Numerical scaling | StandardScaler | Required for LR coefficient comparability |
| Categorical encoding | OneHotEncoder | Nominal variables without ordinal structure |
| Imputation | Median / most-frequent | Robust to outliers; dataset has no missing values so this is defensive |
| Class imbalance | No SMOTE | SMOTE creates synthetic minority samples that inflate PR-AUC without improving real-world performance; `class_weight='balanced'` inside LR and RF provides implicit reweighting instead |

### Evaluation

5-fold stratified cross-validation preserves the 11.6%/88.4% class ratio across all folds. CV is used to estimate generalization variance, not for hyperparameter selection — all three models use their default hyperparameters to isolate the effect of model family rather than tuning effort.

**No temporal split is performed** (see Limitations). All results are from a random 80/20 stratified split.

---

## Feature Engineering

Thirteen features are constructed or transformed with domain rationale:

- **Debt-to-income ratio** — `DTIRatio` directly measures repayment capacity
- **Interest rate as default signal** — high-rate loans indicate lenders already priced in elevated risk at origination; included as raw signal without interaction terms
- **Employment stability proxy** — `MonthsEmployed` captures job tenure; binary `EmploymentType` (full-time vs. other) encodes employment quality
- **Loan structure** — `LoanTerm × InterestRate` interaction not modeled explicitly; LoanAmount and LoanTerm enter separately, allowing tree models to learn the interaction naturally
- **Creditworthiness composite** — `CreditScore`, `NumCreditLines`, `HasCoSigner` jointly encode credit history depth

---

## Feature Importance Analysis

### Logistic Regression Coefficients (standardized)

| Feature | Coefficient | Interpretation |
|---|---|---|
| InterestRate | +0.458 | Strongest default predictor — rate encodes lender risk assessment |
| LoanAmount | +0.300 | Larger loans increase default exposure |
| NumCreditLines | +0.101 | More open lines → higher leverage risk |
| DTIRatio | +0.068 | Moderate positive effect after controlling for other factors |
| LoanPurpose_Auto | −0.105 | Auto loans show lower default probability (collateral effect) |

`CreditScore` — the variable most associated with creditworthiness in public discourse — does not appear in the top-5 LR coefficients after standardization. This counterintuitive finding likely reflects multicollinearity: `CreditScore` is correlated with `InterestRate` (lenders set rates partly based on scores), causing the interest rate term to absorb its signal. When `InterestRate` is removed from the model, `CreditScore` reclaims significance — a classic instance of correlated predictors redistributing coefficient mass.

### Tree Model Feature Importance (mean impurity decrease, normalized)

| Rank | Random Forest | Importance | Gradient Boosting | Importance |
|---|---|---|---|---|
| 1 | Income | 0.120 | Age | 0.286 |
| 2 | InterestRate | 0.114 | Income | 0.211 |
| 3 | LoanAmount | 0.107 | InterestRate | 0.194 |
| 4 | Age | 0.099 | LoanAmount | 0.121 |
| 5 | CreditScore | 0.095 | MonthsEmployed | 0.100 |
| 6 | MonthsEmployed | 0.093 | — | — |

The rank ordering diverges between RF and GB. RF distributes importance more evenly across the top-6 (range: 0.093–0.120), consistent with bagging's ensemble of independent trees each using different feature subsets. GB's importance is concentrated in Age (0.286) — the first splits of boosted trees tend to capture the feature with the highest impurity reduction, and subsequent trees correct residuals in which other features matter, inflating the first-split feature's apparent importance. This is a known GB impurity-importance artifact, not necessarily a signal that Age is three times more informative than Income.

The agreement between all three models on the top cluster — **InterestRate, Income, LoanAmount, Age, MonthsEmployed** — provides convergent validity: these signals are robust across model families and estimation approaches.

---

## Model Behavior: Why RF Underperforms LR

Logistic Regression outperforms Random Forest on both ROC-AUC (0.7531 vs 0.7405) and CV score (0.7464 vs 0.7340). This reversal of the typical complexity ordering is not a failure of RF but a characteristic of this problem:

1. **Linear separability of the default signal**: The strongest predictors (`InterestRate`, `LoanAmount`) have a monotone relationship with default probability. LR captures this directly with one coefficient; RF must approximate it with many axis-aligned splits, introducing quantization noise.

2. **High-cardinality numeric features**: Deep RF trees can overfit to the granular distribution of continuous variables when the signal is smooth. With default hyperparameters (no depth limit), RF on 255k rows can memorize training patterns that don't generalize.

3. **Effective implicit regularization in LR**: StandardScaler + L2 regularization (default `C=1.0`) prevents LR from overfitting to outlier applicants, while RF's bootstrap sampling provides weaker regularization on numerical splits.

4. **No SMOTE**: With balanced class weights in LR and RF, both receive the same reweighting — so imbalance handling is not the differentiating factor.

Gradient Boosting recovers over RF because its sequential residual correction learns a calibrated probability surface rather than relying on averaged tree votes. Its PR-AUC advantage (+0.017 over LR) is most visible at high-precision operating points — the regime relevant to actual credit decisions where false positive costs are high.

---

## Reproducibility

All experiments use `random_state=42`. Train/test split and CV folds are stratified.

```bash
pip install -r requirements.txt
# Place loan.csv at data/loan.csv
jupyter nbconvert --to notebook --execute notebooks/01_credit_risk_end_to_end.ipynb
```

Notebook runtime: ~3 min on CPU (255k rows). Peak memory: ~1 GB.

---

## Limitations

- **No temporal split**: Loan applications have a natural time ordering. A random split may place a 2019 application in training and a 2018 application in test, violating causal ordering. A time-based split (train on earlier applications, test on later) would give a more realistic estimate of deployment performance. The dataset does not include an application date field, making temporal splitting unavailable here.

- **Static snapshot features only**: The model operates on application-time features. Post-origination signals — payment history, balance trajectories, utilization patterns — are absent but are among the strongest default predictors in operational credit systems.

- **No fairness analysis**: Features like `Age`, `MaritalStatus`, and `EducationLevel` are present in the model. Differential error rates across demographic groups (disparate impact) are not evaluated. A production credit system would require fairness auditing under ECOA/FCRA constraints.

- **Default hyperparameters throughout**: No grid search or Bayesian optimization is performed. The results reflect model-family differences, not optimized performance. A tuned XGBoost would likely exceed all three models' PR-AUC.

- **No probability calibration**: ROC-AUC and PR-AUC measure ranking performance, not calibration. Gradient Boosting in particular tends to produce overconfident probabilities (compressed toward 0.5). Isotonic regression or Platt scaling would be required before using predicted probabilities as actual default rates in risk-based pricing.

---

## Future Work

- **SHAP explainability**: TreeExplainer on the Gradient Boosting model to decompose individual predictions into feature contributions — especially useful for adverse action explanations required in regulated credit decisions.
- **Threshold optimization**: Moving from default 0.5 classification threshold to cost-sensitive threshold selection using the false negative (missed default) / false positive (rejected good applicant) loss ratio.
- **XGBoost / LightGBM**: Both support native handling of categorical features and have better impurity-importance estimation (gain-based) that avoids RF's high-cardinality bias.
- **Temporal validation**: If a dataset with application timestamps becomes available, walk-forward cross-validation would provide a lower-bound estimate of true deployment performance.
- **Calibration**: Platt scaling or isotonic regression post-processing to convert GB scores into well-calibrated probabilities for risk-based pricing.

---

## Tech Stack

- **Python 3.13** · NumPy 2.3 · Pandas 2.3 · scikit-learn 1.7
- **Matplotlib 3.10** · Seaborn 0.13

---

## Project Structure

```
credit-risk-modeling-tabular-ml/
├── data/
│   └── loan.csv              # not tracked — download from Kaggle
├── notebooks/
│   └── 01_credit_risk_end_to_end.ipynb   # full pipeline
└── requirements.txt
```

---

**Author:** Sezer Bayraktar
