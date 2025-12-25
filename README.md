# Credit Card Fraud Detection – End-to-End ML System

## Overview
This project implements a **production-grade fraud detection system**
designed for highly imbalanced financial transaction data.

The solution follows industry best practices used in banking and fintech:
- Robust handling of extreme class imbalance
- Proper evaluation using PR-AUC and recall
- Ensemble modeling for stability
- Explainable predictions
- Deployment-ready pipelines

---

## Business Problem
Fraud detection is a **cost-sensitive classification problem** where:
- False negatives are extremely expensive
- Accuracy is misleading due to class imbalance
- Models must be explainable and auditable

This system prioritizes **fraud recall** while maintaining strong precision.

---

## Dataset
- Source: Kaggle – Credit Card Fraud Detection (ULB)
- Transactions: 284,807
- Fraud rate: 0.17%
- Features:
  - V1–V28: PCA-transformed
  - Time, Amount
  - Engineered features (log-amount, cyclic time)

---

## Modeling Approach
### Models Trained
- Logistic Regression (baseline, interpretable)
- Random Forest (non-linear ensemble)
- XGBoost (primary, Optuna-tuned)

### Imbalance Handling
- SMOTE oversampling
- SMOTE + Tomek Links
- Class-weighted baselines for comparison

### Evaluation Metrics
- PR-AUC (primary)
- ROC-AUC
- Fraud Recall
- Precision-Recall trade-off at optimized threshold

---

## Results (Typical)
- PR-AUC: > 0.90
- Fraud Recall: ~90–94%
- Stable generalization via cross-validation

Note: Accuracy is reported for completeness but not used for optimization.

---

## Explainability
- SHAP used for global and local explanations
- Supports regulatory and audit requirements

---

## Production Readiness
- Full preprocessing + model pipelines saved
- Threshold stored explicitly
- Ready for Streamlit inference and AWS deployment
- Designed for drift monitoring

---

## Tech Stack
- Python, Pandas, NumPy
- Scikit-learn, XGBoost
- Imbalanced-learn
- Optuna
- SHAP
- Streamlit (deployment)
- AWS EC2 (deployment)
