# Stroke Risk Prediction – Healthcare ML Project

## Overview
This project implements an end-to-end *stroke risk screening model* using machine learning.  
The system is designed to *prioritize recall (sensitivity)* to minimize missed stroke cases, which is critical in healthcare applications.

This is a *screening model*, not a diagnostic tool.

---

## Problem Statement
Stroke prediction involves:
- Highly imbalanced data
- High cost of false negatives
- Need for medically meaningful features
- Evaluation beyond accuracy

The objective is to detect as many true stroke cases as possible while controlling false positives through *decision threshold tuning*.

---

## Dataset
Structured healthcare dataset with:
- Demographics: age, gender, residence
- Medical conditions: hypertension, heart disease
- Lifestyle factors: smoking
- Clinical measurements: BMI, glucose level

Target:
- stroke (0 = No, 1 = Yes)

---

## Feature Engineering
Custom feature engineering is applied inside a pipeline to avoid data leakage:

- high_glucose: avg_glucose_level > 140  
- high_bmi: BMI > 30  
- smoking: exposure indicator  
- risk_factor_count: aggregated medical risk  
- age_group: clinically meaningful age bins  
- age_glucose_interaction: cumulative vascular risk  
- bmi_missing: missingness signal  

---

## Preprocessing
- Median imputation for numeric features  
- Standard scaling  
- One-hot encoding with unknown category handling  
- Binary features passed without scaling  

All preprocessing is pipeline-based.

---

## Model
- *RandomForestClassifier*
- class_weight='balanced'
- Robust to non-linearities and mixed feature types

---

## Hyperparameter Tuning
- Performed using *Optuna*
- Cross-validated optimization
- Objective: *maximize recall*

---

## Evaluation Strategy
Accuracy is not the primary metric.

Reported metrics:
- Recall (primary)
- Precision
- F1-score
- Confusion matrix

Reason:
> Missing a stroke case is clinically more costly than a false alarm.

---

## Threshold Tuning
Instead of using the default 0.5 cutoff:
- Prediction probabilities are evaluated across multiple thresholds
- Final threshold selected to maintain high recall while improving precision

Final threshold:0.53
---

## Final Performance (Test Set)
- Stroke recall ≈ *0.90*
- Precision ≈ *0.14*
- False negatives minimized
- Expected increase in false positives (acceptable for screening)

---

## Prediction Interface
A production-ready predict_stroke() function:
- Accepts raw user inputs
- Applies the full pipeline
- Returns:
  - Risk category (Low / Moderate / High / Extreme)
  - Stroke probability
  - Clinical disclaimer
- Includes input validation and error handling

---

## Model Saving
The trained pipeline and threshold are serialized using joblib:

```python
joblib.dump(pipe, "pipeline.pkl")
joblib.dump(best_threshold, "best_threshold.pkl")
