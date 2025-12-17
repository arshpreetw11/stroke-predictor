Stroke Risk Prediction (Healthcare ML)
Summary

End-to-end healthcare ML pipeline for stroke risk screening, designed with recall prioritization, threshold tuning, and leakage-safe preprocessing.

Focus: don’t miss stroke cases.

Key Highlights (What Recruiters Care About)

Recall-optimized model for imbalanced medical data

Full scikit-learn pipeline (feature engineering → preprocessing → model)

Threshold tuning instead of blind 0.5 cutoff

Proper handling of class imbalance (class_weight)

Production-ready prediction function with error handling

Model + threshold serialized for deployment

Tech Stack

Python

scikit-learn

Optuna (hyperparameter tuning)

Pandas / NumPy

Joblib

ML Workflow

Feature Engineering

Medical risk flags (high_glucose, high_bmi)

Smoking exposure abstraction

Risk aggregation (risk_factor_count)

Interaction feature (age × glucose)

Missingness indicators

Preprocessing

Median imputation

Standard scaling

One-hot encoding with unknown handling

No data leakage (pipeline-based)

Model

RandomForestClassifier

class_weight='balanced'

Hyperparameter tuning with Optuna

Objective: maximize recall

Evaluation

Recall, Precision, F1, Confusion Matrix

Accuracy intentionally de-emphasized

Threshold tuning for recall–precision tradeoff

Final Performance (Test Set)

Stroke Recall: ~0.90

Precision: ~0.14

False negatives minimized

Behavior aligned with medical screening use-case

Threshold Strategy

Instead of default prediction:

Probabilities evaluated across thresholds

Final cutoff selected to preserve high recall

FINAL_THRESHOLD = 0.51

Prediction Interface

Single function:

predict_stroke(...)


Returns:

Risk category (Low / Moderate / High / Extreme)

Stroke probability

Clinical disclaimer

Robust error handling for invalid inputs

Model Persistence
joblib.dump(pipe, "final_pipeline.pkl")
joblib.dump(best_threshold, "best_threshold.pkl")


Ready for:

API

UI demo

Deployment

Why This Project Matters

Most ML projects stop at accuracy.
This project demonstrates:

Domain-aware metric selection

Decision-threshold optimization

Production-grade ML hygiene

Disclaimer

Screening model only.
Not a diagnostic tool.

Author

ML Intern Project
Healthcare Risk Modeling
