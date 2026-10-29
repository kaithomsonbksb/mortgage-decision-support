"""Train and evaluate the XGBoost mortgage-risk model."""

from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from src.data import TARGET

NUMERIC_FEATURES = [
    "credit_score",
    "debt_to_income_ratio",
    "missed_payments_24m",
    "previous_default_flag",
    "credit_utilisation_pct",
    "employment_years",
    "savings_gbp",
    "annual_income_gbp",
    "loan_to_value_pct",
    "requested_loan_gbp",
    "term_years",
]
CATEGORICAL_FEATURES = ["employment_type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Retain this name for the scoring example's public import.
BASELINE_FEATURES = FEATURES
MODEL_FILE = Path(__file__).resolve().parents[1] / "models" / "baseline_model.joblib"


def build_xgb_pipeline():
    """Return preprocessing and a reproducible first XGBoost model."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    classifier = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        n_jobs=1,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate(model, X_test, y_test):
    """Report held-out binary classification metrics."""
    predicted = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, predicted),
        "precision_1": precision_score(y_test, predicted, zero_division=0),
        "recall_1": recall_score(y_test, predicted, zero_division=0),
        "f1_1": f1_score(y_test, predicted, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=[0, 1]),
    }


def run_xgb(df):
    """Fit the XGBoost model, save it, and return held-out metrics."""
    X = df[FEATURES].copy()
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    pipeline = build_xgb_pipeline()
    pipeline.fit(X_train, y_train)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    return evaluate(pipeline, X_test, y_test)
