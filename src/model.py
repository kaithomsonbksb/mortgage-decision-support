from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
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
BASELINE_FEATURES = ["credit_score", "debt_to_income_ratio"]
MODEL_FILE = Path(__file__).resolve().parents[1] / "models" / "xgb_model.joblib"
CV_FOLDS = 5
METRICS = ("accuracy", "precision_1", "recall_1", "f1_1", "roc_auc")


# module baseline, kept as benchmak
def build_baseline_pipeline():
    """Return the supplied two-feature logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


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


def run_models(df):
    """Fit three comparable models on one held-out split."""
    X = df[FEATURES].copy()
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    reference = DummyClassifier(strategy="most_frequent")
    reference.fit(X_train, y_train)

    tutor_lr = build_baseline_pipeline()
    tutor_lr.fit(X_train[BASELINE_FEATURES], y_train)

    xgb_pipeline = build_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_pipeline, MODEL_FILE)
    return {
        "Base benchmark": evaluate(reference, X_test, y_test),
        "Logistic Regression benchmark": evaluate(
            tutor_lr, X_test[BASELINE_FEATURES], y_test
        ),
        "XGBoost (my model)": evaluate(xgb_pipeline, X_test, y_test),
    }


def cross_validate_models(df, folds=CV_FOLDS):
    """Evaluate the three models with stratified k-fold CV; nothing is saved.

    Returns per-metric mean and standard deviation across folds, plus the
    confusion matrix summed over all test folds.
    """
    X = df[FEATURES].copy()
    y = df[TARGET]
    builders = {
        "Base benchmark": (
            lambda: DummyClassifier(strategy="most_frequent"),
            FEATURES,
        ),
        "Logistic Regression benchmark": (build_baseline_pipeline, BASELINE_FEATURES),
        "XGBoost (my model)": (build_xgb_pipeline, FEATURES),
    }
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    fold_results = {name: [] for name in builders}
    for train_idx, test_idx in splitter.split(X, y):
        for name, (build, cols) in builders.items():
            model = build()
            model.fit(X.iloc[train_idx][cols], y.iloc[train_idx])
            fold_results[name].append(
                evaluate(model, X.iloc[test_idx][cols], y.iloc[test_idx])
            )

    summary = {}
    for name, results in fold_results.items():
        summary[name] = {
            metric: (
                float(np.mean([r[metric] for r in results])),
                float(np.std([r[metric] for r in results])),
            )
            for metric in METRICS
        }
        summary[name]["confusion_matrix"] = sum(r["confusion_matrix"] for r in results)
    return summary
