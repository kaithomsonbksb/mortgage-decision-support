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
TEST_SIZE = 0.20
METRICS = ("accuracy", "precision_1", "recall_1", "f1_1", "roc_auc")
FINAL_XGB_PARAMS = {
    "n_estimators": 300,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "max_depth": 2,
    "learning_rate": 0.05,
    "scale_pos_weight": 1,
}


# base model to test against
def build_baseline_pipeline():
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def build_xgb_pipeline(**xgb_params):
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
        **xgb_params,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate(model, X_test, y_test):
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
    X = df[FEATURES].copy()
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )

    reference = DummyClassifier(strategy="most_frequent")
    reference.fit(X_train, y_train)

    tutor_lr = build_baseline_pipeline()
    tutor_lr.fit(X_train[BASELINE_FEATURES], y_train)

    xgb_pipeline = build_xgb_pipeline()
    xgb_pipeline.fit(X_train, y_train)

    final_xgb_pipeline = build_xgb_pipeline(**FINAL_XGB_PARAMS)
    final_xgb_pipeline.fit(X_train, y_train)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_xgb_pipeline, MODEL_FILE)
    return {
        "Base benchmark": evaluate(reference, X_test, y_test),
        "Logistic Regression benchmark": evaluate(
            tutor_lr, X_test[BASELINE_FEATURES], y_test
        ),
        "XGBoost (my model)": evaluate(xgb_pipeline, X_test, y_test),
        "Regularised XGBoost (final)": evaluate(final_xgb_pipeline, X_test, y_test),
    }


def cross_validate_models(df, folds=CV_FOLDS):
    X = df[FEATURES].copy()
    y = df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )
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
    for train_idx, test_idx in splitter.split(X_train, y_train):
        for name, (build, cols) in builders.items():
            model = build()
            model.fit(X_train.iloc[train_idx][cols], y_train.iloc[train_idx])
            fold_results[name].append(
                evaluate(model, X_train.iloc[test_idx][cols], y_train.iloc[test_idx])
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
