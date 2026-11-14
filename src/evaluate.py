from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data import TARGET, load_data
from src.model import (
    BASELINE_FEATURES,
    FEATURES,
    TEST_SIZE,
    build_baseline_pipeline,
    build_xgb_pipeline,
    evaluate,
)

VALIDATION_SIZE = 0.20
THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"


def split_train_validation_test(df):
    X = df[FEATURES].copy()
    y = df[TARGET]
    X_train, _, y_train, _ = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        stratify=y_train,
        random_state=42,
    )
    return X_fit, X_val, y_fit, y_val


def evaluate_on_validation(X_fit, X_val, y_fit, y_val):
    reference = DummyClassifier(strategy="most_frequent")
    reference.fit(X_fit, y_fit)
    reference_results = evaluate(reference, X_val, y_val)

    baseline = build_baseline_pipeline()
    baseline.fit(X_fit[BASELINE_FEATURES], y_fit)
    baseline_results = evaluate(baseline, X_val[BASELINE_FEATURES], y_val)

    xgb_model = build_xgb_pipeline()
    xgb_model.fit(X_fit, y_fit)
    xgb_results = evaluate(xgb_model, X_val, y_val)
    train_scores = xgb_model.predict_proba(X_fit)[:, 1]
    validation_scores = xgb_model.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_fit, train_scores)
    xgb_results["train_auc"] = train_auc
    xgb_results["overfitting_gap"] = train_auc - xgb_results["roc_auc"]
    xgb_results["extreme_score_share"] = float(
        ((validation_scores < 0.05) | (validation_scores > 0.95)).mean()
    )

    results = {
        "Majority reference": reference_results,
        "Starter Logistic Regression": baseline_results,
        "Default XGBoost": xgb_results,
    }
    return results, xgb_model


def threshold_sweep(y_true, scores, thresholds=THRESHOLDS):
    rows = []
    for threshold in thresholds:
        predicted = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": threshold,
                "flagged_pct": predicted.mean() * 100,
                "precision": precision_score(y_true, predicted, zero_division=0),
                "recall": recall_score(y_true, predicted, zero_division=0),
                "f1": f1_score(y_true, predicted, zero_division=0),
                "false_positives": fp,
                "false_negatives": fn,
            }
        )
    return pd.DataFrame(rows)


def plot_sweep(sweep, title, filename):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = Path(filename).stem
    metrics_path = FIGURES_DIR / f"{stem}_metrics.png"
    errors_path = FIGURES_DIR / f"{stem}_errors.png"

    plt.figure(figsize=(8, 5))
    for metric in ("precision", "recall", "f1"):
        plt.plot(sweep["threshold"], sweep[metric], marker="o", label=metric)
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{title}: precision, recall and F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(metrics_path)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        sweep["threshold"],
        sweep["false_positives"],
        marker="o",
        label="False positives",
    )
    plt.plot(
        sweep["threshold"],
        sweep["false_negatives"],
        marker="o",
        label="False negatives",
    )
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.title(f"{title}: false positives and false negatives")
    plt.legend()
    plt.tight_layout()
    plt.savefig(errors_path)
    plt.close()
    return metrics_path, errors_path


def main():
    df = load_data()
    X_fit, X_val, y_fit, y_val = split_train_validation_test(df)
    print(
        f"Rows: {len(y_fit)} fit ({int(y_fit.sum())} risky), "
        f"{len(y_val)} validation ({int(y_val.sum())} risky)"
    )

    results, xgb_model = evaluate_on_validation(X_fit, X_val, y_fit, y_val)
    print("\nValidation results")
    for name, result in results.items():
        print(
            f"{name}: AUC {result['roc_auc']:.3f}, accuracy "
            f"{result['accuracy']:.3f}, precision "
            f"{result['precision_1']:.3f}, recall {result['recall_1']:.3f}"
        )
    xgb_results = results["Default XGBoost"]
    print(f"XGBoost train AUC: {xgb_results['train_auc']:.3f}")

    validation_scores = xgb_model.predict_proba(X_val)[:, 1]
    sweep = threshold_sweep(y_val, validation_scores)
    print("\nXGBoost threshold sweep")
    print(sweep.round(3).to_string(index=False))
    plot_sweep(
        sweep,
        "Default XGBoost threshold sweep",
        "default_xgboost_threshold_sweep",
    )
    print("Charts saved in figures/")


if __name__ == "__main__":
    main()
