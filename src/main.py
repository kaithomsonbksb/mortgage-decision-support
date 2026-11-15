from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.data import TARGET, load_data, print_audit
from src.decisions import advisory_decision
from src.model import (
    FEATURES,
    MODEL_FILE,
    METRICS,
    TEST_SIZE,
    cross_validate_models,
    run_models,
)


def report(name, result):
    print(f"\n{name}")
    for metric in ("accuracy", "precision_1", "recall_1", "f1_1", "roc_auc"):
        print(f"{metric}: {result[metric]:.3f}")
    print("Confusion matrix [actual rows 0,1; predicted columns 0,1]:")
    print(result["confusion_matrix"])


def report_cv(summary):
    print("\n5-fold stratified cross-validation (mean +/- std across folds)")
    for name, result in summary.items():
        print(f"\n{name}")
        for metric in METRICS:
            mean, std = result[metric]
            print(f"{metric}: {mean:.3f} +/- {std:.3f}")
        print("Confusion matrix summed over all folds:")
        print(result["confusion_matrix"])


def evaluate_advisory_system():
    df = load_data()
    X = df[FEATURES].copy()
    y = df[TARGET]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=42,
    )
    model = joblib.load(MODEL_FILE)
    outcomes = (
        "Standard review",
        "Enhanced review",
        "Senior review",
        "Unable to assess",
    )
    counts = {outcome: {"applicants": 0, "risky": 0} for outcome in outcomes}
    escalated = 0
    escalated_risky = 0

    for (index, row), actual in zip(X_test.iterrows(), y_test):
        application = {
            field: None if pd.isna(value) else value
            for field, value in row.items()
        }
        if application["credit_score"] is None:
            score = None
        else:
            score = model.predict_proba(X_test.loc[[index]])[0, 1]
        decision = advisory_decision(score, application)
        outcome = decision["outcome"]
        risky = int(actual)
        counts[outcome]["applicants"] += 1
        counts[outcome]["risky"] += risky
        if outcome in ("Enhanced review", "Senior review"):
            escalated += 1
            escalated_risky += risky

    print("\nAdvisory-system test-set results")
    print(f"{'Outcome':<22}{'Applicants':>12}{'Risky':>8}{'Risky rate':>13}")
    for outcome in outcomes:
        applicants = counts[outcome]["applicants"]
        risky = counts[outcome]["risky"]
        risky_rate = risky / applicants * 100 if applicants else 0
        print(f"{outcome:<22}{applicants:>12}{risky:>8}{risky_rate:>12.1f}%")
    print(
        f"Escalated (Enhanced + Senior): {escalated}; "
        f"risky applicants caught: {escalated_risky}"
    )


def main():
    df = load_data()
    print_audit(df)
    results = run_models(df)
    print("\nModel features:", FEATURES)
    print("\nTest-set results")
    for name, result in results.items():
        report(name, result)
    final_matrix = results["Regularised XGBoost (final)"]["confusion_matrix"]
    ConfusionMatrixDisplay(
        confusion_matrix=final_matrix,
        display_labels=["Not risky", "Risky"],
    ).plot(cmap="Blues", values_format="d")
    plt.title("Regularised XGBoost (final) - test set")
    plt.tight_layout()
    figures_dir = Path(__file__).resolve().parents[1] / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    confusion_path = figures_dir / "final_confusion_matrix.png"
    plt.savefig(confusion_path)
    plt.close()
    print(f"\nFinal XGBoost model saved: {MODEL_FILE}")
    print(f"Test confusion matrix saved: {confusion_path}")
    report_cv(cross_validate_models(df))
    evaluate_advisory_system()


if __name__ == "__main__":
    main()
