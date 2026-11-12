"""Compare the supplied models; assessed advisory rules are NOT provided."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.data import load_data, print_audit
from src.model import FEATURES, MODEL_FILE, METRICS, cross_validate_models, run_models


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


if __name__ == "__main__":
    main()
