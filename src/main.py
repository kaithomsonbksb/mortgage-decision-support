"""Compare the supplied models; assessed advisory rules are NOT provided."""

from src.data import load_data, print_audit
from src.model import FEATURES, MODEL_FILE, run_models


def report(name, result):
    print(f"\n{name}")
    for metric in ("accuracy", "precision_1", "recall_1", "f1_1", "roc_auc"):
        print(f"{metric}: {result[metric]:.3f}")
    print("Confusion matrix [actual rows 0,1; predicted columns 0,1]:")
    print(result["confusion_matrix"])


def main():
    df = load_data()
    print_audit(df)
    results = run_models(df)
    print("\nModel features:", FEATURES)
    for name, result in results.items():
        report(name, result)
    print(f"\nTrained XGBoost model saved: {MODEL_FILE}")


if __name__ == "__main__":
    main()
