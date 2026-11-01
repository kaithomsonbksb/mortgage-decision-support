"""Score one sample new applicant with the trained XGBoost model.

Shows loading a fitted mortgage model, formatting the same features in the
same order, and predicting a class-1 score.
NOT provided: learner's mortgage-specific validation and decision rules.

Run `python -m src.main` first, from the project root.
Then run `python -m src.applicant_example`.
"""

import joblib
import pandas as pd

from src.model import FEATURES, MODEL_FILE
from src.decisions import validate_application_fields, validate_score

# Sample values for the model. Do not use actual customer information.
EXAMPLE_APPLICATION = {
    "credit_score": 680,
    "debt_to_income_ratio": 0.32,
    "missed_payments_24m": 0,
    "previous_default_flag": 0,
    "credit_utilisation_pct": 35.0,
    "employment_years": 5.0,
    "savings_gbp": 20000.0,
    "annual_income_gbp": 60000.0,
    "loan_to_value_pct": 80.0,
    "requested_loan_gbp": 200000.0,
    "term_years": 25,
    "employment_type": "Employed",
}


def score_example(application):
    """Generate model's estimated probability for class 1; no advice."""
    validate_application_fields(application, FEATURES)
    row = pd.DataFrame([application], columns=FEATURES)
    if not MODEL_FILE.is_file():
        raise FileNotFoundError("No trained model: first run python -m src.main")
    # Load only the trusted, locally generated model artefact.
    model = joblib.load(MODEL_FILE)
    score = model.predict_proba(row)[0, 1]
    return validate_score(float(score))


def main():
    score = score_example(EXAMPLE_APPLICATION)
    print("MORTGAGE APPLICATION ASSESSMENT")
    print("Applicant details:")
    print(f"credit_score={EXAMPLE_APPLICATION['credit_score']}")
    print(f"debt_to_income_ratio={EXAMPLE_APPLICATION['debt_to_income_ratio']:.2f}")
    print(f"Repayment-difficulty class-1 score: {score:.3f}")
    print("Calibration not assessed; this is not a validated PD.")
    print("Provisional recommendation: PENDING — rules not yet implemented")
    print("Status: AWAITING HUMAN REVIEW — no mortgage approved or declined")


if __name__ == "__main__":
    main()
