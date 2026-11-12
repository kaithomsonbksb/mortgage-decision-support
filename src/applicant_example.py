import joblib
import pandas as pd

from src.model import FEATURES, MODEL_FILE
from src.decisions import (
    advisory_decision,
    validate_application,
    validate_application_fields,
    validate_score,
)

# Sample values for the model. Do not use actual customer information.
BASE_APPLICATION = {
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

EXAMPLE_APPLICATIONS = {
    "Clean application": BASE_APPLICATION,
    "High-LTV application": {
        **BASE_APPLICATION,
        "loan_to_value_pct": 95.0,
    },
    "Adverse-credit application": {
        **BASE_APPLICATION,
        "missed_payments_24m": 3,
        "previous_default_flag": 1,
    },
    "Missing-credit-score application": {
        **BASE_APPLICATION,
        "credit_score": None,
    },
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


def print_decision(name, application):
    """Validate, optionally score, and print one advisory decision."""
    problems = validate_application(application)
    if problems:
        decision = advisory_decision(None, application)
    else:
        score = score_example(application)
        decision = advisory_decision(score, application)

    print(f"\n{name}")
    print(f"Outcome: {decision['outcome']}")
    if decision["score"] is None:
        print("Score: not calculated")
    else:
        print(f"Score: {decision['score']:.3f}")
    print(f"Band: {decision['band'] or 'not available'}")
    print(f"Flags: {', '.join(decision['flags']) or 'None'}")
    print(f"Notes: {', '.join(decision['notes']) or 'None'}")
    print(f"Reviewer: {decision['reviewer']}")


def main():
    print("MORTGAGE APPLICATION ADVISORY EXAMPLES")
    for name, application in EXAMPLE_APPLICATIONS.items():
        print_decision(name, application)


if __name__ == "__main__":
    main()
