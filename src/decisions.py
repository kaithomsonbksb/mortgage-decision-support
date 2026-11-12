import math
from numbers import Real

REQUIRED_FIELDS = [
    "credit_score",
    "debt_to_income_ratio",
    "missed_payments_24m",
    "previous_default_flag",
    "annual_income_gbp",
    "loan_to_value_pct",
    "requested_loan_gbp",
    "term_years",
    "employment_type",
]
OPTIONAL_FIELDS = [
    "employment_years",
    "savings_gbp",
    "credit_utilisation_pct",
]
VALID_RANGE = {
    "credit_score": (0, 850),
    "debt_to_income_ratio": (0, 2),
    "missed_payments_24m": (0, 24),
    "previous_default_flag": (0, 1),
    "credit_utilisation_pct": (0, 100),
    "employment_years": (0, 50),
    "savings_gbp": (0, float("inf")),
    "annual_income_gbp": (0, float("inf")),
    "loan_to_value_pct": (0, 100),
    "requested_loan_gbp": (0, float("inf")),
    "term_years": (0, 50),
}
VALID_EMPLOYMENT_TYPES = ["Employed", "Contractor", "Self-employed"]

BAND_MEDIUM = 0.20
BAND_HIGH = 0.40
LTV_FLAG = 90
MISSED_PAYMENTS_FLAG = 2
HIGH_LTI = 4.5


def validate_application_fields(application, required_fields):
    """Reject absent, blank or non-finite values in the selected fields."""
    invalid = []
    for field in required_fields:
        value = application.get(field)
        if (
            value is None
            or (isinstance(value, str) and not value.strip())
            or (isinstance(value, Real) and not math.isfinite(value))
        ):
            invalid.append(field)
    if invalid:
        raise ValueError(f"Missing or invalid fields: {invalid}")
    return True


def _is_missing_optional(value):
    return (
        value is None
        or (isinstance(value, str) and not value.strip())
        or (isinstance(value, Real) and math.isnan(value))
    )


def validate_score(score):
    """Check that a binary classifier's reported score is in the interval [0, 1]."""
    if isinstance(score, bool) or not isinstance(score, Real):
        raise ValueError("Model score must be numeric")
    numeric_score = float(score)
    if not math.isfinite(numeric_score) or not 0.0 <= numeric_score <= 1.0:
        raise ValueError("Model score must be finite and between 0 and 1")
    return numeric_score


def validate_application(application):
    """Return all missing, invalid, out-of-range, and unknown-value problems."""
    problems = []
    for field in REQUIRED_FIELDS + OPTIONAL_FIELDS:
        value = application.get(field)
        if field in OPTIONAL_FIELDS and _is_missing_optional(value):
            continue
        if value is None or value == "":
            problems.append(f"{field}: missing")
            continue
        if field == "employment_type":
            if value not in VALID_EMPLOYMENT_TYPES:
                problems.append(f"{field}: unknown employment type")
            continue
        if not isinstance(value, Real):
            problems.append(f"{field}: must be numeric")
            continue
        if not math.isfinite(value):
            problems.append(f"{field}: must be finite")
            continue
        minimum, maximum = VALID_RANGE[field]
        if not minimum <= value <= maximum:
            problems.append(f"{field}: outside valid range")
    return problems


def policy_flags(application):
    """Return reasons for adverse-credit and high-LTV policy flags."""
    flags = []
    if application.get("missed_payments_24m", 0) >= MISSED_PAYMENTS_FLAG:
        flags.append(f"F1: missed payments exceed {MISSED_PAYMENTS_FLAG} in 24 months")
    if application.get("previous_default_flag") == 1:
        flags.append("F1: previous default recorded")
    if application.get("loan_to_value_pct", 0) > LTV_FLAG:
        flags.append(f"F2: loan-to-value exceeds {LTV_FLAG}%")
    return flags


def info_notes(application):
    """Return non-escalating information notes for the application."""
    notes = []
    annual_income = application.get("annual_income_gbp")
    requested_loan = application.get("requested_loan_gbp")
    if annual_income and requested_loan and requested_loan / annual_income >= HIGH_LTI:
        notes.append(f"High loan-to-income ratio: exceeds {HIGH_LTI:.1f}")
    for field in OPTIONAL_FIELDS:
        value = application.get(field)
        if _is_missing_optional(value):
            notes.append(f"Incomplete data: {field} is missing")
    return notes


def score_band(score):
    """Classify a validated score using inclusive thresholds."""
    score = validate_score(score)
    if score >= BAND_HIGH:
        return "High"
    if score >= BAND_MEDIUM:
        return "Medium"
    return "Low"


def advisory_decision(score, application):
    """Return a provisional review outcome; never approve or decline."""
    problems = validate_application(application)
    if problems:
        return {
            "outcome": "Unable to assess",
            "score": None,
            "band": None,
            "flags": [],
            "notes": problems + info_notes(application),
            "reviewer": "Human review required: application data is incomplete or invalid.",
        }

    validated_score = validate_score(score)
    band = score_band(validated_score)
    flags = policy_flags(application)
    notes = info_notes(application)

    outcomes = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
    }
    level = outcomes[band]
    if flags:
        level = max(level, 2)

    outcome_labels = {
        1: "Standard review",
        2: "Enhanced review",
        3: "Senior review",
    }

    return {
        "outcome": outcome_labels[level],
        "score": validated_score,
        "band": band,
        "flags": flags,
        "notes": notes,
        "reviewer": "Human review required: advisory output only; no automatic approval or decline.",
    }
