"""Examples of testing generic helper functions; not mortgage policy tests."""

import unittest
from src.decisions import (
    REQUIRED_FIELDS,
    advisory_decision,
    validate_application,
    validate_application_fields,
    validate_score,
)


def valid_application():
    return {
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


class GenericValidationTests(unittest.TestCase):
    def test_valid_score(self):
        self.assertEqual(validate_score(0.4), 0.4)

    def test_score_outside_range(self):
        with self.assertRaises(ValueError):
            validate_score(1.5)

    def test_required_value_present(self):
        self.assertTrue(
            validate_application_fields({"credit_score": 680}, ["credit_score"])
        )

    def test_required_value_missing(self):
        with self.assertRaises(ValueError):
            validate_application_fields({}, ["credit_score"])


class FlagEscalationTests(unittest.TestCase):
    def setUp(self):
        self.application = valid_application()
        self.application["missed_payments_24m"] = 3
        self.application["previous_default_flag"] = 1

    def test_multiple_flags_only_raise_low_band_to_enhanced(self):
        decision = advisory_decision(0.1, self.application)

        self.assertEqual(decision["outcome"], "Enhanced review")

    def test_high_band_remains_senior_with_flags(self):
        decision = advisory_decision(0.4, self.application)

        self.assertEqual(decision["outcome"], "Senior review")


class OptionalFieldTests(unittest.TestCase):
    def test_missing_optional_fields_are_skipped_and_noted(self):
        application = valid_application()
        application["credit_score"] = None
        application["employment_years"] = None
        application["savings_gbp"] = float("nan")
        application["credit_utilisation_pct"] = None

        decision = advisory_decision(None, application)

        self.assertEqual(validate_application(application), ["credit_score: missing"])
        self.assertEqual(decision["notes"].count("credit_score: missing"), 1)
        self.assertEqual(
            decision["notes"][1:],
            [
                "Incomplete data: employment_years is missing",
                "Incomplete data: savings_gbp is missing",
                "Incomplete data: credit_utilisation_pct is missing",
            ],
        )

    def test_absent_optional_fields_are_skipped(self):
        application = valid_application()
        for field in ("employment_years", "savings_gbp", "credit_utilisation_pct"):
            del application[field]

        self.assertEqual(validate_application(application), [])
        self.assertTrue(validate_application_fields(application, REQUIRED_FIELDS))

    def test_present_optional_fields_are_range_checked(self):
        cases = (
            ("employment_years", 51, "employment_years: outside valid range"),
            ("savings_gbp", -1, "savings_gbp: outside valid range"),
            (
                "credit_utilisation_pct",
                101,
                "credit_utilisation_pct: outside valid range",
            ),
        )
        for field, value, expected_problem in cases:
            with self.subTest(field=field):
                application = valid_application()
                application[field] = value
                self.assertIn(expected_problem, validate_application(application))


if __name__ == "__main__":
    unittest.main()
