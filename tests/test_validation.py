"""Examples of testing generic helper functions; not mortgage policy tests."""

import unittest
from src.decisions import (
    advisory_decision,
    validate_application_fields,
    validate_score,
)


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
        self.application = {
            "credit_score": 680,
            "debt_to_income_ratio": 0.32,
            "missed_payments_24m": 3,
            "previous_default_flag": 1,
            "credit_utilisation_pct": 35.0,
            "employment_years": 5.0,
            "savings_gbp": 20000.0,
            "annual_income_gbp": 60000.0,
            "loan_to_value_pct": 80.0,
            "requested_loan_gbp": 200000.0,
            "term_years": 25,
            "employment_type": "Employed",
        }

    def test_multiple_flags_only_raise_low_band_to_enhanced(self):
        decision = advisory_decision(0.1, self.application)

        self.assertEqual(decision["outcome"], "Enhanced review")

    def test_high_band_remains_senior_with_flags(self):
        decision = advisory_decision(0.4, self.application)

        self.assertEqual(decision["outcome"], "Senior review")


if __name__ == "__main__":
    unittest.main()
