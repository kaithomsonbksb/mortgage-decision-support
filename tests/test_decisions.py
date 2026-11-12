import unittest

from src.decisions import advisory_decision


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


class AdvisoryDecisionTests(unittest.TestCase):
    def setUp(self):
        self.application = valid_application()

    def test_low_score_without_flags_is_standard_review(self):
        decision = advisory_decision(0.19, self.application)

        self.assertEqual(decision["outcome"], "Standard review")
        self.assertEqual(decision["band"], "Low")
        self.assertEqual(decision["flags"], [])

    def test_medium_score_is_enhanced_review(self):
        decision = advisory_decision(0.20, self.application)

        self.assertEqual(decision["outcome"], "Enhanced review")
        self.assertEqual(decision["band"], "Medium")

    def test_high_score_is_senior_review(self):
        decision = advisory_decision(0.40, self.application)

        self.assertEqual(decision["outcome"], "Senior review")
        self.assertEqual(decision["band"], "High")

    def test_low_score_with_high_ltv_is_enhanced_review(self):
        self.application["loan_to_value_pct"] = 91

        decision = advisory_decision(0.10, self.application)

        self.assertEqual(decision["outcome"], "Enhanced review")
        self.assertIn("F2: loan-to-value exceeds 90%", decision["flags"])

    def test_low_score_with_previous_default_is_enhanced_review(self):
        self.application["previous_default_flag"] = 1

        decision = advisory_decision(0.10, self.application)

        self.assertEqual(decision["outcome"], "Enhanced review")
        self.assertIn("F1: previous default recorded", decision["flags"])

    def test_two_missed_payments_are_enhanced_review(self):
        self.application["missed_payments_24m"] = 2

        decision = advisory_decision(0.10, self.application)

        self.assertEqual(decision["outcome"], "Enhanced review")
        self.assertIn("F1: 2 or more missed payments in 24 months", decision["flags"])

    def test_previous_default_and_high_ltv_are_enhanced_review(self):
        self.application["previous_default_flag"] = 1
        self.application["loan_to_value_pct"] = 92

        decision = advisory_decision(0.10, self.application)

        self.assertEqual(decision["outcome"], "Enhanced review")
        self.assertEqual(len(decision["flags"]), 2)

    def test_loan_to_income_threshold_note_is_inclusive(self):
        self.application["requested_loan_gbp"] = 270000

        decision = advisory_decision(0.10, self.application)

        self.assertIn(
            "High loan-to-income ratio: at or above 4.5x income", decision["notes"]
        )

    def test_debt_to_income_ratio_of_1_2_can_be_assessed(self):
        self.application["debt_to_income_ratio"] = 1.2

        decision = advisory_decision(0.10, self.application)

        self.assertNotEqual(decision["outcome"], "Unable to assess")

    def test_missing_credit_score_cannot_be_assessed(self):
        self.application["credit_score"] = None

        decision = advisory_decision(0.10, self.application)

        self.assertEqual(decision["outcome"], "Unable to assess")
        self.assertIn("credit_score: missing", decision["notes"])

    def test_score_band_boundaries(self):
        cases = (
            (0.20, "Medium", "Enhanced review"),
            (0.40, "High", "Senior review"),
        )
        for score, expected_band, expected_outcome in cases:
            with self.subTest(score=score):
                decision = advisory_decision(score, self.application)
                self.assertEqual(decision["band"], expected_band)
                self.assertEqual(decision["outcome"], expected_outcome)

    def test_ltv_boundary(self):
        for ltv, has_flag in ((90, False), (91, True)):
            with self.subTest(ltv=ltv):
                self.application["loan_to_value_pct"] = ltv
                decision = advisory_decision(0.10, self.application)
                self.assertEqual(
                    any(
                        "loan-to-value exceeds 90%" in flag
                        for flag in decision["flags"]
                    ),
                    has_flag,
                )


if __name__ == "__main__":
    unittest.main()
