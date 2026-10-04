import unittest
from pathlib import Path

from eval.detector import golden_cases
from eval.metrics import evaluate_band, max_false_accepts_to_pass, poisson_upper_bound

GOLDEN = Path(__file__).resolve().parents[2] / "testdata" / "golden" / "detector_cases.json"


class DetectorParityTests(unittest.TestCase):
    def test_matches_the_shared_golden_cases(self):
        for name, expected, actual in golden_cases(GOLDEN):
            with self.subTest(name):
                self.assertEqual(expected, actual)


class MetricsTests(unittest.TestCase):
    def test_poisson_upper_bound_matches_known_values(self):
        self.assertAlmostEqual(poisson_upper_bound(0), 2.9957, places=3)  # "rule of three"
        self.assertAlmostEqual(poisson_upper_bound(4), 9.1535, places=3)
        self.assertAlmostEqual(poisson_upper_bound(10), 16.9622, places=3)

    def test_plan_section_3_sizing(self):
        # 100 h quiet band passes with at most 12 false accepts, as stated in the plan.
        self.assertEqual(12, max_false_accepts_to_pass(0.2, 100))
        # v4's 20 h set could only ever pass by observing zero, and 14 h could not pass at all.
        self.assertEqual(0, max_false_accepts_to_pass(0.2, 20))
        self.assertEqual(-1, max_false_accepts_to_pass(0.2, 14))

    def test_band_fails_on_raw_average_that_hides_uncertainty(self):
        r = evaluate_band("quiet", negative_hours=20, false_accepts=3, positives_total=200, positives_detected=196)
        self.assertAlmostEqual(r.fa_per_hour, 0.15)
        self.assertFalse(r.passed)
        self.assertEqual(2, len(r.failures))  # too few hours, and the bound exceeds 0.2/h

    def test_band_passes(self):
        r = evaluate_band("loud", negative_hours=40, false_accepts=20, positives_total=300, positives_detected=250)
        self.assertTrue(r.passed, r.failures)


if __name__ == "__main__":
    unittest.main()
