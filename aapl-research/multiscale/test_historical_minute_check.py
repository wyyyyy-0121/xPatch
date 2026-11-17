"""Check metric weighting, directional abstentions and protected-time boundaries."""
import unittest

import numpy as np

from .historical_minute_check import CUTOFF_NS, EvidenceError, MODELS, calculate, pool, verify_times


class MinuteMetricsTest(unittest.TestCase):
    def test_return_metrics_keep_abstentions_out_of_accuracy(self):
        actual = np.array([[0.1, 0.2], [-0.1, -0.2]])
        prediction = np.array([[0.0, 0.1], [0.0, 0.0]])
        result = calculate(actual, np.array([100.0, 100.0]), prediction, "future_log_return")
        self.assertAlmostEqual(result["endpoint_mse"], 0.025)
        self.assertAlmostEqual(result["endpoint_mae"], 0.15)
        self.assertAlmostEqual(result["path_mse"], 0.0175)
        self.assertEqual((result["active_count"], result["correct_active_count"]), (1, 1))
        self.assertEqual((result["direction_coverage"], result["direction_accuracy_when_active"]), (0.5, 1.0))

    def test_price_direction_is_relative_to_origin_anchor(self):
        actual = np.array([[105.0], [95.0]])
        prediction = np.array([[100.0], [90.0]])
        result = calculate(actual, np.array([100.0, 100.0]), prediction, "future_close")
        self.assertEqual(result["endpoint_mse"], 25.0)
        self.assertEqual((result["active_count"], result["correct_active_count"]), (1, 1))

    def test_invalid_or_nonfinite_values_are_rejected(self):
        with self.assertRaises(EvidenceError):
            calculate(np.array([[1.0]]), np.array([1.0]), np.array([[np.nan]]), "future_close")
        with self.assertRaises(EvidenceError):
            calculate(np.array([[1.0]]), np.array([1.0, 2.0]), np.array([[1.0]]), "future_close")

    def test_protected_boundary_is_strict(self):
        verify_times(np.array([CUTOFF_NS - 3]), np.array([CUTOFF_NS - 1]))
        with self.assertRaises(EvidenceError):
            verify_times(np.array([CUTOFF_NS - 2]), np.array([CUTOFF_NS]))

    def test_pool_weights_folds_by_number_of_origins(self):
        def fold(n, mse, first, last):
            record = {"endpoint_mse": mse, "endpoint_mae": mse, "path_mse": mse,
                      "path_mae": mse, "active_count": 0, "correct_active_count": 0}
            return {"origins": n, "first_decision_ns": first, "last_decision_ns": last,
                    "models": {name: dict(record) for name in MODELS}}
        result = pool([fold(1, 1.0, 1, 1), fold(3, 3.0, 2, 4)])
        self.assertEqual(result["origins"], 4)
        self.assertEqual(result["models"]["gru"]["endpoint_mse"], 2.5)
        self.assertEqual(result["models"]["gru"]["relative_improvement"], 0.0)
        self.assertIsNone(result["models"]["gru"]["direction_accuracy_when_active"])

    def test_pool_rejects_overlapping_fold_decisions(self):
        rows = [{"first_decision_ns": 1, "last_decision_ns": 3},
                {"first_decision_ns": 3, "last_decision_ns": 4}]
        with self.assertRaises(EvidenceError):
            pool(rows)


if __name__ == "__main__":
    unittest.main()
