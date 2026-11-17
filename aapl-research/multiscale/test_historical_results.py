"""Regression checks for honest coverage, origin weighting and elapsed accounting."""
from copy import deepcopy
import unittest

import numpy as np

from .historical_coverage import EvidenceError, TASKS
from .historical_minute_check import MODELS, calculate
from .historical_results import (checked_coverage, descriptive_interval, elapsed_record,
                                 pooled_paths)


class CoverageTest(unittest.TestCase):
    def setUp(self):
        self.coverage = {"status": "private_verified_coverage_snapshot", "script_sha256": "code",
                         "completed_folds": 100, "folds": [
                             {"profile": profile, "horizon": horizon, "target": target,
                              "fold": fold, "status": "completed", "evidence": {"identity": fold}}
                             for profile, horizon, target in sorted(TASKS) for fold in range(1, 6)]}
        self.current = deepcopy(self.coverage)

    def test_complete_grid_contains_all_cells(self):
        self.assertEqual(len(checked_coverage(self.coverage, self.current, allow_partial=False)), 100)

    def test_partial_reporting_requires_explicit_flag(self):
        self.coverage["folds"][-1]["status"] = "running"
        self.coverage["folds"][-1].pop("evidence")
        self.coverage["completed_folds"] = 99
        with self.assertRaises(EvidenceError):
            checked_coverage(self.coverage, self.current, allow_partial=False)
        rows = checked_coverage(self.coverage, self.current, allow_partial=True)
        self.assertEqual(sum(row["status"] == "running" for row in rows), 1)

    def test_duplicate_fold_cannot_replace_a_missing_fold(self):
        self.coverage["folds"][-1] = deepcopy(self.coverage["folds"][0])
        with self.assertRaises(EvidenceError):
            checked_coverage(self.coverage, self.current, allow_partial=True)

    def test_changed_completed_evidence_is_rejected(self):
        self.current["folds"][0]["evidence"]["identity"] = "changed"
        with self.assertRaises(EvidenceError):
            checked_coverage(self.coverage, self.current, allow_partial=True)


class PoolingTest(unittest.TestCase):
    def make_fold(self, first, truth, prediction):
        values = calculate(np.asarray(truth, dtype=float), np.ones(len(truth)),
                           np.asarray(prediction, dtype=float), "future_log_return")
        return {"origins": len(truth), "first_decision_ns": first,
                "last_decision_ns": first + len(truth), "models": {m: deepcopy(values) for m in MODELS}}

    def test_origin_weighted_pool_matches_direct_all_path_errors(self):
        y1, p1 = [[1., 2.]], [[0., 1.]]
        y2, p2 = [[2., 3.], [3., 4.], [4., 5.]], [[2., 4.], [2., 5.], [1., 7.]]
        result = pooled_paths([self.make_fold(1, y1, p1), self.make_fold(10, y2, p2)])
        expected = calculate(np.array(y1 + y2), np.ones(4), np.array(p1 + p2), "future_log_return")
        for model in MODELS:
            for metric in ("endpoint_mse", "path_mse", "endpoint_mae", "direction_coverage"):
                self.assertAlmostEqual(result["models"][model][metric], expected[metric])
            np.testing.assert_allclose(result["models"][model]["per_step_mse"], expected["per_step_mse"])

    def test_overlapping_folds_are_rejected(self):
        f = self.make_fold(1, [[1., 2.]], [[0., 1.]])
        with self.assertRaises(EvidenceError):
            pooled_paths([f, f])


class DescriptiveIntervalsTest(unittest.TestCase):
    def test_full_order_preserves_missing_session_groups(self):
        groups = np.array(["01", "01", "03", "04", "06"])
        grid = np.array(["01", "02", "03", "04", "05", "06"])
        result = descriptive_interval([1., -2., 3., -1., 2.], groups, grid, 2)
        self.assertEqual((result["groups"], result["observed_groups"], result["resamples"]), (6, 4, 5000))
        self.assertEqual(result["group_grid"], "caller_supplied_full_order")
        self.assertNotIn("centered_one_sided_p_improvement", result)

    def test_degenerate_baseline_does_not_get_a_zero_width_interval(self):
        result = descriptive_interval(np.zeros(6), np.arange(6), np.arange(6), 2)
        self.assertEqual(result["status"], "degenerate_bootstrap")
        self.assertIsNone(result["confidence_interval"])

    def test_insufficient_grid_keeps_observed_loss_difference(self):
        result = descriptive_interval([1., 3.], [1, 1], [1, 2], 2)
        self.assertEqual(result["status"], "insufficient_groups")
        self.assertEqual(result["mean_paired_loss_difference"], 2.)
        self.assertIsNone(result["confidence_interval"])


class ElapsedTest(unittest.TestCase):
    def test_raw_envelope_survives_external_pause_estimate(self):
        result = elapsed_record(15000., 14000.)
        self.assertEqual(result["raw_wall_seconds"], 15000.)
        self.assertEqual(result["suspension_adjusted_elapsed_estimate_seconds"], 1000.)
        self.assertIsNone(result["exact_cpu_seconds"])

    def test_pause_larger_than_envelope_does_not_invent_negative_cost(self):
        self.assertEqual(elapsed_record(10., 12.)["suspension_adjusted_elapsed_estimate_seconds"], 0.)

    def test_nonfinite_cost_is_rejected(self):
        with self.assertRaises(EvidenceError):
            elapsed_record(float("nan"))


if __name__ == "__main__":
    unittest.main()
