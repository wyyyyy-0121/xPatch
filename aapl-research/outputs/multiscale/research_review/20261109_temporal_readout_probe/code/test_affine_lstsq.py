"""Pure-synthetic checks of affine solver and frozen uncertainty estimates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

import numpy as np

from affine_lstsq import (ESTIMATE_KIND, compare_losses, fit_affine, loss_summary,
                         transform_prediction)


RECORDS = []


def fixture(seed=20301, n=259, p=34, h=5, noise=0.1):
    rng = np.random.default_rng(seed)
    x = np.column_stack((rng.normal(size=(n, p-1)), np.ones(n)))
    coefficient = rng.normal(size=(p, h))
    y = x@coefficient+noise*rng.normal(size=(n, h))
    query = np.column_stack((rng.normal(size=(129, p-1)), np.ones(129)))
    return x, y, query


class Acceptance(unittest.TestCase):
    def compare_dense(self, name, x, y, query, should_qualify=True):
        started = time.perf_counter()
        fitted = fit_affine(x, y)
        report = fitted.summary()
        prediction, qualification = fitted.predict(query)
        direct_coefficient, _, direct_rank, direct_singular = np.linalg.lstsq(
            x, y, rcond=np.finfo(float).eps*max(x.shape))
        direct_prediction = query@direct_coefficient
        difference = np.abs(prediction-direct_prediction)
        margin = qualification["absolute_error_margin"]
        ratio = float(np.max(difference/np.maximum(margin, np.finfo(float).tiny)))
        self.assertEqual(report["rank"], direct_rank)
        self.assertEqual(report["solver_qualified"], should_qualify)
        self.assertLessEqual(ratio, 1.0)
        self.assertEqual(qualification["estimate_kind"], ESTIMATE_KIND)
        self.assertFalse(qualification["certified_numerical_interval"])
        self.assertFalse(report["coefficient_serialized"])
        self.assertNotIn("coefficient", report)
        singular_relative = float(np.linalg.norm(
            np.asarray(report["singular_values"])-direct_singular)/max(
                1.0, float(np.linalg.norm(direct_singular))))
        self.assertLess(singular_relative, 1e-10)
        data = {"name": name, "n": len(x), "p": x.shape[1], "H": y.shape[1],
                "query_rows": len(query), "independent_reference": "direct original dense X np.linalg.lstsq",
                "rank": report["rank"], "rank_ambiguous": report["rank_ambiguous"],
                "solver_qualified": report["solver_qualified"], "condition_number": report["condition_number"],
                "estimated_perturbation_to_sigma_min": report["estimated_perturbation_to_sigma_min"],
                "max_prediction_difference": float(difference.max()),
                "max_estimated_prediction_margin": float(margin.max()),
                "max_actual_difference_over_estimated_margin": ratio,
                "singular_normalized_difference": singular_relative,
                "QR_reconstruction_relative": report["QR_reconstruction_relative"],
                "QR_orthogonality": report["QR_orthogonality"],
                "SVD_reconstruction_relative": report["SVD_reconstruction_relative"],
                "stationarity_normalized": report["stationarity_normalized"],
                "elapsed_seconds": time.perf_counter()-started}
        RECORDS.append(data)
        return fitted, prediction, qualification

    def test_full_rank(self):
        self.compare_dense("full_rank34", *fixture())
        self.compare_dense("anchor_intercept2", *fixture(seed=20302, p=2))

    def test_exact_and_zero(self):
        x, _, query = fixture(seed=20303)
        coefficient = np.arange(34*3).reshape(34, 3)*0.001
        fitted, prediction, qualification = self.compare_dense("exact_fit", x, x@coefficient, query)
        self.assertLess(fitted.summary()["residual_MSE"], 1e-24)
        _, prediction, qualification = self.compare_dense("zero_target", x, np.zeros((len(x), 3)), query)
        self.assertEqual(float(np.max(np.abs(prediction))), 0.0)
        self.assertEqual(qualification["max_absolute_error_margin"], 0.0)

    def test_ill_conditioned(self):
        x, y, query = fixture(seed=20304)
        for j in range(33):
            factor = 10.0**(-6*j/32)
            x[:, j] *= factor
            query[:, j] *= factor
        fitted, prediction, qualification = self.compare_dense("ill_conditioned_1e6", x, y, query)
        self.assertGreater(fitted.summary()["condition_number"], 1e5)
        # Increasing query norms must increase the estimate before losses are seen.
        larger_query = query.copy()
        larger_query[:, :-1] *= 100
        _, larger = fitted.predict(larger_query)
        self.assertGreater(larger["max_absolute_error_margin"], qualification["max_absolute_error_margin"])
        same = loss_summary(prediction, prediction+0.01, qualification["absolute_error_margin"])
        check = compare_losses(same, same)
        self.assertFalse(check["numerically_separated_under_frozen_estimate"])
        self.assertFalse(check["certified_exact_inequality"])

    def test_rankdef_and_cutoff_rejected(self):
        x, y, query = fixture(seed=20305)
        x[:, 5], query[:, 5] = x[:, 4], query[:, 4]
        fitted, _, _ = self.compare_dense("duplicate_column", x, y, query, False)
        self.assertTrue(fitted.summary()["rank_truncation_requires_exclusion_from_continuation"])
        n, p = 68, 34
        base = np.zeros((n, p))
        for column in range(p-1):
            base[column, column] = 1/np.sqrt(2)
            base[column+34, column] = -1/np.sqrt(2)
        base[:, -1] = 1
        y = np.ones((n, 2))
        cutoff = np.finfo(float).eps*n*np.sqrt(n)
        for multiplier, rank, ambiguous in [(2.0, 34, True), (0.01, 33, False)]:
            near = base.copy()
            near[:, 32] *= multiplier*cutoff
            fitted, _, _ = self.compare_dense(
                f"cutoff_multiplier_{multiplier}", near, y, near, False)
            report = fitted.summary()
            self.assertEqual(report["rank"], rank)
            self.assertEqual(report["rank_ambiguous"], ambiguous)
            self.assertFalse(report["full_affine_class_claim_allowed"])
        # Rank34/noambiguity alone does not bypass the prospective eta safety gate.
        near = base.copy()
        near[:, 32] *= 100*cutoff
        fitted = fit_affine(near, y)
        self.assertEqual(fitted.summary()["rank"], 34)
        self.assertFalse(fitted.summary()["rank_ambiguous"])
        self.assertFalse(fitted.summary()["numerical_gates"]["perturbation_stable"])
        self.assertFalse(fitted.summary()["solver_qualified"])
        RECORDS.append({"name": "fullrank_but_eta_unstable_rejected",
                        "rank": 34, "solver_qualified": False,
                        "estimated_perturbation_to_sigma_min": fitted.summary()["estimated_perturbation_to_sigma_min"]})

    def test_fp32_promotion_and_unit_map(self):
        x, y, query = fixture(seed=20306, n=129, h=60)
        x, query = x.astype(np.float32).astype(np.float64), query.astype(np.float32).astype(np.float64)
        ym = np.linspace(-0.01, 0.01, 60)
        ys = np.linspace(1e-4, 0.05, 60)
        standardized = ((y-ym)/ys).astype(np.float32).astype(np.float64)
        fitted, prediction, qualification = self.compare_dense("FP32_promoted_n129_H60", x, standardized, query)
        direct = np.linalg.lstsq(x, standardized, rcond=np.finfo(float).eps*max(x.shape))[0]
        raw, raw_margin = transform_prediction(prediction, qualification["absolute_error_margin"], ys, ym)
        reference = (query@direct)*ys+ym
        ratio = float(np.max(np.abs(raw-reference)/np.maximum(raw_margin, np.finfo(float).tiny)))
        self.assertLessEqual(ratio, 1.0)
        # Negative scales also propagate magnitudes, rather than negative margins.
        _, negative_margin = transform_prediction(prediction, qualification["absolute_error_margin"], -ys, ym)
        self.assertTrue((negative_margin >= 0).all())
        RECORDS.append({"name": "original_unit_output_map", "max_actual_difference_over_estimated_margin": ratio,
                        "negative_scale_nonnegative_margin": True,
                        "standardized_label_route": "FP64 standardize; FP32 cast; FP64 promotion"})

    def test_loss_comparison_estimate(self):
        y = np.array([[1e-7, 1.0], [-1e-7, -1.0]])
        good, bad = y.copy(), np.zeros_like(y)
        small = np.full_like(y, 1e-15)
        proposed, control = loss_summary(good, y, small), loss_summary(bad, y, small)
        compared = compare_losses(proposed, control)
        self.assertTrue(compared["numerically_separated_under_frozen_estimate"])
        self.assertFalse(compared["certified_exact_inequality"])
        large = np.full_like(y, 100.0)
        proposed = loss_summary(good, y, large)
        compared = compare_losses(proposed, control)
        self.assertFalse(compared["numerically_separated_under_frozen_estimate"])
        self.assertFalse(compared["proposed_worse_under_frozen_estimate"])
        # Rescaling both values and estimated prediction errors preserves a
        # clear direction while MSE and estimated uncertainty transform quadratically.
        pred1, m1 = transform_prediction(good, small, 1e6, 0)
        pred2, m2 = transform_prediction(bad, small, 1e6, 0)
        scaled = compare_losses(loss_summary(pred1, y*1e6, m1), loss_summary(pred2, y*1e6, m2))
        self.assertTrue(scaled["numerically_separated_under_frozen_estimate"])
        self.assertAlmostEqual(scaled["control_minus_proposed_MSE"] / 1e12,
                               control["MSE"], places=12)
        RECORDS.append({"name": "scale_aware_loss_comparison", "clear_difference_resolved": True,
                        "overlapping_estimates_unresolved": True, "output_rescale_preserves_direction": True,
                        "claim_is_not_certified_inequality_or_statistical_significance": True})

    def test_query_actual_rounding(self):
        x, y, query = fixture(seed=20307, p=2)
        fitted, prediction, qualification = self.compare_dense("p2_querynorm_and_scale", x, y, query)
        self.assertEqual(qualification["absolute_error_margin"].shape, prediction.shape)
        self.assertEqual(fitted.summary()["features"], 2)

    def test_reject_invalid(self):
        x, y, query = fixture(seed=20308)
        bad = x.copy()
        bad[0, 0] = np.nan
        badintercept = x.copy()
        badintercept[:, -1] = 0
        cases = [("nonfinite", lambda: fit_affine(bad, y)),
                 ("no_intercept", lambda: fit_affine(badintercept, y)),
                 ("wrong_p", lambda: fit_affine(x[:, :-1], y)),
                 ("wrong_Y", lambda: fit_affine(x, y[:-1])),
                 ("underdetermined", lambda: fit_affine(x[:10], y[:10]))]
        fit = fit_affine(x, y)
        cases.extend([("queryshape", lambda: fit.predict(query[:, :-1])),
                      ("negative_margin", lambda: loss_summary(y, y, -np.ones_like(y))),
                      ("unit_shape", lambda: transform_prediction(y, np.zeros_like(y), np.ones(7)))])
        for name, operation in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                operation()
        RECORDS.append({"name": "invalid_inputs", "hard_rejected": [name for name, _ in cases]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if set(os.sched_getaffinity(0)) != {26}:
        raise SystemExit("Synthetic acceptance requires CPU26")
    env = {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
    if env != {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "-1"}:
        raise SystemExit("Synthetic acceptance requires BLASthreads1 and GPUhidden")
    started_at, started = datetime.now(timezone.utc).isoformat(), time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    code = Path(__file__).resolve().parent
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    review = {
        "status": "passed_synthetic_solver_and_estimate_acceptance" if result.wasSuccessful() else "failed",
        "started_at_UTC": started_at, "finished_at_UTC": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter()-started, "tests_run": result.testsRun,
        "failures": [{"test": str(t), "traceback": msg} for t,msg in result.failures],
        "errors": [{"test": str(t), "traceback": msg} for t,msg in result.errors],
        "source_sha256": {path.name: sha(path) for path in (code/"affine_lstsq.py", Path(__file__).resolve())},
        "stage_scope_sha256": sha(code.parent/"STAGE_SCOPE.json"),
        "records": RECORDS,
        "contract": {
            "solver": "FP64 QR then R-SVD; no Gram inverse, penalty, tolerance rescue or model saved",
            "required_design_columns": [2,34], "query_maximum_batch_fixture": 129,
            "rank_cutoff": "eps64*max(n,p)*smax; ambiguity[0.1tau,10tau]",
            "solver_qualification": "Full column rank, noambiguity, measured QR/SVD/stationarity<=1e-10 and estimated_deltaX/sigma_min<0.1",
            "frozen_perturbation_safety_factor": 32,
            "coefficient_margin": "Residual/conditioning-aware first-order estimate plus disclosed normal-residual/sigma_min^2 sensitivity; querynorm and arithmetic propagate it",
            "comparison": "MSE difference must exceed both estimated margins; output label numerically_separated_under_frozen_estimate",
            "estimate_kind": ESTIMATE_KIND,
            "certified_numerical_interval": False, "statistical_uncertainty_covered": False,
        },
        "execution": {"actual_python_executable": sys.executable, "NumPy_version": np.__version__,
                      "CPU_affinity": sorted(os.sched_getaffinity(0)), "environment": env},
        "market_arrays_or_weights_read": False, "model_forward_or_training_run": False,
        "stop_outer_Q3_read_or_scored": False, "GPU": False,
        "market_admission_or_predictive_candidate": False, "independent_market_confirmation": False,
        "limitations": [
            "Synthetic empirical containment in the estimates does not prove a certified error bound for arbitrary market designs.",
            "Independent direct-SVD reconstruction on all actual final summaries is still required by the parent stage.",
            "Caller must reject solver_qualified=false before tail scoring, and keep numerical qualification distinct from estimated comparison separation.",
            "No data/state/resume/pause or model implementation is tested here; root owns process,checkpoint and budget acceptance.",
            "Estimated perturbation margins do not cover observation noise, model misspecification, training randomness or sampling uncertainty.",
        ],
    }
    args.report.write_text(json.dumps(review, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": review["status"], "tests_run": result.testsRun,
                      "records": len(RECORDS), "elapsed_seconds": review["elapsed_seconds"]}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
