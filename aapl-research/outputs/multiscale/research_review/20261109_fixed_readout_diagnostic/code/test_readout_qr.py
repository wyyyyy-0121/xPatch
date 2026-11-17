"""Pure synthetic acceptance for the frozen fixed-readout QR kernel.

Run with CPU26 affinity and NumPy/BLAS thread counts one. No market data,
weights, old experiment modules, GPU, network or predictive fitting is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest
from datetime import datetime, timezone

import numpy as np

from readout_qr import N_FEATURES, ReadoutQR


RESULTS: dict = {"fixtures": [], "rejection_checks": []}
EPS = np.finfo(np.float64).eps
NORMALIZED_TOL = 1e-10


def normalized_error(left, right) -> float:
    a, b = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(1.0, float(np.linalg.norm(a)),
                                           float(np.linalg.norm(b))))


def direct_dense(x, y, b0, baseline):
    """Independent original-matrix LAPACK least squares, without streamed R."""
    coefficient, _, rank, singular = np.linalg.lstsq(
        x, y, rcond=EPS * max(len(x), N_FEATURES))
    fitted = x @ coefficient
    original = x @ b0
    return {
        "rank": int(rank), "singular_values": singular,
        "SSE_original": float(np.linalg.norm(original - y) ** 2),
        "SSE_projected": float(np.linalg.norm(fitted - y) ** 2),
        "SSE_head_gap": float(np.linalg.norm(original - fitted) ** 2),
        "SSE_zero_baseline": float(np.linalg.norm(baseline[None, :] - y) ** 2),
    }


def direct_dense_svd(x, y, b0, baseline):
    """Independent original-matrix SVD projection for threshold fixtures."""
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    cutoff = EPS * max(len(x), N_FEATURES) * singular[0]
    keep = singular > cutoff
    fitted = u[:, keep] @ (u[:, keep].T @ y)
    original = x @ b0
    return {
        "rank": int(np.count_nonzero(keep)), "singular_values": singular,
        "SSE_original": float(np.linalg.norm(original - y) ** 2),
        "SSE_projected": float(np.linalg.norm(fitted - y) ** 2),
        "SSE_head_gap": float(np.linalg.norm(original - fitted) ** 2),
        "SSE_zero_baseline": float(np.linalg.norm(baseline[None, :] - y) ** 2),
    }


def make_fixture(seed=9081, n=257, h=5):
    rng = np.random.default_rng(seed)
    x = np.column_stack((rng.normal(size=(n, 33)), np.ones(n)))
    true = rng.normal(size=(34, h))
    y = x @ true + 0.2 * rng.normal(size=(n, h))
    b0 = rng.normal(size=(34, h))
    baseline = rng.normal(size=h)
    return x, y, b0, baseline


def stream(x, y, chunks=128, identity=None):
    state = ReadoutQR(y.shape[1], expected_n=len(x), identity=identity)
    for start in range(0, len(x), chunks):
        state.update(x[start:start + chunks], y[start:start + chunks])
    return state


def rehash(state):
    result = copy.deepcopy(state)
    result.pop("state_sha256", None)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
    result["state_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return result


class SyntheticAcceptance(unittest.TestCase):
    def compare_dense(self, name, x, y, b0, baseline, chunks=128, svd=False):
        state = stream(x, y, chunks)
        result = state.solve(b0, zero_prediction=baseline)
        dense = (direct_dense_svd if svd else direct_dense)(x, y, b0, baseline)
        errors = {key: normalized_error(result[key], dense[key]) for key in (
            "SSE_original", "SSE_projected", "SSE_head_gap", "SSE_zero_baseline")}
        errors["singular_values"] = normalized_error(
            result["singular_values"], np.pad(dense["singular_values"],
                                               (0, 34-len(dense["singular_values"]))))
        self.assertEqual(result["rank"], dense["rank"])
        self.assertLessEqual(max(errors.values()), NORMALIZED_TOL, (name, errors))
        self.assertTrue(result["numerical_gates_passed"], (name, result["numerical_gates"]))
        self.assertNotIn("Bstar", result)
        self.assertFalse(result["projected_coefficients_retained"])
        self.assertFalse(result["forecasting_candidate_admitted"])
        self.assertFalse(result["independent_market_confirmation"])
        RESULTS["fixtures"].append({
            "name": name, "n": len(x), "H": y.shape[1], "chunk_rows": chunks,
            "independent_reference": "dense_original_X_SVD_projection" if svd
                else "dense_original_X_numpy_linalg_lstsq",
            "normalized_errors": errors, "rank": result["rank"],
            "rank_ambiguous": result["rank_ambiguous"],
            "full_head_class_optimum_identifiable": result["full_head_class_optimum_identifiable"],
            "stationarity_normalized": result["stationarity_normalized"],
            "Pythagorean_MSE_residual": result["Pythagorean_MSE_residual"],
            "max_QR_reconstruction_relative": result["max_QR_reconstruction_relative"],
            "max_QR_orthogonality_frobenius": result["max_QR_orthogonality_frobenius"],
            "numerical_gates_passed": result["numerical_gates_passed"],
        })
        return state, result

    def test_full_rank(self):
        _, result = self.compare_dense("full_rank", *make_fixture())
        self.assertEqual(result["rank"], 34)
        self.assertTrue(result["full_head_class_optimum_identifiable"])
        self.assertTrue(result["gap_numerically_resolved"])

    def test_exact_fit(self):
        x, _, b0, baseline = make_fixture(seed=9082)
        y = x @ b0
        _, result = self.compare_dense("exact_fit_original_head", x, y, b0, baseline)
        self.assertLessEqual(result["MSE_projected"], 1e-24)
        self.assertFalse(result["gap_numerically_resolved"])

    def test_zero_target(self):
        x, y, b0, _ = make_fixture(seed=9083)
        y[:] = 0.0
        baseline = np.zeros(y.shape[1])
        _, result = self.compare_dense("zero_target_nonzero_head", x, y, b0, baseline)
        self.assertEqual(result["SSE_projected"], 0.0)
        self.assertEqual(result["SSE_zero_baseline"], 0.0)
        self.assertIsNone(result["gap_over_zero_baseline"])
        _, result = self.compare_dense("zero_target_zero_head", x, y, b0 * 0, baseline)
        self.assertFalse(result["gap_numerically_resolved"])

    def test_duplicate_column(self):
        x, y, b0, baseline = make_fixture(seed=9084)
        x[:, 30] = x[:, 5]
        _, result = self.compare_dense("exact_duplicate_column", x, y, b0, baseline)
        self.assertEqual(result["rank"], 33)
        self.assertFalse(result["full_head_class_optimum_identifiable"])
        self.assertIn("resolved_projection_only", result["attribution_status"])
        self.assertFalse(result["exact_algebraic_dependence_certified_by_kernel"])

    def test_near_cutoff(self):
        n, h = 68, 3
        # Pair-supported orthogonal coordinates, exact observed constant column.
        # This isolates cutoff handling from arbitrary ill-conditioned rotations.
        base = np.zeros((n, 34))
        for column in range(33):
            base[column, column] = 1 / np.sqrt(2)
            base[column+34, column] = -1 / np.sqrt(2)
        base[:, -1] = 1.0
        y = np.column_stack((base[:, 32], base[:, 4] + 0.3*base[:, 32],
                             2.0 * np.ones(n) + base[:, 7]))
        b0 = np.zeros((34, h))
        b0[-1, :] = [0.1, -0.2, 0.4]
        baseline = np.zeros(h)
        for multiplier, rank, ambiguous in ((2.0, 34, True), (0.01, 33, False)):
            with self.subTest(multiplier=multiplier):
                x = base.copy()
                x[:, 32] *= EPS * n * np.sqrt(n) * multiplier
                _, result = self.compare_dense(
                    f"cutoff_multiplier_{multiplier}", x, y, b0, baseline, chunks=17, svd=True)
                self.assertEqual(result["rank"], rank)
                self.assertEqual(result["rank_ambiguous"], ambiguous)
                self.assertFalse(result["full_head_class_optimum_identifiable"])
                self.assertIn("resolved_projection_only", result["attribution_status"])

    def test_discarded_original_gate(self):
        n = 68
        x = np.zeros((n, 34))
        for column in range(33):
            x[column, column] = 1.0
            x[column+34, column] = -1.0
        x[:, -1] = 1.0
        x[:, 32] *= EPS * n * np.sqrt(n) * 0.001
        y = np.zeros((n, 2))
        b0 = np.zeros((34, 2))
        b0[32, :] = 1e16
        result = stream(x, y, 17).solve(b0)
        self.assertFalse(result["numerical_gates"]["original_in_resolved_subspace"])
        self.assertFalse(result["numerical_gates_passed"])
        self.assertFalse(result["full_head_class_optimum_identifiable"])
        RESULTS["fixtures"].append({"name": "nonnegligible_discarded_original",
            "rank": result["rank"], "full_head_class_optimum_identifiable": False,
            "discarded_original_prediction_relative_frobenius":
                result["discarded_original_prediction_relative_frobenius"],
            "required_gate_rejected": True})

    def test_chunks_order_and_resume(self):
        x, y, b0, baseline = make_fixture(seed=9085, n=259, h=4)
        identity = {"fixture": "chunks", "source_sha256": "abc123", "expected_n": len(x)}
        reference = stream(x, y, 128, identity)
        expected = reference.solve(b0, baseline)
        r0 = np.asarray(reference.state_dict()["augmented_R"])
        errors = []
        for size in (1, 17, 128, len(x)):
            other = stream(x, y, size, identity)
            actual = other.solve(b0, baseline)
            r = np.asarray(other.state_dict()["augmented_R"])
            errors.extend(normalized_error(actual[k], expected[k]) for k in (
                "SSE_original", "SSE_projected", "SSE_head_gap", "SSE_zero_baseline"))
            # Products are independent-test invariants, never a solving method.
            errors.append(normalized_error(r.T @ r, r0.T @ r0))
        rng = np.random.default_rng(9832)
        permutation = rng.permutation(len(x))
        shuffled = stream(x[permutation], y[permutation], 17, identity)
        actual = shuffled.solve(b0, baseline)
        errors.extend(normalized_error(actual[k], expected[k]) for k in (
            "SSE_original", "SSE_projected", "SSE_head_gap", "SSE_zero_baseline"))
        midpoint = 119
        partial = ReadoutQR(y.shape[1], expected_n=len(x), identity=identity)
        for start in range(0, midpoint, 17):
            partial.update(x[start:min(start+17, midpoint)], y[start:min(start+17, midpoint)])
        stored = json.loads(json.dumps(partial.state_dict(), allow_nan=False))
        recovered = ReadoutQR.from_state(stored, expected_identity=identity, expected_n=len(x))
        for start in range(midpoint, len(x), 17):
            recovered.update(x[start:start+17], y[start:start+17])
        resumed = recovered.solve(b0, baseline)
        self.assertEqual(recovered.n_rows, len(x))
        errors.extend(normalized_error(resumed[k], expected[k]) for k in (
            "SSE_original", "SSE_projected", "SSE_head_gap", "SSE_zero_baseline"))
        self.assertLessEqual(max(errors), NORMALIZED_TOL)
        self.assertEqual(recovered.identity, identity)
        empty = ReadoutQR(4, expected_n=len(x), identity=identity)
        self.assertEqual(ReadoutQR.from_state(empty.state_dict(), expected_identity=identity,
                                             expected_n=len(x)).n_rows, 0)
        RESULTS["fixtures"].append({"name": "chunks_order_JSON_save_resume",
            "chunk_sizes": [1, 17, 128, len(x)], "random_row_permutation": True,
            "resume_at_row": midpoint, "no_double_count": True,
            "empty_resume": True, "max_normalized_error": max(errors)})

    def test_rejections_transactional(self):
        x, y, b0, _ = make_fixture(seed=9086, n=257, h=3)
        identity = {"fixture": "damaged", "source": "frozen"}
        partial = ReadoutQR(3, expected_n=len(x), identity=identity)
        partial.update(x[:17], y[:17])
        before = partial.state_dict()["state_sha256"]
        bad_x = x[17:20].copy()
        bad_x[0, 0] = np.nan
        wrong_intercept = x[17:20].copy()
        wrong_intercept[:, -1] = 0.0
        failures = {
            "wrong_X_dimension": lambda: partial.update(x[17:20, :-1], y[17:20]),
            "wrong_Y_dimension": lambda: partial.update(x[17:20], y[17:20, :-1]),
            "nonfinite_X": lambda: partial.update(bad_x, y[17:20]),
            "nonfinite_Y": lambda: partial.update(x[17:20], y[17:20] * np.nan),
            "nonconstant_intercept": lambda: partial.update(wrong_intercept, y[17:20]),
            "empty_update": lambda: partial.update(x[:0], y[:0]),
            "expected_n_overflow": lambda: partial.update(x, y),
            "incomplete_solve": lambda: partial.solve(b0),
            "wrong_B0_shape": lambda: partial.solve(b0[:-1]),
        }
        for name, operation in failures.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                operation()
            self.assertEqual(partial.state_dict()["state_sha256"], before)
            RESULTS["rejection_checks"].append({"name": name, "rejected": True,
                                                  "state_unchanged": True})
        stored = partial.state_dict()
        cases = {}
        damaged = copy.deepcopy(stored)
        damaged["augmented_R"][0][0] += 1
        cases["checksum_damage"] = damaged
        for name, field, value in (
            ("rehash_wrong_features", "n_features", 33),
            ("rehash_wrong_shape", "augmented_R", [[1.0]]),
            ("rehash_invalid_cursor", "n_rows", 18),
            ("rehash_fractional_expected_n", "expected_n", 257.0),
            ("rehash_invalid_QR_gate", "max_qr_reconstruction_relative", 1e-5),
            ("rehash_unknown_schema", "schema", "wrong"),
        ):
            damaged = copy.deepcopy(stored)
            damaged[field] = value
            cases[name] = rehash(damaged)
        damaged = copy.deepcopy(stored)
        damaged["augmented_R"][1][0] = 1.0
        cases["rehash_nontriangular"] = rehash(damaged)
        damaged = copy.deepcopy(stored)
        damaged["augmented_R"][0][0] = float("nan")
        cases["nan_state"] = damaged
        damaged = copy.deepcopy(stored)
        damaged.pop("expected_n")
        cases["rehash_missing_field"] = rehash(damaged)
        for name, state in cases.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                ReadoutQR.from_state(state, expected_identity=identity, expected_n=len(x))
            RESULTS["rejection_checks"].append({"name": name, "rejected": True})
        for name, kwargs in (
            ("wrong_identity", {"expected_identity": {"fixture": "other"}}),
            ("wrong_frozen_n", {"expected_identity": identity, "expected_n": len(x)+1}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                ReadoutQR.from_state(stored, **kwargs)
            RESULTS["rejection_checks"].append({"name": name, "rejected": True})
        completed = stream(x, y, 128)
        with self.assertRaises(ValueError):
            completed.solve(b0, np.zeros(2))
        RESULTS["rejection_checks"].append({"name": "wrong_baseline_shape", "rejected": True})

    def test_maximum_target_shape(self):
        x, y, b0, baseline = make_fixture(seed=9087, n=2048, h=60)
        started = time.perf_counter()
        _, result = self.compare_dense("H60_augmented94_columns", x, y, b0, baseline)
        RESULTS["QR_only_synthetic_benchmark"] = {
            "n": len(x), "H": 60, "features": 34, "augmented_columns": 94,
            "chunk_rows": 128, "seconds_including_independent_dense_lstsq":
                time.perf_counter() - started,
            "full_rank_qualified": result["full_head_class_optimum_identifiable"],
            "not_a_model_replay_runtime_estimate": True,
        }


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    affinity = sorted(os.sched_getaffinity(0))
    required_env = {key: os.environ.get(key) for key in (
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
    if affinity != [26] or required_env != {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
        "CUDA_VISIBLE_DEVICES": "-1"}:
        raise SystemExit("Frozen synthetic execution requires CPU26, BLASthreads1 and GPUhidden")
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SyntheticAcceptance)
    execution = unittest.TextTestRunner(verbosity=2).run(suite)
    code = Path(__file__).resolve().parent
    repo = code.parents[4]
    protocol = repo / "outputs/multiscale/research_review/20261109_residual_gap_audit"
    report = {
        "status": "passed_synthetic_engineering_only" if execution.wasSuccessful() else "failed",
        "started_at_UTC": started_at, "finished_at_UTC": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "tests_run": execution.testsRun,
        "failures": [{"test": str(t), "traceback": trace} for t, trace in execution.failures],
        "errors": [{"test": str(t), "traceback": trace} for t, trace in execution.errors],
        "source_sha256": {p.name: sha(p) for p in (code/"readout_qr.py", Path(__file__).resolve())},
        "protocol_sha256": {name: sha(protocol/name) for name in (
            "NEXT_FIXED_READOUT_DIAGNOSTIC_PLAN.json", "IDENTIFICATION_CONFIRMATION_REVIEW.json")},
        "numerical_contract": {
            "features": 34, "arithmetic": "FP64", "solver": "stream augmented[X,Y] QR; final X-block SVD",
            "rank_cutoff": "eps64*max(n_fit,34)*sigma_max", "retained": "singular>cutoff",
            "ambiguity_band": "[0.1*cutoff,10*cutoff] inclusive",
            "independent_dense_normalized_tolerance": NORMALIZED_TOL,
            "rank_truncation_never_qualifies_full_class": True,
            "no_projected_coefficient_return_or_checkpoint": True,
            "baseline_uses_frozen_standardized_prediction_vector": True,
        },
        **RESULTS,
        "execution": {"affinity": affinity, "environment": required_env,
                      "NumPy_version": np.__version__, "Python_version": sys.version},
        "market_arrays_opened": False, "checkpoint_weights_opened": False,
        "stop_or_outer_scores_read": False, "Q3_values_read": False,
        "GPU_used": False, "network_used": False, "training_started": False,
        "old_engine_or_controller_imported": False,
        "market_execution_admitted_by_this_report": False,
        "empirical_market_diagnostic_executed": False,
        "forecasting_candidate_admitted": False, "predictive_success": False,
        "independent_confirmation": False,
        "limitations": [
            "Synthetic qualification only; checkpoint/scaler/split/source binding and model replay cost are caller responsibilities.",
            "Caller owns exact contiguous fit-row cursor, atomic journal and process signal/deadline behavior.",
            "SHA256 detects corruption or identity mismatch; it does not authenticate adversarially rewritten data.",
            "Near-cutoff fixture has deliberately isolated orthogonal coordinates; arbitrary ill-conditioned market states must retain their numerical/rank failure status.",
        ],
    }
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "tests_run": execution.testsRun,
                      "elapsed_seconds": report["elapsed_seconds"],
                      "fixtures": len(RESULTS["fixtures"]),
                      "rejection_checks": len(RESULTS["rejection_checks"])}))
    return 0 if execution.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
