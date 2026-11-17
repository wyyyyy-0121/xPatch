"""FP64 streaming augmented QR for a fixed 34-column diagnostic readout.

No model/data I/O and no forecast candidate or fitted coefficient is returned.
The caller owns identity freezing, contiguous row coverage and atomic file I/O.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


N_FEATURES = 34
QR_TOLERANCE = 1e-10
STATIONARITY_TOLERANCE = 1e-10
DISCARDED_ORIGINAL_TOLERANCE = 1e-10
LOSS_TOLERANCE_FACTOR = 1e-9
SCHEMA = "fixed_readout_augmented_qr_v1"


def _digest(value: dict[str, Any]) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return result


def _squared_norm(value: np.ndarray) -> float:
    result = float(np.linalg.norm(value, ord="fro") ** 2)
    if not np.isfinite(result):
        raise ValueError("SSE/norm overflow; state is not numerically qualified")
    return result


class ReadoutQR:
    """Accumulate QR([X,Y]); X has hidden32, observed anchor and intercept1.

    ``zero_prediction`` in solve is the *prediction* of the original zero-return
    baseline in standardized units, normally -y_mean/y_std (one value per H).
    It is not the standardized observed target and is not reestimated here.
    """

    def __init__(self, n_targets: int, expected_n: int | None = None,
                 identity: Any = None):
        if (isinstance(n_targets, bool) or not isinstance(n_targets, (int, np.integer))
                or n_targets < 1):
            raise ValueError("n_targets must be a positive integer")
        if expected_n is not None and (isinstance(expected_n, bool)
                or not isinstance(expected_n, (int, np.integer)) or expected_n < 1):
            raise ValueError("expected_n must be a positive integer or None")
        # JSON roundtrip rejects unserializable identity values and detaches aliases.
        self.identity = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
        self.n_targets = int(n_targets)
        self.expected_n = None if expected_n is None else int(expected_n)
        self.n_rows = 0
        self.blocks = 0
        self._r = np.empty((0, N_FEATURES + self.n_targets), dtype=np.float64)
        self.max_qr_reconstruction_relative = 0.0
        self.max_qr_orthogonality_frobenius = 0.0

    def update(self, X: Any, Y: Any) -> dict[str, Any]:
        """Commit one complete block after finite/dimension and QR acceptance.

        Failed validation/QR leaves the previous committed state unchanged.
        Block ordering and cursor binding remain the caller's responsibility.
        """
        x, y = _array(X, "X", 2), _array(Y, "Y", 2)
        if x.shape[1] != N_FEATURES or y.shape != (len(x), self.n_targets):
            raise ValueError("Require X=(batch,34), Y=(same_batch,n_targets)")
        if len(x) == 0:
            raise ValueError("Empty blocks are not committed")
        if not np.array_equal(x[:, -1], np.ones(len(x), dtype=np.float64)):
            raise ValueError("X column33 must be the exact intercept1")
        if self.expected_n is not None and self.n_rows + len(x) > self.expected_n:
            raise ValueError("Block exceeds frozen expected_n")
        augmented = np.concatenate((x, y), axis=1)
        stack = np.concatenate((self._r, augmented), axis=0)
        q, reduced = np.linalg.qr(stack, mode="reduced")
        reconstruction = float(np.linalg.norm(stack - q @ reduced, ord="fro")
                               / max(1.0, float(np.linalg.norm(stack, ord="fro"))))
        orthogonality = float(np.linalg.norm(q.T @ q - np.eye(q.shape[1]), ord="fro"))
        if (not np.isfinite(reduced).all() or not np.isfinite(reconstruction)
                or not np.isfinite(orthogonality)
                or reconstruction > QR_TOLERANCE or orthogonality > QR_TOLERANCE):
            raise ValueError("Frozen per-block QR reconstruction/orthogonality gate failed")
        self._r = reduced
        self.n_rows += len(x)
        self.blocks += 1
        self.max_qr_reconstruction_relative = max(self.max_qr_reconstruction_relative, reconstruction)
        self.max_qr_orthogonality_frobenius = max(self.max_qr_orthogonality_frobenius, orthogonality)
        return {"n_rows": self.n_rows, "blocks": self.blocks,
                "qr_reconstruction_relative": reconstruction,
                "qr_orthogonality_frobenius": orthogonality}

    def solve(self, B0: Any, zero_prediction: Any = None) -> dict[str, Any]:
        """Return scalar projection diagnostics; never return/store B_star.

        Numerical truncation always limits the result to the resolved subspace.
        This kernel cannot independently certify exact algebraic dependencies.
        """
        b0 = _array(B0, "B0", 2)
        if b0.shape != (N_FEATURES, self.n_targets):
            raise ValueError("B0 must have shape (34,n_targets)")
        if not self.n_rows or (self.expected_n is not None and self.n_rows != self.expected_n):
            raise ValueError("Cannot solve an empty or incomplete frozen row sequence")
        baseline = (np.zeros(self.n_targets, dtype=np.float64) if zero_prediction is None
                    else _array(zero_prediction, "zero_prediction", 1))
        if baseline.shape != (self.n_targets,):
            raise ValueError("zero_prediction must have shape (n_targets,)")
        a, c = self._r[:, :N_FEATURES], self._r[:, N_FEATURES:]
        u, values, _ = np.linalg.svd(a, full_matrices=False)
        singulars = np.zeros(N_FEATURES, dtype=np.float64)
        singulars[:len(values)] = values
        smax = float(singulars[0])
        cutoff = float(np.finfo(np.float64).eps * max(self.n_rows, N_FEATURES) * smax)
        retained = values > cutoff
        rank = int(np.count_nonzero(retained))
        band = (singulars >= 0.1 * cutoff) & (singulars <= 10.0 * cutoff)
        rank_ambiguous = bool(np.any(band))
        ur = u[:, retained]
        projected = ur @ (ur.T @ c)
        residual = projected - c
        original_fit = a @ b0
        original_residual = original_fit - c
        difference = original_fit - projected
        dropped_original = original_fit - ur @ (ur.T @ original_fit)
        zero_residual = a[:, -1, None] * baseline[None, :] - c
        sse_original = _squared_norm(original_residual)
        sse_projected = _squared_norm(residual)
        sse_gap = _squared_norm(difference)
        sse_zero = _squared_norm(zero_residual)
        denom = self.n_rows * self.n_targets
        loss_original, loss_projected, gap, zero_loss = (
            x / denom for x in (sse_original, sse_projected, sse_gap, sse_zero))
        loss_tol = LOSS_TOLERANCE_FACTOR * max(1.0, loss_original, loss_projected, gap)
        identity_residual = loss_original - loss_projected - gap
        stationarity = float(np.linalg.norm(a.T @ residual, ord="fro") / max(
            1.0, smax * float(np.linalg.norm(c, ord="fro")),
            smax * float(np.linalg.norm(residual, ord="fro"))))
        dropped_relative = float(np.linalg.norm(dropped_original, ord="fro")
                                 / max(1.0, float(np.linalg.norm(original_fit, ord="fro"))))
        gates = {"QR_reconstruction_and_orthogonality": (
                    self.max_qr_reconstruction_relative <= QR_TOLERANCE
                    and self.max_qr_orthogonality_frobenius <= QR_TOLERANCE),
                 "effective_space_stationarity": stationarity <= STATIONARITY_TOLERANCE,
                 "projection_not_above_original": loss_projected <= loss_original + loss_tol,
                 "nonnegative_gap": gap >= -loss_tol,
                 "Pythagorean_identity": abs(identity_residual) <= loss_tol,
                 "original_in_resolved_subspace": dropped_relative <= DISCARDED_ORIGINAL_TOLERANCE}
        numerical_passed = all(gates.values())
        full_class = numerical_passed and rank == N_FEATURES and not rank_ambiguous
        if rank_ambiguous:
            attribution = "nonidentifiable_rank_ambiguous_resolved_projection_only"
        elif rank < N_FEATURES:
            attribution = "nonidentifiable_rank_truncated_resolved_projection_only"
        elif not numerical_passed:
            attribution = "nonidentifiable_numerical_gate_failed"
        else:
            attribution = "qualified_full_34_column_empirical_head_class"
        return {"status": "completed", "n_rows": self.n_rows, "n_targets": self.n_targets,
                "features": N_FEATURES, "blocks": self.blocks,
                "method": "FP64 stream augmented[X,Y] reducedQR; final X-block SVD projection",
                "SSE_original": sse_original, "SSE_projected": sse_projected,
                "SSE_head_gap": sse_gap, "SSE_zero_baseline": sse_zero,
                "MSE_original": loss_original, "MSE_projected": loss_projected,
                "MSE_head_gap": gap, "MSE_zero_baseline": zero_loss,
                "gap_over_zero_baseline": gap / zero_loss if zero_loss > 0.0 else None,
                "gap_numerically_resolved": gap > loss_tol,
                "singular_values": singulars.tolist(), "rank": rank,
                "rank_cutoff": cutoff, "rank_ambiguity_band": [0.1*cutoff, 10.0*cutoff],
                "rank_ambiguous": rank_ambiguous,
                "ambiguous_singular_indices": np.flatnonzero(band).tolist(),
                "retained_condition_number": smax / float(values[retained][-1]) if rank else None,
                "stationarity_normalized": stationarity,
                "Pythagorean_SSE_residual": sse_original-sse_projected-sse_gap,
                "Pythagorean_MSE_residual": identity_residual,
                "Pythagorean_MSE_tolerance": loss_tol,
                "discarded_original_prediction_relative_frobenius": dropped_relative,
                "max_QR_reconstruction_relative": self.max_qr_reconstruction_relative,
                "max_QR_orthogonality_frobenius": self.max_qr_orthogonality_frobenius,
                "numerical_gates": gates, "numerical_gates_passed": numerical_passed,
                "full_head_class_optimum_identifiable": full_class,
                "attribution_status": attribution,
                "exact_algebraic_dependence_certified_by_kernel": False,
                "projected_coefficients_retained": False,
                "forecasting_candidate_admitted": False,
                "independent_market_confirmation": False}

    def state_dict(self) -> dict[str, Any]:
        state = {"schema": SCHEMA, "n_features": N_FEATURES,
                 "n_targets": self.n_targets, "expected_n": self.expected_n,
                 "identity": json.loads(json.dumps(self.identity, allow_nan=False)),
                 "n_rows": self.n_rows, "blocks": self.blocks,
                 "augmented_R": self._r.tolist(),
                 "max_qr_reconstruction_relative": self.max_qr_reconstruction_relative,
                 "max_qr_orthogonality_frobenius": self.max_qr_orthogonality_frobenius}
        state["state_sha256"] = _digest(state)
        return state

    @classmethod
    def from_state(cls, state: dict[str, Any], expected_identity: Any = None,
                   expected_n: int | None = None) -> "ReadoutQR":
        if not isinstance(state, dict):
            raise ValueError("QR state must be a JSON object")
        value = dict(state)
        checksum = value.pop("state_sha256", None)
        if checksum != _digest(value) or value.get("schema") != SCHEMA:
            raise ValueError("Damaged or unknown QR state")
        required = {"schema", "n_features", "n_targets", "expected_n", "identity",
                    "n_rows", "blocks", "augmented_R", "max_qr_reconstruction_relative",
                    "max_qr_orthogonality_frobenius"}
        if set(value) != required:
            raise ValueError("QR state fields do not match the frozen schema")
        if value.get("n_features") != N_FEATURES:
            raise ValueError("QR state feature dimension changed")
        if expected_identity is not None and value.get("identity") != expected_identity:
            raise ValueError("QR state identity does not match frozen run")
        if expected_n is not None and value.get("expected_n") != expected_n:
            raise ValueError("QR state expected_n does not match frozen row count")
        result = cls(value["n_targets"], value["expected_n"], value["identity"])
        n, blocks = value["n_rows"], value["blocks"]
        if (isinstance(n, bool) or not isinstance(n, int) or n < 0
                or isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 0
                or (n == 0) != (blocks == 0) or blocks > n
                or result.expected_n is not None and n > result.expected_n):
            raise ValueError("Invalid QR row/block cursor")
        factor = np.asarray(value["augmented_R"], dtype=np.float64)
        expected_shape = (min(n, N_FEATURES+result.n_targets), N_FEATURES+result.n_targets)
        if n == 0 and factor.shape == (0,):
            factor = np.empty(expected_shape, dtype=np.float64)
        if factor.shape != expected_shape or not np.isfinite(factor).all():
            raise ValueError("Invalid augmented QR factor dimensions or values")
        if factor.size and np.linalg.norm(np.tril(factor, -1), ord="fro") != 0.0:
            raise ValueError("Saved reducedQR factor is not upper trapezoidal")
        for key in ("max_qr_reconstruction_relative", "max_qr_orthogonality_frobenius"):
            x = value[key]
            if not isinstance(x, (int, float)) or not np.isfinite(x) or not 0 <= x <= QR_TOLERANCE:
                raise ValueError("Saved QR numerical gate is invalid")
            setattr(result, key, float(x))
        result.n_rows, result.blocks, result._r = n, blocks, factor.copy()
        return result
