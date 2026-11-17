"""Temporary affine LS diagnostics with explicit numerical-sensitivity estimates.

No data/model I/O, optimizer, regularization, coefficient serialization or
candidate admission lives here. The perturbation margins are conservative
first-order ESTIMATES, not certified intervals or statistical uncertainty.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


EPS64 = np.finfo(np.float64).eps
NUMERIC_TOLERANCE = 1e-10
PERTURBATION_SAFETY = 32.0
MAX_PERTURBATION_TO_SIGMA_MIN = 0.1
ESTIMATE_KIND = "conservative_first_order_not_certified_interval"


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim}D array")
    return result


def _gamma(dimension: int) -> float:
    z = float(dimension) * EPS64
    if z >= 0.5:
        raise ValueError("Dimension is beyond the frozen rounding-estimate regime")
    return z / (1.0-z)


def _sumsq(array: np.ndarray) -> float:
    result = float(np.sum(array*array, dtype=np.float64))
    if not math.isfinite(result):
        raise ValueError("Squared norm overflow")
    return result


class AffineFit:
    """An in-memory diagnostic solution; summary never exposes coefficients."""

    def __init__(self, coefficient, coefficient_margin, details):
        self._coefficient = coefficient.copy()
        self._coefficient_margin = coefficient_margin.copy()
        self._details = details

    def summary(self) -> dict[str, Any]:
        return {**self._details,
                "singular_values": list(self._details["singular_values"]),
                "numerical_gates": dict(self._details["numerical_gates"]),
                "coefficient_l2_error_margin_by_target": self._coefficient_margin.tolist(),
                "coefficient_serialized": False, "forecasting_candidate_admitted": False}

    def predict(self, Xquery: Any) -> tuple[np.ndarray, dict[str, Any]]:
        query = _array(Xquery, "Xquery", 2)
        if query.shape[1] != self._coefficient.shape[0] or not len(query):
            raise ValueError("Query dimension/row count changed")
        if not np.array_equal(query[:, -1], np.ones(len(query))):
            raise ValueError("Query last column must be exact intercept1")
        prediction = query @ self._coefficient
        arithmetic = _gamma(query.shape[1]) * (np.abs(query) @ np.abs(self._coefficient))
        margin = (np.linalg.norm(query, axis=1)[:, None]
                  * self._coefficient_margin[None, :] + arithmetic)
        if not np.isfinite(prediction).all() or not np.isfinite(margin).all():
            raise ValueError("Prediction or estimated uncertainty overflow")
        return prediction, {
            "absolute_error_margin": margin,
            "max_absolute_error_margin": float(np.max(margin)),
            "max_query_l2_norm": float(np.max(np.linalg.norm(query, axis=1))),
            "solver_qualified": self._details["solver_qualified"],
            "estimate_kind": ESTIMATE_KIND,
            "statistical_confidence_interval": False,
            "certified_numerical_interval": False,
        }


def fit_affine(X: Any, Y: Any) -> AffineFit:
    """Fit full affine class by reduced QR then R-SVD; reject unstable claims.

    X is already the intended design (p=34 learned-feature or p=2 anchor-only),
    including an exact constant-one last column. Inputs/labels are treated as
    exact FP64 values; callers reproduce the original FP32-to-FP64 route.
    A rank-truncated fit is retained descriptively, never solver-qualified.
    """
    x, y = _array(X, "X", 2), _array(Y, "Y", 2)
    n, p = x.shape
    if p not in (2, 34) or n < p or y.shape[0] != n or y.shape[1] < 1:
        raise ValueError("Require n>=p, p in {2,34}, and aligned nonempty targets")
    if not np.array_equal(x[:, -1], np.ones(n)):
        raise ValueError("Design last column must be exact intercept1")
    h = y.shape[1]
    q, r = np.linalg.qr(x, mode="reduced")
    qr_error = float(np.linalg.norm(x-q@r, ord="fro"))
    xnorm_f = float(np.linalg.norm(x, ord="fro"))
    qr_relative = qr_error/max(1.0, xnorm_f)
    q_orth = float(np.linalg.norm(q.T@q-np.eye(p), ord="fro"))
    u, singular, vh = np.linalg.svd(r, full_matrices=False)
    svd_error = float(np.linalg.norm(r-(u*singular)@vh, ord="fro"))
    smax, smin = float(singular[0]), float(singular[-1])
    cutoff = float(EPS64*max(n, p)*smax)
    retained = singular > cutoff
    rank = int(np.count_nonzero(retained))
    ambiguous = (singular >= 0.1*cutoff) & (singular <= 10.0*cutoff)
    c = q.T@y
    coefficient = vh[retained].T @ ((u[:, retained].T@c)/singular[retained, None])
    residual = x@coefficient-y
    normal = x.T@residual
    y_norm = np.linalg.norm(y, axis=0)
    residual_norm = np.linalg.norm(residual, axis=0)
    coefficient_norm = np.linalg.norm(coefficient, axis=0)
    stationarity = float(np.linalg.norm(normal, ord="fro")/max(
        1.0, smax*float(np.linalg.norm(y, ord="fro")),
        smax*float(np.linalg.norm(residual, ord="fro"))))
    # E and F estimates are frozen functions of arithmetic dimension, factor
    # reconstruction, orthogonality and scale, never fitted to a tail loss.
    delta_x = (qr_error+svd_error+smax*q_orth
               + PERTURBATION_SAFETY*_gamma(p)*max(xnorm_f, np.finfo(float).tiny))
    delta_y = PERTURBATION_SAFETY*_gamma(max(n, p, h))*y_norm
    inverse_scale = float(singular[retained][-1]) if rank else float("inf")
    eta = delta_x/inverse_scale if rank else 1.0
    solve_residual = np.linalg.norm(r@coefficient-c, axis=0)
    # These are sensitivity estimates, including a residual-dependent term
    # that grows as sigma_min^-2 in ill-conditioned, inconsistent LS.
    if rank:
        first_order = ((delta_y+delta_x*coefficient_norm+solve_residual)/inverse_scale
                       + delta_x*residual_norm/(inverse_scale*inverse_scale))
        first_order /= max(np.finfo(float).eps, 1.0-min(eta, 1.0))
        normal_sensitivity = np.linalg.norm(normal, axis=0)/(inverse_scale*inverse_scale)
        coefficient_margin = np.maximum(first_order, normal_sensitivity)
    else:
        normal_sensitivity = np.zeros(h)
        coefficient_margin = np.full(h, np.finfo(float).max/1e100)
    if not np.isfinite(coefficient).all() or not np.isfinite(coefficient_margin).all():
        raise ValueError("Coefficient or perturbation estimate overflow; no qualified result")
    reconstruction_scaled = float(np.linalg.norm(r-(u*singular)@vh, ord="fro")/max(1.0, smax))
    svd_orth = max(float(np.linalg.norm(u.T@u-np.eye(p), ord="fro")),
                   float(np.linalg.norm(vh@vh.T-np.eye(p), ord="fro")))
    gates = {
        "full_column_rank": rank == p,
        "no_rank_ambiguity": not bool(np.any(ambiguous)),
        "QR_reconstruction": qr_relative <= NUMERIC_TOLERANCE,
        "QR_orthogonality": q_orth <= NUMERIC_TOLERANCE,
        "SVD_reconstruction": reconstruction_scaled <= NUMERIC_TOLERANCE,
        "SVD_orthogonality": svd_orth <= NUMERIC_TOLERANCE,
        "stationarity": stationarity <= NUMERIC_TOLERANCE,
        "perturbation_stable": eta < MAX_PERTURBATION_TO_SIGMA_MIN,
    }
    gates = {name: bool(value) for name, value in gates.items()}
    qualified = all(gates.values())
    details = {
        "method": "FP64 reduced QR then small-R SVD; no Gram inverse or penalty",
        "n_rows": n, "features": p, "n_targets": h, "rank": rank,
        "singular_values": singular.tolist(), "rank_cutoff": cutoff,
        "rank_ambiguous": bool(np.any(ambiguous)),
        "ambiguous_singular_indices": np.flatnonzero(ambiguous).tolist(),
        "condition_number": smax/smin if smin > 0 else None,
        "QR_reconstruction_relative": qr_relative, "QR_orthogonality": q_orth,
        "SVD_reconstruction_relative": reconstruction_scaled, "SVD_orthogonality": svd_orth,
        "stationarity_normalized": stationarity,
        "residual_SSE": _sumsq(residual), "residual_MSE": _sumsq(residual)/(n*h),
        "residual_relative_to_target": float(np.linalg.norm(residual, ord="fro")
                                               / max(1.0, float(np.linalg.norm(y, ord="fro")))),
        "normal_residual_norm_by_target": np.linalg.norm(normal, axis=0).tolist(),
        "normal_residual_sigma_min_squared_sensitivity_by_target": normal_sensitivity.tolist(),
        "estimated_delta_X_operator_upper_scale": delta_x,
        "estimated_delta_Y_l2_by_target": delta_y.tolist(),
        "estimated_perturbation_to_sigma_min": eta,
        "frozen_max_perturbation_to_sigma_min": MAX_PERTURBATION_TO_SIGMA_MIN,
        "frozen_perturbation_safety_factor": PERTURBATION_SAFETY,
        "estimate_kind": ESTIMATE_KIND,
        "margin_formula": "max(normal_residual/smin^2,[(dY+dX*||B||+||RB-C||)/smin+dX*||res||/smin^2]/(1-eta)); querynorm*coefficientmargin+gamma_p*abs(query)@abs(B)",
        "numerical_gates": gates, "solver_qualified": qualified,
        "full_affine_class_claim_allowed": qualified,
        "certified_numerical_interval": False, "statistical_confidence_interval": False,
        "rank_truncation_requires_exclusion_from_continuation": rank < p or bool(np.any(ambiguous)),
    }
    return AffineFit(coefficient, coefficient_margin, details)


def transform_prediction(prediction: Any, margin: Any, scale: Any, offset: Any = 0.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """Propagate estimated prediction uncertainty through original-unit map."""
    pred, error = _array(prediction, "prediction", 2), _array(margin, "margin", 2)
    if pred.shape != error.shape or np.any(error < 0):
        raise ValueError("Prediction and nonnegative uncertainty dimensions must match")
    factors, offsets = np.asarray(scale, dtype=np.float64), np.asarray(offset, dtype=np.float64)
    if not np.isfinite(factors).all() or not np.isfinite(offsets).all():
        raise ValueError("Unit map must be finite")
    result = pred*factors+offsets
    transformed_margin = np.abs(factors)*error+_gamma(2)*(np.abs(pred*factors)+np.abs(offsets))
    if result.shape != pred.shape or transformed_margin.shape != pred.shape:
        raise ValueError("Unit map changed prediction dimensions")
    return result, transformed_margin


def loss_summary(prediction: Any, target: Any, margin: Any) -> dict[str, Any]:
    pred, y, error = (_array(prediction, "prediction", 2), _array(target, "target", 2),
                      _array(margin, "margin", 2))
    if pred.shape != y.shape or pred.shape != error.shape or not pred.size or np.any(error < 0):
        raise ValueError("Loss arrays must align and have nonnegative estimated uncertainty")
    residual = pred-y
    # Additional uncertainty for residual subtraction and loss summation.
    residual_round = _gamma(1)*(np.abs(pred)+np.abs(y))
    total_margin = error+residual_round
    sse = _sumsq(residual)
    rounding = _gamma(pred.size+2)*sse
    uncertainty = float(np.sum(2*np.abs(residual)*total_margin+total_margin**2))+rounding
    if not math.isfinite(uncertainty):
        raise ValueError("Estimated loss uncertainty overflow")
    return {"SSE": sse, "MSE": sse/pred.size, "elements": pred.size,
            "estimated_SSE_error_margin": uncertainty,
            "estimated_MSE_error_margin": uncertainty/pred.size,
            "estimate_kind": ESTIMATE_KIND, "certified_numerical_interval": False,
            "statistical_confidence_interval": False}


def compare_losses(proposed: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    if proposed["elements"] != control["elements"]:
        raise ValueError("Paired losses must have the same number of target elements")
    difference = control["MSE"]-proposed["MSE"]
    margin = control["estimated_MSE_error_margin"]+proposed["estimated_MSE_error_margin"]
    return {"control_minus_proposed_MSE": difference,
            "combined_estimated_MSE_error_margin": margin,
            "numerically_separated_under_frozen_estimate": bool(difference > margin),
            "proposed_worse_under_frozen_estimate": bool(difference < -margin),
            "estimate_kind": ESTIMATE_KIND, "certified_exact_inequality": False,
            "statistical_significance": False}
