"""Pure-array core for the frozen session-alignment representation diagnostic.

No file, network, calendar-source, model-weight or market-data access lives here.
The caller supplies origin-known original relative-close windows, their complete
historical timestamps/session identifiers, and the original eight clock columns.
All fitted arms share those inputs.  The real and fixed mirror summaries differ
only in alignment; any effect includes representation and ridge-penalty geometry.

Streaming recovery must replay the SAME ordered blocks after the saved cursor.
The caller owns atomic serialization, immutable identity, deadline and admission.
"""
from __future__ import annotations

import numpy as np


RIDGE = 0.1
SCALE_FLOOR = 1e-12
ARMS = ("base", "real", "mirror")


def _positive_integer(value, name):
    scalar = np.asarray(value)
    if scalar.ndim != 0 or scalar.dtype.kind not in "iu" or int(scalar) <= 0:
        raise ValueError(name + " must be a positive integer scalar")
    return int(scalar)


def _integer_array(value, name, ndim):
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError(name + " must be an integer array with the required rank")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(name + " exceeds signed int64")
    return np.asarray(array, dtype=np.int64)


def _finite_matrix(value, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(name + " must be a finite two-dimensional matrix")
    return array


def feature_dimensions(lookback):
    """Return JSON-safe dimensions. Common order is x, g, log-gap, clock."""
    lookback = _positive_integer(lookback, "lookback")
    if lookback < 2:
        raise ValueError("lookback must contain at least two closes")
    common = 3 * lookback + 6
    return {"lookback": lookback, "common": common, "total": common + 2,
            "real_index": common, "mirror_index": common + 1}


def columns(lookback):
    """The two summaries are NEVER jointly selected in any fitted arm."""
    dims = feature_dimensions(lookback)
    common = np.arange(dims["common"], dtype=np.int64)
    return {"base": common,
            "real": np.r_[common, dims["real_index"]],
            "mirror": np.r_[common, dims["mirror_index"]]}


def features(x, end_window_ns, session_window, clock, native_ns):
    """Construct common information plus real and fixed mirror summaries.

    x/end_window_ns/session_window have shape (n,L); clock has shape (n,8).
    The chronological window includes the current origin.  No value after that
    supplied origin enters this function. Clock already uses the original New
    York completed-time harmonics/weekday rule; this module does not derive it
    from a future/current realized session length.

    g[:,j-1] = 1{session[:,j] != session[:,j-1]}, j=1,...,L-1.
    mirror reverses g along its L-1 historical increment positions only.
    """
    x = _finite_matrix(x, "x")
    if len(x) == 0:
        raise ValueError("Empty feature batches are not admitted")
    n, lookback = x.shape
    dims = feature_dimensions(lookback)
    if not np.all(np.abs(x[:, -1]) <= SCALE_FLOOR):
        raise ValueError("Original origin-relative history must end at zero")
    ends = _integer_array(end_window_ns, "end_window_ns", 2)
    sessions = np.asarray(session_window)
    if ends.shape != x.shape or sessions.shape != x.shape:
        raise ValueError("Complete historical time/session windows must match x")
    if sessions.dtype.kind not in "iuU":
        raise ValueError("Session identifiers must be integer codes or Unicode dates")
    clock = _finite_matrix(clock, "clock")
    if clock.shape != (n, 8):
        raise ValueError("Exactly eight original origin-clock columns are required")
    native = _positive_integer(native_ns, "native_ns")
    elapsed = np.diff(ends, axis=1)
    if np.any(elapsed <= 0):
        raise ValueError("Historical completed-bar timestamps must increase strictly")
    g = (sessions[:, 1:] != sessions[:, :-1]).astype(np.float64)
    log_gap = np.log(elapsed.astype(np.float64) / native)
    increments = np.diff(x, axis=1)
    real = np.sum(g * increments, axis=1)
    mirror = np.sum(g[:, ::-1] * increments, axis=1)
    result = np.column_stack((x, g, log_gap, clock, real, mirror))
    if result.shape != (n, dims["total"]) or not np.isfinite(result).all():
        raise ValueError("Invalid full feature matrix")
    return result


def empty_stats(p, horizon):
    """Serializable full-matrix moments; no fitted-arm-specific row selection."""
    p = _positive_integer(p, "p")
    horizon = _positive_integer(horizon, "horizon")
    return {"count": 0, "sx": np.zeros(p, dtype=np.float64),
            "sy": np.zeros(horizon, dtype=np.float64),
            "xx": np.zeros((p, p), dtype=np.float64),
            "xy": np.zeros((p, horizon), dtype=np.float64)}


def validate_stats(stats):
    """Validate an in-memory or restored checkpoint without changing its state."""
    needed = {"count", "sx", "sy", "xx", "xy"}
    if not needed.issubset(stats):
        raise ValueError("Incomplete sufficient statistics")
    count = np.asarray(stats["count"])
    if count.ndim != 0 or count.dtype.kind not in "iu" or int(count) < 0:
        raise ValueError("Statistics count must be a nonnegative integer scalar")
    sx, sy, xx, xy = (np.asarray(stats[k]) for k in ("sx", "sy", "xx", "xy"))
    if (sx.ndim != 1 or sy.ndim != 1 or len(sx) == 0 or len(sy) == 0
            or xx.shape != (len(sx), len(sx)) or xy.shape != (len(sx), len(sy))):
        raise ValueError("Statistics dimensions disagree")
    if any(a.dtype != np.dtype("float64") or not np.isfinite(a).all()
           for a in (sx, sy, xx, xy)):
        raise ValueError("Statistics must contain finite float64 arrays")
    return int(count), len(sx), len(sy)


def update_stats(stats, X, Y):
    """Add one fixed ordered block in-place; increment count after all moments.

    This intentionally preserves the familiar raw-moment accumulation order.
    Resume must use the original block boundaries and must not recompute a
    partially accepted block. Atomic acceptance of the whole block is the
    worker's responsibility; mutation here is not a disk transaction.
    """
    count, p, horizon = validate_stats(stats)
    X, Y = _finite_matrix(X, "X"), _finite_matrix(Y, "Y")
    if X.shape != (len(Y), p) or Y.shape[1] != horizon or len(X) == 0:
        raise ValueError("Feature/target block differs from frozen dimensions")
    additions = (X.sum(axis=0), Y.sum(axis=0), X.T @ X, X.T @ Y)
    if not all(np.isfinite(a).all() for a in additions):
        raise ValueError("Nonfinite block moments")
    for key, addition in zip(("sx", "sy", "xx", "xy"), additions):
        stats[key] += addition
    stats["count"] = count + len(X)
    return stats


def _normalized_system(stats, cols):
    n, p, _ = validate_stats(stats)
    if n == 0:
        raise ValueError("Cannot fit empty sufficient statistics")
    cols = _integer_array(cols, "cols", 1)
    if (len(cols) == 0 or np.any(cols < 0) or np.any(cols >= p)
            or not np.all(np.diff(cols) > 0)):
        raise ValueError("Columns must be a nonempty strictly increasing subset")
    mx, my = stats["sx"] / n, stats["sy"] / n
    raw_variance = np.diag(stats["xx"]) / n - mx * mx
    sd = np.sqrt(np.maximum(raw_variance, 0.0))
    sd = np.where(sd > SCALE_FLOOR, sd, 1.0)
    selected_xx = stats["xx"][np.ix_(cols, cols)]
    gram = (selected_xx / n - np.outer(mx[cols], mx[cols])) / sd[cols, None] / sd[None, cols]
    rhs = (stats["xy"][cols] / n - np.outer(mx[cols], my)) / sd[cols, None]
    if not np.isfinite(gram).all() or not np.isfinite(rhs).all():
        raise ValueError("Nonfinite normalized system")
    return n, cols, mx[cols], sd[cols], my, gram, rhs, raw_variance[cols]


def stationarity_diagnostics(gram, rhs, coef, ridge=RIDGE):
    """Report normal-equation residuals; admission thresholds belong to protocol.

    No market-score-based tolerance is set here. The relative residual is the
    normwise backward error ||AB-rhs||inf/(||A||inf||B||inf+||rhs||inf).
    The coefficient gradient of the normalized penalized objective is twice the
    normal-equation residual. There is no target-standard-deviation scaling.
    """
    if float(ridge) != RIDGE:
        raise ValueError("The diagnostic fixes normalized ridge penalty to 0.1")
    gram, rhs, coef = (_finite_matrix(a, name) for a, name in
                        ((gram, "gram"), (rhs, "rhs"), (coef, "coef")))
    if gram.shape != (len(coef), len(coef)) or rhs.shape != coef.shape:
        raise ValueError("Stationarity system shapes disagree")
    system = gram + RIDGE * np.eye(len(coef), dtype=np.float64)
    residual = system @ coef - rhs
    if not np.isfinite(residual).all():
        raise ValueError("Nonfinite normal-equation residual")
    abs_inf = float(np.linalg.norm(residual, ord=np.inf))
    denominator = (float(np.linalg.norm(system, ord=np.inf)) * float(np.linalg.norm(coef, ord=np.inf))
                   + float(np.linalg.norm(rhs, ord=np.inf)))
    if not np.isfinite(abs_inf) or not np.isfinite(denominator):
        raise ValueError("Nonfinite stationarity norm")
    relative = abs_inf / denominator if denominator > 0 else (0.0 if abs_inf == 0 else float("inf"))
    return {"normal_equation_absolute_inf": abs_inf,
            "normal_equation_relative_inf": relative,
            "normal_equation_denominator_inf": denominator,
            "normal_equation_max_abs_entry": float(np.max(np.abs(residual))),
            "objective_gradient_max_abs_entry": float(2.0 * np.max(np.abs(residual))),
            "gram_symmetry_max_abs": float(np.max(np.abs(gram - gram.T))),
            "coefficient_max_abs": float(np.max(np.abs(coef))),
            "normalized_ridge_penalty": RIDGE, "diagnostic_tolerance_applied": False}


def solve(stats, cols, ridge=RIDGE):
    """Fit one predeclared arm from the common full-matrix moments."""
    if float(ridge) != RIDGE:
        raise ValueError("Only the frozen normalized ridge penalty 0.1 is allowed")
    n, cols, mx, sd, my, gram, rhs, raw_variance = _normalized_system(stats, cols)
    coef = np.linalg.solve(gram + RIDGE * np.eye(len(cols), dtype=np.float64), rhs)
    if not np.isfinite(coef).all():
        raise ValueError("Nonfinite Ridge coefficient")
    diagnostics = stationarity_diagnostics(gram, rhs, coef, ridge)
    diagnostics.update({"fit_count": n, "selected_feature_count": len(cols),
                        "minimum_raw_column_variance": float(np.min(raw_variance)),
                        "columns_using_unit_scale": int(np.sum(np.sqrt(np.maximum(raw_variance, 0.0)) <= SCALE_FLOOR))})
    return {"coef": coef, "x_mean": mx, "x_std": sd, "y_mean": my,
            "columns": cols, "diagnostics": diagnostics}


def predict(model, X):
    """Predict the original full cumulative-return path without shrinkage."""
    X = _finite_matrix(X, "X")
    cols = _integer_array(model["columns"], "model columns", 1)
    coef = _finite_matrix(model["coef"], "model coef")
    mx, sd, my = (np.asarray(model[name], dtype=np.float64)
                  for name in ("x_mean", "x_std", "y_mean"))
    if (len(cols) == 0 or np.any(cols < 0) or np.any(cols >= X.shape[1])
            or not np.all(np.diff(cols) > 0) or coef.shape[0] != len(cols)
            or mx.shape != (len(cols),) or sd.shape != mx.shape or my.shape != (coef.shape[1],)
            or not all(np.isfinite(a).all() for a in (mx, sd, my)) or np.any(sd <= 0)):
        raise ValueError("Model/input dimensions or scaling disagree")
    prediction = my + ((X[:, cols] - mx) / sd) @ coef
    if not np.isfinite(prediction).all():
        raise ValueError("Nonfinite Ridge prediction")
    return prediction


def partition_fit(fit, end_ns, horizon):
    """Original fit 80/20 split, followed only by the strict prefix label purge.

    The boundary is fixed on the ORIGINAL fit before any other decision. No
    availability mask, feature value, score, stop/outer origin or target value
    can select rows. Returns (purged_prefix, unchanged_tail, boundary_index).
    """
    fit = _integer_array(fit, "fit", 1)
    ends = _integer_array(end_ns, "end_ns", 1)
    horizon = _positive_integer(horizon, "horizon")
    if (len(fit) < 2 or np.any(fit < 0) or np.any(fit + horizon >= len(ends))
            or not np.all(np.diff(fit) > 0) or not np.all(np.diff(ends) > 0)):
        raise ValueError("Invalid original fit/calendar sequence")
    boundary = 4 * len(fit) // 5
    first_tail = int(fit[boundary])
    prefix, tail = fit[:boundary], fit[boundary:].copy()
    prefix = prefix[ends[prefix + horizon] < ends[first_tail]].copy()
    if len(prefix) == 0 or len(tail) == 0:
        raise ValueError("Strictly purged internal fit partition is empty")
    return prefix, tail, boundary
