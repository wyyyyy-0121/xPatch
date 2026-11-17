"""Fixed-design analytic risk arithmetic. Importing this module runs no arrays.

NumPy is passed by the guarded worker. No labels/noise are sampled, and no
training, inference, market loader, or independent verifier is called here.
"""
from __future__ import annotations

import hashlib
import math

SHAPES = ((480, 60), (96, 8))
ESTIMATORS = ("zero", "constant", "half_constant", "OLS", "Ridge", "half_Ridge")
ARMS = ("I", "O")
FIELDS = ("bias_total", "bias_level", "bias_dynamic", "variance_total", "variance_level",
          "variance_dynamic", "risk_total", "risk_level", "risk_dynamic")
RTOL, ATOL = 1e-9, 1e-11


def require(value, message):
    if not value:
        raise ValueError(message)


def near(np, left, right, message):
    require(np.isfinite(left).all() and np.isfinite(right).all() and
            np.allclose(left, right, rtol=RTOL, atol=ATOL), message)


def array_hash(np, value, dtype):
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def construct_inputs(np, lookback, horizon, guard):
    require((lookback, horizon) in SHAPES, "Unfrozen shape")
    guard()
    rng = np.random.Generator(np.random.PCG64(20261110 + lookback))
    increments = rng.standard_normal((6656, lookback - 1))
    raw = np.zeros((6656, lookback), dtype=np.float64)
    raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(lookback)
    del increments
    fit = raw[:2048]
    x_mean = float(fit.sum(dtype=np.float64) / fit.size)
    x_variance = float(np.square(fit).sum(dtype=np.float64) / fit.size - x_mean * x_mean)
    require(math.isfinite(x_variance) and x_variance > 1e-24, "Input scalar variance unqualified")
    x_std = math.sqrt(x_variance)
    x = ((raw - x_mean) / x_std).astype(np.float32)
    del fit, raw
    require(np.isfinite(x).all(), "Visible FP32 input nonfinite")
    z = math.sqrt(lookback) * x_std * (x[:, -1].astype(np.float64) - x[:, -2].astype(np.float64))
    require(z.shape == (6656,) and np.isfinite(z).all(), "Invalid visible signal")
    z_mean = float(z[:2048].sum(dtype=np.float64) / 2048)
    z_variance = float(np.square(z[:2048] - z_mean).sum(dtype=np.float64) / 2048)
    require(math.isfinite(z_variance) and z_variance > 1e-24, "Fit z standard deviation unqualified")
    z_std = math.sqrt(z_variance)
    u = (z - z_mean) / z_std
    X = np.column_stack((np.ones(2048, dtype=np.float64), u[:2048]))
    Z = np.column_stack((np.ones(4096, dtype=np.float64), u[2560:]))
    require(X.shape == (2048, 2) and Z.shape == (4096, 2) and np.isfinite(X).all() and
            np.isfinite(Z).all(), "Two-column design malformed")
    qualification = {"lookback": lookback, "horizon": horizon, "x_mean": x_mean, "x_std": x_std,
                     "z_fit_mean": z_mean, "z_fit_std": z_std,
                     "x_sha256": array_hash(np, x, "<f4"), "z_sha256": array_hash(np, z, "<f8"),
                     "X_sha256": array_hash(np, X, "<f8"), "Z_sha256": array_hash(np, Z, "<f8"),
                     "x_shape": [6656, lookback], "z_shape": [6656],
                     "X_shape": [2048, 2], "Z_shape": [4096, 2],
                     "x_dtype": "<f4", "z_X_Z_dtype": "<f8", "hash_rule": "C_order_raw_bytes_no_header"}
    guard()
    return X, Z, z[:2048].copy(), z[2560:].copy(), qualification


def operator_set(np, X):
    n = len(X)
    gram = X.T @ X / n
    condition = float(np.linalg.cond(gram))
    require(math.isfinite(condition) and condition <= 1e8, "Design Gram condition failed")
    right = X.T / n
    output, solutions = {}, {}
    for name, penalty in (("OLS", 0.0), ("Ridge", 0.1)):
        matrix = gram + np.diag(np.asarray([0.0, penalty]))
        coefficient = np.linalg.solve(matrix, right)
        denominator = float(np.linalg.norm(matrix, 2) * np.linalg.norm(coefficient, "fro") +
                            np.linalg.norm(right, "fro"))
        residual = float(np.linalg.norm(matrix @ coefficient - right, "fro"))
        require(math.isfinite(denominator) and denominator > 0 and
                math.isfinite(residual) and residual / denominator <= 1e-11,
                "Frozen linear-operator stationarity failed")
        require(coefficient.shape == (2, n) and np.isfinite(coefficient).all(), "Invalid coefficient operator")
        output[name] = coefficient
        solutions[name] = {"penalty": penalty, "stationarity_relative_residual": residual / denominator}
    constant = np.zeros((2, n), dtype=np.float64)
    constant[0, :] = 1.0 / n
    output.update(zero=np.zeros((2, n), dtype=np.float64), constant=constant,
                  half_constant=0.5 * constant, half_Ridge=0.5 * output["Ridge"])
    require(set(output) == set(ESTIMATORS), "Incomplete operator set")
    return output, {"gram": gram.tolist(), "gram_condition": condition, "solutions": solutions}


def coefficient_covariance(np, operator, h, arm):
    require(arm in ARMS and h >= 1, "Invalid covariance request")
    value = h * (operator @ operator.T)
    if arm == "O":
        n = operator.shape[1]
        for lag in range(1, min(h, n)):
            cross = operator[:, :n - lag] @ operator[:, lag:].T
            value = value + (h - lag) * (cross + cross.T)
    require(np.isfinite(value).all(), "Coefficient covariance nonfinite")
    near(np, value, value.T, "Coefficient covariance symmetry failed")
    minimum = float(np.linalg.eigvalsh(value)[0])
    norm = float(np.linalg.norm(value, 2))
    require(minimum >= -(ATOL + RTOL * norm), "Coefficient covariance PSD failed")
    return value


def risk_components(np, operator, Z, mu_fit, mu_test, covariance):
    mean_coefficient = operator @ mu_fit
    bias_vector = Z @ mean_coefficient - mu_test
    mean_bias = float(bias_vector.mean(dtype=np.float64))
    centered_bias = bias_vector - mean_bias
    Z_mean = Z.mean(axis=0, dtype=np.float64)
    Z_centered = Z - Z_mean
    moment = Z.T @ Z / len(Z)
    dynamic_moment = Z_centered.T @ Z_centered / len(Z)
    bias_total = float(np.square(bias_vector).mean(dtype=np.float64))
    bias_level = mean_bias * mean_bias
    bias_dynamic = float(np.square(centered_bias).mean(dtype=np.float64))
    variance_total = float(np.trace(moment @ covariance))
    variance_level = float(Z_mean @ covariance @ Z_mean)
    variance_dynamic = float(np.trace(dynamic_moment @ covariance))
    values = {"bias_total": bias_total, "bias_level": bias_level, "bias_dynamic": bias_dynamic,
              "variance_total": variance_total, "variance_level": variance_level, "variance_dynamic": variance_dynamic,
              "risk_total": bias_total + variance_total, "risk_level": bias_level + variance_level,
              "risk_dynamic": bias_dynamic + variance_dynamic}
    require(all(math.isfinite(v) for v in values.values()), "Risk record nonfinite")
    margin = ATOL + RTOL * abs(values["risk_total"])
    require(all(v >= -margin for v in values.values()), "Risk or variance is negative beyond frozen tolerance")
    # No clipping: small negative raw values, if present, remain in the record.
    near(np, values["bias_total"], bias_level + bias_dynamic, "Bias projection identity failed")
    near(np, variance_total, variance_level + variance_dynamic, "Variance projection identity failed")
    near(np, values["risk_total"], values["risk_level"] + values["risk_dynamic"], "Risk projection identity failed")
    return values


def horizon_records(np, lookback, horizon, h, X, Z, z_fit, z_test, operators):
    mu_fit = 0.25 * h * z_fit / math.sqrt(horizon)
    mu_test = 0.25 * h * z_test / math.sqrt(horizon)
    rows, covariances = [], {}
    for estimator in ESTIMATORS:
        for arm in ARMS:
            covariance = coefficient_covariance(np, operators[estimator], h, arm)
            covariances[(estimator, arm)] = covariance
            components = risk_components(np, operators[estimator], Z, mu_fit, mu_test, covariance)
            eigen_minimum = float(np.linalg.eigvalsh(covariance)[0])
            rows.append({"lookback": lookback, "horizon": horizon, "h": h,
                         "estimator": estimator, "arm": arm, **components,
                         "small_negative_components": [k for k, v in components.items() if v < 0],
                         "covariance_minimum_eigenvalue": eigen_minimum,
                         "small_negative_covariance_eigenvalue": bool(eigen_minimum < 0)})
    lookup = {(row["estimator"], row["arm"]): row for row in rows}
    for estimator in ESTIMATORS:
        for field in ("bias_total", "bias_level", "bias_dynamic"):
            near(np, lookup[estimator, "I"][field], lookup[estimator, "O"][field], "Arm-invariant bias failed")
    for arm in ARMS:
        require(all(lookup["zero", arm][field] == 0 for field in
                    ("variance_total", "variance_level", "variance_dynamic")), "Zero variance not exactly zero")
        for small, full in (("half_constant", "constant"), ("half_Ridge", "Ridge")):
            near(np, covariances[small, arm], 0.25 * covariances[full, arm], "Half shrink covariance identity failed")
    if h == 1:
        for estimator in ESTIMATORS:
            near(np, covariances[estimator, "I"], covariances[estimator, "O"], "h1 arm covariance differs")
    for estimator in ("constant", "half_constant"):
        near(np, lookup[estimator, "O"]["variance_dynamic"], 0.0, "Constant dynamic variance nonzero")
        near(np, lookup[estimator, "I"]["variance_dynamic"], 0.0, "Constant dynamic variance nonzero")
    if h == horizon:
        require(lookup["zero", "I"]["risk_dynamic"] > 1e-12, "Test endpoint dynamic signal unqualified")
    return rows


def expected_unit_keys(mode):
    return [f"L{L}_H{H}_h{h}" for L, H in (SHAPES if mode == "analytic" else ((7, 4),))
            for h in range(1, H + 1)]


def expected_record_keys(mode):
    return [(L, H, h, estimator, arm) for L, H in (SHAPES if mode == "analytic" else ((7, 4),))
            for h in range(1, H + 1) for estimator in ESTIMATORS for arm in ARMS]


def record_key(row):
    return (row["lookback"], row["horizon"], row["h"], row["estimator"], row["arm"])


def summarize(np, records):
    require([record_key(row) for row in records] == expected_record_keys("analytic"), "816-row coverage/order failed")
    paths, endpoints = [], []
    for L, H in SHAPES:
        shape_rows = [row for row in records if (row["lookback"], row["horizon"]) == (L, H)]
        for estimator in ESTIMATORS:
            for arm in ARMS:
                selected = [row for row in shape_rows if row["estimator"] == estimator and row["arm"] == arm]
                require(len(selected) == H, "Path summary missing horizons")
                paths.append({"lookback": L, "horizon": H, "estimator": estimator, "arm": arm,
                              **{field: float(np.asarray([row[field] for row in selected], dtype=np.float64).mean())
                                 for field in FIELDS}})
        last = {(row["estimator"], row["arm"]): row for row in shape_rows if row["h"] == H}
        zero, first, second = last["zero", "I"], last["Ridge", "I"], last["Ridge", "O"]
        controls = {name + "_half_zero": bool(last[name, "I"]["risk_total"] <= 0.5 * zero["risk_total"])
                    for name in ("OLS", "Ridge")}
        total = second["risk_total"] - first["risk_total"]
        level = second["risk_level"] - first["risk_level"]
        dynamic = second["risk_dynamic"] - first["risk_dynamic"]
        eta = 1e-10 * (second["risk_total"] + first["risk_total"] + zero["risk_total"] + 1.0)
        gap = dynamic - 0.01 * zero["risk_dynamic"]
        science = {"total_cost_positive": bool(total > eta), "dynamic_cost_positive": bool(dynamic > eta),
                   "dynamic_cost_at_least_one_percent_signal": bool(gap > eta)}
        near(np, total, level + dynamic, "Risk-difference projection identity failed")
        endpoints.append({"lookback": L, "horizon": H, "zero_risk_total": zero["risk_total"],
                          "zero_risk_dynamic": zero["risk_dynamic"], "ridge_I_risk_total": first["risk_total"],
                          "ridge_O_risk_total": second["risk_total"],
                          "ridge_I_risk_dynamic": first["risk_dynamic"], "ridge_O_risk_dynamic": second["risk_dynamic"],
                          "delta_total": total, "delta_level": level, "delta_dynamic": dynamic, "eta": eta,
                          "dynamic_materiality_gap": gap, "control_checks": controls, "scientific_checks": science,
                          "control_qualified": all(controls.values()), "scientific_conjunction": all(science.values())})
    qualified = all(row["control_qualified"] for row in endpoints)
    return paths, endpoints, {"numerical_qualification_passed": True, "control_qualification_passed": qualified,
                             "supports_fixed_artificial_dynamic_estimation_cost": bool(qualified and all(
                                 row["scientific_conjunction"] for row in endpoints)),
                             "market_candidate_admitted": False, "independent_market_confirmation": False,
                             "research_goal_achieved": False}


def toy_inputs(np):
    z_fit = np.asarray([-3., -2., -1., 0., 1., 2., 3.], dtype=np.float64)
    z_test = np.asarray([-2.5, -0.5, 0.5, 2.5], dtype=np.float64)
    X = np.column_stack((np.ones(7), z_fit / 2.0))
    Z = np.column_stack((np.ones(4), z_test / 2.0))
    return X, Z, z_fit, z_test


def structural_checks(np):
    n, H = 7, 4
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    incidence = {}
    checks = 0
    for h in range(1, H + 1):
        matrix = np.zeros((n, n + H - 1), dtype=np.float64)
        for origin in range(n):
            matrix[origin, origin:origin + h] = 1.0
        incidence[h] = matrix
    for h in range(1, H + 1):
        for k in range(1, H + 1):
            actual = incidence[h] @ incidence[k].T
            formula = np.maximum(0, np.minimum(i + h, j + k) - np.maximum(i, j)).astype(np.float64)
            require(np.array_equal(actual, formula), "Toy cross-origin/cross-horizon covariance failed")
            require(np.array_equal(np.diag(actual), np.full(n, min(h, k))), "Toy same-row marginal failed")
            require(np.array_equal(actual.T, incidence[k] @ incidence[h].T), "Toy cross covariance transpose failed")
            independent_row_covariance = np.eye(n, dtype=np.float64) * min(h, k)
            require(np.array_equal(np.diag(independent_row_covariance), np.full(n, min(h, k))) and
                    np.count_nonzero(independent_row_covariance - np.diag(np.diag(independent_row_covariance))) == 0,
                    "Toy I-arm full h/k row independence failed")
            checks += 4
    require(np.array_equal(incidence[1] @ incidence[1].T, np.eye(n)), "h1 two arms differ")
    contrast = np.asarray([1., -1., 0., 0., 0., 0., 0.])
    difference = incidence[2] @ incidence[2].T - 2 * np.eye(n)
    negative = float(contrast @ difference @ contrast)
    require(negative == -2.0, "Frozen non-PSD difference contrast failed")
    X, Z, z_fit, z_test = toy_inputs(np)
    operators, opcheck = operator_set(np, X)
    for estimator in ESTIMATORS:
        for h in range(1, H + 1):
            lag = coefficient_covariance(np, operators[estimator], h, "O")
            response = operators[estimator] @ incidence[h]
            near(np, lag, response @ response.T, "Toy lag/incidence coefficient covariance differs")
            checks += 1
    return {"status": "passed_small_artificial_structure_only", "exact_or_tolerance_checks": checks + 2,
            "negative_difference_quadratic_form": negative, "n": n, "H": H,
            "X_sha256": array_hash(np, X, "<f8"), "Z_sha256": array_hash(np, Z, "<f8"),
            "z_fit_sha256": array_hash(np, z_fit, "<f8"), "z_test_sha256": array_hash(np, z_test, "<f8"),
            "operator_qualification": opcheck}
