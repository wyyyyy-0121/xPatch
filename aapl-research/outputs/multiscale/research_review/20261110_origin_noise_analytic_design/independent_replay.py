"""Independent fsum/inverse/incidence reconstruction. Does not import analytic_core."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import time
import traceback

# Shared helpers are limited to execution identity, resource guard, and JSON I/O.
# All data generation, operators, covariance, risk, gates, and comparisons below
# are separate from the main numerical implementation.
from runtime_common import (atomic_json, check_resources, install_file_guard, load_protocol,
                            memory_limit, peak_rss_bytes, read_json, require, set_environment)

RULES = ("zero", "constant", "half_constant", "OLS", "Ridge", "half_Ridge")
METRICS = ("bias_total", "bias_level", "bias_dynamic", "variance_total", "variance_level",
           "variance_dynamic", "risk_total", "risk_level", "risk_dynamic")


def reconstruct(np, L, H, check):
    import hashlib
    generator = np.random.Generator(np.random.PCG64(20261110 + L))
    g = generator.standard_normal(size=(6656, L - 1))
    raw = np.empty((6656, L), dtype=np.float64)
    raw[:, L - 1] = 0.0
    # The same prescribed cumsum/RNG backend is deliberately disclosed.
    raw[:, :L - 1] = -np.cumsum(g[:, ::-1], axis=1)[:, ::-1] / math.sqrt(L)
    del g
    train = raw[:2048]
    count = train.size
    xm = math.fsum(float(q) for q in train.flat) / count
    xs2 = math.fsum(float(q) * float(q) for q in train.flat) / count - xm * xm
    require(math.isfinite(xs2) and xs2 > 1e-24, "Independent history variance failed")
    xs = math.sqrt(xs2)
    visible = ((raw - xm) / xs).astype(np.float32)
    del train, raw
    require(np.isfinite(visible).all(), "Independent visible history nonfinite")
    signal_z = math.sqrt(L) * xs * (visible[:, -1].astype(np.float64) - visible[:, -2].astype(np.float64))
    zm = math.fsum(float(q) for q in signal_z[:2048]) / 2048
    zs2 = math.fsum((float(q) - zm) ** 2 for q in signal_z[:2048]) / 2048
    require(math.isfinite(zs2) and zs2 > 1e-24, "Independent signal variance failed")
    zs = math.sqrt(zs2)
    uf = (signal_z[:2048] - zm) / zs
    ut = (signal_z[2560:] - zm) / zs
    check()
    hashes = {"x_sha256": hashlib.sha256(np.asarray(visible, dtype="<f4").tobytes()).hexdigest(),
              "z_sha256": hashlib.sha256(np.asarray(signal_z, dtype="<f8").tobytes()).hexdigest()}
    return uf, ut, signal_z[:2048], signal_z[2560:], {"x_mean": xm, "x_std": xs,
            "z_fit_mean": zm, "z_fit_std": zs, **hashes}


def rules(np, u, compare):
    n = len(u)
    b = math.fsum(float(v) for v in u) / n
    c = math.fsum(float(v) * float(v) for v in u) / n
    gram = np.array([[1.0, b], [b, c]], dtype=np.float64)
    require(float(np.linalg.cond(gram)) <= 1e8, "Independent Gram condition failed")
    operators = {}
    for name, penalty in (("OLS", 0.0), ("Ridge", 0.1)):
        determinant = (c + penalty) - b * b
        require(math.isfinite(determinant) and determinant > 0, "Independent inverse determinant failed")
        a0 = ((c + penalty) - b * u) / (n * determinant)
        a1 = (u - b) / (n * determinant)
        op = np.stack((a0, a1))
        matrix = gram + np.diag([0.0, penalty])
        right = np.stack((np.ones(n), u)) / n
        numerator = float(np.linalg.norm(matrix @ op - right, "fro"))
        denominator = float(np.linalg.norm(matrix, 2) * np.linalg.norm(op, "fro") + np.linalg.norm(right, "fro"))
        require(denominator > 0 and math.isfinite(denominator) and numerator / denominator <= 1e-11,
                "Independent inverse stationarity failed")
        operators[name] = op
    constant = np.stack((np.full(n, 1.0 / n), np.zeros(n)))
    operators.update(zero=np.zeros((2, n)), constant=constant, half_constant=constant / 2,
                     half_Ridge=operators["Ridge"] / 2)
    return operators


def covariance(np, op, h, arm):
    if arm == "I":
        response = op * math.sqrt(h)
    else:
        # Innovation j receives a_i iff i <= j < i+h. This is the explicit
        # interval-incidence response, not the main lag covariance formula.
        response = np.stack([np.convolve(row, np.ones(h, dtype=np.float64), mode="full") for row in op])
        require(response.shape == (2, op.shape[1] + h - 1), "Independent innovation boundary failed")
    return np.array([[math.fsum(float(a) * float(b) for a, b in zip(response[i], response[j]))
                      for j in range(2)] for i in range(2)], dtype=np.float64)


def risk(np, op, ut, mf, mt, cov):
    beta = [math.fsum(float(a) * float(v) for a, v in zip(row, mf)) for row in op]
    errors = beta[0] + beta[1] * ut - mt
    m = len(ut)
    e_mean = math.fsum(float(x) for x in errors) / m
    t = math.fsum(float(x) for x in ut) / m
    v = math.fsum((float(x) - t) ** 2 for x in ut) / m
    bt = math.fsum(float(x) ** 2 for x in errors) / m
    bl = e_mean ** 2
    bd = math.fsum((float(x) - e_mean) ** 2 for x in errors) / m
    vl = float(cov[0, 0] + 2 * t * cov[0, 1] + t * t * cov[1, 1])
    vd = float(v * cov[1, 1])
    # Direct moment expansion, independently of main trace/matrix form.
    t2 = math.fsum(float(x) ** 2 for x in ut) / m
    vt = float(cov[0, 0] + 2 * t * cov[0, 1] + t2 * cov[1, 1])
    return {"bias_total": bt, "bias_level": bl, "bias_dynamic": bd,
            "variance_total": vt, "variance_level": vl, "variance_dynamic": vd,
            "risk_total": bt + vt, "risk_level": bl + vl, "risk_dynamic": bd + vd}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    set_environment(cpu=26)
    protocol, identity = load_protocol(args.protocol)
    output = Path(protocol["output_root"])
    memory_limit()
    install_file_guard(protocol, args.protocol)
    import numpy as np
    stopped = []
    signal.signal(signal.SIGTERM, lambda number, frame: stopped.append(number))
    signal.signal(signal.SIGUSR1, lambda number, frame: stopped.append(number))
    check = lambda: check_resources(protocol, stopped=bool(stopped), respect_reserve=False, cpu=26)
    began = time.monotonic()
    destination = output / "INDEPENDENT_REPLAY.json"
    require(not destination.exists(), "Independent terminal receipt is immutable")
    replay_checkpoint = output / "INDEPENDENT_CHECKPOINT.json"
    counts = 0
    maximum_absolute_difference = 0.0
    def compare(left, right, label):
        nonlocal counts, maximum_absolute_difference
        a, b = float(left), float(right)
        require(math.isfinite(a) and math.isfinite(b) and abs(a - b) <= 1e-11 + 1e-9 * abs(b),
                "Independent numerical mismatch: " + label)
        counts += 1
        maximum_absolute_difference = max(maximum_absolute_difference, abs(a - b))
    previous = None
    if replay_checkpoint.exists():
        previous = read_json(replay_checkpoint)
        require(previous["protocol_sha256"] == identity and previous["deadline_epoch"] == protocol["deadline_epoch"],
                "Independent resume identity differs")
    records = [] if previous is None else previous["records"]
    done = [] if previous is None else previous["completed_units"]
    replay_identity = {} if previous is None else previous["input_qualifications"]
    try:
        check()
        reference = read_json(output / "main" / "CHECKPOINT.json")
        require(reference["status"] == "completed" and reference["protocol_sha256"] == identity,
                "Main terminal identity not qualified")
        summary = read_json(output / "SUMMARY.json")
        require(len(reference["records"]) == 816, "Main record coverage differs")
        expected_keys = [(L, H, h, rule, arm) for L, H in ((480, 60), (96, 8))
                         for h in range(1, H + 1) for rule in RULES for arm in ("I", "O")]
        expected_units = [f"L{L}_H{H}_h{h}" for L, H in ((480, 60), (96, 8)) for h in range(1, H + 1)]
        key = lambda r: (r["lookback"], r["horizon"], r["h"], r["estimator"], r["arm"])
        require([key(r) for r in reference["records"]] == expected_keys, "Main key/order differs")
        require(done == expected_units[:len(done)] and [key(r) for r in records] == expected_keys[:len(records)]
                and len(records) == 12 * len(done),
                "Independent resume cursor/records mismatch")
        endpoint_results = []
        for L, H in ((480, 60), (96, 8)):
            check()
            uf, ut, zf, zt, iq = reconstruct(np, L, H, check)
            if f"L{L}_H{H}" in replay_identity:
                require(replay_identity[f"L{L}_H{H}"] == iq, "Independent resumed input identity differs")
            replay_identity[f"L{L}_H{H}"] = iq
            main_iq = reference["input_qualifications"][f"L{L}_H{H}"]
            for field in ("x_mean", "x_std", "z_fit_mean", "z_fit_std"):
                compare(iq[field], main_iq[field], f"input {L} {field}")
            # Own hashes are reported, not misrepresented as bitwise guarantees
            # for fsum versus pairwise-sum FP64 intermediate values.
            ops = rules(np, uf, compare)
            for h in range(1, H + 1):
                unit = f"L{L}_H{H}_h{h}"
                if unit in done:
                    continue
                check()
                new = []
                mf, mt = 0.25 * h * zf / math.sqrt(H), 0.25 * h * zt / math.sqrt(H)
                for rule in RULES:
                    covs = {}
                    for arm in ("I", "O"):
                        cov = covariance(np, ops[rule], h, arm)
                        covs[arm] = cov
                        compare(cov[0, 1], cov[1, 0], "covariance symmetry")
                        require(float(np.linalg.eigvalsh(cov)[0]) >= -(1e-11 + 1e-9 * float(np.linalg.norm(cov, 2))),
                                "Independent PSD qualification failed")
                        values = risk(np, ops[rule], ut, mf, mt, cov)
                        require(all(math.isfinite(v) and v >= -(1e-11 + 1e-9 * abs(values["risk_total"]))
                                    for v in values.values()), "Independent negative risk qualification failed")
                        for prefix in ("bias", "variance", "risk"):
                            compare(values[prefix + "_total"], values[prefix + "_level"] + values[prefix + "_dynamic"],
                                    "projection " + prefix)
                        new.append({"lookback": L, "horizon": H, "h": h, "estimator": rule, "arm": arm, **values})
                    if h == 1:
                        for a, b in zip(covs["I"].flat, covs["O"].flat):
                            compare(a, b, "h1 arm equality")
                lookup = {(r["estimator"], r["arm"]): r for r in new}
                for rule in RULES:
                    for field in ("bias_total", "bias_level", "bias_dynamic"):
                        compare(lookup[rule, "I"][field], lookup[rule, "O"][field], "bias arm equality")
                for arm in ("I", "O"):
                    for field in ("variance_total", "variance_level", "variance_dynamic"):
                        require(lookup["zero", arm][field] == 0.0, "Independent zero variance failed")
                        for half, full in (("half_constant", "constant"), ("half_Ridge", "Ridge")):
                            compare(lookup[half, arm][field], 0.25 * lookup[full, arm][field], "half variance")
                    for rule in ("constant", "half_constant"):
                        compare(lookup[rule, arm]["variance_dynamic"], 0.0, "constant dynamic variance")
                offset = len(records)
                for index, row in enumerate(new):
                    ref = reference["records"][offset + index]
                    require(key(row) == key(ref), "Independent record key mismatch")
                    for field in METRICS:
                        compare(row[field], ref[field], f"row{offset + index} {field}")
                records.extend(new)
                done.append(unit)
                atomic_json(replay_checkpoint, {"protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                            "completed_units": done, "records": records, "input_qualifications": replay_identity,
                            "pid": os.getpid(), "status": "running"})
            rows = {(r["estimator"], r["arm"]): r for r in records if r["lookback"] == L and r["h"] == H}
            zero, ri, ro = rows["zero", "I"], rows["Ridge", "I"], rows["Ridge", "O"]
            require(zero["risk_dynamic"] > 1e-12, "Independent signal denominator failed")
            controls = {name + "_half_zero": bool(rows[name, "I"]["risk_total"] <= zero["risk_total"] / 2)
                        for name in ("OLS", "Ridge")}
            dt, dl, dd = (ro["risk_" + field] - ri["risk_" + field] for field in ("total", "level", "dynamic"))
            eta = 1e-10 * (ro["risk_total"] + ri["risk_total"] + zero["risk_total"] + 1)
            gap = dd - 0.01 * zero["risk_dynamic"]
            scientific = {"total_cost_positive": bool(dt > eta), "dynamic_cost_positive": bool(dd > eta),
                          "dynamic_cost_at_least_one_percent_signal": bool(gap > eta)}
            own = {"lookback": L, "horizon": H, "zero_risk_total": zero["risk_total"],
                   "zero_risk_dynamic": zero["risk_dynamic"], "ridge_I_risk_total": ri["risk_total"],
                   "ridge_O_risk_total": ro["risk_total"], "ridge_I_risk_dynamic": ri["risk_dynamic"],
                   "ridge_O_risk_dynamic": ro["risk_dynamic"], "delta_total": dt, "delta_level": dl,
                   "delta_dynamic": dd, "eta": eta, "dynamic_materiality_gap": gap,
                   "control_checks": controls, "scientific_checks": scientific,
                   "control_qualified": all(controls.values()), "scientific_conjunction": all(scientific.values())}
            compare(dt, dl + dd, "delta projection")
            main_endpoint = next(r for r in summary["endpoints"] if r["lookback"] == L)
            for k, value in own.items():
                if isinstance(value, (dict, bool)):
                    require(main_endpoint[k] == value, "Independent gate disagrees: " + k)
                else:
                    compare(value, main_endpoint[k], "endpoint " + k)
            endpoint_results.append(own)
            del uf, ut, zf, zt, ops
        require([key(r) for r in records] == expected_keys, "Independent full coverage differs")
        for main_path in summary["paths"]:
            group = [r for r in records if all(r[k] == main_path[k] for k in ("lookback", "horizon", "estimator", "arm"))]
            require(len(group) == main_path["horizon"], "Independent path coverage failed")
            for field in METRICS:
                compare(math.fsum(r[field] for r in group) / len(group), main_path[field], "path " + field)
        contrasts = read_json(output / "ALL_HORIZON_CONTRASTS.json")
        require(len(contrasts) == 408, "Independent contrast coverage differs")
        lookup_all = {key(r): r for r in records}
        expected_contrast_keys = [(L, H, h, rule) for L, H in ((480, 60), (96, 8))
                                  for h in range(1, H + 1) for rule in RULES]
        for reference_contrast, ck in zip(contrasts, expected_contrast_keys):
            require(tuple(reference_contrast[k] for k in ("lookback", "horizon", "h", "estimator")) == ck,
                    "Independent contrast key differs")
            for field in ("total", "level", "dynamic"):
                compare(lookup_all[*ck, "O"]["risk_" + field] - lookup_all[*ck, "I"]["risk_" + field],
                        reference_contrast["delta_" + field], "all-horizon contrast " + field)
        decision = {"numerical_qualification_passed": True,
                    "control_qualification_passed": all(e["control_qualified"] for e in endpoint_results),
                    "supports_fixed_artificial_dynamic_estimation_cost": bool(all(e["control_qualified"] and
                        e["scientific_conjunction"] for e in endpoint_results)), "market_candidate_admitted": False,
                    "independent_market_confirmation": False, "research_goal_achieved": False}
        require(decision == summary["decision"], "Independent final decision differs")
        check()
        atomic_json(destination, {"status": "passed", "protocol_sha256": identity,
                    "deadline_epoch": protocol["deadline_epoch"], "records_independently_rebuilt": len(records),
                    "checks": counts, "maximum_absolute_difference": maximum_absolute_difference,
                    "endpoints": endpoint_results, "decision": decision,
                    "input_qualifications": replay_identity, "shared_numpy_rng_and_runtime_helpers": True,
                    "different_moments_inverse_covariance_and_risk_implementation": True,
                    "peak_rss_bytes": peak_rss_bytes(), "elapsed_seconds": time.monotonic() - began}, exclusive=True)
        atomic_json(replay_checkpoint, {"protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                    "status": "completed", "completed_units": done, "records": records,
                    "input_qualifications": replay_identity}, exclusive=False)
        print({"status": "passed", "checks": counts, "records": len(records)}, flush=True)
        return 0
    except BaseException as error:
        atomic_json(output / f"INDEPENDENT_FAILURE_{os.getpid()}.json", {"error": repr(error),
                    "traceback": traceback.format_exc(), "protocol_sha256": identity,
                    "deadline_epoch": protocol["deadline_epoch"], "completed_units": done}, exclusive=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
