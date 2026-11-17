"""Independent calendar/support/risk-bound replay; import performs no numerics.

No main numerical or decision implementation is imported. Shared helpers are
limited to CPU/thread setup, protocol identity, deadlines, durable JSON I/O,
and environment/dependency qualification. No shared calendar or risk function.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from runtime import BASE, setup, read, write, sha, load_protocol, guard, qualify_dependencies

STATES = ("minute_f1", "minute_f2", "hourly_f1", "hourly_f2")
CONTROLS = ("zero", "prefix_update_mean", "same_information_Ridge")
ARCHIVE_KEYS = ("origins", "labels", "D", "Z", "C", "zero",
                "prefix_update_mean", "same_information_Ridge")
PREDICTION_READS = ("origins", "labels", "zero", "prefix_update_mean",
                    "same_information_Ridge")
BAR_READS = ("start_ns", "end_ns", "observed", "session_date")
MINUTE_NS = 60_000_000_000


def independent_grid(np, xc, calendar_spec, check):
    """Build every planned bar by session, independently of observed timestamps."""
    check(calendar_spec["name"] == "XNYS" and
          calendar_spec["start"] == "2021-01-04" and
          calendar_spec["end"] == "2026-06-30", "Fixed calendar boundary differs")
    calendar = xc.get_calendar("XNYS", start=calendar_spec["start"],
                               end=calendar_spec["end"])
    blocks = {profile: {key: [] for key in
              ("start_ns", "end_ns", "session_date", "open_ns", "close_ns")}
              for profile in ("minute", "hourly")}
    for session in calendar.sessions_in_range(calendar_spec["start"], calendar_spec["end"]):
        label = session.strftime("%Y-%m-%d")
        opening = int(calendar.session_open(session).value)
        closing = int(calendar.session_close(session).value)
        duration = closing - opening
        check(duration > 0 and duration % MINUTE_NS == 0, "Nonintegral planned session")
        for profile, step in (("minute", MINUTE_NS), ("hourly", 60 * MINUTE_NS)):
            count = duration // step
            # Full groups only. An hourly residual never becomes a bar.
            starts = opening + np.arange(count, dtype=np.int64) * step
            blocks[profile]["start_ns"].append(starts)
            blocks[profile]["end_ns"].append(starts + step)
            blocks[profile]["session_date"].append(np.full(count, label, dtype="U10"))
            blocks[profile]["open_ns"].append(np.full(count, opening, dtype=np.int64))
            blocks[profile]["close_ns"].append(np.full(count, closing, dtype=np.int64))
    return {profile: {key: np.concatenate(parts) for key, parts in values.items()}
            for profile, values in blocks.items()}


def independent_columns(y, means, ridge, support, ensure):
    """Columnwise scalar fsum; S contributes zero oracle error, never a predictor."""
    n, horizon = y.shape
    eps, tiny = 2.0 ** -52, float.fromhex("0x1.0000000000000p-1022")
    records = []
    for h in range(horizon):
        b = float(means[0, h])
        yc = y[:, h]
        rc = ridge[:, h]
        sc = support[:, h]
        b_losses = [(float(value) - b) ** 2 for value in yc]
        zero_risk = math.fsum(float(value) ** 2 for value in yc) / n
        b_risk = math.fsum(b_losses) / n
        ridge_risk = math.fsum((float(value) - float(pred)) ** 2
                              for value, pred in zip(yc, rc)) / n
        inside = math.fsum(loss for loss, selected in zip(b_losses, sc) if bool(selected)) / n
        floor = math.fsum(loss for loss, selected in zip(b_losses, sc) if not bool(selected)) / n
        risks = dict(zip(CONTROLS, (zero_risk, b_risk, ridge_risk)))
        scales = {
            "zero": math.fsum(abs(float(value)) ** 2 for value in yc) / n,
            "prefix_update_mean": math.fsum((abs(float(value)) + abs(b)) ** 2
                                             for value in yc) / n,
            "same_information_Ridge": math.fsum((abs(float(value)) + abs(float(pred))) ** 2
                                                for value, pred in zip(yc, rc)) / n,
        }
        # h in the frozen envelope denotes the full path width, not column j.
        factor = 128 * eps * (n + horizon + 16)
        bounds = {name: factor * max(scale, tiny) for name, scale in scales.items()}
        bounds["floor"] = bounds["prefix_update_mean"]
        tolerance = 256 * eps * max(*risks.values(), tiny)
        residual = b_risk - floor - inside
        ensure(abs(residual) <= tolerance, "Independent b/floor/support identity fails")
        ensure(all(math.isfinite(v) and v > 0 for v in risks.values()),
               "Independent comparator denominator is not finite/positive")
        ensure(math.isfinite(floor) and floor >= 0 and inside >= 0,
               "Independent risk floor invalid")
        raw = {name: 100 * (1 - floor / risk) for name, risk in risks.items()}
        safe = {name: 100 * (1 - max(0.0, floor - bounds["floor"]) /
                            (risk + bounds[name])) for name, risk in risks.items()}
        records.append({"horizon": h + 1, "support_count": int(sum(bool(v) for v in sc)),
                        "support_sha256": hashlib.sha256(sc.astype("uint8").tobytes()).hexdigest(),
                        "b": b, "b_MSE": b_risk, "support_b_MSE": inside,
                        "floor_MSE": floor, "MSE": risks,
                        "risk_error_bounds": bounds,
                        "maximum_improvement_pct": raw,
                        "safe_upper_maximum_improvement_pct": safe,
                        "identity_residual": residual, "identity_tolerance": tolerance})
    metrics = {"floor": {"endpoint_MSE": records[-1]["floor_MSE"],
                           "path_MSE": math.fsum(r["floor_MSE"] for r in records) / horizon}}
    floor_errors = {"endpoint_MSE": records[-1]["risk_error_bounds"]["floor"],
                    "path_MSE": math.fsum(r["risk_error_bounds"]["floor"] for r in records) / horizon}
    metrics["floor"].update(endpoint_error_bound=floor_errors["endpoint_MSE"],
                             path_error_bound=floor_errors["path_MSE"])
    for control in CONTROLS:
        values = {"endpoint_MSE": records[-1]["MSE"][control],
                  "path_MSE": math.fsum(r["MSE"][control] for r in records) / horizon}
        errors = {"endpoint_MSE": records[-1]["risk_error_bounds"][control],
                  "path_MSE": math.fsum(r["risk_error_bounds"][control] for r in records) / horizon}
        for metric in ("endpoint", "path"):
            key = metric + "_MSE"
            values[metric + "_maximum_improvement_pct"] = 100 * (1 - metrics["floor"][key] / values[key])
            values[metric + "_safe_upper_maximum_improvement_pct"] = 100 * (
                1 - max(0.0, metrics["floor"][key] - floor_errors[key]) /
                (values[key] + errors[key]))
        metrics[control] = values
    return records, metrics


def independent_decision(records):
    """Rebuild the necessary subset, without importing a main decision helper."""
    margin = 1e-8

    def classify(value, safe, threshold):
        if safe < threshold - margin:
            return "excluded"
        if value <= threshold + margin:
            return "within_numerical_margin"
        return "not_ruled_out"

    checks = []
    for row in records:
        for control in CONTROLS:
            for metric in ("endpoint", "path"):
                prefix = metric
                raw = row["metrics"][control][prefix + "_maximum_improvement_pct"]
                safe = row["metrics"][control][prefix + "_safe_upper_maximum_improvement_pct"]
                checks.append({"state": row["state"], "control": control, "metric": metric,
                               "maximum_improvement_pct": raw,
                               "safe_upper_maximum_improvement_pct": safe,
                               "threshold_pct": .1, "status": classify(raw, safe, .1)})
    tasks = []
    for prefix in ("minute", "hourly"):
        rows = [r for r in records if r["state"].startswith(prefix + "_")]
        if len(rows) != 2:
            raise RuntimeError("Exactly two original folds per task required")
        raw = math.fsum(r["metrics"]["zero"]["endpoint_maximum_improvement_pct"] for r in rows) / 2
        safe = math.fsum(r["metrics"]["zero"]["endpoint_safe_upper_maximum_improvement_pct"]
                         for r in rows) / 2
        tasks.append({"task": prefix,
                      "equal_fold_endpoint_maximum_improvement_over_zero_pct": raw,
                      "safe_upper_maximum_improvement_pct": safe,
                      "threshold_pct": 1.0, "status": classify(raw, safe, 1.0)})
    flags = [r["status"] for r in checks + tasks]
    return {"necessary_checks": checks, "tasks": tasks,
            "necessary_check_count": len(flags),
            "excluded_necessary_conditions": flags.count("excluded"),
            "fixed_support_class_ruled_out_on_development_gates": "excluded" in flags,
            "model_fitting_admitted": False, "candidate_admitted": False,
            "independent_market_confirmation": False,
            "original_58_gate_recipe_changed": False, "full_research_goal_achieved": False}


def main():
    setup(26)
    started = time.monotonic()
    output = BASE / "run"
    protocol, identity = None, None
    completed, signals = [], []
    checks = 0
    maxima = {"loss_absolute": 0.0, "loss_relative": 0.0,
              "raw_identity_absolute": 0.0, "percentage_points": 0.0}

    def ensure(ok, message):
        nonlocal checks
        if not bool(ok):
            raise RuntimeError(message)
        checks += 1

    def stop(number, frame):
        signals.append({"signal": int(number), "pid": os.getpid(), "epoch": time.time()})

    def tick():
        if signals:
            raise InterruptedError("Independent verifier received actual signal")
        guard(protocol)
        ensure(os.sched_getaffinity(0) == {26}, "Independent affinity changed")

    def close_loss(a, b, label):
        a, b = float(a), float(b)
        scale = max(abs(a), abs(b), float.fromhex("0x1.0000000000000p-1022"))
        delta = abs(a - b)
        ensure(math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0 and
               delta <= 1e-11 * scale, "Independent loss/bound mismatch: " + label)
        maxima["loss_absolute"] = max(maxima["loss_absolute"], delta)
        maxima["loss_relative"] = max(maxima["loss_relative"], delta / scale)

    def close_pp(a, b, label):
        delta = abs(float(a) - float(b))
        ensure(math.isfinite(float(a)) and math.isfinite(float(b)) and delta <= 1e-8,
               "Independent percentage-point mismatch: " + label)
        maxima["percentage_points"] = max(maxima["percentage_points"], delta)

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        protocol = load_protocol()
        identity = sha(BASE / "EXECUTION_PROTOCOL.json")
        spec = read(BASE / "DESIGN_SPEC.json")
        ensure(spec["schema"] == "intraday_opportunity_bound_design_v1", "Wrong design schema")
        ensure([s["state"] for s in spec["states"]] == list(STATES), "Fixed states/order differ")
        ensure(spec["controls_for_gate"] == list(CONTROLS), "Three fixed controls differ")
        ensure(spec["total_necessary_checks"] == 26 and spec["Q3_allowed"] is False and
               spec["GPU"] is False and spec["state_minimum_improvement_pct"] == .1 and
               spec["task_minimum_improvement_pct"] == 1.0 and
               spec["percentage_point_margin"] == 1e-8 and spec["fp64_error_factor"] == 128 and
               spec["independent_loss_rtol"] == 1e-11 and spec["numeric_identity_eps_factor"] == 256 and
               spec["new_market_fitting_allowed"] is False and
               spec["market_inference_allowed"] is False, "Scientific scope differs")
        ensure(spec["prediction_decode_members"] == list(PREDICTION_READS) and
               spec["bar_decode_members"] == list(BAR_READS), "Decode whitelist changed")
        ensure(sorted(spec["prediction_archive_members"]) == sorted(k + ".npy" for k in ARCHIVE_KEYS) and
               set(spec["bar_archive_members"]) == set(BAR_READS) | {"close"},
               "Fixed archive directory whitelist changed")
        ensure(not (output / "INDEPENDENT_REPLAY.json").exists(), "Do not replace terminal verifier receipt")
        done = read(output / "WORKER_DONE.json")
        parent = read(output / "RESULTS.json")
        ensure(done["protocol_sha256"] == identity and done["completed_states"] == 4,
               "Main four-state completion missing")
        ensure(parent["protocol_sha256"] == identity and
               [r["state"] for r in parent["records"]] == list(STATES), "Main result identity differs")
        ensure(parent["market_fits"] == 0 and parent["training_updates"] == 0 and
               parent["model_inference"] == 0 and parent["close_member_opened"] is False and
               parent["Q3_read"] is False and parent["GPU"] is False, "Main scope receipt differs")
        dependency_receipt = qualify_dependencies(spec)
        import numpy as np
        import exchange_calendars as xc
        for package, version in spec["package_versions"].items():
            ensure(importlib.metadata.version(package) == version, "Numerical/calendar dependency changed")
        planned = independent_grid(np, xc, spec["calendar"], ensure)
        tick()
        bars = {}
        decoded = []
        for profile in ("minute", "hourly"):
            entry = spec["bars"][profile]
            path = Path(entry["path"])
            ensure(sha(path) == entry["sha256"], "Bar container identity differs")
            with np.load(path, allow_pickle=False) as archive:
                ensure(len(archive.files) == 5 and set(archive.files) == set(BAR_READS) | {"close"},
                       "Bar members differ")
                # close is only a directory name: its compressed member is never opened.
                value = {key: archive[key] for key in BAR_READS}
            ensure(sha(path) == entry["sha256"], "Bar container changed during replay")
            rows = int(entry["rows"])
            ensure(all(v.shape == (rows,) for v in value.values()), "Bar shapes differ")
            ensure(value["start_ns"].dtype == np.dtype("int64") and
                   value["end_ns"].dtype == np.dtype("int64") and
                   value["observed"].dtype == np.dtype("bool") and
                   value["session_date"].dtype.kind == "U", "Bar dtype differs")
            ensure(all(np.array_equal(value[k], planned[profile][k]) for k in
                       ("start_ns", "end_ns", "session_date")), "Actual grid differs from planned calendar")
            ensure(str(value["session_date"][0]) == "2021-01-04" and
                   str(value["session_date"][-1]) == "2026-06-30", "Bar/Q3 boundary differs")
            bars[profile] = value
            decoded.append({"path": str(path), "members": list(BAR_READS), "close_decoded": False})
        records = []
        for entry in spec["states"]:
            tick()
            state, n, H = entry["state"], int(entry["count"]), int(entry["horizon"])
            profile = "minute" if state.startswith("minute") else "hourly"
            ensure(H == (60 if profile == "minute" else 8), "Primary horizon differs")
            path = Path(entry["path"])
            ensure(sha(path) == entry["sha256"], "Prediction archive identity differs")
            with np.load(path, allow_pickle=False) as archive:
                ensure(len(archive.files) == 8 and set(archive.files) == set(ARCHIVE_KEYS),
                       "Prediction archive member directory differs")
                arrays = {key: archive[key] for key in PREDICTION_READS}
            ensure(sha(path) == entry["sha256"], "Prediction archive changed during replay")
            origins = arrays["origins"]
            ensure(origins.dtype == np.dtype("int64") and origins.shape == (n,) and
                   np.all(np.diff(origins) > 0), "Origin identity/dtype differs")
            digest = hashlib.sha256(np.asarray(origins, dtype="<i8").tobytes()).hexdigest()
            ensure(digest == entry["tail_sha256"] and entry["partition"]["id"] == state and
                   entry["partition"]["tail_count"] == n and
                   entry["partition"]["tail_sha256"] == digest, "Original tail/partition differs")
            for key in PREDICTION_READS[1:]:
                ensure(arrays[key].dtype == np.dtype("float64") and arrays[key].shape == (n, H) and
                       np.isfinite(arrays[key]).all(), "Prediction dtype/shape/finite qualification: " + key)
            ensure(np.all(arrays["zero"] == 0), "zero control differs")
            ensure(np.all(arrays["prefix_update_mean"] == arrays["prefix_update_mean"][0]),
                   "Prefix b is not constant")
            grid, actual = planned[profile], bars[profile]
            lookback = 480 if profile == "minute" else 96
            ensure(origins[0] >= lookback - 1 and origins[-1] + H < len(grid["end_ns"]),
                   "Origin outside planned history/target grid")
            missing = np.r_[0, np.cumsum(~actual["observed"], dtype=np.int64)]
            ensure(np.all(missing[origins + H + 1] == missing[origins - lookback + 1]),
                   "Original full history/target observation identity fails")
            support = np.zeros((n, H), dtype=bool)
            for i, raw in enumerate(origins):
                o = int(raw)
                opening, closing = int(grid["open_ns"][o]), int(grid["close_ns"][o])
                decision = int(grid["end_ns"][o])
                if profile == "minute":
                    ensure(grid["session_date"][o] == grid["session_date"][o + H],
                           "Minute target crosses a planned session")
                allowed = opening + 30 * MINUTE_NS <= decision <= closing - 30 * MINUTE_NS
                for h in range(1, H + 1):
                    support[i, h - 1] = allowed and int(grid["end_ns"][o + h]) >= closing
                if profile == "minute":
                    ensure(not support[i, :-1].any() and
                           (not support[i, -1] or decision == closing - 60 * MINUTE_NS),
                           "Minute structural support identity fails")
                if i % 4096 == 0:
                    tick()
            if profile == "hourly":
                ensure(support[:, -1].all(), "H8 endpoint planned superset should cover all original origins")
            own_h, own_metrics = independent_columns(np.asarray(arrays["labels"]),
                arrays["prefix_update_mean"], arrays["same_information_Ridge"], support, ensure)
            reference = read(output / (state + ".json"))
            ensure(reference["state"] == state and reference["count"] == n and
                   reference["horizon"] == H and reference["tail_sha256"] == digest,
                   "Main state identity differs")
            ensure(reference["identity_qualified"] is True, "Main identity qualification missing")
            ensure(len(reference["all_horizons"]) == H, "Main horizon coverage differs")
            for own, ref in zip(own_h, reference["all_horizons"]):
                ensure(own["horizon"] == ref["horizon"] and
                       own["support_count"] == ref["support_count"], "Independent support count differs")
                if "support_sha256" in ref:
                    ensure(own["support_sha256"] == ref["support_sha256"], "Support membership hash differs")
                if "b" in ref:
                    ensure(own["b"] == float(ref["b"]), "Fixed prefix b differs")
                for key in ("b_MSE", "support_b_MSE", "floor_MSE"):
                    close_loss(own[key], ref[key], state + " " + key)
                for control in CONTROLS:
                    close_loss(own["MSE"][control], ref["MSE"][control], state + " " + control)
                    close_pp(own["maximum_improvement_pct"][control],
                             ref["maximum_improvement_pct"][control], state + " raw maximum")
                    close_pp(own["safe_upper_maximum_improvement_pct"][control],
                             ref["safe_upper_maximum_improvement_pct"][control], state + " safe maximum")
                for key in ("floor",) + CONTROLS:
                    close_loss(own["risk_error_bounds"][key], ref["risk_error_bounds"][key], state + " envelope")
                ensure(abs(float(ref["identity_residual"])) <= float(ref["identity_tolerance"]),
                       "Main partition-risk identity fails its own band")
                close_loss(own["identity_tolerance"], ref["identity_tolerance"], state + " identity band")
                delta = abs(own["identity_residual"] - float(ref["identity_residual"]))
                ensure(delta <= own["identity_tolerance"] + float(ref["identity_tolerance"]),
                       "Independent raw partition-identity mismatch")
                maxima["raw_identity_absolute"] = max(maxima["raw_identity_absolute"], delta)
            for predictor, values in own_metrics.items():
                for key, value in values.items():
                    if key.endswith("_pct"):
                        close_pp(value, reference["metrics"][predictor][key], state + " aggregate upper bound")
                    else:
                        close_loss(value, reference["metrics"][predictor][key], state + " aggregate loss")
            if profile == "minute":
                endpoint_delta = own_metrics["prefix_update_mean"]["endpoint_MSE"] - own_metrics["floor"]["endpoint_MSE"]
                path_delta = own_metrics["prefix_update_mean"]["path_MSE"] - own_metrics["floor"]["path_MSE"]
                tolerance = 512 * (2.0 ** -52) * max(
                    own_metrics["prefix_update_mean"]["endpoint_MSE"],
                    own_metrics["prefix_update_mean"]["path_MSE"],
                    float.fromhex("0x1.0000000000000p-1022"))
                ensure(abs(path_delta - endpoint_delta / H) <= tolerance,
                       "Minute endpoint/path support-opportunity identity fails")
            support_hash = hashlib.sha256(support.astype("uint8").tobytes()).hexdigest()
            ensure(support_hash == reference["support_sha256"], "Full C-order support mask differs")
            records.append({"state": state, "count": n, "horizon": H, "tail_sha256": digest,
                            "support_sha256": support_hash,
                            "metrics": own_metrics, "all_horizons": own_h})
            completed.append({"state": state, "archive_sha256": entry["sha256"],
                              "main_state_sha256": sha(output / (state + ".json"))})
            write(output / "INDEPENDENT_PROGRESS.json", {"protocol_sha256": identity,
                "deadline_epoch": protocol["deadline_epoch"], "completed_states": completed,
                "pid": os.getpid(), "status": "running"})
            del arrays, origins, support, own_h, own_metrics
        tick()
        own_decision = independent_decision(records)
        main_decision = read(output / "DECISION.json")
        ensure(len(main_decision["necessary_checks"]) == 24 and len(main_decision["tasks"]) == 2,
               "Main 26-condition necessary subset differs")
        for own, ref in zip(own_decision["necessary_checks"], main_decision["necessary_checks"]):
            ensure(all(own[k] == ref[k] for k in ("state", "control", "metric", "status")),
                   "Independent state necessary-gate classification differs")
            for key in ("maximum_improvement_pct", "safe_upper_maximum_improvement_pct"):
                close_pp(own[key], ref[key], "Necessary state upper bound")
        for own, ref in zip(own_decision["tasks"], main_decision["tasks"]):
            ensure(own["task"] == ref["task"] and own["status"] == ref["status"],
                   "Independent equal-fold necessary-gate classification differs")
            for key in ("equal_fold_endpoint_maximum_improvement_over_zero_pct",
                        "safe_upper_maximum_improvement_pct"):
                close_pp(own[key], ref[key], "Necessary equal-fold upper bound")
        for key in ("necessary_check_count", "excluded_necessary_conditions",
                    "fixed_support_class_ruled_out_on_development_gates", "model_fitting_admitted",
                    "candidate_admitted", "independent_market_confirmation",
                    "original_58_gate_recipe_changed", "full_research_goal_achieved"):
            ensure(own_decision[key] == main_decision[key], "Independent final decision differs: " + key)
        tick()
        receipt = {"passed": True, "status": "passed", "protocol_sha256": identity,
                   "deadline_epoch": protocol["deadline_epoch"], "pid": os.getpid(),
                   "checks": checks, "max_differences": maxima,
                   "records_independently_rebuilt": records, "decision": own_decision,
                   "state_horizon_positions": sum(r["horizon"] for r in records),
                   "necessary_gate_checks": 26, "decoded_bar_members": decoded,
                   "actual_dependency_receipt": dependency_receipt,
                   "elapsed_seconds": time.monotonic() - started,
                   "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                   "scope": {"CPU": [26], "numeric_threads": 1, "GPU": False, "Q3_read": False,
                       "training_updates": 0, "model_inference": 0, "new_model_fit": False,
                       "close_member_decoded": False, "q_computed": False,
                       "prediction_members_decoded": list(PREDICTION_READS),
                       "market_confirmation": False, "own_calendar_S_fsum_and_decision": True,
                       "shared_calendar_dependency_numpy_decoder_and_runtime": True}}
        write(output / "INDEPENDENT_REPLAY.json", receipt)
        write(output / "INDEPENDENT_PROGRESS.json", {"protocol_sha256": identity,
            "deadline_epoch": protocol["deadline_epoch"], "completed_states": completed,
            "pid": os.getpid(), "status": "completed"})
        print({"passed": True, "checks": checks,
               "excluded_necessary_conditions": own_decision["excluded_necessary_conditions"]}, flush=True)
        return 0
    except BaseException as error:
        failure = {"passed": False, "status": "failed_preserved", "error": repr(error),
                   "traceback": traceback.format_exc(), "protocol_sha256": identity,
                   "deadline_epoch": None if protocol is None else protocol["deadline_epoch"],
                   "pid": os.getpid(), "completed_states": completed, "checks": checks,
                   "max_differences_so_far": maxima, "received_signals": signals,
                   "elapsed_seconds": time.monotonic() - started,
                   "training_updates": 0, "model_inference": 0, "Q3_read": False}
        path = output / ("INDEPENDENT_FAILURE_" + str(os.getpid()) + ".json")
        if not path.exists():
            write(path, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
