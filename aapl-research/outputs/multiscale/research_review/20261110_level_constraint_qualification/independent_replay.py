"""Independent saved-prediction level-pair replay; import performs no arithmetic.

No main numerical function or decision helper is imported. NumPy import and
all archive access occur only inside main(), after CPU/thread/scope setup.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import resource
import signal
import time
import traceback

from runtime import BASE, setup, read, write, sha, load_protocol, guard

STATES = ("minute_f1", "minute_f2", "hourly_f1", "hourly_f2")
MEMBERS = ("origins", "labels", "D", "Z", "C", "zero",
           "prefix_update_mean", "same_information_Ridge")
PREDICTORS = ("D", "Z", "C", "zero", "prefix_update_mean",
              "same_information_Ridge", "g", "half_Ridge", "half_g")
CONTROLS = ("same_information_Ridge", "zero", "prefix_update_mean",
            "half_Ridge", "half_g")


def main():
    setup(26)
    started = time.monotonic()
    output = BASE / "run"
    protocol = None
    identity = None
    completed = []
    received = []
    checks = 0
    maxima = {"loss_absolute": 0.0, "loss_relative": 0.0,
              "identity_component_absolute": 0.0, "percentage_points": 0.0}

    def ensure(value, message):
        nonlocal checks
        if not bool(value):
            raise RuntimeError(message)
        checks += 1

    def stop(number, frame):
        received.append({"signal": int(number), "pid": os.getpid(),
                         "received_epoch": time.time()})

    signal.signal(signal.SIGUSR1, stop)
    signal.signal(signal.SIGTERM, stop)

    def check_time():
        if received:
            raise InterruptedError("Independent replay received actual stop signal")
        guard(protocol)
        ensure(os.sched_getaffinity(0) == {26}, "Independent CPU affinity changed")

    try:
        protocol = load_protocol()
        identity = sha(BASE / "EXECUTION_PROTOCOL.json")
        spec = read(BASE / "DESIGN_SPEC.json")
        ensure(spec["schema"] == "level_constraint_design_v1", "Wrong fixed design schema")
        ensure([s["state"] for s in spec["states"]] == list(STATES), "Four fixed states/order required")
        ensure(sorted(spec["permitted_members"]) == sorted(k + ".npy" for k in MEMBERS),
               "Design member whitelist differs")
        ensure(spec["controls_for_gate"] == list(CONTROLS), "Frozen controls differ")
        ensure(spec["state_minimum_fractional_improvement"] == .001 and
               spec["equal_fold_primary_minimum_fractional_improvement"] == .01 and
               spec["total_gate_checks"] == 42, "Scientific gate changed")
        ensure(spec["numeric_identity_eps_factor"] == 256 and
               spec["independent_loss_rtol"] == 1e-11 and
               spec["independent_percentage_point_atol"] == 1e-8,
               "Frozen numerical qualifications differ")
        ensure(spec["Q3_allowed"] is False and spec["GPU"] is False and
               spec["new_market_fitting_allowed"] is False and
               spec["market_neural_inference_allowed"] is False,
               "Independent scope changed")
        ensure(not (output / "INDEPENDENT_REPLAY.json").exists(),
               "Do not replace an independent terminal receipt")
        # Require main invocation/completion before any archive member decode.
        done = read(output / "WORKER_DONE.json")
        ensure(done["protocol_sha256"] == identity and done["completed_states"] == 4,
               "Main has not completed the same four-state protocol")
        parent = read(output / "RESULTS.json")
        ensure(parent["protocol_sha256"] == identity and
               [s["state"] for s in parent["records"]] == list(STATES),
               "Main result identity/order differs")
        ensure(parent["training_updates"] == 0 and parent["model_inference"] == 0 and
               parent["Q3_read"] is False and parent["GPU"] is False,
               "Main scope receipt differs")
        import numpy as np
        eps = float(np.finfo(np.float64).eps)
        tiny = float(np.finfo(np.float64).tiny)
        loss_rtol = 1e-11
        pp_atol = 1e-8

        def loss_close(a, b, label):
            a, b = float(a), float(b)
            scale = max(abs(a), abs(b), tiny)
            delta = abs(a - b)
            ensure(math.isfinite(a) and math.isfinite(b) and
                   a >= 0 and b >= 0 and delta <= loss_rtol * scale,
                   "Independent loss mismatch: " + label)
            maxima["loss_absolute"] = max(maxima["loss_absolute"], delta)
            maxima["loss_relative"] = max(maxima["loss_relative"], delta / scale)

        def raw_close(a, b, tolerance, label):
            a, b = float(a), float(b)
            delta = abs(a - b)
            ensure(math.isfinite(a) and math.isfinite(b) and
                   math.isfinite(tolerance) and tolerance >= 0 and delta <= tolerance,
                   "Independent raw-unit mismatch: " + label)
            maxima["identity_component_absolute"] = max(maxima["identity_component_absolute"], delta)

        def pp_close(a, b, label):
            a, b = float(a), float(b)
            delta = abs(a - b)
            ensure(math.isfinite(a) and math.isfinite(b) and delta <= pp_atol,
                   "Independent percentage-point mismatch: " + label)
            maxima["percentage_points"] = max(maxima["percentage_points"], delta)

        def square_mean(column):
            # Streams Python scalars; never materializes n*h Python objects.
            return math.fsum(float(v) * float(v) for v in column) / len(column)

        records = []
        for entry in spec["states"]:
            check_time()
            state = entry["state"]
            n, H = int(entry["count"]), int(entry["horizon"])
            ensure(n > 0 and H == (60 if state.startswith("minute") else 8),
                   "Count/horizon identity differs")
            path = Path(entry["path"])
            ensure(sha(path) == entry["sha256"], "Saved archive changed: " + state)
            # Separate loader from main's ZIP/BytesIO member implementation.
            with np.load(path, allow_pickle=False) as archive:
                ensure(len(archive.files) == 8 and sorted(archive.files) == sorted(MEMBERS),
                       "Unexpected, missing or duplicate archive member")
                arrays = {name: archive[name] for name in MEMBERS}
            ensure(sha(path) == entry["sha256"], "Archive changed during independent read")
            origins = arrays["origins"]
            ensure(origins.dtype == np.dtype("int64") and origins.shape == (n,) and
                   bool(np.all(np.diff(origins) > 0)), "Independent origin dtype/shape/order")
            origin_hash = hashlib.sha256(np.asarray(origins, dtype="<i8").tobytes(order="C")).hexdigest()
            ensure(origin_hash == entry["tail_sha256"], "Independent tail origin hash differs")
            ensure(entry["partition"]["id"] == state and entry["partition"]["tail_count"] == n and
                   entry["partition"]["tail_sha256"] == origin_hash,
                   "Stored original partition identity differs")
            for name in MEMBERS[1:]:
                value = arrays[name]
                ensure(value.dtype == np.dtype("float64") and value.shape == (n, H) and
                       bool(np.isfinite(value).all()), "Independent member qualification: " + name)
            ensure(bool(np.all(arrays["zero"] == 0.0)), "Saved zero is not zero")
            ensure(bool(np.all(arrays["prefix_update_mean"] == arrays["prefix_update_mean"][0])),
                   "Saved prefix mean is not a constant path")
            reference = read(output / (state + ".json"))
            ensure(reference["state"] == state and reference["count"] == n and
                   reference["horizon"] == H and reference["tail_sha256"] == origin_hash,
                   "Main state identity differs")
            ensure(len(reference["all_horizons"]) == H and
                   [r["horizon"] for r in reference["all_horizons"]] == list(range(1, H + 1)),
                   "Main horizon coverage/order differs")
            ensure(set(reference["metrics"]) == set(PREDICTORS) and
                   reference["identity_qualified"] is True and
                   reference["same_slopes_by_construction"] is True,
                   "Main predictor coverage/qualification differs")
            own_rows = []
            per_h = {name: [] for name in PREDICTORS}
            for h in range(H):
                check_time()
                ycol = arrays["labels"][:, h]
                pcol = arrays["same_information_Ridge"][:, h]
                b = float(arrays["prefix_update_mean"][0, h])
                gcol = pcol - b
                halfp, halfg = pcol * .5, gcol * .5
                cols = {name: arrays[name][:, h] for name in MEMBERS[2:]}
                cols.update(g=gcol, half_Ridge=halfp, half_g=halfg)
                mse = {}
                for name in PREDICTORS:
                    mse[name] = math.fsum((float(y) - float(f)) ** 2
                                          for y, f in zip(ycol, cols[name])) / n
                    ensure(math.isfinite(mse[name]) and mse[name] >= 0,
                           "Independent nonfinite/negative MSE")
                    per_h[name].append(mse[name])
                b2 = b * b
                cross = 2 * b * math.fsum(float(y) - float(g)
                                           for y, g in zip(ycol, gcol)) / n
                difference = mse["same_information_Ridge"] - mse["g"]
                scale = max(square_mean(ycol), square_mean(pcol), square_mean(gcol), b2, tiny)
                tolerance = 256 * eps * scale
                residual = difference - (b2 - cross)
                ensure(math.isfinite(scale) and math.isfinite(tolerance) and
                       math.isfinite(residual) and abs(residual) <= tolerance,
                       "Independent paired loss identity fails")
                ensure(bool(np.all(np.abs((gcol + b) - pcol) <=
                                   8 * eps * np.maximum(np.abs(pcol) + abs(b), tiny))),
                       "Independent fixed-level mapping fails")
                ref = reference["all_horizons"][h]
                ensure(set(ref["MSE"]) == set(PREDICTORS), "Main per-h predictor keys differ")
                raw_close(b, ref["b"], 0.0, state + " b")
                for name in PREDICTORS:
                    loss_close(mse[name], ref["MSE"][name], state + " h" + str(h + 1) + " " + name)
                for name, value in (("b_square", b2), ("twice_b_residual_mean", cross),
                                    ("L_p_minus_L_g", difference)):
                    raw_close(value, ref[name], tolerance, state + " " + name)
                ensure(abs(float(ref["identity_residual"])) <= float(ref["identity_tolerance"]),
                       "Main identity residual exceeds its saved tolerance")
                raw_close(residual, ref["identity_residual"], 2 * tolerance,
                          state + " identity residual")
                loss_close(tolerance, ref["identity_tolerance"], state + " identity tolerance")
                own_rows.append({"horizon": h + 1, "b": b, "b_square": b2,
                                 "twice_b_residual_mean": cross, "L_p_minus_L_g": difference,
                                 "identity_residual": residual, "identity_tolerance": tolerance,
                                 "MSE": mse})
                del cols, halfp, halfg, gcol
            metrics = {}
            for name in PREDICTORS:
                metrics[name] = {"endpoint_MSE": per_h[name][-1],
                                 "path_MSE": math.fsum(per_h[name]) / H}
                for metric, value in metrics[name].items():
                    loss_close(value, reference["metrics"][name][metric], state + " " + name + " " + metric)
            # Cancellation residuals use the raw-unit identity band, never the
            # relative predictive-loss comparator near zero.
            raw_close(max(abs(r["identity_residual"]) for r in own_rows),
                      reference["maximum_identity_residual"],
                      2 * max(r["identity_tolerance"] for r in own_rows),
                      state + " maximum identity residual")
            records.append({"state": state, "count": n, "horizon": H,
                            "tail_sha256": origin_hash, "metrics": metrics,
                            "all_horizons": own_rows})
            completed.append({"state": state, "archive_sha256": entry["sha256"],
                              "main_state_sha256": sha(output / (state + ".json")),
                              "horizons": H})
            write(output / "INDEPENDENT_PROGRESS.json",
                  {"protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                   "completed_states": completed, "pid": os.getpid(), "status": "running"})
            del arrays, per_h, own_rows, ycol, pcol, origins
        check_time()
        ensure([r["state"] for r in records] == list(STATES), "Independent full coverage differs")
        own_checks = []
        for row in records:
            for control in CONTROLS:
                for metric in ("endpoint_MSE", "path_MSE"):
                    denominator = row["metrics"][control][metric]
                    ensure(math.isfinite(denominator) and denominator > 0,
                           "Positive independent control denominator required")
                    gain = 1 - row["metrics"]["g"][metric] / denominator
                    own_checks.append({"state": row["state"], "control": control, "metric": metric,
                                       "improvement_pct": 100 * gain, "passed": bool(gain >= .001)})
        own_tasks = []
        for prefix in ("minute", "hourly"):
            rows = [r for r in records if r["state"].startswith(prefix + "_")]
            ensure(len(rows) == 2, "Exactly two original folds per primary")
            gain = math.fsum(1 - r["metrics"]["g"]["endpoint_MSE"] /
                             r["metrics"]["zero"]["endpoint_MSE"] for r in rows) / 2
            own_tasks.append({"task": prefix, "equal_fold_endpoint_improvement_over_zero_pct": 100 * gain,
                              "passed": bool(gain >= .01)})
        own_decision = {"checks": own_checks, "tasks": own_tasks,
                        "passed_of42": sum(int(x["passed"]) for x in own_checks + own_tasks),
                        "continue_to_full_development_design_review": all(x["passed"] for x in own_checks + own_tasks),
                        "candidate_qualified": False, "independent_confirmation": False,
                        "full_goal_achieved": False}
        main_decision = read(output / "DECISION.json")
        ensure(len(main_decision["checks"]) == 40 and len(main_decision["tasks"]) == 2,
               "Main 42-gate coverage differs")
        for own, ref in zip(own_checks, main_decision["checks"]):
            ensure(all(own[k] == ref[k] for k in ("state", "control", "metric", "passed")),
                   "Independent state gate disagrees")
            pp_close(own["improvement_pct"], ref["improvement_pct"], "state gate")
        for own, ref in zip(own_tasks, main_decision["tasks"]):
            ensure(own["task"] == ref["task"] and own["passed"] == ref["passed"],
                   "Independent primary gate disagrees")
            pp_close(own["equal_fold_endpoint_improvement_over_zero_pct"],
                     ref["equal_fold_endpoint_improvement_over_zero_pct"], "primary equal-fold gate")
        for name in ("passed_of42", "continue_to_full_development_design_review",
                     "candidate_qualified", "independent_confirmation", "full_goal_achieved"):
            ensure(own_decision[name] == main_decision[name], "Independent final decision differs: " + name)
        check_time()
        receipt = {"passed": True, "status": "passed", "protocol_sha256": identity,
                   "deadline_epoch": protocol["deadline_epoch"], "pid": os.getpid(),
                   "checks": checks, "max_differences": maxima,
                   "records_independently_rebuilt": records, "completed_states": completed,
                   "state_horizon_positions": sum(r["horizon"] for r in records),
                   "MSE_predictors_per_horizon": len(PREDICTORS), "scientific_gate_checks": 42,
                   "decision": own_decision, "elapsed_seconds": time.monotonic() - started,
                   "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
                   "scope": {"CPU": [26], "numeric_threads": 1, "GPU": False, "Q3_read": False,
                             "training_updates": 0, "model_inference": 0,
                             "new_model_fit": False, "bars_or_splits_or_weights_read": False,
                             "saved_exposed_development_predictions_read": True,
                             "old_accounting_rerun": False, "independent_market_confirmation": False,
                             "shared_runtime_and_numpy_decode_backend": True,
                             "own_fsum_risk_identity_and_gate_functions": True}}
        write(output / "INDEPENDENT_REPLAY.json", receipt)
        write(output / "INDEPENDENT_PROGRESS.json",
              {"protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
               "completed_states": completed, "pid": os.getpid(), "status": "completed"})
        print({"passed": True, "checks": checks, "states": len(records),
               "passed_of42": own_decision["passed_of42"]}, flush=True)
        return 0
    except BaseException as error:
        failure = {"passed": False, "status": "failed_preserved", "error": repr(error),
                   "traceback": traceback.format_exc(), "protocol_sha256": identity,
                   "deadline_epoch": None if protocol is None else protocol["deadline_epoch"],
                   "pid": os.getpid(), "completed_states": completed, "checks": checks,
                   "max_differences_so_far": maxima, "received_signals": received,
                   "elapsed_seconds": time.monotonic() - started,
                   "training_updates": 0, "model_inference": 0, "Q3_read": False,
                   "old_accounting_rerun": False}
        destination = output / ("INDEPENDENT_FAILURE_" + str(os.getpid()) + ".json")
        if not destination.exists():
            write(destination, failure)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
