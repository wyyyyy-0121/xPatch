"""Guarded analytic worker; samples inputs only, never labels or model weights."""
from __future__ import annotations

import argparse
import csv
import io
import os
from pathlib import Path
import signal
import time
import traceback

from runtime_common import (BASE, atomic_json, canonical_digest, check_resources,
                            install_file_guard, load_protocol, memory_limit,
                            peak_rss_bytes, read_json, require, set_environment, utc_now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--mode", choices=("toy", "analytic"), required=True)
    parser.add_argument("--output-name", choices=("toy_continuous", "toy_resumed", "main"), required=True)
    parser.add_argument("--signal-handshake", action="store_true")
    args = parser.parse_args()
    set_environment()
    protocol, identity = load_protocol(args.protocol)
    # This reviewed module has definitions only. Load it after identity checks
    # but before the guard so importlib's attempted absent local pyc read cannot
    # bypass/contradict the deliberately narrow post-install read whitelist.
    import analytic_core as core
    require((args.mode == "analytic") == (args.output_name == "main"), "Output/mode mismatch")
    require(not args.signal_handshake or args.output_name == "toy_resumed", "Only toy lifecycle may handshake")
    output = Path(protocol["output_root"]) / args.output_name
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "CHECKPOINT.json"
    memory_limit()
    install_file_guard(protocol, args.protocol)
    import numpy as np

    received = []
    def stop_handler(number, frame):
        received.append({"signal": number, "received_epoch": time.time(), "pid": os.getpid()})
    signal.signal(signal.SIGUSR1, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    guard = lambda: check_resources(protocol, stopped=bool(received))
    started = time.monotonic()
    state = {"schema": "origin_noise_checkpoint_v1", "protocol_sha256": identity,
             "deadline_epoch": protocol["deadline_epoch"], "runtime": protocol["runtime"],
             "mode": args.mode, "completed_units": [], "records": [], "input_qualifications": {},
             "operator_qualifications": {}, "pid_history": [], "signal_history": []}
    if checkpoint_path.exists():
        old = read_json(checkpoint_path)
        for field in ("schema", "protocol_sha256", "deadline_epoch", "runtime", "mode"):
            require(old[field] == state[field], "Resume identity differs: " + field)
        require(old["status"] in ("paused", "running"), "Cannot rerun terminal numerical state")
        state = old
        units = core.expected_unit_keys(args.mode)
        require(state["completed_units"] == units[:len(state["completed_units"])], "Nonprefix resume cursor")
        keys = core.expected_record_keys(args.mode)
        require([core.record_key(row) for row in state["records"]] == keys[:12 * len(state["completed_units"])],
                "Committed records do not match cursor")
    state["pid_history"].append(os.getpid())

    def commit(status):
        state.update(status=status, saved_utc=utc_now(), pid=os.getpid(), peak_rss_bytes=peak_rss_bytes(),
                     current_process_elapsed_seconds=time.monotonic() - started)
        known = {(r["pid"], r["signal"], r["received_epoch"]) for r in state["signal_history"]}
        state["signal_history"].extend(r for r in received if (r["pid"], r["signal"], r["received_epoch"]) not in known)
        atomic_json(checkpoint_path, state)

    try:
        guard()
        if args.mode == "toy":
            structure = core.structural_checks(np)
            atomic_json(output / "STRUCTURAL_ACCEPTANCE.json", structure)
            shapes = ((7, 4),)
        else:
            lifecycle = read_json(Path(protocol["output_root"]) / "LIFECYCLE_ACCEPTANCE.json")
            require(lifecycle["passed"] and lifecycle["protocol_sha256"] == identity,
                    "Actual signal/new-process lifecycle is required before main")
            shapes = core.SHAPES
        commit("running")
        for lookback, horizon in shapes:
            guard()
            shape_started = time.monotonic()
            if args.mode == "toy":
                X, Z, z_fit, z_test = core.toy_inputs(np)
                qualification = {"X_sha256": core.array_hash(np, X, "<f8"),
                                 "Z_sha256": core.array_hash(np, Z, "<f8"),
                                 "z_fit_sha256": core.array_hash(np, z_fit, "<f8"),
                                 "z_test_sha256": core.array_hash(np, z_test, "<f8")}
            else:
                X, Z, z_fit, z_test, qualification = core.construct_inputs(np, lookback, horizon, guard)
            label = f"L{lookback}_H{horizon}"
            if label in state["input_qualifications"]:
                require(state["input_qualifications"][label] == qualification, "Resumed fixed design differs")
            state["input_qualifications"][label] = qualification
            operators, op_qualification = core.operator_set(np, X)
            state["operator_qualifications"][label] = op_qualification
            for h in range(1, horizon + 1):
                unit = f"{label}_h{h}"
                if unit in state["completed_units"]:
                    continue
                guard()
                rows = core.horizon_records(np, lookback, horizon, h, X, Z, z_fit, z_test, operators)
                state["records"].extend(rows)
                state["completed_units"].append(unit)
                commit("running")
                if args.mode == "analytic" and lookback == 480 and h == 1:
                    measured = time.monotonic() - shape_started
                    # Includes full maximum-shape generation and first unit; a deliberately
                    # conservative linear multiplier covers every remaining h and I/O.
                    bound = measured * 68 * 4 + 5
                    remaining = protocol["deadline_epoch"] - time.time()
                    cost = {"maximum_shape_first_unit_seconds": measured,
                            "conservative_remaining_plus_replay_seconds": bound,
                            "remaining_seconds": remaining, "reserved_seconds": 60,
                            "passed": bool(bound + 60 <= remaining)}
                    atomic_json(Path(protocol["output_root"]) / "COST_ACCEPTANCE.json", cost)
                    require(cost["passed"], "Maximum-shape cost does not fit unchanged stage window")
                if args.signal_handshake and len(state["completed_units"]) == 1:
                    atomic_json(output / "SIGNAL_READY.json", {"pid": os.getpid(), "protocol_sha256": identity,
                                "deadline_epoch": protocol["deadline_epoch"], "completed_units": list(state["completed_units"]),
                                "checkpoint_sha256": canonical_digest(state), "ready_epoch": time.time()})
                    handshake_end = min(time.time() + 5, protocol["deadline_epoch"] - 60)
                    while not received and time.time() < handshake_end:
                        time.sleep(0.005)
                    require(bool(received), "External OS signal was not received")
                    guard()
            del X, Z, z_fit, z_test, operators
        guard()
        require(state["completed_units"] == core.expected_unit_keys(args.mode), "Incomplete unit coverage")
        require([core.record_key(row) for row in state["records"]] == core.expected_record_keys(args.mode),
                "Record coverage/order differs")
        if args.mode == "analytic":
            paths, endpoints, decision = core.summarize(np, state["records"])
            summary = {"protocol_sha256": identity, "endpoints": endpoints, "paths": paths,
                       "decision": decision, "records": len(state["records"]),
                       "shared_numpy_rng_backend": True, "noise_realizations_generated": 0,
                       "labels_fitted": 0, "training_updates": 0,
                       "bias_computed_from_explicit_noise_free_label_response": True,
                       "small_negative_risk_or_variance_components": sum(bool(r["small_negative_components"]) for r in state["records"]),
                       "small_negative_covariance_eigenvalues": sum(r["small_negative_covariance_eigenvalue"] for r in state["records"])}
            atomic_json(Path(protocol["output_root"]) / "SUMMARY.json", summary, exclusive=True)
            atomic_json(Path(protocol["output_root"]) / "INPUT_QUALIFICATION.json",
                        {"passed": True, "inputs": state["input_qualifications"],
                         "operators": state["operator_qualifications"]}, exclusive=True)
            text = io.StringIO()
            writer = csv.DictWriter(text, fieldnames=list(state["records"][0]))
            writer.writeheader()
            writer.writerows(state["records"])
            csv_path = Path(protocol["output_root"]) / "RISK_COMPONENTS.csv"
            temporary = csv_path.with_suffix(".csv.tmp")
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                stream.write(text.getvalue()); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, csv_path)
            contrasts = []
            lookup = {core.record_key(r): r for r in state["records"]}
            for L, H in core.SHAPES:
                for h in range(1, H + 1):
                    for estimator in core.ESTIMATORS:
                        ri, ro = lookup[L, H, h, estimator, "I"], lookup[L, H, h, estimator, "O"]
                        contrasts.append({"lookback": L, "horizon": H, "h": h, "estimator": estimator,
                            **{"delta_" + suffix: ro["risk_" + suffix] - ri["risk_" + suffix]
                               for suffix in ("total", "level", "dynamic")}})
            atomic_json(Path(protocol["output_root"]) / "ALL_HORIZON_CONTRASTS.json", contrasts, exclusive=True)
            state["outcome"] = decision
        else:
            state["outcome"] = {"toy_complete": True, "structural_qualification_passed": True,
                                "no_scientific_inference_from_toy": True}
        commit("completed")
        print({"status": "completed", "units": len(state["completed_units"]),
               "records": len(state["records"]), "pid": os.getpid()}, flush=True)
        return 0
    except InterruptedError as error:
        commit("paused")
        atomic_json(output / f"PAUSE_{os.getpid()}.json", {"reason": str(error), "pid": os.getpid(),
                    "protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                    "completed_units": state["completed_units"], "signals": state["signal_history"]}, exclusive=True)
        print({"status": "paused", "units": len(state["completed_units"]), "pid": os.getpid()}, flush=True)
        return 75
    except BaseException as error:
        commit("failed")
        atomic_json(output / f"FAILURE_{os.getpid()}.json", {"error": repr(error), "traceback": traceback.format_exc(),
                    "protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"]}, exclusive=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
