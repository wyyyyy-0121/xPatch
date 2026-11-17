"""Continuous replay and independent fsum verification of an already seen cell.

Shared data-admission helpers are disclosed: independence concerns the execution
trajectory and metric reduction, not data access or independent market evidence.
Requires a separately frozen protocol; importing this file starts no replay.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import zipfile
import numpy as np
import torch
import torch._dynamo
import cap_diagnostic_threads2 as helper

STAGE = Path(__file__).resolve().parents[1]
MAIN = STAGE / "thread_matched_run"
TOLERANCE = 1e-15


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def alarm(*_):
    raise TimeoutError("Immutable independent replay deadline or signal")


def registered_json(p, path):
    path = Path(path).resolve()
    require(str(path) in p["metadata_sha256"], "Unregistered metadata: " + str(path))
    require(helper.sha(path) == p["metadata_sha256"][str(path)], "Metadata hash differs")
    return json.loads(path.read_text())


def load_pt(path, p):
    path = Path(path).resolve()
    output = Path(p["output_directory"]).resolve()
    require(path.is_relative_to(output) or str(path) in p["input_sha256"], "Unregistered tensor")
    if not path.is_relative_to(output):
        require(helper.sha(path) == p["input_sha256"][str(path)], "Tensor hash differs")
    return torch.load(path, weights_only=True, map_location="cpu")


def decode_npz(path, members):
    """Decode exactly the named NPY members, without the main score helper."""
    decoded = {}
    with zipfile.ZipFile(path) as archive:
        require(set(archive.namelist()) == {name + ".npy" for name in members}, "Unexpected NPZ members")
        for name in members:
            member = name + ".npy"
            require(archive.namelist().count(member) == 1, "Duplicate NPZ member")
            with archive.open(member) as stream:
                decoded[name] = np.lib.format.read_array(io.BytesIO(stream.read()), allow_pickle=False)
    return decoded


def independent_metrics(prediction, truth):
    rows, horizon = truth.shape
    squared = [[(float(prediction[i, j]) - float(truth[i, j])) ** 2
                for j in range(horizon)] for i in range(rows)]
    per_step = [math.fsum(row[j] for row in squared) / rows for j in range(horizon)]
    return {"endpoint_mse": math.fsum(row[-1] for row in squared) / rows,
            "path_mse": math.fsum(value for row in squared for value in row) / (rows * horizon),
            "per_step_mse": per_step}


def verify_metrics(p, main_results, receipt, windows, stop):
    path = MAIN / "DEVELOPMENT_PREDICTIONS.npz"
    require(str(path) in p["input_sha256"], "Prediction archive not admitted")
    require(helper.sha(path) == p["input_sha256"][str(path)], "Prediction bytes differ")
    arrays = decode_npz(path, ("origins", "cap20", "cap60", "truth"))
    origins = arrays["origins"]
    require(origins.dtype == np.int64 and origins.shape == (1356,), "Outer shape/dtype differs")
    require(bool(np.all(np.diff(origins) > 0)), "Outer origins are not ordered and unique")
    digest = hashlib.sha256(np.asarray(origins, dtype="<i8").tobytes()).hexdigest()
    require(digest == receipt["identity_fields"]["validation_indices_sha256"], "Outer canonical hash differs")
    with zipfile.ZipFile(helper.SPLIT) as archive:
        require(archive.namelist().count("validation.npy") == 1, "Ambiguous validation member")
        with archive.open("validation.npy") as stream:
            original_origins = np.lib.format.read_array(io.BytesIO(stream.read()), allow_pickle=False)
    require(helper.same(original_origins, origins), "NPZ origins differ from original validation")
    require(windows.bars.end_ns[stop[-1] + 8] < windows.bars.end_ns[origins[0]], "Stop/outer purge differs")
    require(origins[0] >= 95 and origins[-1] + 8 < len(windows.bars.close), "Outer label bounds differ")
    require(bool(np.all(windows.bars.session_date[origins + 8] <= "2026-06-30")), "Q3 labels forbidden")
    for name in ("cap20", "cap60", "truth"):
        value = arrays[name]
        require(value.dtype == np.float64 and value.shape == (1356, 8)
                and bool(np.isfinite(value).all()), "Invalid prediction/truth array: " + name)
    require(helper.same(arrays["cap20"], arrays["cap60"]), "Old/new predictions are not byte-identical")
    future = origins[:, None] + np.arange(1, 9, dtype=np.int64)
    require(bool(windows.bars.observed[origins].all() and windows.bars.observed[future].all()), "Unobserved outer target")
    reconstructed_truth = np.log(windows.bars.close[future] / windows.bars.close[origins, None])
    require(helper.same(reconstructed_truth, arrays["truth"]), "Saved truth differs from source bars")
    results, maximum_difference, comparisons = {}, 0.0, 0
    for name in ("cap20", "cap60", "zero"):
        prediction = np.zeros_like(arrays["truth"]) if name == "zero" else arrays[name]
        score = independent_metrics(prediction, arrays["truth"])
        expected = main_results["scores"][name]
        for key in ("endpoint_mse", "path_mse", "per_step_mse"):
            pairs = zip(score[key], expected[key]) if key == "per_step_mse" else [(score[key], expected[key])]
            for actual, reference in pairs:
                difference = abs(actual - reference)
                require(difference <= TOLERANCE, "Frozen metric tolerance exceeded: " + name + "/" + key)
                maximum_difference = max(maximum_difference, difference)
                comparisons += 1
        results[name] = score
    require(comparisons == 30, "Expected three models times ten metrics")
    return {"metrics": results, "MSE_comparisons": comparisons, "absolute_tolerance": TOLERANCE,
            "maximum_absolute_difference": maximum_difference, "validation_sha256": digest,
            "origins": len(origins), "prediction_old_new_exactbytes": True,
            "truth_reconstructed_exactbytes": True, "method": "Python scalar squares and math.fsum"}


def run(p, protocol_path):
    helper.ACTIVE = p
    require(p["status"] == "frozen_before_execution" and p["mode"] == "independent_replay", "Wrong phase")
    require(p.get("independent_replay_admitted") is True and p.get("market_arrays_admitted") is True,
            "Replay/data admission missing")
    require(p.get("training_updates_admitted") is True and p.get("new_market_periods_admitted") is False,
            "Only repeated original-cell updates admitted")
    require(p.get("Q3_allowed") is False and p.get("network_allowed") is False, "Forbidden access")
    require(math.isfinite(p["deadline_epoch"]) and p.get("deadline_may_reset") is False, "Finite immutable deadline required")
    helper.check(p)
    signal.signal(signal.SIGALRM, alarm)
    signal.signal(signal.SIGTERM, alarm)
    signal.signal(signal.SIGINT, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, p["deadline_epoch"] - time.time()))
    out = Path(p["output_directory"]).resolve()
    require(out == STAGE / "thread_matched_independent_run", "Fixed isolated output required")
    access = helper.install_guard(p, protocol_path)
    for mapping in ("code_sha256", "metadata_sha256", "input_sha256"):
        for path, expected in p[mapping].items():
            require(helper.sha(path) == expected, "Frozen input changed: " + path)
    require(str(Path(__file__).resolve()) in p["code_sha256"], "Replay code not registered")
    require(str(STAGE / "code/cap_diagnostic_threads2.py") in p["code_sha256"], "Shared admission helper not frozen")
    market_protocol = registered_json(p, STAGE / "THREAD_MATCHED_PROTOCOL.json")
    require(p["deadline_epoch"] == market_protocol["deadline_epoch"], "Stage deadline changed")
    main_results = registered_json(p, MAIN / "RESULTS.json")
    require(main_results["status"] == "completed_single_cap_diagnostic", "Main run not complete")
    require(main_results["protocol_sha256"] == helper.sha(STAGE / "THREAD_MATCHED_PROTOCOL.json"), "Main protocol mismatch")
    require(main_results["training_updates"] == 54 and main_results["call_count"] == 55, "Main trajectory differs")
    require(p.get("maximum_repeated_updates") == 54, "Repeated update budget must be 54")
    cost = float(p["replay_cost_estimate_seconds"])
    require(math.isfinite(cost) and cost >= 3 * 55 * main_results["max_call_seconds"] + 10,
            "Replay cost reserve below predeclared main timing margin")
    helper.check(p, cost + 10)
    require(not out.exists(), "Refusing to reset prior replay output")
    out.mkdir()
    status = {"status": "running", "protocol_sha256": helper.sha(protocol_path),
              "deadline_epoch": p["deadline_epoch"], "repeated_updates": 0, "Q3_read": False}
    helper.atomic_json(out / "STATUS.json", status)
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(1)
        sys.dont_write_bytecode = True
        sys.pycache_prefix = str(out / "unused_pycache")
        sys.path.insert(0, str(helper.SOURCE))
        windows, fit, stop, c20, dataid, latest, best, receipt, evidence = helper.load_data_and_verify(p)
        from multiscale.training import identity, atomic_torch_save, run_fit
        c60 = replace(c20, max_epochs=60)
        require(c60.patience == 3, "Original patience changed")
        new_identity = identity(c60, dataid, fit, stop, "cpu")
        extended = out / "extended60"; extended.mkdir()
        for name, original in (("latest", latest), ("best", best)):
            value = copy.deepcopy(original)
            value["identity"] = new_identity
            if name == "best":
                value["config"]["max_epochs"] = 60
            restored = copy.deepcopy(value); restored["identity"] = original["identity"]
            if name == "best":
                restored["config"]["max_epochs"] = 20
            require(helper.same(original, restored), "Bridge changed scientific payload")
            atomic_torch_save(value, extended / (name + ".pt"))
            require(helper.same(value, load_pt(extended / (name + ".pt"), p)), "Bridge roundtrip differs")
        helper.check(p, cost)
        update_count = [0]
        def count_update(optimizer, args, kwargs):
            helper.check(p, 2)
            require(update_count[0] < 54, "Replay exceeded 54 repeated updates; abort before update 55")
            update_count[0] += 1
        from torch.optim.optimizer import register_optimizer_step_pre_hook
        observer = register_optimizer_step_pre_hook(count_update)
        started = time.monotonic()
        try:
            # One uninterrupted original call; the hook only observes/bounds updates.
            summary = run_fit(windows, fit, stop, c60, extended, dataid, device="cpu",
                              resume=True, stop_after_steps=None)
        finally:
            observer.remove()
        elapsed = time.monotonic() - started
        require(update_count[0] == 54 and summary["epochs"] == 23 and summary["steps"] == 414
                and summary["best_epoch"] == 20 and summary["stopped_early"] is True,
                "Replay did not reproduce epoch23/414/best20/patience3")
        require(helper.same(summary, main_results["cap60_summary"]), "Returned summary differs from main")
        actual_summary = json.loads((extended / "training_summary.json").read_text())
        reference_summary = registered_json(p, MAIN / "extended60/training_summary.json")
        require(helper.same(actual_summary, reference_summary), "Saved summary differs bytewise")
        for name in ("latest.pt", "best.pt"):
            require(helper.same(load_pt(extended / name, p), load_pt(MAIN / "extended60" / name, p)),
                    "Continuous vs 55-call final payload differs: " + name)
        independent_score = verify_metrics(p, main_results, receipt, windows, stop)
        for mapping in ("code_sha256", "metadata_sha256", "input_sha256"):
            for path, expected in p[mapping].items():
                require(helper.sha(path) == expected, "Frozen input modified during replay")
        helper.check(p, 2)
        status.update(status="passed", completed_epoch=time.time(), repeated_updates=update_count[0], replay_seconds=elapsed,
                      continuous_vs_main_latest_best_summary_exactbytes=True,
                      main_call_count=55, independent_training_calls=1,
                      independent_metric_verification=independent_score,
                      data_verification=evidence, access_guard=access,
                      shared_admission_helpers=True, independent_data_admission_claimed=False,
                      independent_market_confirmation=False, new_candidates=0,
                      predictive_method_success=False, full_goal_achieved=False)
        helper.atomic_json(out / "RESULTS.json", status)
        helper.atomic_json(out / "STATUS.json", status)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        status.update(status="deadline_or_signal" if isinstance(exc, TimeoutError) else "failed",
                      error_type=type(exc).__name__, error=str(exc),
                      repeated_updates=update_count[0] if "update_count" in locals() else 0)
        helper.atomic_json(out / "STATUS.json", status)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    path = args.protocol.resolve()
    try:
        run(json.loads(path.read_text()), path)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
