"""Independent original-forward/direct-X SVD replay of four existing fit states.

This script never imports the streamed QR kernel or an old experiment engine.
Only scalar summaries are persisted. Dense designs, labels and SVD factors stay
in memory for one state at a time; no projected coefficient/predictor is formed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time


BASE = Path(__file__).resolve().parents[1]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def check_deadline(deadline):
    if STOP or time.time() >= deadline:
        raise InterruptedError("Original stage deadline or explicit stop reached; deadline not reset")


def check_files(files, deadline):
    for path, expected in files.items():
        check_deadline(deadline)
        if sha(path) != expected:
            raise ValueError("Frozen input bytes changed: " + path)


def normalized_scalar_difference(left, right):
    return abs(float(left) - float(right)) / max(1.0, abs(float(left)), abs(float(right)))


def dense_projection(np, x, y, b0, zero_prediction, deadline):
    """Direct SVD of original dense X; no QR factors and no Gram inverse."""
    check_deadline(deadline)
    n, features = x.shape
    h = y.shape[1]
    if features != 34 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid original dense design or targets")
    u, singular, _ = np.linalg.svd(x, full_matrices=False)
    check_deadline(deadline)
    if singular.shape != (34,) or not np.isfinite(singular).all():
        raise ValueError("Unexpected direct dense singular values")
    smax = float(singular[0])
    cutoff = float(np.finfo(np.float64).eps * max(n, 34) * smax)
    retained = singular > cutoff
    ambiguous = (singular >= 0.1 * cutoff) & (singular <= 10.0 * cutoff)
    rank = int(np.count_nonzero(retained))
    # Only U_r^T Y is formed. There is no B_star or projected predictor.
    ur = u if rank == 34 else u[:, retained]
    compressed_target = ur.T @ y
    original_projection = np.zeros((rank, h), dtype=np.float64)
    for start in range(0, n, 128):
        check_deadline(deadline)
        end = min(start + 128, n)
        original_projection += ur[start:end].T @ (x[start:end] @ b0)
    pieces = {key: [] for key in ("original", "projected", "gap", "zero", "dropped", "original_norm")}
    normal_residual = np.zeros((34, h), dtype=np.float64)
    for start in range(0, n, 128):
        check_deadline(deadline)
        end = min(start + 128, n)
        xb, yb, ub = x[start:end], y[start:end], ur[start:end]
        original = xb @ b0
        projected = ub @ compressed_target
        residual = projected - yb
        original_dropped = original - ub @ original_projection
        for name, array in (("original", original-yb), ("projected", residual),
                            ("gap", original-projected), ("zero", zero_prediction[None, :]-yb),
                            ("dropped", original_dropped), ("original_norm", original)):
            pieces[name].append(float(np.sum(array * array, dtype=np.float64)))
        normal_residual += xb.T @ residual
    sse = {key: math.fsum(value) for key, value in pieces.items()}
    if not all(math.isfinite(value) for value in sse.values()):
        raise ValueError("Nonfinite direct dense residual norm")
    denominator = n * h
    losses = {key: sse[key] / denominator for key in ("original", "projected", "gap", "zero")}
    tolerance = 1e-9 * max(1.0, losses["original"], losses["projected"], losses["gap"])
    pythagorean = losses["original"] - losses["projected"] - losses["gap"]
    stationarity = float(np.linalg.norm(normal_residual, ord="fro") / max(
        1.0, smax * float(np.linalg.norm(y, ord="fro")), smax * math.sqrt(sse["projected"])))
    dropped_relative = math.sqrt(sse["dropped"]) / max(1.0, math.sqrt(sse["original_norm"]))
    gates = {
        "effective_space_stationarity": stationarity <= 1e-10,
        "projection_not_above_original": losses["projected"] <= losses["original"] + tolerance,
        "nonnegative_gap": losses["gap"] >= -tolerance,
        "Pythagorean_identity": abs(pythagorean) <= tolerance,
        "original_in_resolved_subspace": dropped_relative <= 1e-10,
    }
    numerically_qualified = all(gates.values())
    rank_ambiguous = bool(np.any(ambiguous))
    return {
        "method": "direct original dense X FP64 SVD orthogonal projection; no streamed QR",
        "n_rows": n, "n_targets": h, "features": 34,
        "SSE_original": sse["original"], "SSE_projected": sse["projected"],
        "SSE_head_gap": sse["gap"], "SSE_zero_baseline": sse["zero"],
        "MSE_original": losses["original"], "MSE_projected": losses["projected"],
        "MSE_head_gap": losses["gap"], "MSE_zero_baseline": losses["zero"],
        "gap_over_zero_baseline": losses["gap"] / losses["zero"] if losses["zero"] > 0 else None,
        "gap_numerically_resolved": losses["gap"] > tolerance,
        "singular_values": singular.tolist(), "rank": rank, "rank_cutoff": cutoff,
        "rank_ambiguous": rank_ambiguous,
        "ambiguous_singular_indices": np.flatnonzero(ambiguous).tolist(),
        "rank_ambiguity_band": [0.1*cutoff, 10.0*cutoff],
        "retained_condition_number": smax / float(singular[retained][-1]) if rank else None,
        "stationarity_normalized": stationarity,
        "Pythagorean_SSE_residual": sse["original"]-sse["projected"]-sse["gap"],
        "Pythagorean_MSE_residual": pythagorean,
        "Pythagorean_MSE_tolerance": tolerance,
        "discarded_original_prediction_relative_frobenius": dropped_relative,
        "numerical_gates": gates, "numerical_gates_passed": numerically_qualified,
        "full_head_class_optimum_identifiable": rank == 34 and not rank_ambiguous and numerically_qualified,
        "projected_coefficients_formed_or_retained": False,
        "exact_algebraic_dependence_certified": False,
    }


def compare(np, direct, primary, protocol):
    expected = primary["solution"]
    errors = {}
    for key in protocol["scalar_comparison_keys"]:
        left, right = direct[key], expected[key]
        errors[key] = (0.0 if left is None and right is None else
                       1.0 if left is None or right is None else
                       normalized_scalar_difference(left, right))
    left, right = np.asarray(direct["singular_values"]), np.asarray(expected["singular_values"])
    singular_difference = float(np.linalg.norm(left-right) / max(
        1.0, float(np.linalg.norm(left)), float(np.linalg.norm(right))))
    exact = {key: direct[key] == expected[key] for key in protocol["exact_comparison_keys"]}
    # Conditioning is disclosed but is not assigned an extra threshold beyond
    # the frozen singular-value/rank/numerical contracts.
    condition_difference = normalized_scalar_difference(
        direct["retained_condition_number"], expected["retained_condition_number"])
    passed = (all(error <= protocol["normalized_comparison_tolerance"] for error in errors.values())
              and singular_difference <= protocol["normalized_comparison_tolerance"]
              and all(exact.values()))
    return {"passed": passed, "normalized_scalar_differences": errors,
            "normalized_singular_value_difference": singular_difference,
            "exact_comparisons": exact,
            "retained_condition_number_relative_difference_disclosed": condition_difference}


def replay_one(np, torch, source, inputs, primary, protocol, deadline, progress):
    from multiscale.bars import Bars, targets
    from multiscale.features import CloseWindows
    from multiscale.models import make_model

    cfg = source["cfg"]
    if not (cfg["model"] == "lstm" and cfg["variant"] == "anchor_baseline"
            and cfg["anchor"] == "observed" and cfg["output"] == "return"
            and cfg["target"] == "future_log_return" and cfg["loss"] == "path"
            and cfg["seed"] == 2026 and cfg["fold"] in (1, 2) and cfg["batch_size"] == 128):
        raise ValueError("State no longer matches the four-state frozen diagnostic")
    check_deadline(deadline)
    with np.load(source["split_file"], allow_pickle=False) as archive:
        fit = archive["fit"]
    if not (fit.dtype.kind == "i" and fit.ndim == 1 and len(fit) == source["fit_count"]
            and (np.diff(fit) > 0).all()):
        raise ValueError("Invalid frozen fit row sequence")
    if hashlib.sha256(np.asarray(fit, dtype="<i8").tobytes()).hexdigest() != source["fit_array_sha256"]:
        raise ValueError("Frozen fit origin digest changed")
    with np.load(source["source_bars"], allow_pickle=False) as archive:
        if set(archive.files) != {"start_ns", "end_ns", "close", "observed", "session_date"}:
            raise ValueError("Unexpected admitted source bar members")
        arrays = {name: archive[name] for name in archive.files}
    cutoff = np.datetime64("2026-07-01T04:00:00", "ns").astype("int64")
    if (np.any(arrays["session_date"] > "2026-06-30")
            or not (arrays["end_ns"][arrays["observed"]] < cutoff).all()):
        raise ValueError("Source bars exceed original pre-Q3 boundary")
    bars = Bars(cfg["frequency"], **arrays)
    windows = CloseWindows(bars, cfg["lookback"], load(inputs["source_admission_path"])["split_actions"])
    saved = torch.load(source["checkpoint"], map_location="cpu", weights_only=True)
    if (saved["identity"] != source["training_identity"] or saved["cfg"] != cfg
            or saved["epoch"] != source["best_epoch"] or saved["scaler"] != source["scaler_original"]):
        raise ValueError("Original checkpoint/scaler identity changed")
    net = make_model("lstm", cfg["lookback"], cfg["horizon"])
    net.load_state_dict(saved["model"], strict=True)
    if saved["parameters"] != sum(p.numel() for p in net.parameters()):
        raise ValueError("Checkpoint architecture parameter count changed")
    net.eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    h, n = cfg["horizon"], len(fit)
    x_dense, y_dense = np.empty((n, 34), dtype=np.float64), np.empty((n, h), dtype=np.float64)
    scaler = saved["scaler"]
    ym, ys = np.asarray(scaler["y_mean"]), np.asarray(scaler["y_std"])
    zero_prediction = -ym/ys
    b0 = np.vstack((net.head.weight.detach().numpy().T,
                    net.head.bias.detach().numpy())).astype(np.float64)
    captured = []

    def capture_head_inputs(_module, arguments):
        if len(arguments) != 1 or arguments[0].ndim != 2 or arguments[0].shape[1] != 33:
            raise ValueError("Original head hook did not expose exactly 33 inputs")
        captured.append(arguments[0].detach().numpy().astype(np.float64, copy=True))

    handle = net.head.register_forward_pre_hook(capture_head_inputs)
    native_sse, reconstruction_max = [], 0.0
    started = time.perf_counter()
    try:
        for start in range(0, n, 128):
            check_deadline(deadline)
            end = min(start+128, n)
            indices = fit[start:end]
            x, anchor = windows.arrays(indices)
            x = ((x-scaler["x_mean"])/scaler["x_std"]).astype(np.float32)
            anchor = ((anchor-scaler["anchor_mean"])/scaler["anchor_std"]).astype(np.float32)
            labels = targets(bars, indices, h, cfg["target"])
            y_dense[start:end] = ((labels-ym)/ys).astype(np.float32).astype(np.float64)
            captured.clear()
            # This is the independent representation route: original model
            # forward plus pre-hook, with no manual call to its recurrent core.
            with torch.inference_mode():
                native = net(torch.from_numpy(x[:, :, None]), torch.from_numpy(anchor)).numpy().astype(np.float64)
            if len(captured) != 1 or captured[0].shape != (end-start, 33):
                raise ValueError("Original head invocation count or feature shape changed")
            x_dense[start:end, :33], x_dense[start:end, 33] = captured[0], 1.0
            original = x_dense[start:end] @ b0
            denominator = max(1.0, float(np.abs(native).max()),
                              float((np.abs(x_dense[start:end]) @ np.abs(b0)).max()))
            discrepancy = float(np.max(np.abs(original-native))) / denominator
            if not math.isfinite(discrepancy) or discrepancy > 64*np.finfo(np.float32).eps:
                raise ValueError("Independent original-forward affine reconstruction failed")
            reconstruction_max = max(reconstruction_max, discrepancy)
            native_sse.append(float(np.sum((native-y_dense[start:end])**2, dtype=np.float64)))
            progress(source["id"], end, n)
    finally:
        handle.remove()
    replay_seconds = time.perf_counter()-started
    check_deadline(deadline)
    solved = dense_projection(np, x_dense, y_dense, b0, zero_prediction, deadline)
    compared = compare(np, solved, primary, protocol)
    native_difference = normalized_scalar_difference(
        math.fsum(native_sse), primary["totals"]["native_fp32_training_target_SSE"])
    compared["native_forward_SSE_normalized_difference"] = native_difference
    compared["passed"] &= native_difference <= protocol["normalized_comparison_tolerance"]
    return {"state_id": source["id"], "n_rows": n, "horizon": h,
            "replay_seconds": replay_seconds, "dense_X_Y_bytes": x_dense.nbytes+y_dense.nbytes,
            "dense_SVD_U_bytes": n*34*8,
            "maximum_affine_reconstruction_scaled_error": reconstruction_max,
            "solution": solved, "comparison": compared,
            "only_fit_targets_generated_or_scored": True, "whole_preQ3_bars_loaded": True,
            "no_dense_cache_or_coefficients_saved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=BASE/"INDEPENDENT_DENSE_REPLAY.json")
    args = parser.parse_args()
    destination = args.report.resolve()
    destination.relative_to(BASE)
    if set(os.sched_getaffinity(0)) != {26}:
        raise ValueError("Independent replay requires CPU26")
    for key, expected in {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                          "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": "-1"}.items():
        if os.environ.get(key) != expected:
            raise ValueError("Independent replay environment changed: "+key)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    protocol_path = BASE/"INDEPENDENT_REPLAY_PROTOCOL.json"
    protocol = load(protocol_path)
    if sha(Path(__file__)) != protocol["independent_code_sha256"]:
        raise ValueError("Independent replay source changed after freeze")
    for name, expected in protocol["bound_metadata_sha256"].items():
        if sha(BASE/name) != expected:
            raise ValueError("Bound metadata changed after independent replay freeze: "+name)
    scope = load(BASE/"STAGE_SCOPE.json")
    deadline = scope["deadline_epoch"]
    if deadline != protocol["stage_deadline_epoch"]:
        raise ValueError("Original stage deadline changed")
    check_deadline(deadline)
    admission = load(BASE/"MARKET_ADMISSION.json")
    if (admission["status"] != "admitted_four_fit_only_states"
            or admission["code_freeze_sha256"] != sha(BASE/"CODE_FREEZE.json")
            or admission["scope_sha256"] != sha(BASE/"STAGE_SCOPE.json")
            or admission["input_identities_sha256"] != sha(BASE/"INPUT_IDENTITIES.json")):
        raise ValueError("Root market admission is absent or not identity-bound")
    inputs = load(BASE/"INPUT_IDENTITIES.json")
    states = inputs["states"]
    if ([state["id"] for state in states] != protocol["state_ids"]
            or admission["state_ids"] != protocol["state_ids"]):
        raise ValueError("Frozen four-state list changed")
    primary_results, primary_hashes = {}, {}
    for state in states:
        path = BASE/"runs"/state["id"]/"RESULT.json"
        result = load(path)
        binding = result["binding"]
        if (result["status"] != "completed" or not result["market_fit_replay"]
                or result["stop_outer_scored"] or result["Q3_values_read"]
                or binding["provenance"]["state_id"] != state["id"]
                or binding["provenance"]["training_identity"] != state["training_identity"]
                or binding["provenance"]["selected_epoch"] != state["best_epoch"]
                or binding["scope_sha256"] != sha(BASE/"STAGE_SCOPE.json")
                or binding["code_freeze_sha256"] != sha(BASE/"CODE_FREEZE.json")
                or binding["provenance"]["input_identities_sha256"] != sha(BASE/"INPUT_IDENTITIES.json")
                or binding["stage_deadline_epoch"] != deadline
                or binding["rows"] != state["fit_count"] or binding["batch_size"] != 128):
            raise ValueError("All four primary fit-only results must complete before independent replay")
        primary_results[state["id"]] = result
        primary_hashes[state["id"]] = sha(path)
    # Verify only now, after four primary completions, before array/weight load.
    check_files(load(BASE/"CODE_FREEZE.json")["files"], deadline)
    check_files(inputs["verified_files"], deadline)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    sys.path.insert(0, inputs["reference_root"])
    from multiscale import bars, features, models
    for module in (bars, features, models):
        Path(module.__file__).resolve().relative_to(Path(inputs["reference_root"]).resolve())
    binding = {"protocol_sha256": sha(protocol_path),
               "input_identities_sha256": sha(BASE/"INPUT_IDENTITIES.json"),
               "scope_sha256": sha(BASE/"STAGE_SCOPE.json"),
               "code_freeze_sha256": sha(BASE/"CODE_FREEZE.json"),
               "market_admission_sha256": sha(BASE/"MARKET_ADMISSION.json"),
               "primary_results_sha256": primary_hashes, "stage_deadline_epoch": deadline}
    rows = []
    if destination.exists():
        previous = load(destination)
        if previous["binding"] != binding:
            raise ValueError("Independent replay report binding changed; no permissive restart")
        rows = previous["states"]
        if [row["state_id"] for row in rows] != protocol["state_ids"][:len(rows)]:
            raise ValueError("Independent replay completed-state cursor changed")
    report = {"status": "running", "binding": binding, "states": rows,
              "started_at_UTC": datetime.now(timezone.utc).isoformat(),
              "actual_python_executable": sys.executable, "NumPy_version": np.__version__,
              "CPU_affinity": sorted(os.sched_getaffinity(0)), "GPU": False,
              "new_market_states_or_training": False, "stop_outer_scored": False,
              "Q3_values_read": False, "new_predictor_saved": False,
              "independent_market_confirmation": False, "research_goal_achieved": False}

    def progress(state_id, cursor, total):
        report.update(current_state=state_id, transient_fit_cursor=cursor, current_state_rows=total,
                      in_memory_only_partial_state_replayed_from_zero_on_resume=True)
        atomic(destination, report)

    atomic(destination, report)
    try:
        for state in states[len(rows):]:
            check_deadline(deadline)
            row = replay_one(np, torch, state, inputs, primary_results[state["id"]],
                             protocol, deadline, progress)
            rows.append(row)
            report["states"] = rows
            report["all_completed_comparisons_passed"] = all(r["comparison"]["passed"] for r in rows)
            atomic(destination, report)
        report["status"] = ("completed_all_four_comparisons_passed" if
            all(row["comparison"]["passed"] for row in rows) else "completed_comparison_failure_retained")
    except InterruptedError as error:
        report["status"] = "partial_stopped_completed_scalar_states_preserved"
        report["stop_reason"] = str(error)
    except Exception as error:
        report["status"] = "error_completed_scalar_states_preserved"
        report["error_type"], report["error"] = type(error).__name__, str(error)
        atomic(destination, report)
        raise
    report["finished_at_UTC"] = datetime.now(timezone.utc).isoformat()
    atomic(destination, report)
    print(json.dumps({"status": report["status"], "completed_states": len(rows),
                      "report": str(destination)}, indent=2))
    return 0 if report["status"] == "completed_all_four_comparisons_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
