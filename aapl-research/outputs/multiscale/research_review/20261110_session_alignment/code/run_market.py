"""Frozen four-state, fit-internal session-alignment pipeline.

No computation runs at import. The worker first completes every prefix-only,
label-free feature gate. Only a unanimous frozen gate permits construction of
original fit labels. There is one continuation form and no recipe selection.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import alignment_core as core
from engineering import atomic_json, atomic_npz, validate_moments
from market_adapter import load_admitted
from market_guard import install_guard


BASE = Path(__file__).resolve().parents[1]
STOP = False
REQUIRED = {"block_rows": 256, "checkpoint_every_blocks": 16,
            "stationarity_relative_tolerance": 1e-10,
            "raw_variance_relative_tolerance": 1e-10,
            "summary_sd_floor": 1e-12,
            "summary_standardized_distance_floor": 1e-8,
            "summary_residual_eigenvalue_floor": 1e-8,
            "summary_pinv_relative_cutoff": 1e-10,
            "continuation_relative_margin": 1e-6}
MOMENTS = ("count", "sx", "sy", "xx", "xy")
METHODS = ("base", "real", "mirror", "zero", "prefix_mean")


class Paused(Exception):
    pass


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_sha(value):
    return hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def receive_signal(*_):
    global STOP
    STOP = True


def stop_reason(protocol):
    if STOP:
        return "signal"
    if time.time() >= float(protocol["deadline_epoch"]):
        return "deadline"
    return None


def maybe_pause(protocol):
    reason = stop_reason(protocol)
    if reason:
        raise Paused(reason)


def gate(protocol, protocol_path):
    if protocol.get("status") != "frozen_before_execution":
        raise ValueError("Execution requires a separately frozen market protocol")
    for key, value in REQUIRED.items():
        if key not in protocol or protocol[key] != value:
            raise ValueError("Missing or changed frozen gate: " + key)
    if (protocol.get("market_arrays_admitted") is not True
            or protocol.get("fit_labels_admitted") is not True
            or protocol.get("Q3_allowed", False) or protocol.get("network_allowed", False)):
        raise ValueError("Wrong market/fit-only admission")
    affinity = set(os.sched_getaffinity(0))
    if not affinity or not affinity.issubset({24, 26}) or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CPU24/26 only and GPU hidden are required")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Numerical thread count must be frozen to one: " + key)
    hashes = protocol["code_sha256"]
    for name in ("run_market.py", "alignment_core.py", "market_adapter.py", "market_guard.py", "engineering.py"):
        path = str(BASE / "code" / name)
        if path not in hashes:
            raise ValueError("Source missing from exact freeze: " + name)
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError("Frozen source changed: " + path)
    for path, expected in protocol.get("metadata_sha256", {}).items():
        if sha(path) != expected:
            raise ValueError("Frozen engineering/science metadata changed")
    deadline = float(protocol["deadline_epoch"])
    if not np.isfinite(deadline):
        raise ValueError("Finite immutable deadline required")
    output = Path(protocol["output_directory"])
    if not output.is_absolute() or not output.resolve().is_relative_to(BASE.resolve()):
        raise ValueError("Output must remain in this diagnostic stage")
    install_guard(protocol, protocol_path)
    output.mkdir(parents=True, exist_ok=True)
    return output, sha(protocol_path)


def state_identity(protocol_sha, profile, fold, prefix, tail):
    return hashlib.sha256(encoded({"protocol_sha256": protocol_sha, "profile": profile,
        "fold": fold, "prefix_sha256": index_sha(prefix), "tail_sha256": index_sha(tail)})).hexdigest()


def metadata(protocol, protocol_sha, identity, phase, cursor):
    return {"protocol_sha256": np.asarray(protocol_sha), "state_identity": np.asarray(identity),
            "phase": np.asarray(phase), "cursor": np.int64(cursor),
            "deadline_epoch": np.float64(protocol["deadline_epoch"])}


def verify_checkpoint(z, protocol, protocol_sha, identity, phase, total):
    if (str(z["protocol_sha256"]) != protocol_sha or str(z["state_identity"]) != identity
            or str(z["phase"]) != phase or float(z["deadline_epoch"]) != float(protocol["deadline_epoch"])):
        raise ValueError("Checkpoint identity/source/deadline mismatch")
    cursor = int(z["cursor"])
    if cursor < 0 or cursor > total or (cursor != total and cursor % protocol["block_rows"] != 0):
        raise ValueError("Checkpoint is not at an original complete-block boundary")
    return cursor


def load_stats(path, protocol, protocol_sha, identity, phase, total, p, horizon):
    stats, cursor = core.empty_stats(p, horizon), 0
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            cursor = verify_checkpoint(z, protocol, protocol_sha, identity, phase, total)
            stats = {key: z[key].copy() for key in MOMENTS}
        stats["count"] = int(stats["count"])
        count, actual_p, actual_h = core.validate_stats(stats)
        if count != cursor or actual_p != p or actual_h != horizon:
            raise ValueError("Statistics dimensions/count do not close at saved cursor")
        validate_moments(stats)
    return stats, cursor


def save_stats(path, stats, protocol, protocol_sha, identity, phase, cursor):
    atomic_npz(path, **stats, **metadata(protocol, protocol_sha, identity, phase, cursor))


def accumulate(data, prefix, folder, phase, protocol, protocol_sha, identity):
    """Fixed-block prefix moments; features phase never calls targets()."""
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "CHECKPOINT.npz"
    p = core.feature_dimensions(data.lookback)["total"]
    stats, cursor = load_stats(checkpoint, protocol, protocol_sha, identity, phase, len(prefix), p, data.horizon)
    try:
        maybe_pause(protocol)
        blocks = 0
        while cursor < len(prefix):
            maybe_pause(protocol)
            block = prefix[cursor:cursor + protocol["block_rows"]]
            X = data.batch_features(block)
            Y = np.zeros((len(block), data.horizon), dtype=np.float64) if phase == "features" else data.targets(block)
            core.update_stats(stats, X, Y)
            cursor += len(block)
            validate_moments(stats)
            blocks += 1
            reason = stop_reason(protocol)
            if (blocks % protocol["checkpoint_every_blocks"] == 0 or cursor == len(prefix) or reason):
                save_stats(checkpoint, stats, protocol, protocol_sha, identity, phase, cursor)
                atomic_json(folder / "STATUS.json", {"status": "paused" if reason else "running",
                    "phase": phase, "cursor": cursor, "total": len(prefix), "reason": reason,
                    "protocol_sha256": protocol_sha, "state_identity": identity,
                    "deadline_epoch": protocol["deadline_epoch"]})
            if reason:
                raise Paused(reason)
        return stats
    except (Paused, TimeoutError) as exc:
        save_stats(checkpoint, stats, protocol, protocol_sha, identity, phase, cursor)
        atomic_json(folder / "STATUS.json", {"status": "paused", "phase": phase,
            "cursor": cursor, "total": len(prefix), "reason": str(exc),
            "protocol_sha256": protocol_sha, "state_identity": identity,
            "deadline_epoch": protocol["deadline_epoch"]})
        raise Paused(str(exc)) from exc
    except (ValueError, np.linalg.LinAlgError) as exc:
        save_stats(checkpoint, stats, protocol, protocol_sha, identity, phase, cursor)
        atomic_json(folder / "STATUS.json", {"status": "numerically_unqualified", "phase": phase,
            "cursor": cursor, "total": len(prefix), "error_type": type(exc).__name__, "error": str(exc),
            "protocol_sha256": protocol_sha, "state_identity": identity,
            "deadline_epoch": protocol["deadline_epoch"]})
        raise


def summary_gate(stats, lookback, protocol):
    """No Y is inspected. Both summary columns are tested without selection."""
    n, p, _ = core.validate_stats(stats)
    if not n or np.any(stats["sy"] != 0) or np.any(stats["xy"] != 0):
        raise ValueError("Feature gate requires label-free zero-Y moments")
    validate_moments(stats)
    dims = core.feature_dimensions(lookback)
    if p != dims["total"]:
        raise ValueError("Wrong feature dimensions")
    mean = stats["sx"] / n
    variance = np.diag(stats["xx"]) / n - mean ** 2
    raw_sd = np.sqrt(np.maximum(variance, 0.0))
    sd = np.where(raw_sd > core.SCALE_FLOOR, raw_sd, 1.0)
    gram = (stats["xx"] / n - np.outer(mean, mean)) / sd[:, None] / sd[None, :]
    common = dims["common"]
    summary_cov = gram[common:, common:]
    minus = float(np.sqrt(max(0.0, summary_cov[0, 0] + summary_cov[1, 1] - 2 * summary_cov[0, 1])))
    plus = float(np.sqrt(max(0.0, summary_cov[0, 0] + summary_cov[1, 1] + 2 * summary_cov[0, 1])))
    base = gram[:common, :common]
    eigenvalues, vectors = np.linalg.eigh((base + base.T) / 2)
    largest = float(np.max(eigenvalues))
    cutoff = protocol["summary_pinv_relative_cutoff"] * largest if largest > 0 else 0.0
    keep = eigenvalues > cutoff
    projection = vectors[:, keep].T @ gram[:common, common:]
    explained = (projection / eigenvalues[keep, None]).T @ projection if np.any(keep) else np.zeros((2, 2))
    residual = summary_cov - explained
    residual = (residual + residual.T) / 2
    residual_eigenvalues = np.linalg.eigvalsh(residual)
    checks = {"real_sd": bool(raw_sd[common] > protocol["summary_sd_floor"]),
              "mirror_sd": bool(raw_sd[common + 1] > protocol["summary_sd_floor"]),
              "standardized_distinguishable_up_to_sign": bool(min(minus, plus) > protocol["summary_standardized_distance_floor"]),
              "both_summaries_residual_rank_two": bool(np.min(residual_eigenvalues) > protocol["summary_residual_eigenvalue_floor"])}
    return {"passed": all(checks.values()), "checks": checks, "count": n,
            "real_sd": float(raw_sd[common]), "mirror_sd": float(raw_sd[common + 1]),
            "standardized_summary_rms_difference": minus, "standardized_summary_rms_sum": plus,
            "baseline_gram_rank": int(np.sum(keep)), "baseline_feature_count": common,
            "baseline_largest_eigenvalue": largest, "baseline_minimum_eigenvalue": float(np.min(eigenvalues)),
            "baseline_pinv_cutoff": cutoff, "residual_summary_gram": residual.tolist(),
            "residual_summary_eigenvalues": residual_eigenvalues.tolist(),
            "raw_variance_minimum": float(np.min(variance)), "targets_constructed": False}


def score_tail(data, tail, models, folder, protocol, protocol_sha, identity):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "CHECKPOINT.npz"
    cursor = 0
    sse = np.zeros((len(METHODS), data.horizon), dtype=np.float64)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            cursor = verify_checkpoint(z, protocol, protocol_sha, identity, "tail", len(tail))
            if int(z["count"]) != cursor:
                raise ValueError("Tail count/cursor mismatch")
            sse = z["sse"].copy()
        if sse.shape != (len(METHODS), data.horizon) or not np.isfinite(sse).all() or np.any(sse < 0):
            raise ValueError("Invalid tail SSE checkpoint")

    def save():
        atomic_npz(path, count=np.int64(cursor), sse=sse,
                   **metadata(protocol, protocol_sha, identity, "tail", cursor))
    try:
        maybe_pause(protocol)
        blocks = 0
        while cursor < len(tail):
            maybe_pause(protocol)
            block = tail[cursor:cursor + protocol["block_rows"]]
            X, Y = data.batch_features(block), data.targets(block)
            predictions = {arm: core.predict(models[arm], X) for arm in core.ARMS}
            predictions["zero"] = np.zeros_like(Y)
            predictions["prefix_mean"] = np.broadcast_to(models["base"]["y_mean"], Y.shape)
            for i, name in enumerate(METHODS):
                sse[i] += np.sum((predictions[name] - Y) ** 2, axis=0)
            if not np.isfinite(sse).all():
                raise ValueError("Nonfinite tail SSE")
            cursor += len(block)
            blocks += 1
            reason = stop_reason(protocol)
            if blocks % protocol["checkpoint_every_blocks"] == 0 or cursor == len(tail) or reason:
                save()
                atomic_json(folder / "STATUS.json", {"status": "paused" if reason else "running",
                    "cursor": cursor, "total": len(tail), "reason": reason,
                    "protocol_sha256": protocol_sha, "state_identity": identity,
                    "deadline_epoch": protocol["deadline_epoch"]})
            if reason:
                raise Paused(reason)
    except (Paused, TimeoutError) as exc:
        save()
        raise Paused(str(exc)) from exc
    result = {name: {"endpoint_mse": float(sse[i, -1] / cursor),
                     "path_mse": float(np.sum(sse[i]) / (cursor * data.horizon)),
                     "per_step_mse": (sse[i] / cursor).tolist()}
              for i, name in enumerate(METHODS)}
    margin = protocol["continuation_relative_margin"]
    checks = {}
    for control in ("base", "mirror"):
        for metric in ("endpoint_mse", "path_mse"):
            checks[metric + "_vs_" + control] = result[control][metric] - result["real"][metric] > margin * result[control][metric]
    for control in ("zero", "prefix_mean"):
        checks["endpoint_mse_vs_" + control] = result[control]["endpoint_mse"] - result["real"]["endpoint_mse"] > margin * result[control]["endpoint_mse"]
    return {"methods": result, "checks": checks, "passed": all(checks.values()),
            "tail_count": cursor, "relative_margin": margin, "tail_checkpoint_sha256": sha(path),
            "tail_prediction_arrays_saved": False}


def fit_and_score(state, feature_stats, protocol, protocol_sha, output):
    data, prefix, tail, identity = state["data"], state["prefix"], state["tail"], state["identity"]
    folder = output / state["name"]
    stats = accumulate(data, prefix, folder / "fit", "fit", protocol, protocol_sha, identity)
    if not np.array_equal(stats["sx"], feature_stats["sx"]) or not np.array_equal(stats["xx"], feature_stats["xx"]):
        raise ValueError("Actual-label and feature-only prefix X moments differ bitwise")
    models, diagnostics = {}, {}
    for arm, cols in core.columns(data.lookback).items():
        maybe_pause(protocol)
        model = core.solve(stats, cols)
        d = model["diagnostics"]
        if not np.isfinite(d["normal_equation_relative_inf"]) or d["normal_equation_relative_inf"] > protocol["stationarity_relative_tolerance"]:
            raise ValueError("Frozen stationarity threshold failed for " + arm)
        diagnostics[arm] = d
        models[arm] = model
        atomic_npz(folder / "fit" / ("MODEL_" + arm + ".npz"),
                   **{key: value for key, value in model.items() if key != "diagnostics"},
                   protocol_sha256=np.asarray(protocol_sha), state_identity=np.asarray(identity))
    atomic_json(folder / "fit" / "SOLVER_DIAGNOSTICS.json", {"protocol_sha256": protocol_sha,
        "state_identity": identity, "models": diagnostics,
        "feature_moments_bitwise_equal": True, "named_models_are_not_formal_candidates": True})
    scores = score_tail(data, tail, models, folder / "tail", protocol, protocol_sha, identity)
    result = {"status": "completed_fit_internal_diagnostic", "name": state["name"],
              "profile": data.profile, "fold": state["fold"], "state_identity": identity,
              "protocol_sha256": protocol_sha, "prefix_count": len(prefix),
              "feature_moments_bitwise_equal": True, "solver_diagnostics": diagnostics, **scores,
              "continuation_form": "real_only", "candidate_admitted": False,
              "independent_confirmation": False, "Q3_read": False}
    atomic_json(folder / "RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    output, protocol_sha = gate(protocol, args.protocol)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, receive_signal)
    lock = (output / "WORKER_LOCK.json").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("Another worker already owns this diagnostic output") from exc
    states, features_by_name, feature_results, results = [], {}, [], []
    labels_started = False
    active = {"phase": "admission"}
    try:
        maybe_pause(protocol)
        for profile in ("minute", "hourly"):
            data = load_admitted(profile, protocol)
            atomic_json(output / (profile + "_INPUT_RECEIPT.json"), data.receipt)
            for fold in (1, 2):
                fit = data.fit(fold)
                prefix, tail, boundary = core.partition_fit(fit, data.end_ns, data.horizon)
                name = profile + "_f" + str(fold)
                identity = state_identity(protocol_sha, profile, fold, prefix, tail)
                state = {"name": name, "data": data, "fold": fold, "prefix": prefix, "tail": tail, "identity": identity}
                states.append(state)
                atomic_json(output / name / "PARTITION.json", {"protocol_sha256": protocol_sha,
                    "state_identity": identity, "original_fit_count": len(fit), "original_boundary": boundary,
                    "first_tail_origin": int(tail[0]), "prefix_count": len(prefix), "tail_count": len(tail),
                    "prefix_sha256": index_sha(prefix), "tail_sha256": index_sha(tail),
                    "last_prefix_label_end_ns": int(data.end_ns[prefix[-1] + data.horizon]),
                    "first_tail_origin_end_ns": int(data.end_ns[tail[0]]),
                    "no_feature_availability_row_removal": True})
        for state in states:
            active = {"phase": "features", "state": state["name"]}
            folder = output / state["name"] / "features"
            try:
                stats = accumulate(state["data"], state["prefix"], folder, "features", protocol, protocol_sha, state["identity"])
                features_by_name[state["name"]] = stats
                maybe_pause(protocol)
                result = summary_gate(stats, state["data"].lookback, protocol)
            except (ValueError, np.linalg.LinAlgError) as exc:
                result = {"passed": False, "status": "numerically_unqualified", "targets_constructed": False,
                          "error_type": type(exc).__name__, "error": str(exc)}
            result.update({"state": state["name"], "state_identity": state["identity"],
                           "protocol_sha256": protocol_sha, "checkpoint_sha256": sha(folder / "CHECKPOINT.npz")})
            atomic_json(folder / "FEATURE_GATE.json", result)
            feature_results.append(result)
        admitted = len(feature_results) == 4 and all(r["passed"] for r in feature_results)
        global_features = {"status": "passed" if admitted else "rejected", "protocol_sha256": protocol_sha,
                           "deadline_epoch": protocol["deadline_epoch"], "states": feature_results,
                           "all_four_states_required": True, "targets_constructed_before_gate": False}
        atomic_json(output / "GLOBAL_FEATURE_GATE.json", global_features)
        if not admitted:
            atomic_json(output / "RESULTS.json", {"status": "feature_gate_rejected", "protocol_sha256": protocol_sha,
                "feature_states": feature_results, "targets_constructed": False, "continuation_supported": False,
                "candidate_admitted": False, "research_goal_achieved": False, "Q3_read": False})
            atomic_json(output / "STATUS.json", {"status": "feature_gate_rejected", "protocol_sha256": protocol_sha,
                "deadline_epoch": protocol["deadline_epoch"], "targets_constructed": False})
            return
        for state in states:
            active = {"phase": "fit_and_tail", "state": state["name"]}
            maybe_pause(protocol)
            labels_started = True
            try:
                result = fit_and_score(state, features_by_name[state["name"]], protocol, protocol_sha, output)
            except (ValueError, np.linalg.LinAlgError) as exc:
                result = {"status": "numerically_unqualified", "name": state["name"], "passed": False,
                          "protocol_sha256": protocol_sha, "state_identity": state["identity"],
                          "error_type": type(exc).__name__, "error": str(exc), "candidate_admitted": False}
                atomic_json(output / state["name"] / "RESULT.json", result)
            results.append(result)
        maybe_pause(protocol)
        continuation = len(results) == 4 and all(r["passed"] for r in results)
        final = {"status": "completed_fit_internal_diagnostic", "protocol_sha256": protocol_sha,
                 "deadline_epoch": protocol["deadline_epoch"], "states": results,
                 "feature_gate_sha256": sha(output / "GLOBAL_FEATURE_GATE.json"),
                 "continuation_supported": continuation, "unique_continuation_form": "real_only",
                 "four_nested_development_states_are_not_independent_replication": True,
                 "candidate_admitted": False, "independent_confirmation": False,
                 "research_goal_achieved": False, "Q3_read": False}
        atomic_json(output / "RESULTS.json", final)
        atomic_json(output / "STATUS.json", {"status": final["status"], "protocol_sha256": protocol_sha,
            "deadline_epoch": protocol["deadline_epoch"], "targets_constructed": labels_started,
            "completed_states": len(results), "continuation_supported": continuation})
        print(json.dumps({"status": final["status"], "states": len(results), "continuation_supported": continuation}))
    except (Paused, TimeoutError) as exc:
        atomic_json(output / "STATUS.json", {"status": "paused", "reason": str(exc), **active,
            "protocol_sha256": protocol_sha, "deadline_epoch": protocol["deadline_epoch"],
            "completed_feature_states": len(feature_results), "completed_scored_states": len(results),
            "targets_constructed": labels_started, "checkpoints_retained": True,
            "deadline_must_not_reset_on_resume": True})
    except BaseException as exc:
        atomic_json(output / "STATUS.json", {"status": "failed", "error_type": type(exc).__name__,
            "error": str(exc), **active, "protocol_sha256": protocol_sha,
            "deadline_epoch": protocol["deadline_epoch"], "targets_constructed": labels_started,
            "checkpoints_retained": True, "research_goal_achieved": False})
        raise
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


if __name__ == "__main__":
    main()
