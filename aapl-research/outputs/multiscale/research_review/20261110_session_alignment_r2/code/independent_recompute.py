"""Independent direct-centered verification of all four fixed historical states.

Only access control and atomic JSON writing are reused. No numerical function
from the main core, adapter, or worker is called, and no saved model is read.
All prices remain the registered pre-Q3 caches. Split ZIP decoding is fit.npy
only. Descriptive error identities were requested after main MSE was known;
they cannot select, repair, or revive the failed hypothesis.

CLI: --protocol /absolute/path/to/INDEPENDENT_PROTOCOL.json
The protocol copies the parent's inputs and scientific constants, adds this
source to code_sha256, and registers parent_protocol_path, main_results_path,
and main_feature_gate_path in metadata_sha256. The absolute deadline is shared.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

from engineering import atomic_json
from market_guard import install_guard


BASE = Path(__file__).resolve().parents[1]
ROOT = Path("/home/ayaka/wslcode/aapl-research")
SOURCE = ROOT / "outputs/multiscale/admission/alpaca_20261107T052228573242Z"
CACHE = ROOT / "outputs/multiscale/controlled_followup/20261108_v2/cache"
NY = ZoneInfo("America/New_York")
DEADLINE = 1794249085.7666245
RIDGE = 0.1
METRIC_RTOL = 1e-6
IDENTITY_RTOL = 1e-10
STOP = False
METHODS = ("base", "real", "mirror", "zero", "prefix_mean")
ARMS = METHODS[:3]
ADMISSION_SHA = "f6fc738205532cf5016d74b2baa4691692cb03674c3db9b6a8ccf8ce8126ae86"
SPLITS = [{"date": "2020-08-31", "ratio": 4.0}]
CONSTANTS = {
    "block_rows": 256, "checkpoint_every_blocks": 16,
    "stationarity_relative_tolerance": 1e-10,
    "raw_variance_relative_tolerance": 1e-10,
    "summary_sd_floor": 1e-12,
    "summary_standardized_distance_floor": 1e-8,
    "summary_residual_eigenvalue_floor": 1e-8,
    "summary_pinv_relative_cutoff": 1e-10,
    "continuation_relative_margin": 1e-6,
}
SPECS = {
    "minute": (480, 60, 60_000_000_000, "bars_minute.npz",
               "c81d57cd1e190456e1fdbf59d7493c915704551fa2e6c5870b5bd66f4cd94332"),
    "hourly": (96, 8, 3_600_000_000_000, "bars_hour.npz",
               "c41851d735d179c3243dd3356204d17845b55289c56c59f9e170c0b5f432881c"),
}
FITS = {
    "minute": {
        "1": (64261, "ae69a8e1a37676403b71d91f0f949de5bdb196e2caf31191d448b3ab7ccff326"),
        "2": (128761, "93c5608c9b1ff3b3d1c724589dfd7d049e040405f7a716c59f57218a7ba2808b"),
    },
    "hourly": {
        "1": (1139, "38079b182cd2a01409cb3649596e1838c68fc509652b2d201efc975cc2aeb42a"),
        "2": (2294, "ad5db4224a6fae358341f6e6576deb0d4dddb0489089b76991b1e82758a83bb9"),
    },
}


class Paused(Exception):
    pass


def stop(*_):
    global STOP
    STOP = True


def check(p):
    if STOP or time.time() >= float(p["deadline_epoch"]):
        raise Paused("signal_or_unchanged_absolute_deadline")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path, p=None):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            if p is not None:
                check(p)
            h.update(part)
    return h.hexdigest()


def index_sha(x):
    return hashlib.sha256(np.asarray(x, dtype="<i8").tobytes()).hexdigest()


def state_identity(parent_sha, profile, fold, prefix, tail):
    item = {"protocol_sha256": parent_sha, "profile": profile, "fold": fold,
            "prefix_sha256": index_sha(prefix), "tail_sha256": index_sha(tail)}
    raw = json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def finite(x, name):
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite independent " + name)


def registered_json(p, key):
    path = Path(p[key]).resolve()
    expected = p.get("metadata_sha256", {}).get(str(path))
    if path.suffix.lower() != ".json" or not expected or sha(path, p) != expected:
        raise ValueError("Unregistered immutable JSON: " + key)
    return path


def gate(p, protocol_path):
    if (p.get("status") != "frozen_before_execution"
            or p.get("independent_recompute_admitted") is not True
            or p.get("market_arrays_admitted") is not True
            or p.get("fit_labels_admitted") is not True
            or p.get("Q3_allowed", False) or p.get("network_allowed", False)):
        raise ValueError("Separate frozen independent fit-only admission required")
    if float(p["deadline_epoch"]) != DEADLINE:
        raise ValueError("The existing absolute deadline cannot change")
    check(p)
    parent_path = registered_json(p, "parent_protocol_path")
    if sha(parent_path, p) != p["parent_protocol_sha256"]:
        raise ValueError("Parent protocol identity mismatch")
    parent = read(parent_path)
    for key in ("deadline_epoch", "source_root", "bars", "fits", "split_actions"):
        if p[key] != parent[key]:
            raise ValueError("Independent source/budget differs from parent: " + key)
    for key, value in CONSTANTS.items():
        if p.get(key) != value or parent.get(key) != value:
            raise ValueError("Independent scientific constant changed: " + key)
    if Path(p["source_root"]).resolve() != SOURCE or p["split_actions"] != SPLITS:
        raise ValueError("Wrong source/split registration")
    own = str(Path(__file__).resolve())
    if own not in p["code_sha256"]:
        raise ValueError("Independent source is missing from manifest")
    for mapping in (p["code_sha256"], p.get("metadata_sha256", {}),
                    parent["code_sha256"], parent.get("metadata_sha256", {})):
        for path, expected in mapping.items():
            if Path(path).suffix.lower() not in (".py", ".json"):
                raise ValueError("Source metadata must not hide an archive/weight input")
            if sha(path, p) != expected:
                raise ValueError("Frozen source/metadata changed: " + path)
    main_path = registered_json(p, "main_results_path")
    features_path = registered_json(p, "main_feature_gate_path")
    expected_output = Path(parent["output_directory"]).resolve()
    if main_path != expected_output / "RESULTS.json" or features_path != expected_output / "GLOBAL_FEATURE_GATE.json":
        raise ValueError("Main result paths differ from the parent output")
    main, gates = read(main_path), read(features_path)
    if (main.get("status") != "completed_fit_internal_diagnostic"
            or len(main.get("states", [])) != 4
            or main.get("protocol_sha256") != p["parent_protocol_sha256"]
            or gates.get("protocol_sha256") != p["parent_protocol_sha256"]
            or main.get("feature_gate_sha256") != sha(features_path, p)
            or gates.get("status") != "passed" or len(gates.get("states", [])) != 4):
        raise ValueError("Main completed four-state evidence is not correctly bound")
    output = Path(p["output_directory"]).resolve()
    if output == expected_output or output.is_relative_to(expected_output):
        raise ValueError("Independent outputs must not overwrite parent evidence")
    access = install_guard(p, protocol_path)
    output.mkdir(parents=True, exist_ok=True)
    return parent, main, gates, output, access


def load_source(profile, p):
    """Independent original-cache decoder; no old target or model artifact."""
    L, H, native, filename, expected_sha = SPECS[profile]
    if sha(SOURCE / "admission.json", p) != ADMISSION_SHA:
        raise ValueError("Original admission changed")
    admission = read(SOURCE / "admission.json")
    if admission["split_actions"] != SPLITS:
        raise ValueError("Original split schedule changed")
    rec = p["bars"][profile]
    path = Path(rec["path"]).resolve()
    if (path != SOURCE / filename or rec["sha256"] != expected_sha
            or admission["files"][filename] != expected_sha or sha(path, p) != expected_sha):
        raise ValueError("Wrong original bar file")
    keys = {"start_ns", "end_ns", "close", "observed", "session_date"}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != keys or len(archive.files) != len(keys):
            raise ValueError("Unexpected bar ZIP members")
        bars = {key: archive[key].copy() for key in keys}
    check(p)
    n = len(bars["close"])
    if not n or any(a.ndim != 1 or len(a) != n for a in bars.values()):
        raise ValueError("Invalid independent bar shapes")
    if (bars["observed"].dtype != np.dtype("bool")
            or bars["close"].dtype != np.dtype("float64")
            or bars["session_date"].dtype.kind != "U"
            or bars["start_ns"].dtype != np.dtype("int64")
            or bars["end_ns"].dtype != np.dtype("int64")):
        raise ValueError("Invalid original bar dtypes")
    if (np.any(np.diff(bars["end_ns"]) <= 0) or np.any(np.diff(bars["start_ns"]) <= 0)
            or np.any(bars["end_ns"] - bars["start_ns"] != native)
            or np.any(bars["session_date"] > "2026-06-30")
            or np.any(bars["session_date"][1:] < bars["session_date"][:-1])):
        raise ValueError("Invalid pre-Q3 ordered complete bars")
    observed = bars["observed"]
    finite(bars["close"][observed], "observed prices")
    if np.any(bars["close"][observed] <= 0):
        raise ValueError("Nonpositive observed close")
    fits = {}
    if set(p["fits"][profile]) != {"1", "2"}:
        raise ValueError("Exactly original folds 1 and 2 required")
    for fold in ("1", "2"):
        rec = p["fits"][profile][fold]
        count, expected_fit_sha = FITS[profile][fold]
        path = Path(rec["path"]).resolve()
        wanted = CACHE / (profile + "_h" + str(H) + "_future_log_return") / ("fold_" + fold + ".npz")
        if (path != wanted or rec["count"] != count or rec["fit_sha256"] != expected_fit_sha
                or sha(path, p) != rec["file_sha256"]):
            raise ValueError("Original fit registration changed")
        with zipfile.ZipFile(path, "r") as archive:
            if archive.namelist().count("fit.npy") != 1:
                raise ValueError("Ambiguous fit member")
            with archive.open("fit.npy", "r") as member:
                fit = np.load(io.BytesIO(member.read()), allow_pickle=False)
        if (fit.dtype != np.dtype("int64") or fit.ndim != 1 or len(fit) != count
                or np.any(np.diff(fit) <= 0) or index_sha(fit) != expected_fit_sha
                or np.any(fit < L - 1) or np.any(fit + H >= n)):
            raise ValueError("Independent canonical fit verification failed")
        fits[int(fold)] = fit
    bad = np.r_[0, np.cumsum(~observed, dtype=np.int64)]
    for fit in fits.values():
        if np.any(bad[fit + H + 1] != bad[fit - L + 1]):
            raise ValueError("Original rows unavailable; do not drop them")
        if profile == "minute" and np.any(bars["session_date"][fit] != bars["session_date"][fit + H]):
            raise ValueError("Original minute path crosses session")
    logs = np.full(n, np.nan, dtype=np.float64)
    logs[observed] = np.log(bars["close"][observed])
    split = np.zeros(n, dtype=np.float64)
    for action in SPLITS:
        split[bars["session_date"] >= action["date"]] += np.log(action["ratio"])
    return {"profile": profile, "L": L, "H": H, "native": native,
            "bars": bars, "fits": fits, "log_close": logs, "split": split}


def partitions(data, fit):
    boundary = len(fit) * 4 // 5
    tail = fit[boundary:].copy()
    first = data["bars"]["end_ns"][tail[0]]
    prefix = fit[:boundary]
    prefix = prefix[data["bars"]["end_ns"][prefix + data["H"]] < first].copy()
    if not len(prefix) or not len(tail):
        raise ValueError("Empty strictly purged independent partition")
    return prefix, tail, boundary


def feature_block(data, origins):
    L = data["L"]
    bars = data["bars"]
    positions = origins[:, None] + np.arange(1 - L, 1)
    # Preserve the original split-adjustment arithmetic order exactly.
    x = (data["log_close"][positions] + data["split"][positions]
         - data["split"][origins, None] - data["log_close"][origins, None])
    end = bars["end_ns"][positions]
    labels = bars["session_date"][positions]
    mask = (labels[:, 1:] != labels[:, :-1]).astype(np.float64)
    elapsed = np.diff(end, axis=1)
    if np.any(elapsed <= 0):
        raise ValueError("Non-increasing historical input")
    gap = np.log(elapsed.astype(np.float64) / data["native"])
    civil = []
    for stamp, label in zip(bars["end_ns"][origins], bars["session_date"][origins]):
        seconds, remainder = divmod(int(stamp), 1_000_000_000)
        local = datetime.fromtimestamp(seconds, timezone.utc).astimezone(NY)
        if remainder or local.date().isoformat() != str(label) or local.weekday() > 4:
            raise ValueError("Independent completed-time calendar disagreement")
        minute = local.hour * 60 + local.minute + local.second / 60.0
        if not 570 < minute <= 960:
            raise ValueError("Independent clock outside regular completed interval")
        civil.append((minute, local.weekday()))
    civil = np.asarray(civil, dtype=np.float64)
    u = (civil[:, 0] - 570.0) / 390.0
    clock = np.column_stack((np.sin(2*np.pi*u), np.cos(2*np.pi*u),
                            np.sin(4*np.pi*u), np.cos(4*np.pi*u),
                            *[(civil[:, 1] == day).astype(np.float64) for day in range(4)]))
    changes = x[:, 1:] - x[:, :-1]
    real = np.einsum("ij,ij->i", mask, changes, optimize=False)
    mirror = np.einsum("ij,ij->i", mask[:, ::-1], changes, optimize=False)
    X = np.column_stack((x, mask, gap, clock, real, mirror))
    if X.shape != (len(origins), 3 * L + 8) or np.any(np.abs(x[:, -1]) > 1e-12):
        raise ValueError("Independent complete input dimensions/anchor changed")
    finite(X, "features")
    return X


def full_prefix(data, prefix, p):
    X = np.empty((len(prefix), 3 * data["L"] + 8), dtype=np.float64)
    for start in range(0, len(prefix), 256):
        check(p)
        X[start:start + 256] = feature_block(data, prefix[start:start + 256])
    mean = X.mean(axis=0)
    raw_sd = X.std(axis=0, ddof=0)
    scale = np.where(raw_sd > 1e-12, raw_sd, 1.0)
    X -= mean
    X /= scale
    finite(X, "directly centered standardized prefix")
    check(p)
    gram = X.T @ X / len(X)
    finite(gram, "direct centered Gram")
    check(p)
    return X, mean, scale, raw_sd, gram


def feature_gate(gram, raw_sd, L, n, p):
    common = 3 * L + 6
    cov = gram[common:, common:]
    difference = float(np.sqrt(max(0.0, cov[0, 0] + cov[1, 1] - 2*cov[0, 1])))
    sum_distance = float(np.sqrt(max(0.0, cov[0, 0] + cov[1, 1] + 2*cov[0, 1])))
    block = (gram[:common, :common] + gram[:common, :common].T) / 2
    check(p)
    eigenvalues, vectors = np.linalg.eigh(block)
    check(p)
    largest = float(eigenvalues[-1])
    cutoff = max(largest, 0.0) * p["summary_pinv_relative_cutoff"]
    selected = eigenvalues > cutoff
    coordinates = vectors[:, selected].T @ gram[:common, common:]
    projected = ((coordinates / eigenvalues[selected, None]).T @ coordinates
                 if np.any(selected) else np.zeros((2, 2)))
    residual = cov - projected
    residual = (residual + residual.T) / 2
    values = np.linalg.eigvalsh(residual)
    tests = {
        "real_sd": bool(raw_sd[common] > p["summary_sd_floor"]),
        "mirror_sd": bool(raw_sd[common + 1] > p["summary_sd_floor"]),
        "standardized_distinguishable_up_to_sign": bool(min(difference, sum_distance) > p["summary_standardized_distance_floor"]),
        "both_summaries_residual_rank_two": bool(values.min() > p["summary_residual_eigenvalue_floor"]),
    }
    return {"passed": all(tests.values()), "checks": tests, "count": n,
            "real_sd": float(raw_sd[common]), "mirror_sd": float(raw_sd[common + 1]),
            "standardized_summary_rms_difference": difference,
            "standardized_summary_rms_sum": sum_distance,
            "baseline_gram_rank": int(selected.sum()),
            "baseline_pinv_cutoff": cutoff,
            "residual_summary_gram": residual.tolist(),
            "residual_summary_eigenvalues": values.tolist(), "targets_constructed": False}


def target_block(data, origins):
    future = origins[:, None] + np.arange(1, data["H"] + 1)
    y = np.log(data["bars"]["close"][future] / data["bars"]["close"][origins, None])
    finite(y, "original nominal cumulative target")
    return y


def fit_direct(data, prefix, p):
    Z, mean, scale, raw_sd, gram = full_prefix(data, prefix, p)
    Y = np.empty((len(prefix), data["H"]), dtype=np.float64)
    for start in range(0, len(prefix), 256):
        check(p)
        Y[start:start + 256] = target_block(data, prefix[start:start + 256])
    intercept = Y.mean(axis=0)
    Y -= intercept
    check(p)
    rhs = Z.T @ Y / len(prefix)
    del Z, Y
    finite(rhs, "direct centered RHS")
    common = 3 * data["L"] + 6
    common_cols = np.arange(common)
    choices = {"base": common_cols, "real": np.r_[common_cols, common],
               "mirror": np.r_[common_cols, common + 1]}
    models, diagnostics = {}, {}
    for arm, cols in choices.items():
        check(p)
        A = gram[np.ix_(cols, cols)] + RIDGE * np.eye(len(cols))
        R = rhs[cols]
        coef = np.linalg.solve(A, R)
        finite(coef, "direct coefficients")
        residual = A @ coef - R
        denom = float(np.linalg.norm(A, ord=np.inf) * np.linalg.norm(coef, ord=np.inf)
                      + np.linalg.norm(R, ord=np.inf))
        absolute = float(np.linalg.norm(residual, ord=np.inf))
        backward = absolute / denom if denom > 0 else (0.0 if absolute == 0 else float("inf"))
        if not np.isfinite(backward) or backward > p["stationarity_relative_tolerance"]:
            raise ValueError("Independent direct solve is numerically unqualified")
        models[arm] = {"columns": cols, "mean": mean[cols], "scale": scale[cols],
                       "coef": coef, "intercept": intercept}
        diagnostics[arm] = {"relative_backward_error": backward, "absolute_residual_inf": absolute,
                            "feature_count": len(cols), "prefix_count": len(prefix),
                            "coef_float64_sha256": hashlib.sha256(np.asarray(coef, dtype="<f8").tobytes()).hexdigest()}
        check(p)
    return models, diagnostics


def vectors(values):
    array = np.asarray(values, dtype=np.float64)
    finite(array, "descriptive identity")
    return {"per_step": array.tolist(), "endpoint": float(array[-1]), "path": float(array.mean())}


def tail_scores(data, tail, models, p):
    H = data["H"]
    sse = np.zeros((5, H), dtype=np.float64)
    p2 = np.zeros((3, H), dtype=np.float64)
    py = np.zeros_like(p2)
    delta2 = np.zeros((2, H), dtype=np.float64)
    delta_residual = np.zeros_like(delta2)
    for start in range(0, len(tail), 256):
        check(p)
        origins = tail[start:start + 256]
        X, Y = feature_block(data, origins), target_block(data, origins)
        predictions = {}
        for arm in ARMS:
            m = models[arm]
            predictions[arm] = m["intercept"] + ((X[:, m["columns"]] - m["mean"]) / m["scale"]) @ m["coef"]
        predictions["zero"] = np.zeros_like(Y)
        predictions["prefix_mean"] = np.broadcast_to(models["base"]["intercept"], Y.shape)
        for i, arm in enumerate(METHODS):
            prediction = predictions[arm]
            finite(prediction, "tail prediction")
            sse[i] += np.sum((prediction - Y) ** 2, axis=0)
            if i < 3:
                p2[i] += np.sum(prediction ** 2, axis=0)
                py[i] += np.sum(prediction * Y, axis=0)
        for i, arm in enumerate(("base", "mirror")):
            delta = predictions["real"] - predictions[arm]
            delta2[i] += np.sum(delta ** 2, axis=0)
            delta_residual[i] += np.sum(delta * (Y - predictions[arm]), axis=0)
        for array in (sse, p2, py, delta2, delta_residual):
            finite(array, "tail accumulated moment")
    mse = sse / len(tail)
    metrics = {arm: {"endpoint_mse": float(mse[i, -1]), "path_mse": float(mse[i].mean()),
                     "per_step_mse": mse[i].tolist()} for i, arm in enumerate(METHODS)}
    checks = {}
    margin = p["continuation_relative_margin"]
    for control in ("base", "mirror"):
        for metric in ("endpoint_mse", "path_mse"):
            ref = metrics[control][metric]
            checks[metric + "_vs_" + control] = bool(ref - metrics["real"][metric] > margin * ref)
    for control in ("zero", "prefix_mean"):
        ref = metrics[control]["endpoint_mse"]
        checks["endpoint_mse_vs_" + control] = bool(ref - metrics["real"]["endpoint_mse"] > margin * ref)
    description = {"posthoc_after_main_MSE_known": True, "used_for_selection": False,
                   "economic_causal_interpretation": False, "identity_relative_tolerance": IDENTITY_RTOL,
                   "prediction_vs_zero": {}, "real_vs_control": {}}
    identity_passes = []
    for i, arm in enumerate(ARMS):
        lhs = mse[i] - mse[3]
        rhs = p2[i] / len(tail) - 2 * py[i] / len(tail)
        residual = lhs - rhs
        reference = np.maximum(np.maximum(np.abs(mse[i]), np.abs(mse[3])), 1e-15)
        relative = np.abs(residual) / reference
        passed = bool(np.all(relative <= IDENTITY_RTOL))
        identity_passes.append(passed)
        description["prediction_vs_zero"][arm] = {
            "E_prediction_squared": vectors(p2[i] / len(tail)),
            "E_prediction_times_target": vectors(py[i] / len(tail)),
            "MSE_difference": vectors(lhs), "identity_rhs": vectors(rhs),
            "identity_residual": vectors(residual), "max_relative_identity_residual": float(relative.max()),
            "identity_verified": passed}
    for i, arm in enumerate(("base", "mirror")):
        control = METHODS.index(arm)
        lhs = mse[1] - mse[control]
        rhs = delta2[i] / len(tail) - 2 * delta_residual[i] / len(tail)
        residual = lhs - rhs
        reference = np.maximum(np.maximum(np.abs(mse[1]), np.abs(mse[control])), 1e-15)
        relative = np.abs(residual) / reference
        passed = bool(np.all(relative <= IDENTITY_RTOL))
        identity_passes.append(passed)
        description["real_vs_control"][arm] = {
            "E_delta_squared": vectors(delta2[i] / len(tail)),
            "E_delta_times_control_residual": vectors(delta_residual[i] / len(tail)),
            "MSE_real_minus_control": vectors(lhs), "identity_rhs": vectors(rhs),
            "identity_residual": vectors(residual), "max_relative_identity_residual": float(relative.max()),
            "identity_verified": passed}
    description["all_identities_verified"] = all(identity_passes)
    return metrics, checks, description


def compare_metrics(ours, main):
    comparisons = {}
    for arm in METHODS:
        comparisons[arm] = {}
        for key in ("endpoint_mse", "path_mse", "per_step_mse"):
            actual = np.asarray(ours[arm][key], dtype=np.float64)
            expected = np.asarray(main[arm][key], dtype=np.float64)
            if actual.shape != expected.shape:
                raise ValueError("Independent metric shapes differ")
            relative = np.abs(actual - expected) / np.maximum(expected, 1e-15)
            comparisons[arm][key] = {"max_relative_error": float(relative.max()),
                                     "within_tolerance": bool(np.all(relative <= METRIC_RTOL))}
    passed = all(item["within_tolerance"] for method in comparisons.values() for item in method.values())
    return passed, comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    p = read(protocol_path)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    parent, main_result, main_gate, output, access = gate(p, protocol_path)
    protocol_sha = sha(protocol_path, p)
    result = {"status": "running_independent_direct_centered_verification",
              "protocol_sha256": protocol_sha, "parent_protocol_sha256": p["parent_protocol_sha256"],
              "main_results_sha256": sha(p["main_results_path"], p),
              "deadline_epoch": p["deadline_epoch"], "deadline_reset": False,
              "metric_relative_tolerance": METRIC_RTOL, "new_seed_or_search": False,
              "old_models_or_weights_read": False, "Q3_read": False,
              "posthoc_error_decomposition_after_main_MSE_known": True,
              "posthoc_error_decomposition_used_for_selection": False,
              "feature_states": [], "states": [], "candidate_admitted": False,
              "independent_confirmation": False, "research_goal_achieved": False}
    try:
        check(p)
        sources = {profile: load_source(profile, p) for profile in ("minute", "hourly")}
        states = []
        for profile in ("minute", "hourly"):
            data = sources[profile]
            for fold in (1, 2):
                prefix, tail, boundary = partitions(data, data["fits"][fold])
                identity = state_identity(p["parent_protocol_sha256"], profile, fold, prefix, tail)
                expected = next(r for r in main_result["states"] if r["profile"] == profile and r["fold"] == fold)
                if (identity != expected["state_identity"] or len(prefix) != expected["prefix_count"]
                        or len(tail) != expected["tail_count"]):
                    raise ValueError("Independent original partition does not match main identity")
                states.append({"profile": profile, "fold": fold, "name": profile + "_f" + str(fold),
                               "data": data, "prefix": prefix, "tail": tail,
                               "identity": identity, "expected": expected, "boundary": boundary})
        # Complete every direct, prefix-only feature gate before constructing Y.
        for state in states:
            check(p)
            X, mean, scale, raw_sd, gram = full_prefix(state["data"], state["prefix"], p)
            gate_result = feature_gate(gram, raw_sd, state["data"]["L"], len(X), p)
            del X, mean, scale, raw_sd, gram
            expected = next(r for r in main_gate["states"] if r["state"] == state["name"])
            gate_result.update({"state": state["name"], "state_identity": state["identity"],
                "checks_match_main": gate_result["checks"] == expected["checks"],
                "passed_matches_main": gate_result["passed"] == expected["passed"],
                "prefix_count": len(state["prefix"]), "tail_count": len(state["tail"]),
                "prefix_sha256": index_sha(state["prefix"]), "tail_sha256": index_sha(state["tail"])})
            result["feature_states"].append(gate_result)
            atomic_json(output / "PROGRESS.json", result)
        feature_pass = (len(result["feature_states"]) == 4 and all(
            r["passed"] and r["checks_match_main"] and r["passed_matches_main"] for r in result["feature_states"]))
        if not feature_pass:
            result.update(status="failed_independent_feature_gate_comparison",
                          targets_constructed=False, all_four_states_verified=False)
            atomic_json(output / "RESULTS.json", result)
            raise ValueError("Independent direct feature gate disagrees; no labels constructed")
        for state in states:
            check(p)
            models, diagnostics = fit_direct(state["data"], state["prefix"], p)
            metrics, checks, decomposition = tail_scores(state["data"], state["tail"], models, p)
            metric_match, comparisons = compare_metrics(metrics, state["expected"]["methods"])
            checks_match = checks == state["expected"]["checks"]
            science_pass = all(checks.values())
            state_result = {"state": state["name"], "profile": state["profile"], "fold": state["fold"],
                "state_identity": state["identity"], "prefix_count": len(state["prefix"]),
                "tail_count": len(state["tail"]), "prefix_sha256": index_sha(state["prefix"]),
                "tail_sha256": index_sha(state["tail"]), "methods": metrics, "checks": checks,
                "scientific_state_passed": science_pass, "checks_exactly_match_main": checks_match,
                "scientific_pass_matches_main": science_pass == state["expected"]["passed"],
                "all_metrics_match_main": metric_match, "metric_comparisons": comparisons,
                "direct_solver_diagnostics": diagnostics, "descriptive_error_decomposition": decomposition}
            state_result["independent_state_verified"] = bool(metric_match and checks_match
                and state_result["scientific_pass_matches_main"] and decomposition["all_identities_verified"])
            result["states"].append(state_result)
            atomic_json(output / (state["name"] + "_RESULT.json"), state_result)
            atomic_json(output / "PROGRESS.json", result)
            del models
        check(p)
        continuation = len(result["states"]) == 4 and all(r["scientific_state_passed"] for r in result["states"])
        verified = (len(result["states"]) == 4 and all(r["independent_state_verified"] for r in result["states"])
                    and continuation == main_result["continuation_supported"])
        result.update({"status": "passed_independent_direct_centered_verification" if verified else "failed_independent_comparison",
            "all_four_states_verified": verified, "independent_continuation_supported": continuation,
            "main_continuation_supported": main_result["continuation_supported"],
            "global_continue_decision_matches_main": continuation == main_result["continuation_supported"],
            "targets_constructed_only_after_all_four_direct_feature_gates": True,
            "completed_epoch": time.time(), "access_guard": access,
            "overlapping_development_states_are_not_independent_replications": True})
        atomic_json(output / "RESULTS.json", result)
        atomic_json(output / "STATUS.json", {"status": result["status"], "protocol_sha256": protocol_sha,
                    "deadline_epoch": p["deadline_epoch"], "states": len(result["states"])})
        print(json.dumps({"status": result["status"], "states": len(result["states"]),
                          "continuation_supported": continuation}))
        if not verified:
            raise ValueError("Independent outputs do not meet frozen comparison tolerance")
    except Paused as exc:
        result.update(status="paused_independent_verification", error=str(exc),
                      completed_state_receipts_retained=True, deadline_must_not_reset=True)
        atomic_json(output / "STATUS.json", result)
        raise
    except BaseException as exc:
        result.update(status="failed_independent_verification", error_type=type(exc).__name__, error=str(exc))
        atomic_json(output / "STATUS.json", result)
        raise


if __name__ == "__main__":
    main()
