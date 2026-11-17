"""Post-matrix independent component forward and raw-unit loss reconstruction.

This file is preparation only until a matching frozen protocol/admission exists.
It never calls the experimental wrapper forward or old engine prediction/run.
The original model components are used explicitly; all 24 primary scores must
already be frozen before Data construction or checkpoint tensor loading.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
V3 = ROOT / "outputs/multiscale/mechanism_repair/20261109_v3/code"
STOP = False


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
        temporary = Path(f.name)
    os.replace(temporary, path)


def write_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w+b", dir=path.parent, delete=False) as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
        temporary = Path(f.name)
    os.replace(temporary, path)


def stop(*_):
    global STOP
    STOP = True


def check(deadline):
    if STOP or time.time() >= deadline:
        raise TimeoutError("Independent replay paused at its unchanged absolute deadline; committed prefixes retained")


def expected_ids():
    return [f"{profile}_h{horizon}_{target}_f{fold}_s2026_{arm}"
            for profile, horizon in (("hourly", 1), ("quarterly", 3))
            for target in ("future_close", "future_log_return")
            for fold in (1, 2) for arm in ("A", "B", "C")]


def gate(protocol_path, admission_path):
    protocol, admission = read(protocol_path), read(admission_path)
    ph = sha(protocol_path)
    if (protocol.get("status") != "frozen_before_execution" or
            admission.get("approved") is not True or
            admission.get("protocol_sha256") != ph or
            "independent_replay" not in admission.get("allowed_modes", []) or
            admission.get("engineering_passed") is not True or
            admission.get("complete_matrix_cost_admitted") is not True or
            admission.get("Q3_allowed", False) or admission.get("outer_allowed", False)):
        raise RuntimeError("No matching frozen post-score independent replay admission")
    for key in ("code_sha256", "input_sha256"):
        for name, digest in protocol.get(key, {}).items():
            path = Path(name) if Path(name).is_absolute() else BASE / name
            if sha(path) != digest:
                raise RuntimeError(f"Frozen {key} changed: {name}")
    own = str(Path(__file__).resolve())
    if protocol.get("code_sha256", {}).get(own) != sha(own):
        raise RuntimeError("Independent reconstruction source itself must be frozen")
    deadline = min(float(protocol["deadline_epoch"]), float(admission["deadline_epoch"]))
    check(deadline)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("GPU must remain hidden")
    if not set(os.sched_getaffinity(0)) <= set(protocol["allowed_cpu_union"]) or len(os.sched_getaffinity(0)) > 2:
        raise RuntimeError("Independent CPU affinity exceeds frozen union")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    mf_path, sf_path = BASE / "MODEL_FREEZE.json", BASE / "SCORE_MATRIX.json"
    if sha(mf_path) != admission["model_freeze_sha256"] or sha(sf_path) != admission["score_matrix_sha256"]:
        raise RuntimeError("Post-score admission does not bind model/score matrix")
    models, scores = read(mf_path), read(sf_path)
    ids = expected_ids()
    if (models.get("protocol_sha256") != ph or scores.get("protocol_sha256") != ph or
            scores.get("model_freeze_sha256") != sha(mf_path) or
            len(models.get("entries", [])) != 24 or len(scores.get("entries", [])) != 24 or
            {row["id"] for row in models["entries"]} != set(ids) or
            {row["id"] for row in scores["entries"]} != set(ids)):
        raise RuntimeError("Complete original 24-model/score matrix is required")
    # Verify every small receipt and final tensor/score artifact before the first
    # Data() or torch.load(). Hashing tensor bytes is not tensor deserialization.
    for row in models["entries"]:
        check(deadline)
        for key in ("done", "manifest", "payload"):
            if sha(row[key]) != row[key + "_sha256"]:
                raise RuntimeError("Frozen final model artifact changed")
    for row in scores["entries"]:
        for key in ("result", "predictions"):
            if sha(row[key]) != row[key + "_sha256"]:
                raise RuntimeError("Frozen primary score artifact changed")
        result = read(row["result"])
        if (result.get("status") != "final20_stop_only_development_scored" or
                result.get("cfg", {}).get("id") != row["id"] or
                result.get("protocol_sha256") != ph or
                result.get("model_freeze_sha256") != sha(mf_path) or
                result.get("predictions_sha256") != row["predictions_sha256"]):
            raise RuntimeError("Primary score is not a final frozen logical operation")
    return protocol, ph, admission, deadline, models, scores


class ScalarAffine(nn.Module):
    def __init__(self, layer):
        super().__init__()
        if tuple(layer.normalized_shape) != (1,):
            raise RuntimeError("Only the frozen scalar second normalization is admitted")
        self.weight = nn.Parameter(layer.weight.detach().clone())
        self.bias = nn.Parameter(layer.bias.detach().clone())

    def forward(self, value):
        return value * self.weight + self.bias


class Prefix:
    """Immutable per-batch arrays with a SHA-bound last-complete pointer."""
    def __init__(self, folder, identity):
        self.folder, self.identity = Path(folder), identity
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cursor = 0

    def load(self):
        pointer = self.folder / "LATEST.json"
        if not pointer.exists():
            return None
        receipt = read(pointer)
        path = self.folder / receipt["arrays"]
        if receipt.get("identity") != self.identity or sha(path) != receipt["arrays_sha256"]:
            raise RuntimeError("Independent prefix source/data/hash mismatch")
        with np.load(path, allow_pickle=False) as a:
            value = {key: a[key].copy() for key in a.files}
        self.cursor = int(receipt["cursor"])
        if str(value.pop("identity")) != self.identity or int(value.pop("cursor")) != self.cursor:
            raise RuntimeError("Independent prefix embedded identity/cursor mismatch")
        return value

    def save(self, cursor, **arrays):
        name = f"prefix_{cursor:09d}.npz"
        path = self.folder / name
        write_npz(path, identity=np.asarray(self.identity), cursor=np.asarray(cursor), **arrays)
        write_json(self.folder / "LATEST.json", {"identity": self.identity, "cursor": cursor,
                                                  "arrays": name, "arrays_sha256": sha(path)})
        self.cursor = cursor


def relative_error(value, reference):
    value, reference = np.asarray(value, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    return float(np.max(np.abs(value - reference)) / max(float(np.max(np.abs(reference))), 1e-30))


def losses(prediction, truth):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {"endpoint_mse": float(np.mean(error[:, -1] ** 2)),
            "path_mse": float(np.mean(error ** 2)),
            "endpoint_mae": float(np.mean(np.abs(error[:, -1])))}


def one_job(entry, score, ph, admission, deadline):
    folder = BASE / "independent" / "jobs" / entry["id"]
    identity = hashlib.sha256(canonical({"protocol": ph, "model_freeze": admission["model_freeze_sha256"],
                                        "score_matrix": admission["score_matrix_sha256"], "entry": entry,
                                        "score": score, "source": sha(__file__)})).hexdigest()
    result_path = folder / "REPLAY_RESULT.json"
    if result_path.exists():
        result = read(result_path)
        if result.get("identity") != identity or sha(result["arrays"]) != result["arrays_sha256"]:
            raise RuntimeError("Existing independent result changed")
        return result
    check(deadline)
    sys.path.insert(0, str(V3))
    study = importlib.import_module("study")
    networks, engine = importlib.import_module("networks"), importlib.import_module("engine")
    from multiscale.bars import targets
    from aapl.models import CausalEMA
    done = read(entry["done"])
    cfg = done["cfg"]
    if cfg["id"] != entry["id"] or cfg["arm"] not in ("A", "B", "C"):
        raise RuntimeError("Unexpected fixed job")
    data = engine.Data(cfg)
    if str(data.bars.session_date[-1]) > "2026-06-30":
        raise RuntimeError("Protected data encountered")
    for part, indices in data.splits.items():
        if hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest() != data.rec["split_sha256"][part]:
            raise RuntimeError("Original canonical purged split changed")
    provider = {"task_metadata_sha256": sha(engine.PRIOR / "cache" / data.task / "TASK.json"),
                "split_archive_sha256": sha(data.rec["split_file"]),
                "scaler_file_sha256": sha(data.rec["scaler_file"]), "scaler": data.scaler,
                "fit_rows": len(data.splits["fit"]), "source_admission_sha256": sha(engine.SOURCE / "admission.json"),
                "source_scope": "Full admitted pre-Q3 bar cache; labels/windows restricted by mode"}
    binding = {"protocol_sha256": ph, "mode": "train", "job": cfg, "provider": provider}
    if binding != entry["execution_binding"] or hashlib.sha256(canonical(binding)).hexdigest() != entry["identity"]:
        raise RuntimeError("Independent original data/scaler/model binding differs")
    payload = torch.load(entry["payload"], map_location="cpu", weights_only=True)
    if (payload["identity"] != entry["identity"] or payload["state"]["phase"] != "completed" or
            payload["state"]["epoch"] != 20 or payload["state"]["updates"] != entry["updates"]):
        raise RuntimeError("Final checkpoint/config/updates differ")
    scaler = data.scaler

    def windows(part, start, end):
        indices = data.splits[part][start:end]
        raw, anchor = data.windows.arrays(indices)
        x = torch.from_numpy(((raw - scaler["x_mean"]) / scaler["x_std"]).astype("float32")[:, :, None])
        a = torch.from_numpy(((anchor - scaler["anchor_mean"]) / scaler["anchor_std"]).astype("float32"))
        return x, a, indices

    # Reconstruct the fit-only constant independently of saved accumulators.
    mean_store = Prefix(folder / "mean_prefix", identity)
    mean_previous = mean_store.load()
    total = (torch.zeros((cfg["lookback"], 1), dtype=torch.float64) if mean_previous is None
             else torch.from_numpy(mean_previous["total"]))
    ema = CausalEMA(.1)
    for start in range(mean_store.cursor, len(data.splits["fit"]), 128):
        check(deadline)
        end = min(start + 128, len(data.splits["fit"]))
        x, _, _ = windows("fit", start, end)
        with torch.no_grad():
            total += ema(x).to(torch.float64).sum(0)
        mean_store.save(end, total=total.numpy())
    mean = (total / len(data.splits["fit"])).to(torch.float32)
    mean_exact = torch.equal(mean, payload["state"]["mean"])
    if not mean_exact or payload["state"]["mean_seen"] != len(data.splits["fit"]):
        raise RuntimeError("Independent original fit-only mean failed bitwise identity")
    torch.manual_seed(2026)
    model = networks.create(cfg, scaler)
    if cfg["arm"] in ("B", "C"):
        model.backbone.ln2 = ScalarAffine(model.backbone.ln2)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    prediction_store = Prefix(folder / "prediction_prefix", identity)
    previous = prediction_store.load()
    prediction = np.empty((0, cfg["horizon"]), dtype=np.float64) if previous is None else previous["prediction"]
    truth = np.empty_like(prediction) if previous is None else previous["rawtruth"]
    origins = np.empty(0, dtype="<i8") if previous is None else previous["origins"]
    if len(prediction) != prediction_store.cursor or not np.array_equal(origins, data.splits["stop"][:len(origins)]):
        raise RuntimeError("Independent saved score prefix is inconsistent")
    for start in range(prediction_store.cursor, len(data.splits["stop"]), 128):
        check(deadline)
        end = min(start + 128, len(data.splits["stop"]))
        x, anchor, indices = windows("stop", start, end)
        with torch.no_grad():
            # Explicit original components; never model.forward/worker wrapper.
            actual_trend = model.ema(x)
            seasonal = x - actual_trend
            branch = mean.expand(x.shape[0], -1, -1) if cfg["arm"] == "C" else actual_trend
            normalized = (model.backbone(seasonal, branch)[:, :, 0] + model.anchor_head(anchor))[:, :cfg["horizon"]]
            raw_prediction = normalized.numpy() * np.asarray(scaler["y_std"]) + np.asarray(scaler["y_mean"])
        raw_truth = np.asarray(targets(data.bars, indices, cfg["horizon"], cfg["target"]), dtype=np.float64)
        prediction, truth = np.concatenate((prediction, raw_prediction)), np.concatenate((truth, raw_truth))
        origins = np.concatenate((origins, indices))
        prediction_store.save(end, prediction=prediction, rawtruth=truth, origins=origins)
    with np.load(score["predictions"], allow_pickle=False) as primary:
        original_prediction, original_truth, original_origins = primary["prediction"], primary["actual"], primary["origins"]
    primary_result = read(score["result"])
    prediction_error = relative_error(prediction, original_prediction)
    metrics = losses(prediction, truth)
    metric_errors = {key: relative_error(value, primary_result[key]) for key, value in metrics.items()}
    truth_exact, origins_exact = np.array_equal(truth, original_truth), np.array_equal(origins, original_origins)
    qualified = (np.isfinite(prediction).all() and np.isfinite(truth).all() and mean_exact and truth_exact and
                 origins_exact and prediction_error <= 1e-7 and max(metric_errors.values()) <= 1e-7)
    baseline = (np.broadcast_to(data.bars.close[origins, None], truth.shape).copy()
                if cfg["target"] == "future_close" else np.zeros_like(truth))
    groups = data.bars.session_date[origins].astype("U10")
    if cfg["profile"] == "quarterly":
        groups = np.asarray([str(group)[:7] for group in groups], dtype="U7")
    path = folder / "replay_predictions.npz"
    write_npz(path, prediction=prediction, rawtruth=truth, origins=origins, baseline=baseline, groups=groups)
    result = {"id": cfg["id"], "identity": identity, "qualified": bool(qualified), "cfg": cfg,
              "arrays": str(path), "arrays_sha256": sha(path), "mean_bitwise_exact": mean_exact,
              "mean_sha256": hashlib.sha256(mean.numpy().tobytes()).hexdigest(),
              "rawtruth_exact": truth_exact, "origins_exact": origins_exact,
              "normalized_max_prediction_error": prediction_error, "metric_normalized_errors": metric_errors,
              "normalization_denominator": "max(maxabs(reference),1e-30)", "tolerance": 1e-7,
              "metrics": metrics, "baseline_metrics": losses(baseline, truth), "provider": provider,
              "model_identity": entry["identity"], "checkpoint_payload_sha256": entry["payload_sha256"],
              "protocol_sha256": ph, "outer_read": False, "Q3_read": False,
              "independent_market_confirmation": False, "candidate_admitted": False}
    write_json(result_path, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--admission", required=True)
    args = parser.parse_args()
    protocol, ph, admission, deadline, models, scores = gate(args.protocol, args.admission)
    folder = BASE / "independent"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "REPLAY_LOCK").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        results = []
        by_model, by_score = {r["id"]: r for r in models["entries"]}, {r["id"]: r for r in scores["entries"]}
        for identifier in expected_ids():
            check(deadline)
            result = one_job(by_model[identifier], by_score[identifier], ph, admission, deadline)
            results.append(result)
            write_json(folder / "RECONSTRUCTION_PROGRESS.json", {"status": "partial", "jobs": results,
                                                                  "protocol_sha256": ph})
            if not result["qualified"]:
                raise RuntimeError("Frozen independent numerical threshold failed; stop without tolerance rescue")
        value = {"status": "passed", "jobs": results, "protocol_sha256": ph,
                 "model_freeze_sha256": admission["model_freeze_sha256"],
                 "score_matrix_sha256": admission["score_matrix_sha256"], "source_sha256": sha(__file__),
                 "old_engine_prediction_or_run_called": False, "outer_read": False, "Q3_read": False,
                 "candidate_admitted": False, "full_goal_achieved": False}
        write_json(folder / "RECONSTRUCTION.json", value)
        print(json.dumps({"status": "passed", "jobs": len(results)}, sort_keys=True))


if __name__ == "__main__":
    main()
