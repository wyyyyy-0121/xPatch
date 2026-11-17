"""Isolated, admission-gated short-horizon route diagnostic.

No market provider is constructed by --engineering. Old engine.run,
stop_score and prediction are never called. A market provider reads the
admitted full pre-Q3 bar cache; fit/stop restrictions apply to consumed
windows and labels, not to an OS-level raw-file-access claim.
"""
from __future__ import annotations

import argparse
import copy
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
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import numpy as np
import torch
from torch import nn

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
V3 = ROOT / "outputs/multiscale/mechanism_repair/20261109_v3/code"
STOP_REQUESTED = False


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
        temp = Path(f.name)
    os.replace(temp, path)


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w+b", dir=path.parent, delete=False) as f:
        torch.save(value, f)
        f.flush()
        os.fsync(f.fileno())
        temp = Path(f.name)
    os.replace(temp, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w+b", dir=path.parent, delete=False) as f:
        np.savez_compressed(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
        temp = Path(f.name)
    os.replace(temp, path)


def configure_runtime(protocol):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise RuntimeError("GPU must be hidden")
    affinity = set(os.sched_getaffinity(0))
    allowed = set(protocol.get("allowed_cpu_union", [24, 26]))
    if not affinity or not affinity <= allowed or len(affinity) > 2:
        raise RuntimeError("Process affinity exceeds the frozen CPU allowance")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)


def request_stop(*_):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def frozen_protocol(path):
    path = Path(path).resolve()
    protocol = read(path)
    if protocol.get("status") != "frozen_before_execution":
        raise RuntimeError("A frozen protocol is required")
    if not protocol.get("code_sha256"):
        raise RuntimeError("Protocol lacks a source freeze")
    for section in ("code_sha256", "input_sha256"):
        for name, expected in protocol.get(section, {}).items():
            source = Path(name) if Path(name).is_absolute() else BASE / name
            if digest(source) != expected:
                raise RuntimeError(f"Frozen {section} changed: {name}")
    if time.time() >= float(protocol["deadline_epoch"]):
        raise RuntimeError("Absolute protocol deadline expired; it cannot reset")
    return protocol, digest(path)


def market_admission(path, protocol_hash, mode, protocol):
    if path is None:
        raise RuntimeError("This mode requires a separate admission receipt")
    admission = read(path)
    if (admission.get("approved") is not True or
            admission.get("protocol_sha256") != protocol_hash or
            mode not in admission.get("allowed_modes", [])):
        raise RuntimeError("No matching market admission for this mode")
    if admission.get("engineering_passed") is not True:
        raise RuntimeError("Engineering acceptance is required")
    if time.time() >= min(float(admission["deadline_epoch"]), float(protocol["deadline_epoch"])):
        raise RuntimeError("Admission deadline expired")
    if mode in ("train", "score", "freeze") and admission.get("complete_matrix_cost_admitted") is not True:
        raise RuntimeError("Complete-matrix cost admission is required")
    if admission.get("Q3_allowed", False) or admission.get("outer_allowed", False):
        raise RuntimeError("This implementation never admits Q3 or outer scoring")
    return admission


def original_modules():
    # These imports define classes only; market reads occur only in Data().
    if str(V3) not in sys.path:
        sys.path.insert(0, str(V3))
    return importlib.import_module("study"), importlib.import_module("networks"), importlib.import_module("engine")


def job_matrix():
    jobs = []
    for profile, horizon in (("hourly", 1), ("quarterly", 3)):
        for target in ("future_close", "future_log_return"):
            for fold in (1, 2):
                for arm in ("A", "B", "C"):
                    frequency, lookback = ("hour", 96) if profile == "hourly" else ("month", 36)
                    jobs.append({"id": f"{profile}_h{horizon}_{target}_f{fold}_s2026_{arm}",
                                 "profile": profile, "frequency": frequency, "lookback": lookback,
                                 "horizon": horizon, "target": target, "fold": fold, "seed": 2026,
                                 "arm": arm, "model": "xpatch", "anchor": "observed",
                                 "initialization": "baseline", "output": "price" if target == "future_close" else "return",
                                 "loss": "path", "max_epochs": 20, "batch_size": 128,
                                 "learning_rate": .001, "patience": 3,
                                 "variant": "price_baseline" if target == "future_close" else "anchor_baseline"})
    assert len(jobs) == 24 and len({j["id"] for j in jobs}) == 24
    return jobs


class ScalarAffine(nn.Module):
    def __init__(self, original):
        super().__init__()
        if tuple(original.normalized_shape) != (1,):
            raise ValueError("The sole admitted repair is scalar native H2/H3")
        self.weight = nn.Parameter(original.weight.detach().clone())
        self.bias = nn.Parameter(original.bias.detach().clone())

    def forward(self, value):
        return self.weight * value + self.bias


def route_forward(self, x, anchor):
    actual_trend = self.ema(x)
    seasonal = x - actual_trend
    trend = self._route_mean.to(x).expand(x.shape[0], -1, -1) if self._route_arm == "C" else actual_trend
    value = self.backbone(seasonal, trend)[:, :, 0] + self.anchor_head(anchor)
    return value[:, :self.horizon]


def build_model(cfg, scaler, mean, dtype=torch.float32):
    _, networks, _ = original_modules()
    torch.manual_seed(int(cfg["seed"]))
    model = networks.create(cfg, scaler).to(dtype=dtype)
    if model.native_horizon not in (2, 3):
        raise RuntimeError("Unsupported native horizon")
    if cfg["arm"] in ("B", "C"):
        model.backbone.ln2 = ScalarAffine(model.backbone.ln2)
    model._route_arm = cfg["arm"]
    model._route_mean = mean.detach().clone().to(dtype=dtype)
    model.forward = types.MethodType(route_forward, model)
    return model


class MarketProvider:
    def __init__(self, cfg, *, allow_stop=False):
        _, _, engine = original_modules()
        self.data = engine.Data(cfg)  # Permitted only after CLI market admission.
        self.cfg = cfg
        self.allow_stop = allow_stop
        self.scaler = self.data.scaler
        if str(self.data.bars.session_date[-1]) > "2026-06-30":
            raise RuntimeError("Protected-period bar encountered")
        self.length = len(self.data.splits["fit"])
        rec = self.data.rec
        if set(self.data.splits) != set(rec["split_sha256"]):
            raise RuntimeError("Split index metadata keys changed")
        for part, indices in self.data.splits.items():
            actual = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
            if actual != rec["split_sha256"][part]:
                raise RuntimeError(f"Split index metadata changed: {part}")
            if len(indices) != rec["counts"][part]:
                raise RuntimeError(f"Split index count changed: {part}")
        if self.length < 2:
            raise RuntimeError("BatchNorm training requires at least two fit rows")
        self.identity = {"task_metadata_sha256": digest(engine.PRIOR / "cache" / self.data.task / "TASK.json"),
                         "split_archive_sha256": digest(rec["split_file"]),
                         "scaler_file_sha256": digest(rec["scaler_file"]),
                         "scaler": self.scaler, "fit_rows": self.length,
                         "source_admission_sha256": digest(engine.SOURCE / "admission.json"),
                         "source_scope": "Full admitted pre-Q3 bar cache; labels/windows restricted by mode"}

    def size(self, part):
        self._part(part)
        return len(self.data.splits[part])

    def _part(self, part):
        if part != "fit" and not (part == "stop" and self.allow_stop):
            raise RuntimeError("No outer or unadmitted partition access")

    def inputs(self, part, positions):
        self._part(part)
        indices = self.data.splits[part][np.asarray(positions)]
        raw_x, raw_anchor = self.data.windows.arrays(indices)
        x = ((raw_x - self.scaler["x_mean"]) / self.scaler["x_std"]).astype("float32")
        anchor = ((raw_anchor - self.scaler["anchor_mean"]) / self.scaler["anchor_std"]).astype("float32")
        return torch.from_numpy(x[:, :, None]), torch.from_numpy(anchor)

    def labels(self, part, positions):
        raw = self.raw_labels(part, positions)
        value = ((raw - np.asarray(self.scaler["y_mean"])) / np.asarray(self.scaler["y_std"])).astype("float32")
        return torch.from_numpy(value)

    def raw_labels(self, part, positions):
        self._part(part)
        from multiscale.bars import targets
        indices = self.data.splits[part][np.asarray(positions)]
        return np.asarray(targets(self.data.bars, indices, self.cfg["horizon"], self.cfg["target"]), dtype=np.float64)

    def inverse(self, value):
        return np.asarray(value) * np.asarray(self.scaler["y_std"]) + np.asarray(self.scaler["y_mean"])


class Checkpoints:
    """Generation files committed by a SHA-bound pointer; no silent fallback."""
    def __init__(self, folder, identity):
        self.folder = Path(folder)
        self.identity = identity
        self.generation = 0

    def load(self):
        pointer_path = self.folder / "LATEST.json"
        if not pointer_path.exists():
            return None
        pointer = read(pointer_path)
        manifest_path = self.folder / pointer["manifest"]
        if digest(manifest_path) != pointer["manifest_sha256"]:
            raise RuntimeError("Checkpoint manifest was modified")
        manifest = read(manifest_path)
        if manifest["identity"] != self.identity:
            raise RuntimeError("Checkpoint belongs to another source/protocol/job")
        payload_path = self.folder / manifest["payload"]
        if digest(payload_path) != manifest["payload_sha256"]:
            raise RuntimeError("Checkpoint payload was modified")
        payload = torch.load(payload_path, map_location="cpu", weights_only=True)
        if payload["identity"] != self.identity or payload["generation"] != manifest["generation"]:
            raise RuntimeError("Checkpoint generation identity mismatch")
        self.generation = int(manifest["generation"])
        return payload

    def save(self, state, model, optimizer, rng, elapsed):
        self.folder.mkdir(parents=True, exist_ok=True)
        self.generation += 1
        name = f"generation_{self.generation:08d}"
        payload = {"identity": self.identity, "generation": self.generation,
                   "state": copy.deepcopy(state), "model": None if model is None else model.state_dict(),
                   "optimizer": None if optimizer is None else optimizer.state_dict(),
                   "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                   "elapsed": elapsed}
        payload_path = self.folder / (name + ".pt")
        atomic_torch(payload_path, payload)
        manifest = {"identity": self.identity, "generation": self.generation,
                    "payload": payload_path.name, "payload_sha256": digest(payload_path),
                    "phase": state["phase"], "epoch": state["epoch"], "updates": state["updates"],
                    "cursor": state["cursor"], "mean_cursor": state["mean_cursor"]}
        manifest_path = self.folder / (name + ".json")
        atomic_json(manifest_path, manifest)
        atomic_json(self.folder / "LATEST.json", {"manifest": manifest_path.name,
                                                  "manifest_sha256": digest(manifest_path)})
        return manifest_path, manifest


def train_fixed(cfg, provider, folder, identity, deadline, *, epochs=20,
                checkpoint_every=100, pause_updates=None, artificial_batch_delay=0., execution_binding=None,
                training_deadline_seconds=None, pause_mean_batches=None):
    """No stop/outer access. Also used by synthetic recovery fixtures."""
    clock_wall, clock_cpu = time.perf_counter(), time.process_time()
    store = Checkpoints(folder, identity)
    saved = store.load()
    done_path = Path(folder) / "DONE.json"
    if done_path.exists():
        done = read(done_path)
        if (saved is None or done.get("status") != "completed" or done.get("identity") != identity or
                done.get("cfg") != cfg or done.get("epochs") != epochs or
                done.get("execution_binding") != execution_binding or
                saved["state"]["phase"] != "completed" or saved["state"]["epoch"] != epochs or
                saved["state"]["updates"] != done["updates"]):
            raise RuntimeError("Existing DONE identity/config/state is invalid")
        pointer = read(Path(folder) / "LATEST.json")
        manifest = read(Path(folder) / pointer["manifest"])
        if (pointer["manifest"] != done["checkpoint_manifest"] or
                pointer["manifest_sha256"] != done["checkpoint_manifest_sha256"] or
                manifest["payload_sha256"] != done["payload_sha256"]):
            raise RuntimeError("Existing DONE checkpoint differs from the verified latest generation")
        return done
    if execution_binding is not None and execution_binding.get("mode") == "train" and (BASE / "MODEL_FREEZE.json").exists():
        raise RuntimeError("Model freeze prohibits any training checkpoint modification")
    state = {"phase": "mean", "epoch": 0, "cursor": 0, "updates": 0,
             "permutation": None, "mean_cursor": 0, "mean_seen": 0,
             "mean_sum": torch.zeros((cfg["lookback"], 1), dtype=torch.float64),
             "mean": None, "training_loss_sum": 0., "training_rows": 0, "history": []}
    rng = np.random.default_rng(cfg["seed"])
    previous = {"wall_seconds": 0., "cpu_seconds": 0.}
    if saved is not None:
        state = saved["state"]
        rng.bit_generator.state = saved["numpy_rng"]
        previous = saved["elapsed"]
    model = optimizer = None

    def elapsed():
        return {"wall_seconds": previous["wall_seconds"] + time.perf_counter() - clock_wall,
                "cpu_seconds": previous["cpu_seconds"] + time.process_time() - clock_cpu}

    def save():
        return store.save(state, model, optimizer, rng, elapsed())

    def stopped():
        return STOP_REQUESTED or time.time() >= deadline

    # No labels are constructed to estimate the constant trend input.
    original_modules()
    from aapl.models import CausalEMA
    ema = CausalEMA(.1)
    mean_batches_at_entry = 0
    while state["phase"] == "mean" and state["mean_cursor"] < provider.length:
        if stopped() or (pause_mean_batches is not None and mean_batches_at_entry >= pause_mean_batches):
            save()
            return {"status": "paused", "phase": "mean", "elapsed": elapsed()}
        start = state["mean_cursor"]
        end = min(start + cfg["batch_size"], provider.length)
        x, _ = provider.inputs("fit", np.arange(start, end))
        with torch.no_grad():
            trend = ema(x)
            state["mean_sum"] += trend.to(torch.float64).sum(0)
        state["mean_seen"] += end - start
        state["mean_cursor"] = end
        mean_batches_at_entry += 1
        if end == provider.length or (end // cfg["batch_size"]) % checkpoint_every == 0:
            save()
    if state["phase"] == "mean":
        assert state["mean_seen"] == provider.length > 0
        state["mean"] = (state["mean_sum"] / state["mean_seen"]).to(torch.float32)
        state["phase"] = "train"
    model = build_model(cfg, provider.scaler, state["mean"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    if saved is not None and saved["model"] is not None:
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
    elif state["updates"] != 0:
        raise RuntimeError("Training cursor without a model")
    save()
    updates_at_entry = state["updates"]
    if training_deadline_seconds is not None:
        # Synthetic timeout is anchored after imports/mean/model preparation;
        # the artificial batch delay proves training cannot finish first.
        deadline = min(deadline, time.time() + training_deadline_seconds)
    while state["epoch"] < epochs:
        if state["permutation"] is None:
            state["permutation"] = torch.from_numpy(rng.permutation(provider.length))
            state["cursor"] = 0
        while state["cursor"] < provider.length:
            if stopped() or (pause_updates is not None and state["updates"] - updates_at_entry >= pause_updates):
                save()
                return {"status": "paused", "phase": "train", "epoch": state["epoch"],
                        "updates": state["updates"], "elapsed": elapsed()}
            end = min(state["cursor"] + cfg["batch_size"], provider.length)
            if provider.length - end == 1:
                end = provider.length
            positions = state["permutation"][state["cursor"]:end].numpy()
            x, anchor = provider.inputs("fit", positions)
            y = provider.labels("fit", positions)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            pred = model(x, anchor)
            loss = (pred - y).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite fit loss")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise RuntimeError("Missing/nonfinite gradient")
            optimizer.step()
            state["updates"] += 1
            state["cursor"] = end
            state["training_loss_sum"] += float(loss.detach()) * len(positions)
            state["training_rows"] += len(positions)
            if state["updates"] % checkpoint_every == 0:
                save()
            if artificial_batch_delay:
                time.sleep(min(artificial_batch_delay, .25))
        state["history"].append({"epoch": state["epoch"] + 1,
                                  "fit_path_mse_standardized": state["training_loss_sum"] / state["training_rows"],
                                  "updates": state["updates"]})
        state["epoch"] += 1
        state["cursor"] = 0
        state["permutation"] = None
        state["training_loss_sum"] = 0.
        state["training_rows"] = 0
        save()
    state["phase"] = "completed"
    manifest_path, manifest = save()
    done = {"status": "completed", "identity": identity, "cfg": cfg, "epochs": state["epoch"],
            "updates": state["updates"], "checkpoint_manifest": manifest_path.name,
            "checkpoint_manifest_sha256": digest(manifest_path), "payload_sha256": manifest["payload_sha256"],
            "mean_seen": state["mean_seen"], "mean_sha256": hashlib.sha256(state["mean"].numpy().tobytes()).hexdigest(),
            "elapsed": elapsed(), "stop_scored": False, "outer_read": False, "Q3_read": False,
            "candidate_admitted": False, "execution_binding": execution_binding}
    atomic_json(Path(folder) / "DONE.json", done)
    return done


def freeze_models(protocol_hash):
    entries = []
    for cfg in job_matrix():
        folder = BASE / "runs" / cfg["id"]
        done_path = folder / "DONE.json"
        done = read(done_path)
        manifest_path = folder / done["checkpoint_manifest"]
        binding = done.get("execution_binding")
        if (not isinstance(binding, dict) or binding.get("protocol_sha256") != protocol_hash or
                binding.get("mode") != "train" or binding.get("job") != cfg or
                hashlib.sha256(canonical(binding)).hexdigest() != done["identity"]):
            raise RuntimeError("Completed run does not bind the current protocol/job/provider")
        fit_rows = int(binding["provider"]["fit_rows"])
        batches = (fit_rows + cfg["batch_size"] - 1) // cfg["batch_size"]
        if fit_rows % cfg["batch_size"] == 1:
            batches -= 1
        if done["updates"] != 20 * batches:
            raise RuntimeError("Completed run has the wrong fixed update count")
        if (done["status"] != "completed" or done["epochs"] != 20 or done["cfg"] != cfg or
                digest(manifest_path) != done["checkpoint_manifest_sha256"]):
            raise RuntimeError("Complete fixed-20 matrix is required")
        manifest = read(manifest_path)
        latest = read(folder / "LATEST.json")
        if (manifest["phase"] != "completed" or manifest["epoch"] != 20 or
                latest["manifest"] != done["checkpoint_manifest"] or
                latest["manifest_sha256"] != done["checkpoint_manifest_sha256"] or
                manifest["payload_sha256"] != done["payload_sha256"]):
            raise RuntimeError("Incomplete checkpoint")
        payload = folder / manifest["payload"]
        if digest(payload) != manifest["payload_sha256"]:
            raise RuntimeError("Checkpoint changed")
        saved = Checkpoints(folder, done["identity"]).load()
        if (saved["state"]["phase"] != "completed" or saved["state"]["epoch"] != 20 or
                saved["state"]["updates"] != done["updates"]):
            raise RuntimeError("Checkpoint payload does not match the completed fixed-20 run")
        entries.append({"id": cfg["id"], "done": str(done_path), "done_sha256": digest(done_path),
                        "manifest": str(manifest_path), "manifest_sha256": digest(manifest_path),
                        "payload": str(payload), "payload_sha256": digest(payload), "identity": done["identity"],
                        "execution_binding": binding, "updates": done["updates"]})
    value = {"status": "all_24_final_models_frozen_before_stop_scoring", "protocol_sha256": protocol_hash,
             "entries": entries, "outer_read": False, "Q3_read": False, "candidate_admitted": False}
    path = BASE / "MODEL_FREEZE.json"
    if path.exists() and read(path) != value:
        raise RuntimeError("Existing model freeze cannot be replaced")
    atomic_json(path, value)
    return value


def score_final(cfg, protocol_hash, model_freeze_hash, deadline):
    folder = BASE / "scores" / cfg["id"]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "SCORE_LOCK").open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This logical scoring operation is already active") from error
        try:
            return score_locked(cfg, protocol_hash, model_freeze_hash, deadline, folder)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def score_locked(cfg, protocol_hash, model_freeze_hash, deadline, folder):
    def check_deadline():
        if STOP_REQUESTED or time.time() >= deadline:
            raise RuntimeError("Scoring deadline reached before a resumable prediction operation began")

    check_deadline()
    freeze_path = BASE / "MODEL_FREEZE.json"
    if digest(freeze_path) != model_freeze_hash:
        raise RuntimeError("Admission does not bind the model freeze")
    frozen = read(freeze_path)
    if frozen["protocol_sha256"] != protocol_hash or len(frozen["entries"]) != 24:
        raise RuntimeError("Wrong/incomplete matrix freeze")
    if {e["id"] for e in frozen["entries"]} != {j["id"] for j in job_matrix()}:
        raise RuntimeError("Unexpected matrix identities")
    for entry in frozen["entries"]:
        check_deadline()
        for key in ("done", "manifest", "payload"):
            if digest(entry[key]) != entry[key + "_sha256"]:
                raise RuntimeError("A frozen model changed before scoring")
    entry = next(e for e in frozen["entries"] if e["id"] == cfg["id"])
    result_path = folder / "RESULT.json"
    prediction_path = folder / "stop_predictions.npz"
    binding = {"protocol_sha256": protocol_hash, "model_freeze_sha256": model_freeze_hash,
               "cfg": cfg, "training_identity": entry["identity"],
               "provider": entry["execution_binding"]["provider"]}
    score_identity = hashlib.sha256(canonical(binding)).hexdigest()
    if result_path.exists():
        result = read(result_path)
        if (result.get("cfg") != cfg or result.get("protocol_sha256") != protocol_hash or
                result.get("model_freeze_sha256") != model_freeze_hash or
                result.get("score_identity") != score_identity or
                result.get("predictions_sha256") != digest(prediction_path)):
            raise RuntimeError("Existing score identity/prediction hash differs; no rereading stop labels")
        return result
    receipt_path = folder / "SCORE_STARTED.json"
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("score_identity") != score_identity or receipt.get("binding") != binding:
            raise RuntimeError("Existing scoring operation belongs to another frozen identity")
    else:
        atomic_json(receipt_path, {"score_identity": score_identity, "binding": binding,
                                  "started_epoch": time.time(), "single_logical_operation": True})
    payload = torch.load(entry["payload"], map_location="cpu", weights_only=True)
    if payload["identity"] != entry["identity"] or payload["state"]["phase"] != "completed":
        raise RuntimeError("Wrong final model")
    # This is the first point at which this mode may instantiate a stop provider.
    provider = MarketProvider(cfg, allow_stop=True)
    check_deadline()
    expected_binding = {"protocol_sha256": protocol_hash, "mode": "train", "job": cfg, "provider": provider.identity}
    if (expected_binding != entry["execution_binding"] or
            hashlib.sha256(canonical(expected_binding)).hexdigest() != payload["identity"] or
            payload["state"]["epoch"] != 20 or payload["state"]["updates"] != entry["updates"]):
        raise RuntimeError("Scoring data/scaler/config/update identities differ from the trained model")
    model = build_model(cfg, provider.scaler, payload["state"]["mean"])
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    progress = score_batches(cfg, provider, model, folder, binding, score_identity, deadline)
    if progress["status"] != "completed":
        return progress
    predictions, actual = progress["prediction"], progress["actual"]
    cursor = progress["cursor"]
    total = len(actual)
    prediction, truth = predictions, actual
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise RuntimeError("Nonfinite final prediction")
    error = prediction - truth
    result = {"status": "final20_stop_only_development_scored", "cfg": cfg,
              "protocol_sha256": protocol_hash,
              "score_identity": score_identity,
              "model_freeze_sha256": model_freeze_hash, "origins": len(truth),
              "endpoint_mse": float(np.mean(error[:, -1] ** 2)),
              "path_mse": float(np.mean(error ** 2)), "endpoint_mae": float(np.mean(abs(error[:, -1]))),
              "outer_read": False, "Q3_read": False, "candidate_admitted": False,
              "independent_market_confirmation": False}
    if STOP_REQUESTED or time.time() >= deadline:
        return {"status": "paused", "phase": "score_final_commit", "cursor": cursor,
                "total_origins": total, "score_identity": score_identity,
                "completed_batches_preserved": True, "Q3_read": False, "outer_read": False}
    atomic_npz(prediction_path, prediction=prediction, actual=truth, origins=provider.data.splits["stop"])
    result["predictions_sha256"] = digest(prediction_path)
    atomic_json(folder / "RESULT.json", result)
    return result


def score_batches(cfg, provider, model, folder, binding, score_identity, deadline, *, pause_batches=None):
    """Resume one scoring process from committed raw-truth/prediction prefixes."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if binding["provider"] != provider.identity:
        raise RuntimeError("Score batch provider identity differs from the frozen binding")
    total = provider.size("stop")
    if total <= 0:
        raise RuntimeError("The frozen stop partition is empty")
    predictions = np.empty((0, cfg["horizon"]), dtype=np.float64)
    actual = np.empty_like(predictions)
    origins = np.empty(0, dtype="<i8")
    cursor = generation = 0
    pointer_path = folder / "SCORE_LATEST.json"

    def load_arrays(array_path, expected_generation, expected_cursor=None):
        with np.load(array_path, allow_pickle=False) as arrays:
            if (str(arrays["score_identity"]) != score_identity or int(arrays["generation"]) != expected_generation or
                    int(arrays["total_origins"]) != total):
                raise RuntimeError("Score array metadata differs from the frozen identity")
            value = {key: arrays[key].copy() for key in ("prediction", "actual", "origins")}
            value["cursor"] = int(arrays["cursor"])
            value["previous_cursor"] = int(arrays["previous_cursor"])
        count = value["cursor"]
        if (expected_cursor is not None and count != expected_cursor) or not 0 <= count <= total:
            raise RuntimeError("Score array cursor differs from its manifest")
        if (value["prediction"].shape != (count, cfg["horizon"]) or value["actual"].shape != value["prediction"].shape or
                value["prediction"].dtype != np.float64 or value["actual"].dtype != np.float64 or
                value["origins"].shape != (count,) or
                not np.array_equal(value["origins"], provider.data.splits["stop"][:count]) or
                not np.isfinite(value["prediction"]).all() or not np.isfinite(value["actual"]).all()):
            raise RuntimeError("Score checkpoint cursor/shape/origin/truth state is invalid")
        return value

    if pointer_path.exists():
        pointer = read(pointer_path)
        manifest_path = folder / pointer["manifest"]
        if digest(manifest_path) != pointer["manifest_sha256"]:
            raise RuntimeError("Score checkpoint manifest changed")
        manifest = read(manifest_path)
        if manifest.get("score_identity") != score_identity or manifest.get("binding") != binding:
            raise RuntimeError("Score checkpoint belongs to another frozen model/data/protocol")
        array_path = folder / manifest["arrays"]
        if digest(array_path) != manifest["arrays_sha256"]:
            raise RuntimeError("Score checkpoint arrays changed")
        cursor, generation = int(manifest["cursor"]), int(manifest["generation"])
        if manifest["total_origins"] != total:
            raise RuntimeError("Score checkpoint total origin count changed")
        value = load_arrays(array_path, generation, cursor)
        predictions, actual, origins = value["prediction"], value["actual"], value["origins"]

    # A crash may occur after an immutable npz was fsynced but before its
    # manifest/pointer commit. Adopt its verified complete prefix, without
    # rereading the already persisted raw truth. Never fall back on corruption.
    for array_path in sorted(folder.glob("score_generation_*.npz")):
        candidate_generation = int(array_path.stem.rsplit("_", 1)[1])
        if candidate_generation <= generation:
            continue
        if candidate_generation != generation + 1:
            raise RuntimeError("Unexpected gap in score checkpoint generations")
        value = load_arrays(array_path, candidate_generation)
        count = value["cursor"]
        if (value["previous_cursor"] != cursor or not cursor <= count <= min(cursor + 128, total) or
                (candidate_generation > 1 and count == cursor) or
                not np.array_equal(predictions, value["prediction"][:cursor]) or
                not np.array_equal(actual, value["actual"][:cursor]) or
                not np.array_equal(origins, value["origins"][:cursor])):
            raise RuntimeError("Orphan score checkpoint does not extend the exact committed prefix")
        manifest_path = array_path.with_suffix(".json")
        manifest = {"score_identity": score_identity, "binding": binding,
                    "generation": candidate_generation, "cursor": count, "total_origins": total,
                    "arrays": array_path.name, "arrays_sha256": digest(array_path)}
        if manifest_path.exists() and read(manifest_path) != manifest:
            raise RuntimeError("Orphan score manifest/hash differs from its array payload")
        if not manifest_path.exists():
            atomic_json(manifest_path, manifest)
        atomic_json(pointer_path, {"manifest": manifest_path.name, "manifest_sha256": digest(manifest_path)})
        cursor, generation = count, candidate_generation
        predictions, actual, origins = value["prediction"], value["actual"], value["origins"]

    committed_cursor = cursor

    def save_progress():
        nonlocal generation, committed_cursor
        generation += 1
        name = f"score_generation_{generation:08d}"
        array_path = folder / (name + ".npz")
        atomic_npz(array_path, prediction=predictions, actual=actual, origins=origins,
                   score_identity=np.asarray(score_identity), generation=np.asarray(generation),
                   cursor=np.asarray(cursor), previous_cursor=np.asarray(committed_cursor), total_origins=np.asarray(total))
        manifest = {"score_identity": score_identity, "binding": binding,
                    "generation": generation, "cursor": cursor, "total_origins": total,
                    "arrays": array_path.name, "arrays_sha256": digest(array_path)}
        manifest_path = folder / (name + ".json")
        atomic_json(manifest_path, manifest)
        atomic_json(pointer_path, {"manifest": manifest_path.name, "manifest_sha256": digest(manifest_path)})
        committed_cursor = cursor

    if not pointer_path.exists():
        save_progress()
    batches_at_entry = 0
    with torch.no_grad():
        while cursor < total:
            if (STOP_REQUESTED or time.time() >= deadline or
                    (pause_batches is not None and batches_at_entry >= pause_batches)):
                return {"status": "paused", "phase": "stop_scoring", "cursor": cursor,
                        "total_origins": total, "score_identity": score_identity,
                        "completed_batches_preserved": True, "Q3_read": False, "outer_read": False}
            end = min(cursor + 128, total)
            positions = np.arange(cursor, end)
            x, anchor = provider.inputs("stop", positions)
            batch_prediction = provider.inverse(model(x, anchor).numpy())
            batch_truth = provider.raw_labels("stop", positions)
            if not np.isfinite(batch_prediction).all() or not np.isfinite(batch_truth).all():
                raise RuntimeError("Nonfinite final prediction/truth")
            predictions = np.concatenate((predictions, batch_prediction))
            actual = np.concatenate((actual, batch_truth))
            origins = np.concatenate((origins, provider.data.splits["stop"][cursor:end]))
            cursor = end
            batches_at_entry += 1
            save_progress()
    return {"status": "completed", "prediction": predictions, "actual": actual, "origins": origins, "cursor": cursor}


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("engineering", "cost", "train", "score", "freeze-models"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--job", type=int)
    parser.add_argument("--engineering-child", action="store_true")
    parser.add_argument("--case-folder")
    parser.add_argument("--pause-updates", type=int)
    parser.add_argument("--child-deadline", type=float)
    parser.add_argument("--batch-delay", type=float, default=0.)
    parser.add_argument("--synthetic-arm", choices=("A", "B", "C"), default="B")
    parser.add_argument("--training-deadline-seconds", type=float)
    parser.add_argument("--pause-mean-batches", type=int)
    args = parser.parse_args()
    protocol, protocol_hash = frozen_protocol(args.protocol)
    configure_runtime(protocol)
    if args.engineering:
        from engineering import child_case, run_engineering
        if args.engineering_child:
            result = child_case(args, protocol, protocol_hash)
        else:
            result = run_engineering(args, protocol, protocol_hash)
        print(json.dumps(result, sort_keys=True))
        return
    mode = "freeze" if args.freeze_models else "cost" if args.cost else "train" if args.train else "score"
    admission = market_admission(args.admission, protocol_hash, mode, protocol)
    if args.freeze_models:
        print(json.dumps(freeze_models(protocol_hash), sort_keys=True))
        return
    jobs = job_matrix()
    if args.job is None or not 0 <= args.job < len(jobs):
        raise RuntimeError("A registered job index 0..23 is required")
    cfg = jobs[args.job]
    if mode == "score":
        deadline = min(float(protocol["deadline_epoch"]), float(admission["deadline_epoch"]))
        result = score_final(cfg, protocol_hash, admission["model_freeze_sha256"], deadline)
    else:
        provider = MarketProvider(cfg)
        binding = {"protocol_sha256": protocol_hash, "mode": mode, "job": cfg, "provider": provider.identity}
        identity = hashlib.sha256(canonical(binding)).hexdigest()
        folder = BASE / ("cost" if mode == "cost" else "runs") / cfg["id"]
        deadline = min(float(protocol["deadline_epoch"]), float(admission["deadline_epoch"]))
        result = train_fixed(cfg, provider, folder, identity, deadline,
                             pause_updates=int(admission["maximum_cost_updates"]) if mode == "cost" else None,
                             execution_binding=binding)
        if mode == "cost" and result["status"] == "completed":
            raise RuntimeError("A cost run cannot be recorded as production training")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
