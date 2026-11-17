"""Stage 1: verify the original GRU checkpoint without a training update.

Importing this script does not load a checkpoint. The only admitted tensor-file
inputs are the registered original latest.pt and best.pt. No market array,
forward on market data, target, scaler fitting, backward, optimizer update,
identity rewrite or cap extension is performed. The sole forward input is a
fixed locally generated synthetic tensor. A separate protocol admits execution.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import time

import numpy as np
import torch
import torch._dynamo  # Resolve Adam's lazy runtime cache before the access hook.

from cap_guard import install_guard


ROOT = Path("/home/ayaka/wslcode/aapl-research")
SOURCE = ROOT / "outputs/multiscale/controlled_followup/20261108_v2/code/reference_source"
JOB = ROOT / "outputs/multiscale/private_three_scale_development/20261107T134128870773Z/hourly_h8_future_log_return_f2"
FEASIBILITY = ROOT / "outputs/multiscale/research_review/20261110_session_alignment_r2/NEXT_CAP_FEASIBILITY.json"
IDENTITY = "910553b6471b9fcef06f325b5b30381a1c4c2e764d6d09001f4179e6685815e4"
TENSOR_SHA = {
    "latest": "72ebe367d404094c73588357289d1bddf8f1df031ad0f1a08e2b9eb85bd87f47",
    "best": "e2bf7600a26199e6f6add0fd3e8061d93309512d2611d6ca9639fb05030fc3e8",
}
CONFIG = {"model": "gru", "lookback": 96, "horizon": 8, "target": "future_log_return",
          "max_epochs": 20, "patience": 3, "batch_size": 128, "learning_rate": 0.001,
          "seed": 2026, "checkpoint_batches": 100, "shared_horizons": ()}
TRAINING_SOURCES = ("multiscale/training.py", "multiscale/models.py", "multiscale/features.py",
                    "multiscale/bars.py", "multiscale/mechanism.py", "aapl/models.py",
                    "aapl/vendor/xpatch_network.py")
LATEST_KEYS = {"schema", "identity", "scaler", "state", "model", "optimizer", "numpy_rng", "torch_rng", "cuda_rng"}
STATE_KEYS = {"epoch", "cursor", "permutation", "steps", "best_loss", "best_epoch", "best_state",
              "stale_epochs", "epoch_loss_sum", "epoch_seen", "history"}
STOP = False


class Stopped(Exception):
    pass


def receive_signal(*_):
    global STOP
    STOP = True


def check(protocol):
    if STOP or time.time() >= float(protocol["deadline_epoch"]):
        raise Stopped("signal or immutable stage-1 deadline")


def sha(path, protocol=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            if protocol is not None:
                check(protocol)
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="tmp", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_torch_save(payload, path):
    path = Path(path)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix="tmp", suffix=".pt")
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def same(a, b):
    """Recursive exact comparison, including scalar floating-point bit patterns."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.dtype == b.dtype and a.shape == b.shape and a.layout == b.layout
                and a.device == b.device and a.layout == torch.strided
                and torch.equal(a.detach().reshape(-1).contiguous().view(torch.uint8),
                                b.detach().reshape(-1).contiguous().view(torch.uint8)))
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes())
    if isinstance(a, dict) or isinstance(b, dict):
        return (isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys()
                and all(same(a[k], b[k]) for k in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return type(a) is type(b) and struct.pack("!d", a) == struct.pack("!d", b)
    return type(a) is type(b) and a == b


def require_same(a, b, message):
    if not same(a, b):
        raise ValueError(message)


def registered_json(protocol, key, expected_path):
    path = Path(protocol[key]).resolve()
    if path != expected_path.resolve():
        raise ValueError("Wrong small-metadata path: " + key)
    expected = protocol["metadata_sha256"].get(str(path))
    if not expected or sha(path, protocol) != expected:
        raise ValueError("Unregistered or changed small metadata: " + key)
    return json.loads(path.read_text(encoding="utf-8"))


def gate(protocol, protocol_path):
    if (protocol.get("status") != "frozen_before_execution"
            or protocol.get("checkpoint_no_update_admitted") is not True
            or protocol.get("market_arrays_admitted", False)
            or protocol.get("training_updates_admitted", False)
            or protocol.get("Q3_allowed", False) or protocol.get("network_allowed", False)):
        raise ValueError("Only separate stage-1 no-update admission is allowed")
    if set(os.sched_getaffinity(0)) != {26} or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Stage 1 requires logical CPU26 and hidden GPU")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("Exactly one numerical thread is required")
    if Path(protocol["source_root"]).resolve() != SOURCE.resolve():
        raise ValueError("Wrong original training source snapshot")
    if not np.isfinite(float(protocol["deadline_epoch"])):
        raise ValueError("Stage-1 deadline must be a finite absolute time")
    check(protocol)
    own, guard = Path(__file__).resolve(), Path(__file__).resolve().with_name("cap_guard.py")
    required = {str(own), str(guard)} | {str(SOURCE / rel) for rel in TRAINING_SOURCES}
    if not required.issubset(protocol["code_sha256"]):
        raise ValueError("Missing exact script/guard/original-source registration")
    for mapping in (protocol["code_sha256"], protocol["metadata_sha256"]):
        for path, expected in mapping.items():
            if Path(path).suffix not in (".py", ".json") or sha(path, protocol) != expected:
                raise ValueError("Frozen source/metadata changed or hides a binary input")
    records = protocol["checkpoints"]
    if set(records) != {"latest", "best"}:
        raise ValueError("Exactly the original latest/best pair must be admitted")
    for name in ("latest", "best"):
        if (Path(records[name]["path"]).resolve() != (JOB / "gru" / (name + ".pt")).resolve()
                or records[name]["sha256"] != TENSOR_SHA[name]):
            raise ValueError("Wrong original registered checkpoint")
    output = Path(protocol["output_directory"])
    stage = Path(__file__).resolve().parents[1]
    if not output.is_absolute() or not output.resolve().is_relative_to(stage.resolve()):
        raise ValueError("Output must stay inside the new no-update stage")
    access = install_guard(protocol, protocol_path)
    output.mkdir(parents=True, exist_ok=True)
    return output, access


def validate_metadata(protocol):
    summary = registered_json(protocol, "training_summary_path", JOB / "gru/training_summary.json")
    receipt = registered_json(protocol, "fold_receipt_path", JOB / "fold_receipt.json")
    feasibility = registered_json(protocol, "source_feasibility_path", FEASIBILITY)
    fold_fields = receipt["identity_fields"]
    require_same(receipt["config"], fold_fields["config"], "Receipt/fold configuration identity differs")
    require_same(receipt["data_identity"], fold_fields["data_identity"], "Receipt/fold data identity differs")
    require_same(receipt["source_sha256"], fold_fields["source_sha256"], "Receipt/fold source identity differs")
    require_same(fold_fields["actions"], [{"date": "2020-08-31", "ratio": 4.0}], "Original effective split metadata differs")
    fold_digest = hashlib.sha256(json.dumps(fold_fields, sort_keys=True, separators=(",", ":"),
                                            allow_nan=False).encode("utf8")).hexdigest()
    if fold_digest != receipt["fold_identity"]:
        raise ValueError("Original fold identity cannot be reconstructed from its registered metadata")
    if (summary.get("identity") != IDENTITY or summary.get("status") != "completed"
            or summary.get("epochs") != 20 or summary.get("steps") != 360
            or summary.get("best_epoch") != 20 or summary.get("stopped_early") is not False):
        raise ValueError("Original summary differs from the one capped cell")
    gru = receipt["models"]["gru"]
    for key, expected in (("epochs", 20), ("steps", 360), ("best_epoch", 20),
                          ("training_identity", IDENTITY), ("checkpoint_dir", "gru")):
        if gru[key] != expected:
            raise ValueError("Original receipt GRU identity/configuration differs")
    if (gru["latest_checkpoint_sha256"] != TENSOR_SHA["latest"]
            or gru["best_checkpoint_sha256"] != TENSOR_SHA["best"]
            or gru["training_summary_sha256"] != sha(protocol["training_summary_path"], protocol)
            or receipt["fit_origins"] != 2294 or receipt["stop_origins"] != 406):
        raise ValueError("Original receipt file/cohort identities differ")
    if feasibility["registered_not_freshly_verified_tensor_sha256"] != {name + ".pt": value for name, value in TENSOR_SHA.items()}:
        raise ValueError("Source-only feasibility described a different checkpoint pair")
    for key in ("lookback", "horizon", "target", "max_epochs", "patience", "batch_size", "learning_rate", "seed"):
        if receipt["config"][key] != CONFIG[key]:
            raise ValueError("Original receipt recipe changed: " + key)
    runtime = {"device": "cpu", "python": sys.version.split()[0], "numpy": np.__version__,
               "torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": None}
    old_runtime = receipt["identity_fields"]["runtime"]
    if any(runtime[key] != old_runtime[key] for key in runtime):
        raise ValueError("Original CPU training runtime differs")
    sources = {}
    for rel in TRAINING_SOURCES:
        actual = sha(SOURCE / rel, protocol)
        if actual != receipt["source_sha256"][rel]:
            raise ValueError("Original training source differs from its receipt")
        sources[rel] = actual
    sys.path.insert(0, str(SOURCE))
    from multiscale.training import TrainConfig
    config = asdict(TrainConfig(**CONFIG))
    require_same(config, CONFIG, "Original TrainConfig defaults/schema changed")
    identity_fields = {"config": config,
        "data_identity": receipt["data_identity"] + ":" + receipt["config"]["protocol_identity"] + ":" + receipt["fold_identity"],
        "fit_indices_sha256": receipt["identity_fields"]["fit_indices_sha256"],
        "stop_indices_sha256": receipt["identity_fields"]["stop_indices_sha256"],
        "runtime": runtime, "source_sha256": sources}
    digest = hashlib.sha256(json.dumps(identity_fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if digest != IDENTITY:
        raise ValueError("Original training identity cannot be reconstructed from verified metadata")
    return summary, receipt, runtime


def validate_payload(latest, best, summary):
    if not isinstance(latest, dict) or set(latest) != LATEST_KEYS or latest["schema"] != 1:
        raise ValueError("Unexpected original latest checkpoint schema")
    if not isinstance(best, dict) or set(best) != {"identity", "epoch", "model", "scaler", "config"}:
        raise ValueError("Unexpected original best checkpoint schema")
    if latest["identity"] != IDENTITY or best["identity"] != IDENTITY or best["epoch"] != 20:
        raise ValueError("Original payload identity/epoch differs")
    require_same(best["config"], CONFIG, "Original best payload TrainConfig differs")
    state = latest["state"]
    if not isinstance(state, dict) or set(state) != STATE_KEYS:
        raise ValueError("Missing original sampler/optimizer-history continuation state")
    for key, expected in (("epoch", 20), ("cursor", 0), ("permutation", None), ("steps", 360),
                          ("best_epoch", 20), ("stale_epochs", 0), ("epoch_loss_sum", 0.0), ("epoch_seen", 0)):
        require_same(state[key], expected, "Unexpected original terminal state: " + key)
    require_same(state["history"], summary["history"], "Checkpoint history differs from original summary")
    require_same(state["best_loss"], summary["inner_stop_metric"], "Checkpoint best loss differs from summary")
    require_same(latest["model"], state["best_state"], "Epoch20 model differs from selected epoch20 best state")
    require_same(latest["model"], best["model"], "Original latest and best weights differ")
    require_same(latest["scaler"], best["scaler"], "Original latest/best scalers differ")
    scaler = latest["scaler"]
    if (not isinstance(scaler, dict) or set(scaler) != {"x_mean", "x_std", "anchor_mean", "anchor_std", "y_mean", "y_std", "fitted_origin_count"}
            or scaler["fitted_origin_count"] != 2294 or len(scaler["y_mean"]) != 8 or len(scaler["y_std"]) != 8):
        raise ValueError("Original complete scaler schema/count differs")
    scalars = [scaler[k] for k in ("x_mean", "x_std", "anchor_mean", "anchor_std")] + list(scaler["y_mean"]) + list(scaler["y_std"])
    if not np.isfinite(scalars).all() or scaler["x_std"] <= 0 or scaler["anchor_std"] <= 0 or np.any(np.asarray(scaler["y_std"]) <= 0):
        raise ValueError("Nonfinite or invalid original scaler")
    if latest["cuda_rng"] is not None:
        raise ValueError("Original CPU checkpoint unexpectedly contains CUDA RNG")
    rng = latest["torch_rng"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1 or rng.device.type != "cpu":
        raise ValueError("Missing original CPU Torch RNG state")


def restore_instance(payload, make_model):
    model = make_model("gru", 96, 8).to("cpu")
    if model.rnn.hidden_size != 32 or model.rnn.num_layers != 1:
        raise ValueError("Original GRU architecture differs")
    expected_state = model.state_dict()
    if expected_state.keys() != payload["model"].keys():
        raise ValueError("Original model state keys differ")
    for key, value in payload["model"].items():
        if (not isinstance(value, torch.Tensor) or value.dtype != expected_state[key].dtype
                or value.shape != expected_state[key].shape or value.device.type != "cpu"
                or not torch.isfinite(value).all()):
            raise ValueError("Original model tensor invalid: " + key)
    model.load_state_dict(payload["model"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    saved = payload["optimizer"]
    if not isinstance(saved, dict) or set(saved) != {"state", "param_groups"} or len(saved["param_groups"]) != 1:
        raise ValueError("Unexpected original Adam schema")
    default_group = optimizer.state_dict()["param_groups"][0]
    group = saved["param_groups"][0]
    require_same({k: v for k, v in group.items() if k != "params"},
                 {k: v for k, v in default_group.items() if k != "params"}, "Original Adam hyperparameters/defaults differ")
    if len(group["params"]) != len(list(model.parameters())) or set(saved["state"]) != set(group["params"]):
        raise ValueError("Adam parameter IDs/count do not cover the original model")
    for parameter_id, parameter in zip(group["params"], model.parameters()):
        entry = saved["state"][parameter_id]
        if set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError("Original default Adam moment schema differs")
        step = entry["step"]
        if (not isinstance(step, torch.Tensor) or step.ndim != 0 or step.dtype != torch.float32
                or step.device.type != "cpu" or not torch.isfinite(step) or float(step) != 360):
            raise ValueError("Original Adam parameter step is not360")
        for name in ("exp_avg", "exp_avg_sq"):
            value = entry[name]
            if (not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype
                    or value.device.type != "cpu" or not torch.isfinite(value).all()):
                raise ValueError("Original Adam moment shape/dtype/finiteness differs")
            if name == "exp_avg_sq" and torch.any(value < 0):
                raise ValueError("Original Adam second moment is negative")
    optimizer.load_state_dict(saved)
    require_same(model.state_dict(), payload["model"], "Strict model restore changed original state")
    require_same(optimizer.state_dict(), saved, "Strict Adam restore changed original state")
    numpy_rng = np.random.default_rng(0)
    numpy_rng.bit_generator.state = copy.deepcopy(payload["numpy_rng"])
    require_same(numpy_rng.bit_generator.state, payload["numpy_rng"], "NumPy sampler RNG restore differs")
    torch_rng = torch.Generator(device="cpu")
    torch_rng.set_state(payload["torch_rng"].clone())
    require_same(torch_rng.get_state(), payload["torch_rng"], "Torch local RNG restore differs")
    from multiscale.features import Scaler
    scaler = asdict(Scaler(**copy.deepcopy(payload["scaler"])))
    require_same(scaler, payload["scaler"], "Original Scaler dataclass restore differs")
    return model, optimizer, numpy_rng, torch_rng, scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, receive_signal)
    output, access = gate(protocol, protocol_path)
    protocol_sha = sha(protocol_path, protocol)
    result = {"status": "running_no_update_checkpoint_verification", "protocol_sha256": protocol_sha,
              "deadline_epoch": protocol["deadline_epoch"], "identity_rewritten": False,
              "training_updates": 0, "backward_calls": 0, "market_forward_calls": 0,
              "market_arrays_read": False, "Q3_read": False, "cap_extension_admitted": False,
              "source_artifacts_modified": False, "access_guard": access}
    atomic_json(output / "STATUS.json", result)
    try:
        summary, receipt, runtime = validate_metadata(protocol)
        # Verify both original byte identities BEFORE either torch.load.
        for name in ("latest", "best"):
            if sha(protocol["checkpoints"][name]["path"], protocol) != TENSOR_SHA[name]:
                raise ValueError("Original tensor-file SHA differs: " + name)
        check(protocol)
        latest = torch.load(protocol["checkpoints"]["latest"]["path"], weights_only=True, map_location="cpu")
        check(protocol)
        best = torch.load(protocol["checkpoints"]["best"]["path"], weights_only=True, map_location="cpu")
        validate_payload(latest, best, summary)
        original_snapshot = copy.deepcopy(latest)
        copied_path = output / "CHECKPOINT_COPY.pt"
        check(protocol)
        atomic_torch_save(latest, copied_path)
        copied = torch.load(copied_path, weights_only=True, map_location="cpu")
        require_same(latest, copied, "Whole original payload differs after independent disk roundtrip")
        sys.path.insert(0, str(SOURCE))
        from multiscale.models import make_model
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        global_torch_before = torch.get_rng_state().clone()
        global_numpy_before = copy.deepcopy(np.random.get_state())
        with torch.random.fork_rng(devices=[], enabled=True):
            a = restore_instance(latest, make_model)
            b = restore_instance(copied, make_model)
            model_a, opt_a, np_a, gen_a, scaler_a = a
            model_b, opt_b, np_b, gen_b, scaler_b = b
            model_before = copy.deepcopy(model_a.state_dict())
            optimizer_before = copy.deepcopy(opt_a.state_dict())
            require_same(model_before, model_b.state_dict(), "Two restored model states differ")
            require_same(optimizer_before, opt_b.state_dict(), "Two restored Adam states differ")
            require_same(scaler_a, scaler_b, "Two restored scalers differ")
            x = torch.linspace(-1.0, 1.0, 128 * 96, dtype=torch.float32).reshape(128, 96, 1)
            anchor = torch.linspace(-0.5, 0.5, 128, dtype=torch.float32).reshape(128, 1)
            model_a.eval(); model_b.eval()
            check(protocol)
            with torch.inference_mode():
                out_a, out_b = model_a(x, anchor), model_b(x, anchor)
            if out_a.shape != (128, 8) or not torch.isfinite(out_a).all():
                raise ValueError("Invalid synthetic-only inference output")
            require_same(out_a, out_b, "Synthetic inference differs bitwise after original-payload roundtrip")
            require_same(model_a.state_dict(), model_before, "Synthetic inference changed model parameters/buffers")
            require_same(model_b.state_dict(), model_before, "Roundtrip model changed parameters/buffers")
            require_same(opt_a.state_dict(), optimizer_before, "Synthetic inference changed original Adam state")
            require_same(opt_b.state_dict(), optimizer_before, "Synthetic inference changed copied Adam state")
            require_same(np_a.bit_generator.state, latest["numpy_rng"], "Synthetic inference changed sampler RNG")
            require_same(np_b.bit_generator.state, latest["numpy_rng"], "Copied sampler RNG changed")
            require_same(gen_a.get_state(), latest["torch_rng"], "Synthetic inference changed local Torch RNG")
            require_same(gen_b.get_state(), latest["torch_rng"], "Copied local Torch RNG changed")
            # Advance only independent local generators; no optimizer update.
            permutation_a, permutation_b = np_a.permutation(2294), np_b.permutation(2294)
            require_same(permutation_a, permutation_b, "Next original-fit sampling permutations differ")
            next_a = torch.rand(128, generator=gen_a, dtype=torch.float32)
            next_b = torch.rand(128, generator=gen_b, dtype=torch.float32)
            require_same(next_a, next_b, "Next local Torch random vectors differ")
        require_same(torch.get_rng_state(), global_torch_before, "Verification changed caller global Torch RNG")
        require_same(np.random.get_state(), global_numpy_before, "Verification changed caller global NumPy RNG")
        require_same(latest, original_snapshot, "Verification mutated original loaded payload")
        require_same(copied, original_snapshot, "Verification mutated roundtrip payload")
        for name in ("latest", "best"):
            if sha(protocol["checkpoints"][name]["path"], protocol) != TENSOR_SHA[name]:
                raise ValueError("Original checkpoint file changed during no-update verification")
        check(protocol)
        result.update({"status": "passed_no_update_original_checkpoint_verification",
            "fresh_original_checkpoint_sha256": TENSOR_SHA, "training_identity": IDENTITY,
            "runtime": runtime, "epochs": 20, "steps": 360, "best_epoch": 20,
            "sampler_cursor": 0, "permutation": None, "stale_epochs": 0,
            "whole_payload_roundtrip_bitwise": True, "model_optimizer_scaler_restore_exact": True,
            "synthetic_inference_output_bitwise": True, "synthetic_forward_calls": 2,
            "model_optimizer_payload_unchanged": True, "caller_global_rng_unchanged": True,
            "next_fit_permutation_2294_bitwise": True, "next_torch_vector_128_bitwise": True,
            "checkpoint_copy_sha256": sha(copied_path, protocol), "completed_epoch": time.time(),
            "research_goal_achieved": False, "predictive_gain_established": False})
        atomic_json(output / "RESULTS.json", result)
        atomic_json(output / "STATUS.json", result)
        print(json.dumps({"status": result["status"], "training_updates": 0, "market_arrays_read": False}))
    except BaseException as exc:
        result.update(status="paused_no_update_verification" if isinstance(exc, Stopped) else "failed_no_update_verification",
                      error_type=type(exc).__name__, error=str(exc), deadline_must_not_reset=True)
        atomic_json(output / "STATUS.json", result)
        raise


if __name__ == "__main__":
    main()
