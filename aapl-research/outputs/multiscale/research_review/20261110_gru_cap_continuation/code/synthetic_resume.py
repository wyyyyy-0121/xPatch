"""Synthetic-only engineering check of unchanged run_fit and cap metadata bridge.

No market arrays, original checkpoints, original prediction files, network or
market updates are admitted. Execution requires its own frozen finite protocol.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import sys
import tempfile
import time


ROOT = Path("/home/ayaka/wslcode/aapl-research")
SOURCE = ROOT / "outputs/multiscale/controlled_followup/20261108_v2/code/reference_source"
STAGE = ROOT / "outputs/multiscale/research_review/20261110_gru_cap_continuation"
TRAINING_SOURCES = ("multiscale/training.py", "multiscale/models.py", "multiscale/features.py",
                    "multiscale/bars.py", "multiscale/mechanism.py", "aapl/models.py",
                    "aapl/vendor/xpatch_network.py")
EXPECTED_RUNTIME = {"python": "3.13.13", "numpy": "2.5.4", "torch": "2.14.1+cu130",
                    "cuda": "13.0", "device": "cpu", "gpu": None}
PROTOCOL = None
OUT = None
np = None
torch = None


class DeadlineExpired(Exception):
    pass


def deadline_signal(*_):
    raise DeadlineExpired("signal or immutable synthetic deadline")


def check():
    if PROTOCOL is None or time.time() >= float(PROTOCOL["deadline_epoch"]):
        raise DeadlineExpired("immutable synthetic deadline")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            check()
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix="tmp", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def same(a, b):
    """Exact dtype/shape and storage bytes, including scalar floating bits."""
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
                and all(same(a[key], b[key]) for key in a))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(same(x, y) for x, y in zip(a, b)))
    if isinstance(a, float) or isinstance(b, float):
        return type(a) is type(b) and struct.pack("!d", a) == struct.pack("!d", b)
    return type(a) is type(b) and a == b


def require_same(a, b, reason):
    if not same(a, b):
        raise ValueError(reason)


def install_synthetic_guard(protocol_path):
    """Constrain subsequent file access to code/runtime and this synthetic run."""
    runtime_roots = {Path(sys.base_prefix).resolve(), Path(sys.prefix).resolve(),
                     Path(np.__file__).resolve().parent, Path(torch.__file__).resolve().parent}
    for entry in sys.path:
        if entry:
            candidate = Path(entry).resolve()
            if candidate.name in ("site-packages", "dist-packages"):
                runtime_roots.add(candidate)
    known_reads = {protocol_path.resolve(), Path(__file__).resolve()}
    counts = {"read_open_events": 0, "write_open_events": 0, "blocked_events": 0}

    def below(path, parent):
        return path == parent or path.is_relative_to(parent)

    def path_value(value):
        if isinstance(value, (str, bytes, os.PathLike)):
            return Path(os.fsdecode(value)).resolve()
        return None  # An already-open file descriptor, not a new path.

    def permitted_read(path):
        if path.is_relative_to(SOURCE) and path.suffix == ".pyc":
            return False
        if path.suffix in (".pt", ".npy", ".npz", ".csv", ".parquet"):
            return below(path, OUT)
        return (path in known_reads or below(path, OUT) or below(path, SOURCE)
                or any(below(path, item) for item in runtime_roots)
                or below(path, Path("/proc")) or below(path, Path("/sys"))
                or path in (Path("/dev/urandom"), Path("/dev/null")))

    def deny(event):
        counts["blocked_events"] += 1
        raise PermissionError("Synthetic access guard: " + event)

    def audit(event, args):
        if event.startswith("socket.") or event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.fork"):
            deny(event)
        if event == "open":
            path = path_value(args[0])
            if path is None:
                return
            mode, flags = args[1], args[2]
            writing = ((isinstance(mode, str) and any(c in mode for c in "wax+"))
                       or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))))
            if writing:
                if not below(path, OUT):
                    deny("write " + str(path))
                counts["write_open_events"] += 1
            else:
                if not permitted_read(path):
                    deny("read " + str(path))
                counts["read_open_events"] += 1
        if event in ("os.mkdir", "os.remove", "os.rmdir", "os.rename"):
            values = args[:2] if event == "os.rename" else args[:1]
            for value in values:
                path = path_value(value)
                if path is not None and not below(path, OUT):
                    deny(event + " " + str(path))
        if event in ("os.link", "os.symlink", "os.truncate"):
            deny(event)

    sys.addaudithook(audit)
    return counts


def gate(protocol_path):
    global PROTOCOL, OUT
    protocol_path = protocol_path.resolve()
    PROTOCOL = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (PROTOCOL.get("status") != "frozen_before_execution"
            or PROTOCOL.get("engineering_synthetic_admitted") is not True
            or PROTOCOL.get("market_arrays_admitted") is not False
            or PROTOCOL.get("training_market_admitted") is not False
            or PROTOCOL.get("Q3_allowed", False) or PROTOCOL.get("network_allowed", False)):
        raise ValueError("A frozen synthetic-only protocol is required")
    deadline = float(PROTOCOL["deadline_epoch"])
    if not math.isfinite(deadline):
        raise ValueError("Finite absolute deadline required")
    check()
    signal.signal(signal.SIGALRM, deadline_signal)
    signal.signal(signal.SIGTERM, deadline_signal)
    signal.signal(signal.SIGINT, deadline_signal)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, deadline - time.time()))
    if Path(PROTOCOL["source_root"]).resolve() != SOURCE:
        raise ValueError("Original source snapshot required")
    if set(os.sched_getaffinity(0)) != {26} or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("CPU26 and hidden GPUs required")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(name) != "1":
            raise ValueError("Exactly one numerical thread required: " + name)
    required = {str(Path(__file__).resolve())} | {str(SOURCE / name) for name in TRAINING_SOURCES}
    if not required.issubset(PROTOCOL["code_sha256"]):
        raise ValueError("Script and original training identity sources must be registered")
    for name, expected in PROTOCOL["code_sha256"].items():
        path = Path(name).resolve()
        if path != Path(__file__).resolve() and not path.is_relative_to(SOURCE):
            raise ValueError("Unexpected source registration")
        if sha(path) != expected:
            raise ValueError("Source changed: " + str(path))
    OUT = Path(PROTOCOL["output_directory"]).resolve()
    if not OUT.is_relative_to(STAGE) or OUT == STAGE or OUT.is_relative_to(STAGE / "code"):
        raise ValueError("A separate output directory within this stage is required")
    if OUT.exists():
        raise FileExistsError("Refusing to reset a previous synthetic run: " + str(OUT))
    OUT.mkdir(parents=True, exist_ok=False)
    return protocol_path


def load_pt(directory, name):
    check()
    path = (directory / name).resolve()
    if not path.is_relative_to(OUT) or name not in ("latest.pt", "best.pt"):
        raise ValueError("Only locally generated synthetic checkpoints may be loaded")
    return torch.load(path, map_location="cpu", weights_only=True)


def fixture(Bars, CloseWindows):
    local_rng = np.random.default_rng(2026)
    count = 640
    position = np.arange(count, dtype=np.float64)
    log_close = (4.6 + 0.0001 * position + 0.006 * np.sin(position / 17.0)
                 + 0.002 * np.cos(position / 7.0) + local_rng.normal(0.0, 0.0005, count))
    start = (np.int64(946684800000000000)
             + np.arange(count, dtype=np.int64) * np.int64(3600000000000))
    dates = start.astype("datetime64[ns]").astype("datetime64[D]").astype("U10")
    bars = Bars("hour", start, start + np.int64(3600000000000), np.exp(log_close),
                np.ones(count, dtype=np.bool_), dates)
    windows = CloseWindows(bars, 96)
    fit = np.arange(95, 480, dtype=np.int64)  # 128 + 128 + 129, including singleton merge.
    stop = np.arange(500, 532, dtype=np.int64)
    pieces = {"recipe": "local_synthetic_gru_resume_v1", "lookback": 96, "horizon": 8,
              "fit_count": len(fit), "stop_count": len(stop), "arrays": {}}
    for name, value in (("start", start), ("close", bars.close), ("fit", fit), ("stop", stop)):
        pieces["arrays"][name] = {"dtype": value.dtype.str, "shape": list(value.shape),
                                  "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    data_id = hashlib.sha256(json.dumps(pieces, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return windows, fit, stop, data_id, pieces


def timed_fit(run_fit, windows, fit, stop, config, directory, data_id, *, resume, one_batch):
    check()
    before = time.perf_counter()
    result = run_fit(windows, fit, stop, config, directory, data_id, device="cpu",
                     resume=resume, stop_after_steps=1 if one_batch else None)
    elapsed = time.perf_counter() - before
    check()
    return result, elapsed


def step_to_completion(run_fit, windows, fit, stop, config, directory, data_id, *, resume):
    trace, durations = [], []
    for call in range(32):
        result, elapsed = timed_fit(run_fit, windows, fit, stop, config, directory, data_id,
                                   resume=resume, one_batch=True)
        resume = True
        durations.append(elapsed)
        payload = load_pt(directory, "latest.pt")
        state = payload["state"]
        trace.append({"call": call + 1, "status": result["status"], "wall_seconds": elapsed,
                      "epoch": state["epoch"], "steps": state["steps"], "cursor": state["cursor"],
                      "permutation_present": state["permutation"] is not None,
                      "best_epoch": state["best_epoch"]})
        if result["status"] == "completed":
            return result, trace, durations
        if result["status"] != "interrupted_for_test":
            raise ValueError("Unexpected original run_fit status")
    raise ValueError("Synthetic step loop exceeded its predeclared finite call count")


def compare_outputs(left, right, label):
    for filename in ("latest.pt", "best.pt"):
        require_same(load_pt(left, filename), load_pt(right, filename), label + ": " + filename)
    a = json.loads((left / "training_summary.json").read_text(encoding="utf-8"))
    b = json.loads((right / "training_summary.json").read_text(encoding="utf-8"))
    require_same(a, b, label + ": summary")


def bridge(source_directory, destination, config2, config4, new_identity, atomic_torch_save):
    original = {name: load_pt(source_directory, name) for name in ("latest.pt", "best.pt")}
    require_same(original["best.pt"]["config"], asdict(config2), "Source synthetic best config")
    if original["latest.pt"]["identity"] == new_identity:
        raise ValueError("Cap intervention must have a distinct identity")
    destination.mkdir(exist_ok=False)
    changed = copy.deepcopy(original)
    changed["latest.pt"]["identity"] = new_identity
    changed["best.pt"]["identity"] = new_identity
    changed["best.pt"]["config"] = asdict(config4)
    expected_config = asdict(config2)
    expected_config["max_epochs"] = 4
    require_same(expected_config, asdict(config4), "Only max_epochs may change")
    for name in ("latest.pt", "best.pt"):
        restored = copy.deepcopy(changed[name])
        restored["identity"] = original[name]["identity"]
        if name == "best.pt":
            restored["config"]["max_epochs"] = 2
        require_same(original[name], restored, "Bridge changed scientific payload: " + name)
        atomic_torch_save(changed[name], destination / name)
        require_same(changed[name], load_pt(destination, name), "Bridge disk roundtrip: " + name)
    receipt = {"old_identity": original["latest.pt"]["identity"], "new_identity": new_identity,
               "allowed_changes": {"latest.pt": ["identity"],
                                   "best.pt": ["identity", "config.max_epochs"]},
               "all_scientific_payload_bytes_preserved": True,
               "original_sha256": {name: sha(source_directory / name) for name in original},
               "bridge_sha256": {name: sha(destination / name) for name in original}}
    atomic_json(destination / "BRIDGE_RECEIPT.json", receipt)
    return receipt


def execute(protocol_path):
    global np, torch
    started = time.perf_counter()
    import numpy as numpy_module
    import torch as torch_module
    import torch._dynamo  # Resolve Adam's lazy imports before the file audit hook.
    np, torch = numpy_module, torch_module
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    runtime = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": str(torch.__version__),
               "cuda": torch.version.cuda, "device": "cpu", "gpu": None}
    if runtime != EXPECTED_RUNTIME:
        raise ValueError("Original CPU runtime not reproduced: " + repr(runtime))
    sys.dont_write_bytecode = True
    sys.pycache_prefix = str(OUT / "unused_pycache")
    sys.path.insert(0, str(SOURCE))
    from multiscale.bars import Bars
    from multiscale.features import CloseWindows
    from multiscale.training import TrainConfig, run_fit, identity, atomic_torch_save
    for name in ("multiscale.bars", "multiscale.features", "multiscale.training", "multiscale.models"):
        if not Path(sys.modules[name].__file__).resolve().is_relative_to(SOURCE):
            raise ValueError("Unexpected imported original module: " + name)
    access = install_synthetic_guard(protocol_path)
    atomic_json(OUT / "STATUS.json", {"status": "running", "deadline_epoch": PROTOCOL["deadline_epoch"],
                                      "market_arrays_read": False, "market_updates": 0})
    windows, fit, stop, data_id, fixture_receipt = fixture(Bars, CloseWindows)
    config4 = TrainConfig(model="gru", lookback=96, horizon=8, target="future_log_return", max_epochs=4,
                          patience=10, batch_size=128, learning_rate=0.001, seed=2026,
                          checkpoint_batches=100, shared_horizons=())
    config2 = replace(config4, max_epochs=2)
    direct, stepped, capped, extended = (OUT / name for name in ("direct4", "stepped4", "capped2", "extended4"))
    result_direct, direct_time = timed_fit(run_fit, windows, fit, stop, config4, direct, data_id,
                                          resume=False, one_batch=False)
    result_step, step_trace, step_times = step_to_completion(run_fit, windows, fit, stop, config4, stepped,
                                                           data_id, resume=False)
    require_same(result_direct, result_step, "Continuous vs stepped returned summary")
    compare_outputs(direct, stepped, "Continuous vs one-batch resume")
    if not any(0 < item["cursor"] < len(fit) and item["permutation_present"] for item in step_trace):
        raise ValueError("No checkpoint inside an epoch was exercised")
    if not any(item["cursor"] == len(fit) and item["permutation_present"] for item in step_trace):
        raise ValueError("No final-batch-before-epoch-finalization checkpoint exercised")
    if {item["epoch"] for item in step_trace} != set(range(5)):
        raise ValueError("Not all epoch boundaries were exercised")
    result_cap, cap_time = timed_fit(run_fit, windows, fit, stop, config2, capped, data_id,
                                    resume=False, one_batch=False)
    if result_cap["epochs"] != 2 or result_cap["stopped_early"]:
        raise ValueError("Synthetic cap2 did not finish exactly at its cap")
    bridge_receipt = bridge(capped, extended, config2, config4,
                            identity(config4, data_id, fit, stop, "cpu"), atomic_torch_save)
    result_extended, bridge_trace, bridge_times = step_to_completion(run_fit, windows, fit, stop, config4,
                                                                    extended, data_id, resume=True)
    require_same(result_direct, result_extended, "Direct cap4 vs isolated cap2→4 returned summary")
    compare_outputs(direct, extended, "Direct cap4 vs isolated cap2→4 bridge")
    direct_payload = load_pt(direct, "latest.pt")
    if direct_payload["state"]["epoch"] != 4 or direct_payload["state"]["steps"] != 12:
        raise ValueError("Expected 4 epochs and 12 optimizer updates")
    if result_direct["stopped_early"] or result_direct["epochs"] != 4:
        raise ValueError("Synthetic patience must not truncate the 4-epoch fixture")
    for path, expected in PROTOCOL["code_sha256"].items():
        if sha(path) != expected:
            raise ValueError("Source changed during verification")
    all_step_times = step_times + bridge_times
    wall_total = time.perf_counter() - started
    result = {"status": "passed", "engineering_only": True, "runtime": runtime,
              "market_arrays_read": False, "original_checkpoints_read": False,
              "market_updates": 0, "Q3_read": False, "deadline_epoch": PROTOCOL["deadline_epoch"],
              "fixture": fixture_receipt, "synthetic_config": asdict(config4),
              "scientific_real_patience_unchanged": 3,
              "continuous_vs_step_latest_best_summary_exactbytes": True,
              "cap2_to_4_bridge_latest_best_summary_exactbytes": True,
              "mid_epoch_and_final_batch_checkpoints_exercised": True,
              "step_trace": step_trace, "bridge_trace": bridge_trace, "bridge": bridge_receipt,
              "wall_seconds": {"total_including_imports": wall_total, "direct4": direct_time,
                               "direct2": cap_time, "one_batch_calls": len(all_step_times),
                               "one_batch_total": sum(all_step_times), "one_batch_max": max(all_step_times),
                               "one_batch_mean": sum(all_step_times) / len(all_step_times),
                               "721_calls_at_observed_max": 721 * max(all_step_times)},
              "call_estimate_definition": "720 additional optimizer updates plus one final epoch-finalization-only call",
              "cost_estimate_limit": "Observed synthetic timing is a planning estimate, not a bound on real calls; fit/scaler size differs.",
              "access": access, "predictive_improvement_proved": False,
              "market_continuation_automatically_admitted": False}
    check()
    atomic_json(OUT / "RESULTS.json", result)
    atomic_json(OUT / "STATUS.json", {"status": "passed", "deadline_epoch": PROTOCOL["deadline_epoch"],
                                      "engineering_only": True, "market_arrays_read": False,
                                      "market_updates": 0, "wall_total_seconds": wall_total})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    try:
        protocol_path = gate(args.protocol)
        execute(protocol_path)
    except BaseException as exc:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if OUT is not None and OUT.is_dir():
            atomic_json(OUT / "STATUS.json", {"status": "deadline_or_signal" if isinstance(exc, DeadlineExpired) else "failed",
                                              "error_type": type(exc).__name__, "error": str(exc),
                                              "engineering_only": True, "market_arrays_read": False,
                                              "market_updates": 0, "deadline_epoch": PROTOCOL["deadline_epoch"]})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
