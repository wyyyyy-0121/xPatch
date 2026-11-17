"""One independently frozen lattice-DGP C/D input pair; no market-data inputs.

Original LSTM32, path loss, optimizer, stopping and update checkpointing retained.
A new artificial distribution: this does not repair the closed Gaussian transform.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time

STOP = False
RUNTIME_ROOTS = ["/home/ayaka/.venvs/xpatch-ls/lib/python3.13/site-packages",
                 "/home/ayaka/.venvs/xpatch-v2/lib/python3.13/site-packages",
                 "/home/ayaka/wslcode/aapl-research/.venv/lib/python3.13/site-packages"]


class Paused(Exception):
    pass


def handle_signal(_number, _frame):
    global STOP
    STOP = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def initial_model_sha(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode("utf-8"))
        result.update(str(tensor.dtype).encode("ascii"))
        result.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return result.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("w") as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save_checkpoint(torch, path, payload):
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def check_stop(protocol):
    if STOP or time.time() >= protocol["deadline_epoch"] - 10:
        raise Paused("signal_or_original_absolute_deadline")


def prepare(args):
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    assert protocol["market_data_allowed"] is False and protocol["GPU"] is False
    assert protocol["paid_resources"] is False
    assert protocol.get("Q3_allowed", False) is False
    assert set(os.sched_getaffinity(0)) == {24} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert protocol["counts"] == [2048, 512, 4096] and protocol["shapes"] == [[480, 60], [96, 8]]
    assert protocol["data_seed"] == 20261110 and protocol["alphas"] == [.25]
    assert protocol["training_seeds"] == [2026, 2027, 2028]
    assert protocol["max_epochs"] == 20 and protocol["patience"] == 3
    assert protocol["batch_size"] == 128 and protocol["learning_rate"] == .001
    assert math.isfinite(protocol["deadline_epoch"])
    sources = {Path(p).resolve(): digest for p, digest in protocol["source_sha256"].items()}
    assert sources.get(Path(__file__).resolve()) == sha(__file__)
    for path, digest in sources.items():
        assert path.suffix in {".py", ".json", ".md"} and path.stat().st_size < 2_000_000
        assert sha(path) == digest, "Frozen source changed: " + str(path)
    native_source = Path(protocol["native_model_source"]).resolve()
    assert native_source in sources
    root, output = Path(protocol["output_root"]).resolve(), Path(args.output).resolve()
    assert output.is_relative_to(root)
    output.mkdir(parents=True, exist_ok=True)
    sources = set(sources) | {protocol_path}
    runtime_tmp = root / "runtime_tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(runtime_tmp)
    for key in ["TMPDIR", "TEMP", "TMP"]:
        os.environ[key] = str(runtime_tmp)
    runtime = [Path(p).resolve() for p in RUNTIME_ROOTS +
               ["/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", sys.prefix, sys.base_prefix]]
    for path, digest in protocol.get("runtime_file_sha256", {}).items():
        assert any(Path(path).resolve().is_relative_to(r) for r in runtime) and sha(path) == digest

    def audit(event, values):
        if event.startswith("socket."):
            raise PermissionError("No network input in artificial pair")
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode, flags = values[1], values[2]
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if write and not path.is_relative_to(root) and str(path) != "/dev/null":
            raise PermissionError("Write outside own artificial outputs: " + str(path))
        if not write and path not in sources and not path.is_relative_to(root) and not any(
                path.is_relative_to(r) for r in runtime):
            raise PermissionError("Unadmitted artificial pair read: " + str(path))

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[key] = "1"
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    nodes = [n for n in ast.parse(native_source.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name == "Recurrent"]
    assert len(nodes) == 1
    namespace = {"nn": torch.nn, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native_source), "exec"), namespace)
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    check_stop(protocol)
    return protocol, output, np, torch, namespace["Recurrent"], sha(protocol_path)


def generate(np, protocol, lookback, horizon):
    nfit, nstop, ntest = protocol["counts"]
    n = nfit + nstop + ntest
    rng = np.random.default_rng(protocol["data_seed"] + lookback)
    increments = rng.standard_normal((n, lookback - 1))
    raw = np.zeros((n, lookback), dtype=np.float64)
    raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(lookback)
    assert np.isfinite(raw).all() and bool((np.abs(raw) <= 32).all()), "Raw bound failed; no clipping/resampling"
    quantum = 2.0 ** -12
    k = np.rint(raw / quantum).astype(np.int64)
    assert bool((np.abs(k) <= 131072).all()) and bool((k[:, -1] == 0).all())
    difference = np.zeros_like(k)
    difference[:, 1:] = np.diff(k, axis=1)
    assert bool((np.abs(difference) <= 262144).all())
    z = math.sqrt(lookback) * quantum * (k[:, -1] - k[:, -2]).astype(np.float64)
    noise = rng.standard_normal((n, horizon)).cumsum(axis=1)
    factor = np.arange(1, horizon + 1, dtype=np.float64) / math.sqrt(horizon)
    arms = {}
    for arm, integer in (("C", k), ("D", difference)):
        fit_max = int(np.abs(integer[:nfit]).max())
        scale = 1 << (fit_max - 1).bit_length() if fit_max else 1
        assert 1 <= scale <= 262144
        x = (integer.astype(np.float64) / scale).astype(np.float32)
        arms[arm] = {"x": x, "x_mean": 0.0, "x_std": float(scale), "fit_max": fit_max}
    return {"K": k, "D": difference, "z": z, "noise": noise, "factor": factor,
            "arms": arms, "quantum": quantum, "max_abs_raw": float(np.abs(raw).max())}


def arm_data(base, arm):
    data = dict(base)
    data.update(base["arms"][arm])
    data["arm"] = arm
    data["input_identity"] = {"arm": arm, "K_sha256": array_sha(base["K"]),
        "D_sha256": array_sha(base["D"]), "x_sha256": array_sha(data["x"]),
        "z_sha256": array_sha(base["z"]), "noise_sha256": array_sha(base["noise"]),
        "x_scale": data["x_std"], "quantum": base["quantum"]}
    return data


def qualify_data(np, protocol, base, shape):
    nfit, nstop, _ = protocol["counts"]
    record = {"shape": shape, "quantum": base["quantum"], "max_abs_raw": base["max_abs_raw"],
              "K_sha256": array_sha(base["K"]), "D_sha256": array_sha(base["D"]),
              "z_sha256": array_sha(base["z"]), "noise_sha256": array_sha(base["noise"]), "arms": {}}
    y, mean, _, _ = labels(np, base, .25, nfit)
    assert np.isfinite(y).all() and np.isfinite(mean).all()
    record.update(target_sha256=array_sha(y), known_mean_sha256=array_sha(mean))
    for arm in ("C", "D"):
        data = arm_data(base, arm)
        decoded = data["x"].astype(np.float64) * data["x_std"]
        expected = base["K"] if arm == "C" else base["D"]
        assert np.array_equal(decoded, expected.astype(np.float64))
        integer = decoded.astype(np.int64)
        assert integer.tobytes() == expected.tobytes()
        if arm == "D":
            reconstructed = np.zeros_like(integer)
            reconstructed[:, :-1] = -np.cumsum(integer[:, :0:-1], axis=1, dtype=np.int64)[:, ::-1]
        else:
            reconstructed = integer
        zhat = math.sqrt(shape[0]) * base["quantum"] * (reconstructed[:, -1] - reconstructed[:, -2]).astype(np.float64)
        mhat = .25 * zhat[:, None] * base["factor"][None, :]
        splits = []
        for name, selection in (("fit", slice(0, nfit)), ("stop", slice(nfit, nfit + nstop)),
                                ("test", slice(nfit + nstop, None))):
            check_stop(protocol)
            exact = reconstructed[selection].tobytes() == base["K"][selection].tobytes()
            zexact = zhat[selection].tobytes() == base["z"][selection].tobytes()
            mexact = mhat[selection].tobytes() == mean[selection].tobytes()
            assert exact and zexact and mexact, "Exact lattice qualification failed"
            splits.append({"split": name, "count": len(reconstructed[selection]),
                           "integer_and_full_history_exact": exact, "z_bitwise_exact": zexact,
                           "known_mean_bitwise_exact": mexact})
        record["arms"][arm] = {"input_identity": data["input_identity"],
                              "fit_max": data["fit_max"], "splits": splits}
    return record

def labels(np, data, alpha, nfit):
    mean = alpha * data["z"][:, None] * data["factor"][None, :]
    y = data["noise"] + mean
    ym = y[:nfit].sum(axis=0, dtype=np.float64) / nfit
    ys = np.sqrt(np.maximum(np.square(y[:nfit]).sum(axis=0, dtype=np.float64) / nfit - ym * ym, 0.0))
    ys = np.where(ys > 1e-12, ys, 1.0)
    return y, mean, ym, ys


def predict(np, torch, model, x, ym, ys, protocol):
    out = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(x), protocol["batch_size"]):
            check_stop(protocol)
            xx = torch.as_tensor(x[start:start + protocol["batch_size"], :, None])
            anchor = torch.zeros((len(xx), 1), dtype=xx.dtype)
            out.append(model(xx, anchor).numpy().astype(np.float64) * ys + ym)
    return np.concatenate(out)


def fit_unit(np, torch, cls, protocol, protocol_hash, directory, data, shape, alpha, seed,
             *, resume=False, stop_after=None, ready_after=None):
    directory.mkdir(parents=True, exist_ok=True)
    nfit, nstop, _ = protocol["counts"]
    lookback, horizon = shape
    if protocol["phase"] == "preparation":
        assert stop_after is not None and 1 <= stop_after <= 8
        assert nfit >= protocol["batch_size"] * 9
        assert ready_after is None or 0 < ready_after < stop_after
    y, _, ym, ys = labels(np, data, alpha, nfit)
    yy = ((y[:nfit] - ym) / ys).astype(np.float32)
    torch.manual_seed(seed)
    model = cls("lstm", horizon, width=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    rng = np.random.default_rng(seed)
    runtime = {"torch": str(torch.__version__), "numpy": str(np.__version__),
        "python": sys.version, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "torch_threads": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
        "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
        "mkldnn_deterministic": bool(torch.backends.mkldnn.deterministic),
        "default_dtype": str(torch.get_default_dtype()), "byteorder": sys.byteorder,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    plan_rng = np.random.default_rng(seed)
    permutation_hashes, plan_digest = [], hashlib.sha256()
    for _ in range(protocol["max_epochs"]):
        planned = plan_rng.permutation(nfit).astype(np.int64)
        permutation_hashes.append(array_sha(planned))
        plan_digest.update(planned.tobytes(order="C"))
    permutation_plan_sha = plan_digest.hexdigest()
    identity = {"protocol_sha256": protocol_hash, "shape": list(shape), "alpha": alpha, "seed": seed,
                "runtime": runtime, "arm": data["arm"], "input_identity": data["input_identity"],
                "initial_model_sha256": initial_model_sha(model), "permutation_plan_sha256": permutation_plan_sha,
                "target_scaler_sha256": hashlib.sha256(ym.tobytes() + ys.tobytes()).hexdigest()}
    scaler = {"x_mean": data["x_mean"], "x_std": data["x_std"], "y_mean": ym.tolist(),
              "y_std": ys.tolist(), "anchor_mean": 0.0, "anchor_std": 1.0}
    checkpoint_path = directory / "latest.pt"
    state = {"epoch": 0, "cursor": 0, "permutation": None, "steps": 0,
             "best_loss": float("inf"), "best_epoch": None, "best_model": None,
             "stale": 0, "epoch_loss_sum": 0.0, "epoch_seen": 0, "history": []}
    if resume and checkpoint_path.exists():
        old = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
        assert old["identity"] == identity
        assert old["scaler"] == scaler
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        state = old["state"]
        rng.bit_generator.state = old["numpy_rng"]
        torch.set_rng_state(old["torch_rng"])
        if protocol["phase"] == "preparation":
            assert state["steps"] <= stop_after and state["epoch"] == 0 and not state["history"]
    elif checkpoint_path.exists():
        raise FileExistsError("Existing checkpoint requires explicit --resume")

    def checkpoint():
        save_checkpoint(torch, checkpoint_path, {"identity": identity, "state": state,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "scaler": scaler})

    if not resume or not checkpoint_path.exists():
        checkpoint()
    while state["epoch"] < protocol["max_epochs"] and state["stale"] < protocol["patience"]:
        check_stop(protocol)
        if state["permutation"] is None:
            state["permutation"] = torch.as_tensor(rng.permutation(nfit), dtype=torch.int64)
            assert array_sha(state["permutation"].numpy()) == permutation_hashes[state["epoch"]]
            state["cursor"] = 0
        while state["cursor"] < nfit:
            check_stop(protocol)
            if stop_after is not None and state["steps"] >= stop_after:
                if protocol["phase"] == "preparation":
                    assert state["steps"] <= 8 and state["epoch"] == 0 and not state["history"]
                return {"status": "trajectory_stopped", "steps": state["steps"]}
            if ready_after is not None and state["steps"] == ready_after:
                print("READY_FOR_EXTERNAL_SIGNAL", flush=True)
                while not STOP:
                    check_stop(protocol)
                    time.sleep(0.02)
                check_stop(protocol)
            end = min(state["cursor"] + protocol["batch_size"], nfit)
            indices = state["permutation"][state["cursor"]:end].numpy()
            xx = torch.as_tensor(data["x"][indices, :, None])
            target = torch.as_tensor(yy[indices])
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xx, torch.zeros((len(xx), 1), dtype=xx.dtype))
            loss = (prediction - target).square().mean()
            assert bool(torch.isfinite(loss))
            loss.backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            optimizer.step()
            state["epoch_loss_sum"] += float(loss.detach()) * len(indices)
            state["epoch_seen"] += len(indices)
            state["cursor"] = end
            state["steps"] += 1
            checkpoint()  # Every completed update is committed atomically.
        raw = predict(np, torch, model, data["x"][nfit:nfit + nstop], ym, ys, protocol)
        score = float(np.square(raw[:, -1] - y[nfit:nfit + nstop, -1]).mean())
        assert math.isfinite(score)
        state["history"].append({"epoch": state["epoch"] + 1, "steps": state["steps"],
            "training_scaled_path_loss": state["epoch_loss_sum"] / state["epoch_seen"], "raw_endpoint_stop_mse": score,
            "permutation_sha256": array_sha(state["permutation"].numpy())})
        if score < state["best_loss"]:
            state["best_loss"], state["best_epoch"], state["stale"] = score, state["epoch"] + 1, 0
            state["best_model"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            state["stale"] += 1
        state["epoch"] += 1
        state["cursor"], state["permutation"] = 0, None
        state["epoch_loss_sum"], state["epoch_seen"] = 0.0, 0
        checkpoint()
    summary = {"status": "complete", "epochs": state["epoch"], "steps": state["steps"],
        "best_epoch": state["best_epoch"], "history": state["history"], "identity": identity}
    atomic_json(directory / "TRAINING_SUMMARY.json", summary)
    return summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["qualify", "trajectory", "train"], required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--ready-after-updates", type=int)
    args = parser.parse_args()
    protocol, out, np, torch, cls, protocol_hash = prepare(args)
    started = time.monotonic()
    try:
        if (out / "RESULTS.json").exists() and args.mode != "trajectory":
            raise FileExistsError("Terminal result exists; no automatic rerun")
        if args.mode == "qualify":
            assert protocol["phase"] == "preparation"
            all_data = [(shape, generate(np, protocol, *shape)) for shape in protocol["shapes"]]
            qualification = [qualify_data(np, protocol, base, shape) for shape, base in all_data]
            atomic_json(out / "INPUT_QUALIFICATION.json", {"status": "passed_exact_lattice_input_pair",
                "protocol_sha256": protocol_hash, "shapes": qualification,
                "training_started_before_qualification": False})
            costs = []
            for shape, base in all_data:
                for arm in ("C", "D"):
                    begin = time.monotonic()
                    case = fit_unit(np, torch, cls, protocol, protocol_hash,
                        out / f"L{shape[0]}_H{shape[1]}_{arm}", arm_data(base, arm), shape, .25, 2026, stop_after=8)
                    assert case["steps"] == 8 and case["status"] == "trajectory_stopped"
                    costs.append({"shape": shape, "arm": arm, "steps": 8,
                                  "elapsed_seconds": time.monotonic() - begin})
            result = {"status": "qualification_and_cost_complete", "costs": costs,
                "elapsed_seconds": time.monotonic() - started, "performance_scored": False,
                "oracle_prediction_scored": False, "input_qualification_passed": True,
                "market_data": False, "protocol_sha256": protocol_hash}
            atomic_json(out / "RESULTS.json", result)
            print(json.dumps(result), flush=True)
            return 0
        if args.mode == "trajectory":
            assert protocol["phase"] == "preparation"
            shape = protocol["shapes"][0]
            base = generate(np, protocol, *shape)
            qualify_data(np, protocol, base, shape)
            result = fit_unit(np, torch, cls, protocol, protocol_hash, out / "unit", arm_data(base, "D"),
                shape, .25, 2026, resume=args.resume, stop_after=args.stop_after_updates,
                ready_after=args.ready_after_updates)
            atomic_json(out / "RESULTS.json", result)
            print(json.dumps(result), flush=True)
            return 0
        assert protocol["phase"] == "formal_synthetic" and protocol["synthetic_training_admitted"] is True
        assert args.stop_after_updates is None and args.ready_after_updates is None
        units, pair_identities = [], {}
        for shape in protocol["shapes"]:
            base = generate(np, protocol, *shape)
            qualification = qualify_data(np, protocol, base, shape)
            atomic_json(out / f"INPUT_QUALIFICATION_L{shape[0]}_H{shape[1]}.json", qualification)
            y, _, _, _ = labels(np, base, .25, protocol["counts"][0])
            nfit = protocol["counts"][0]
            design = np.column_stack([np.ones(nfit), base["z"][:nfit]])
            coefficients = np.linalg.lstsq(design, y[:nfit], rcond=None)[0]
            assert np.isfinite(coefficients).all()
            control_path = out / f"control_L{shape[0]}_H{shape[1]}_a0.25.json"
            atomic_json(control_path, {"shape": shape, "alpha": .25, "coefficients": coefficients.tolist()})
            for seed in protocol["training_seeds"]:
                for arm in ("C", "D"):
                    name = f"L{shape[0]}_H{shape[1]}_a0.25_s{seed}_{arm}"
                    data = arm_data(base, arm)
                    print("START " + name, flush=True)
                    summary = fit_unit(np, torch, cls, protocol, protocol_hash, out / name,
                        data, shape, .25, seed, resume=args.resume)
                    identity = summary["identity"]
                    pair_key = (tuple(shape), seed)
                    pair_value = (identity["initial_model_sha256"], identity["permutation_plan_sha256"],
                                  identity["target_scaler_sha256"])
                    if pair_key in pair_identities:
                        assert pair_identities[pair_key] == pair_value
                    else:
                        pair_identities[pair_key] = pair_value
                    units.append({"name": name, "shape": shape, "alpha": .25, "seed": seed, "arm": arm,
                        "checkpoint_sha256": sha(out / name / "latest.pt"),
                        "control_sha256": sha(control_path), "control_path": str(control_path),
                        "input_identity": data["input_identity"], "initial_model_sha256": pair_value[0],
                        "permutation_plan_sha256": pair_value[1], "target_scaler_sha256": pair_value[2],
                        "epochs": summary["epochs"],
                        "steps": summary["steps"], "best_epoch": summary["best_epoch"]})
                    atomic_json(out / "PROGRESS.json", {"completed_units": units, "test_scored": False})
                    print("COMPLETE " + name + " " + str(summary["epochs"]), flush=True)
        assert len(units) == 12
        atomic_json(out / "GLOBAL_FREEZE.json", {"units": units, "test_scored": False, "protocol_sha256": protocol_hash})
        scores = []
        for unit in units:
            check_stop(protocol)
            shape = unit["shape"]
            base = generate(np, protocol, *shape)
            data = arm_data(base, unit["arm"])
            assert data["input_identity"] == unit["input_identity"]
            nfit, nstop, _ = protocol["counts"]
            _, mean, ym, ys = labels(np, base, .25, nfit)
            test = slice(nfit + nstop, None)
            assert sha(out / unit["name"] / "latest.pt") == unit["checkpoint_sha256"]
            checkpoint = torch.load(out / unit["name"] / "latest.pt", weights_only=True, map_location="cpu")
            assert checkpoint["identity"]["arm"] == unit["arm"]
            assert checkpoint["identity"]["input_identity"] == unit["input_identity"]
            model = cls("lstm", shape[1], width=32)
            model.load_state_dict(checkpoint["state"]["best_model"])
            prediction = predict(np, torch, model, data["x"][test], ym, ys, protocol)
            assert sha(unit["control_path"]) == unit["control_sha256"]
            control = json.loads(Path(unit["control_path"]).read_text())
            ols = np.column_stack([np.ones(len(base["z"][test])), base["z"][test]]) @ np.asarray(control["coefficients"])
            known = mean[test]
            assert all(np.isfinite(value).all() for value in (known, prediction, ols))
            np.savez_compressed(out / unit["name"] / "SYNTHETIC_PREDICTIONS.npz", native=prediction, ols=ols,
                                known_mean=known, visible_z=base["z"][test])
            record = dict(unit)
            record["conditional_excess_risk"] = {}
            for metric in ("endpoint", "raw_path"):
                select = (lambda a: a[:, -1]) if metric == "endpoint" else (lambda a: a)
                zero = float(np.square(select(known)).mean())
                native = float(np.square(select(prediction - known)).mean())
                linear = float(np.square(select(ols - known)).mean())
                assert all(math.isfinite(v) and v >= 0 for v in (zero, native, linear)) and zero > 0
                record["conditional_excess_risk"][metric] = {"zero": zero, "native": native, "ols": linear,
                    "native_recovery_fraction": 1 - native / zero, "ols_recovery_fraction": 1 - linear / zero,
                    "native_half_recovery": native <= .5 * zero, "control_half_recovery": linear <= .5 * zero}
            scores.append(record)
        control_gates, pair_gates, half_gates = {}, [], []
        lookup = {(tuple(u["shape"]), u["seed"], u["arm"]): u for u in scores}
        for shape in protocol["shapes"]:
            for seed in protocol["training_seeds"]:
                c, d = lookup[(tuple(shape), seed, "C")], lookup[(tuple(shape), seed, "D")]
                for metric in ("endpoint", "raw_path"):
                    cr, dr = c["conditional_excess_risk"][metric], d["conditional_excess_risk"][metric]
                    key = f"L{shape[0]}_H{shape[1]}:{metric}"
                    gate = cr["control_half_recovery"]
                    assert dr["control_half_recovery"] == gate
                    if key in control_gates:
                        assert control_gates[key] == gate
                    control_gates[key] = gate
                    pair_gates.append({"shape": shape, "seed": seed, "metric": metric,
                        "cumulative": cr["native"], "difference": dr["native"],
                        "passed": cr["native"] > 0 and dr["native"] <= .9 * cr["native"]})
                    half_gates.append({"name": d["name"], "metric": metric, "passed": dr["native_half_recovery"]})
        assert len(control_gates) == 4 and len(pair_gates) == len(half_gates) == 12
        result = {"status": "completed_fixed_lattice_input_pair", "units": scores,
            "control_unique_gates": control_gates, "pair_relative_gates": pair_gates,
            "diff_half_recovery_gates": half_gates, "control_qualified": all(control_gates.values()),
            "all_pair_relative_gates_passed": all(g["passed"] for g in pair_gates),
            "all_diff_half_recovery_gates_passed": all(g["passed"] for g in half_gates),
            "elapsed_seconds": time.monotonic() - started, "protocol_sha256": protocol_hash,
            "market_data_read": False, "Q3_read": False, "new_market_candidate_admitted": False,
            "weak_natural_signal_proven": False, "pure_optimization_or_capacity_cause_identified": False,
            "full_research_goal_achieved": False}
        result["verdict"] = ("control_inconclusive" if not result["control_qualified"] else
            "complete_fixed_recipe_pair_criteria_passed" if result["all_pair_relative_gates_passed"]
            and result["all_diff_half_recovery_gates_passed"] else "fixed_recipe_pair_criteria_not_all_met")
        atomic_json(out / "RESULTS.json", result)
        print(json.dumps({"status": result["status"], "units": len(scores),
                          "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return 0
    except Paused as error:
        atomic_json(out / "PAUSE.json", {"status": "paused_checkpoint_preserved", "reason": str(error),
            "original_deadline_epoch": protocol["deadline_epoch"], "protocol_sha256": protocol_hash})
        print("PAUSED_CHECKPOINT_PRESERVED", flush=True)
        return 75
    except BaseException as error:
        atomic_json(out / "FAILURE.json", {"status": "terminal_noadmit_checkpoints_retained",
            "type": type(error).__name__, "message": str(error), "protocol_sha256": protocol_hash})
        raise


if __name__ == "__main__":
    sys.exit(main())
