"""One frozen artificial known-mean calibration; no market-data loader/import.

The original Recurrent AST is used unchanged. The surrounding artificial iid
origin experiment is deliberately not claimed to be the original market DGP.
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


class Paused(Exception):
    pass


def handle_signal(_number, _frame):
    global STOP
    STOP = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def check_stop(protocol):
    if STOP or time.time() >= protocol["deadline_epoch"] - 10:
        raise Paused("signal_or_original_absolute_deadline")


def prepare(args):
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    assert protocol["market_data_allowed"] is False
    assert protocol["GPU"] is False and protocol["paid_resources"] is False
    assert set(os.sched_getaffinity(0)) == {24}
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    for name, digest in protocol["source_sha256"].items():
        assert sha(name) == digest, "Frozen source changed: " + name
    root = Path(protocol["output_root"]).resolve()
    output = Path(args.output).resolve()
    assert output.is_relative_to(root)
    output.mkdir(parents=True, exist_ok=True)
    repo = Path(protocol["repository"]).resolve()
    sources = {Path(p).resolve() for p in protocol["source_sha256"]}

    def audit(event, values):
        if event.startswith("socket."):
            raise PermissionError("Network disabled for synthetic worker")
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode = values[1] if len(values) > 1 else "r"
        flags = values[2] if len(values) > 2 else 0
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if write and not path.is_relative_to(root) and str(path) != "/dev/null":
            raise PermissionError("Write outside own synthetic stage: " + str(path))
        if not write and path.is_relative_to(repo) and not path.is_relative_to(root) and path not in sources:
            # Installed venv dependencies are not market/research inputs.
            if not path.is_relative_to(repo / ".venv"):
                raise PermissionError("Unadmitted repository input: " + str(path))

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    source = Path(protocol["native_model_source"])
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Recurrent"]
    assert len(nodes) == 1
    namespace = {"nn": torch.nn, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    signal.signal(signal.SIGUSR1, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return protocol, output, np, torch, namespace["Recurrent"], sha(protocol_path)


def generate(np, protocol, lookback, horizon):
    nfit, nstop, ntest = protocol["counts"]
    n = nfit + nstop + ntest
    rng = np.random.default_rng(protocol["data_seed"] + lookback)
    increments = rng.standard_normal((n, lookback - 1))
    raw = np.zeros((n, lookback), dtype=np.float64)
    raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(lookback)
    fit = raw[:nfit]
    # Original scalar history-moment formula and FP32 transformation.
    xm = float(fit.sum(dtype=np.float64) / fit.size)
    xs = math.sqrt(max(float(np.square(fit).sum(dtype=np.float64) / fit.size - xm * xm), 0.0))
    xs = xs if xs > 1e-12 else 1.0
    x = ((raw - xm) / xs).astype(np.float32)
    z = math.sqrt(lookback) * xs * (x[:, -1].astype(np.float64) - x[:, -2].astype(np.float64))
    noise = rng.standard_normal((n, horizon)).cumsum(axis=1)
    factor = np.arange(1, horizon + 1, dtype=np.float64) / math.sqrt(horizon)
    return {"x": x, "z": z, "noise": noise, "factor": factor, "x_mean": xm, "x_std": xs}


def labels(np, data, alpha, nfit):
    mean = alpha * data["z"][:, None] * data["factor"][None, :]
    y = data["noise"] + mean
    ym = y[:nfit].sum(axis=0, dtype=np.float64) / nfit
    ys = np.sqrt(np.maximum(np.square(y[:nfit]).sum(axis=0, dtype=np.float64) / nfit - ym * ym, 0.0))
    ys = np.where(ys > 1e-12, ys, 1.0)
    return y, mean, ym, ys


def save_checkpoint(torch, path, payload):
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


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
    identity = {"protocol_sha256": protocol_hash, "shape": list(shape), "alpha": alpha, "seed": seed,
                "runtime": runtime}
    scaler = {"x_mean": data["x_mean"], "x_std": data["x_std"], "y_mean": ym.tolist(),
              "y_std": ys.tolist(), "anchor_mean": 100.0, "anchor_std": 1.0}
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
            "training_scaled_path_loss": state["epoch_loss_sum"] / state["epoch_seen"], "raw_endpoint_stop_mse": score})
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
            costs = []
            for shape in protocol["shapes"]:
                begin = time.monotonic()
                data = generate(np, protocol, *shape)
                y, mean, _, _ = labels(np, data, .25, protocol["counts"][0])
                assert data["x"].dtype == np.float32 and data["x"].shape[1] == shape[0]
                assert np.isfinite(y).all() and np.isfinite(data["x"]).all()
                visible = math.sqrt(shape[0]) * data["x_std"] * (data["x"][:, -1].astype(np.float64) - data["x"][:, -2].astype(np.float64))
                assert np.array_equal(visible, data["z"])
                assert np.array_equal(mean, .25 * visible[:, None] * data["factor"][None, :])
                case = fit_unit(np, torch, cls, protocol, protocol_hash, out / f"L{shape[0]}_H{shape[1]}", data,
                                shape, .25, 2026, stop_after=8)
                assert case["steps"] == 8 and case["status"] == "trajectory_stopped"
                costs.append({"shape": shape, "steps": case["steps"], "elapsed_seconds": time.monotonic() - begin})
            result = {"status": "qualification_and_cost_complete", "costs": costs,
                      "elapsed_seconds": time.monotonic() - started, "performance_scored": False,
                      "oracle_prediction_scored": False, "market_data": False}
            atomic_json(out / "RESULTS.json", result)
            print(json.dumps(result), flush=True)
            return 0
        if args.mode == "trajectory":
            assert protocol["phase"] == "preparation"
            data = generate(np, protocol, *protocol["shapes"][0])
            result = fit_unit(np, torch, cls, protocol, protocol_hash, out / "unit", data,
                protocol["shapes"][0], .25, 2026, resume=args.resume,
                stop_after=args.stop_after_updates, ready_after=args.ready_after_updates)
            atomic_json(out / "RESULTS.json", result)
            print(json.dumps(result), flush=True)
            return 0
        assert protocol["phase"] == "formal_synthetic" and protocol["synthetic_training_admitted"]
        assert args.stop_after_updates is None and args.ready_after_updates is None
        units = []
        for shape in protocol["shapes"]:
            data = generate(np, protocol, *shape)
            for alpha in protocol["alphas"]:
                y, _, _, _ = labels(np, data, alpha, protocol["counts"][0])
                design = np.column_stack([np.ones(protocol["counts"][0]), data["z"][:protocol["counts"][0]]])
                coefficients = np.linalg.lstsq(design, y[:protocol["counts"][0]], rcond=None)[0]
                assert np.isfinite(coefficients).all()
                control = {"shape": shape, "alpha": alpha, "coefficients": coefficients.tolist()}
                control_path = out / f"control_L{shape[0]}_H{shape[1]}_a{alpha}.json"
                atomic_json(control_path, control)
                for seed in protocol["training_seeds"]:
                    name = f"L{shape[0]}_H{shape[1]}_a{alpha}_s{seed}"
                    print("START " + name, flush=True)
                    summary = fit_unit(np, torch, cls, protocol, protocol_hash, out / name,
                        data, shape, alpha, seed, resume=args.resume)
                    units.append({"name": name, "shape": shape, "alpha": alpha, "seed": seed,
                        "checkpoint_sha256": sha(out / name / "latest.pt"),
                        "control_sha256": sha(control_path), "control_path": str(control_path),
                        "epochs": summary["epochs"], "steps": summary["steps"], "best_epoch": summary["best_epoch"]})
                    atomic_json(out / "PROGRESS.json", {"completed_units": units, "test_scored": False})
                    print("COMPLETE " + name + " " + str(summary["epochs"]), flush=True)
        assert len(units) == len(protocol["shapes"]) * len(protocol["alphas"]) * len(protocol["training_seeds"])
        atomic_json(out / "GLOBAL_FREEZE.json", {"units": units, "test_scored": False, "protocol_sha256": protocol_hash})
        scores = []
        for unit in units:
            check_stop(protocol)
            shape, alpha = unit["shape"], unit["alpha"]
            data = generate(np, protocol, *shape)
            nfit, nstop, _ = protocol["counts"]
            _, mean, ym, ys = labels(np, data, alpha, nfit)
            test = slice(nfit + nstop, None)
            assert sha(out / unit["name"] / "latest.pt") == unit["checkpoint_sha256"]
            checkpoint = torch.load(out / unit["name"] / "latest.pt", weights_only=True, map_location="cpu")
            model = cls("lstm", shape[1], width=32)
            model.load_state_dict(checkpoint["state"]["best_model"])
            prediction = predict(np, torch, model, data["x"][test], ym, ys, protocol)
            assert sha(unit["control_path"]) == unit["control_sha256"]
            control = json.loads(Path(unit["control_path"]).read_text())
            ols = np.column_stack([np.ones(len(data["z"][test])), data["z"][test]]) @ np.asarray(control["coefficients"])
            known = mean[test]
            assert np.isfinite(known).all() and np.isfinite(prediction).all() and np.isfinite(ols).all()
            np.savez_compressed(out / unit["name"] / "SYNTHETIC_PREDICTIONS.npz", native=prediction, ols=ols,
                                known_mean=known, visible_z=data["z"][test])
            record = dict(unit)
            record["conditional_excess_risk"] = {}
            for metric in ("endpoint", "raw_path"):
                select = (lambda a: a[:, -1]) if metric == "endpoint" else (lambda a: a)
                zero = float(np.square(select(known)).mean())
                native = float(np.square(select(prediction - known)).mean())
                linear = float(np.square(select(ols - known)).mean())
                risk = {"zero": zero, "native": native, "ols": linear}
                assert all(math.isfinite(v) and v >= 0 for v in risk.values())
                if alpha > 0:
                    assert zero > 0, "Degenerate known-mean denominator; calibration invalid"
                    risk.update(native_recovery_fraction=1 - native / zero, ols_recovery_fraction=1 - linear / zero,
                                native_half_recovery=native <= .5 * zero, control_half_recovery=linear <= .5 * zero)
                else:
                    noise = float(shape[1]) if metric == "endpoint" else (shape[1] + 1) / 2
                    risk.update(native_energy_over_noise=native / noise, ols_energy_over_noise=linear / noise)
                record["conditional_excess_risk"][metric] = risk
            scores.append(record)
        result = {"status": "completed_fixed_synthetic_calibration", "units": scores,
            "elapsed_seconds": time.monotonic() - started, "protocol_sha256": protocol_hash,
            "market_data_read": False, "Q3_read": False, "new_market_candidate_admitted": False,
            "weak_natural_signal_proven": False, "pure_optimization_or_capacity_cause_identified": False,
            "full_research_goal_achieved": False}
        atomic_json(out / "RESULTS.json", result)
        print(json.dumps({"status": result["status"], "units": len(scores), "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
        return 0
    except Paused as e:
        atomic_json(out / "PAUSE.json", {"status": "paused_checkpoint_preserved", "reason": str(e),
            "original_deadline_epoch": protocol["deadline_epoch"], "protocol_sha256": protocol_hash})
        print("PAUSED_CHECKPOINT_PRESERVED", flush=True)
        return 75


if __name__ == "__main__":
    sys.exit(main())
