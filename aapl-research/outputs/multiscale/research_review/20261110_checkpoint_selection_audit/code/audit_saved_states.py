"""Describe selected and terminal artificial checkpoints; never train or select a candidate."""
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stopped(_signum, _frame):
    global STOP
    STOP = True


def atomic_json(path, value):
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    pp = Path(args.protocol).resolve()
    p = json.loads(pp.read_text())
    output = Path(p["output_root"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / "RESULTS.json").exists(), "Terminal audit exists"
    checks = []

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def deadline():
        if STOP or time.time() >= p["deadline_epoch"] - 5:
            raise TimeoutError("Fixed audit deadline or external pause")

    check("scope", p["training_updates"] == 0 and not p["market_allowed"] and not p["Q3_allowed"]
          and not p["GPU"] and not p["candidate_admitted"])
    check("CPU24", set(os.sched_getaffinity(0)) == {24} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1")
    check("finite_deadline", math.isfinite(p["deadline_epoch"]) and p["maximum_seconds"] == 600
          and p["deadline_epoch"] - p["started_epoch"] == 600)
    deadline()
    allowed = {pp}
    for path, wanted in p["read_file_sha256"].items():
        check("input_identity:" + path, digest(path) == wanted)
        allowed.add(Path(path).resolve())
    check("worker_frozen", Path(__file__).resolve() in allowed)
    prior = json.loads(Path(p["prior_results"]).read_text())
    check("accepted_prior", prior["status"] == "completed_fixed_lattice_input_pair" and len(prior["units"]) == 12)
    report = {"status": "running", "protocol_sha256": digest(pp), "units": [], "checks": checks,
              "training_updates": 0, "candidate_admitted": False, "market_data_read": False,
              "Q3_read": False, "full_goal_achieved": False, "pid": os.getpid()}
    progress = output / "PROGRESS.json"
    if args.resume:
        old = json.loads(progress.read_text())
        check("resume_identity", old["protocol_sha256"] == digest(pp))
        report["units"] = old["units"]
    else:
        check("fresh_audit", not progress.exists())
    tmp = output / "runtime_tmp"
    tmp.mkdir(exist_ok=True)
    tempfile.tempdir = str(tmp)
    runtime = [Path(v).resolve() for v in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", sys.prefix, sys.base_prefix)]
    runtime += [Path(v).resolve() for v in sys.path if v and Path(v).name == "site-packages"]
    for path, wanted in p["runtime_file_sha256"].items():
        check("runtime_identity:" + path, any(Path(path).resolve().is_relative_to(v) for v in runtime) and digest(path) == wanted)

    def audit(event, values):
        if event.startswith("socket."):
            raise PermissionError("Network not admitted")
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode, flags = values[1], values[2]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and not path.is_relative_to(output) and path != Path("/dev/null"):
            raise PermissionError("Unadmitted write: " + str(path))
        if not writing and path not in allowed and not path.is_relative_to(output) and not any(path.is_relative_to(v) for v in runtime):
            raise PermissionError("Unadmitted read: " + str(path))

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        signal.signal(signal.SIGUSR1, stopped)
        signal.signal(signal.SIGTERM, stopped)
        nodes = [n for n in ast.parse(Path(p["native_model_source"]).read_text()).body
                 if isinstance(n, ast.ClassDef) and n.name == "Recurrent"]
        check("native_AST", len(nodes) == 1)
        namespace = {"nn": torch.nn, "torch": torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), p["native_model_source"], "exec"), namespace)
        nfit, nstop, ntest = 2048, 512, 4096
        splits = {"fit": slice(0, nfit), "stop": slice(nfit, nfit + nstop), "test": slice(nfit + nstop, None)}
        completed = {u["name"] for u in report["units"]}
        for L, H in ((480, 60), (96, 8)):
            deadline()
            rng = np.random.default_rng(20261110 + L)
            increments = rng.standard_normal((nfit + nstop + ntest, L - 1))
            raw = np.zeros((len(increments), L), dtype=np.float64)
            raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(L)
            K = np.rint(raw / (2.0 ** -12)).astype(np.int64)
            D = np.zeros_like(K)
            D[:, 1:] = K[:, 1:] - K[:, :-1]
            z = math.sqrt(L) * 2.0 ** -12 * (K[:, -1] - K[:, -2])
            noise = rng.standard_normal((len(K), H)).cumsum(axis=1)
            mean = .25 * z[:, None] * (np.arange(1, H + 1, dtype=np.float64) / math.sqrt(H))[None, :]
            y = noise + mean
            ym = y[:nfit].sum(axis=0, dtype=np.float64) / nfit
            ys = np.sqrt(np.maximum(np.square(y[:nfit]).sum(axis=0, dtype=np.float64) / nfit - ym * ym, 0))
            ys = np.where(ys > 1e-12, ys, 1.)
            for unit in [u for u in prior["units"] if u["shape"] == [L, H]]:
                if unit["name"] in completed:
                    continue
                deadline()
                integers = K if unit["arm"] == "C" else D
                maximum = int(np.abs(integers[:nfit]).max())
                scale = float(1 << (maximum - 1).bit_length()) if maximum else 1.
                x = (integers.astype(np.float64) / scale).astype(np.float32)
                identity = unit["input_identity"]
                for key, array in (("K_sha256", K), ("D_sha256", D), ("x_sha256", x), ("z_sha256", z), ("noise_sha256", noise)):
                    check(unit["name"] + key, hashlib.sha256(array.tobytes()).hexdigest() == identity[key])
                directory = Path(p["prior_run"]) / unit["name"]
                check("accepted_checkpoint_hash:" + unit["name"], digest(directory / "latest.pt") == unit["checkpoint_sha256"])
                cp = torch.load(directory / "latest.pt", map_location="cpu", weights_only=True)
                check("checkpoint_metadata:" + unit["name"], cp["identity"]["input_identity"] == identity
                      and cp["state"]["best_epoch"] == unit["best_epoch"] and cp["state"]["epoch"] == unit["epochs"]
                      and cp["identity"]["shape"] == unit["shape"] and cp["identity"]["seed"] == unit["seed"]
                      and cp["identity"]["arm"] == unit["arm"] and cp["identity"]["protocol_sha256"] == prior["protocol_sha256"])
                check("target_scaler:" + unit["name"], np.asarray(cp["scaler"]["y_mean"], dtype=np.float64).tobytes() == ym.tobytes()
                      and np.asarray(cp["scaler"]["y_std"], dtype=np.float64).tobytes() == ys.tobytes())
                selected = namespace["Recurrent"]("lstm", H, width=32)
                terminal = namespace["Recurrent"]("lstm", H, width=32)
                selected.load_state_dict(cp["state"]["best_model"])
                terminal.load_state_dict(cp["model"])
                record = {"name": unit["name"], "shape": [L, H], "seed": unit["seed"], "arm": unit["arm"],
                          "selected_epoch": unit["best_epoch"], "terminal_epoch": unit["epochs"],
                          "same_parameters": all(torch.equal(cp["state"]["best_model"][k], t) for k, t in cp["model"].items()), "states": {}}
                for label, model in (("selected", selected), ("terminal", terminal)):
                    model.eval()
                    blocks = []
                    with torch.inference_mode():
                        for offset in range(0, len(x), 128):
                            deadline()
                            xx = torch.as_tensor(x[offset:offset + 128, :, None])
                            blocks.append(model(xx, torch.zeros((len(xx), 1), dtype=xx.dtype)).numpy().astype(np.float64) * ys + ym)
                    prediction = np.concatenate(blocks)
                    check("finite_predictions:" + unit["name"] + label, np.isfinite(prediction).all())
                    if label == "selected":
                        with np.load(directory / "SYNTHETIC_PREDICTIONS.npz", allow_pickle=False) as saved:
                            check("accepted_selected_prediction:" + unit["name"], np.allclose(prediction[splits["test"]], saved["native"], atol=1e-11, rtol=1e-9))
                    raw_stop = float(np.square(prediction[splits["stop"], -1] - y[splits["stop"], -1]).mean())
                    expected_stop = cp["state"]["best_loss"] if label == "selected" else cp["state"]["history"][-1]["raw_endpoint_stop_mse"]
                    check("stored_stop_identity:" + unit["name"] + label, math.isclose(raw_stop, expected_stop, abs_tol=1e-11, rel_tol=1e-9))
                    record["states"][label] = {"raw_endpoint_stop_mse": raw_stop, "splits": {}}
                    for split, section in splits.items():
                        metrics = {}
                        for metric, select in (("endpoint", lambda a: a[:, -1]), ("raw_path", lambda a: a)):
                            zero = float(np.square(select(mean[section])).mean())
                            risk = float(np.square(select(prediction[section] - mean[section])).mean())
                            check("finite_risk:" + unit["name"] + label + split + metric, math.isfinite(zero) and zero > 0 and math.isfinite(risk))
                            metrics[metric] = {"zero": zero, "known_mean_risk": risk, "recovery_fraction": 1 - risk / zero, "half_recovery": risk <= .5 * zero}
                        record["states"][label]["splits"][split] = metrics
                report["units"].append(record)
                atomic_json(progress, report)
                print("COMPLETE " + unit["name"], flush=True)
        check("all_units", len(report["units"]) == 12)
        report.update(status="completed_saved_state_diagnostic_no_training", elapsed_seconds=time.monotonic() - started,
                      scalar_metric_records=sum(len(v["splits"]) * 2 for u in report["units"] for v in u["states"].values()))
        check("all144records", report["scalar_metric_records"] == 144)
        atomic_json(output / "RESULTS.json", report)
    except BaseException as error:
        report.update(status="paused" if isinstance(error, TimeoutError) else "failed", error={"type": type(error).__name__, "message": str(error)}, elapsed_seconds=time.monotonic() - started)
        atomic_json(output / ("PAUSE.json" if isinstance(error, TimeoutError) else "FAILURE.json"), report)
        print(json.dumps({"status": report["status"], "error": report["error"]}), flush=True)
        return 75 if isinstance(error, TimeoutError) else 1
    print(json.dumps({"status": report["status"], "units": len(report["units"]), "checks": len(checks), "records": report["scalar_metric_records"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
