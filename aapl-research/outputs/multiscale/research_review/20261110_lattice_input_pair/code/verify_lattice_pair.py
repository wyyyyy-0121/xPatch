"""Independent lattice-pair replay; no worker imports, training or market inputs."""
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

ATOL, RTOL, STOP = 1e-11, 1e-9, False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stop(_number, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-directory", required=True)
    args = parser.parse_args()
    pp = Path(args.protocol).resolve()
    p = json.loads(pp.read_text())
    root, run = Path(p["output_root"]).resolve(), Path(args.run_directory).resolve()
    assert run == root / "formal" and run.is_dir()
    destination, temporary = run / "INDEPENDENT_REPLAY.json", run / (".independent_replay_" + str(os.getpid()) + ".json")
    assert not destination.exists(), "Terminal replay report exists; no automatic retry"
    checks, started = [], time.monotonic()
    report = {"status": "not_complete", "checks": checks, "protocol_sha256": digest(pp),
              "verifier_sha256": digest(__file__), "ATOL": ATOL, "RTOL": RTOL,
              "scientific_gate_tolerance": 0, "training_updates": 0, "worker_functions_imported": False,
              "market_data_or_old_weights_read": False, "Q3_read": False,
              "new_market_candidate_admitted": False, "full_research_goal_achieved": False}

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def deadline():
        if STOP or time.time() >= p["stage_deadline_epoch"] - 10:
            raise TimeoutError("Original stage absolute deadline or external stop")

    def read_json(path):
        path = Path(path)
        check("small_json:" + path.name, path.suffix == ".json" and path.stat().st_size < 2_000_000)
        return json.loads(path.read_text())

    try:
        check("scope", p["phase"] == "formal_synthetic" and p["synthetic_training_admitted"] is True
              and all(p[k] is False for k in ("market_data_allowed", "GPU", "paid_resources")))
        check("CPU24_CUDA_off", set(os.sched_getaffinity(0)) == {24} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1")
        check("fixed_matrix", p["shapes"] == [[480, 60], [96, 8]] and p["counts"] == [2048, 512, 4096]
              and p["training_seeds"] == [2026, 2027, 2028] and p["data_seed"] == 20261110 and p["alphas"] == [.25])
        check("fixed_recipe", p["batch_size"] == 128 and p["max_epochs"] == 20 and p["patience"] == 3 and p["learning_rate"] == .001)
        check("fixed_aggregation", p["aggregation"] == {"OLS_controls": 4, "pair_checks": 12, "minimum_pair_reduction": .1, "difference_half_checks": 12, "minimum_recovery": .5})
        check("finite_absolute_deadline", math.isfinite(p["deadline_epoch"]) and math.isfinite(p["stage_deadline_epoch"])
              and p["stage_deadline_epoch"] >= p["deadline_epoch"])
        deadline()
        sources = {Path(name).resolve(): wanted for name, wanted in p["source_sha256"].items()}
        for path, wanted in sources.items():
            check("source:" + str(path), path.suffix in {".py", ".md", ".json"} and path.stat().st_size < 2_000_000 and digest(path) == wanted)
        native_source = Path(p["native_model_source"]).resolve()
        check("native_and_verifier_frozen", native_source in sources and Path(__file__).resolve() in sources)
        frozen, recorded = read_json(run / "GLOBAL_FREEZE.json"), read_json(run / "RESULTS.json")
        check("freeze_identity", frozen["protocol_sha256"] == digest(pp) and frozen["test_scored"] is False)
        check("result_identity", recorded["protocol_sha256"] == digest(pp) and recorded["status"] == "completed_fixed_lattice_input_pair")
        check("result_scope", all(recorded[k] is False for k in ("market_data_read", "Q3_read", "new_market_candidate_admitted", "full_research_goal_achieved")))
        expected = {f"L{L}_H{H}_a0.25_s{s}_{a}": ([L, H], s, a) for L, H in p["shapes"] for s in p["training_seeds"] for a in ("C", "D")}
        units, saved = {u["name"]: u for u in frozen["units"]}, {u["name"]: u for u in recorded["units"]}
        check("all_twelve_units", len(frozen["units"]) == len(recorded["units"]) == len(units) == len(saved) == 12 and set(units) == set(saved) == set(expected))
        allowed = set(sources) | {pp, run / "GLOBAL_FREEZE.json", run / "RESULTS.json"}
        for name, (shape, seed, arm) in expected.items():
            unit, control = units[name], run / f"control_L{shape[0]}_H{shape[1]}_a0.25.json"
            check("unit_identity:" + name, unit["shape"] == shape and unit["seed"] == seed and unit["arm"] == arm and unit["alpha"] == .25)
            check("artifact_identity:" + name, Path(unit["control_path"]).resolve() == control and digest(control) == unit["control_sha256"] and digest(run / name / "latest.pt") == unit["checkpoint_sha256"])
            allowed.update({control, run / name / "latest.pt", run / name / "TRAINING_SUMMARY.json", run / name / "SYNTHETIC_PREDICTIONS.npz"})
        allowed.update(run / f"INPUT_QUALIFICATION_L{L}_H{H}.json" for L, H in p["shapes"])
        runtime_tmp = root / "runtime_tmp"
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(runtime_tmp)
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = str(runtime_tmp)
        runtime_roots = [Path(v).resolve() for v in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", sys.prefix, sys.base_prefix)]
        runtime_roots += [Path(v).resolve() for v in sys.path if v and Path(v).name == "site-packages"]
        report["runtime_read_roots"] = sorted({str(v) for v in runtime_roots})
        for name, wanted in p.get("runtime_file_sha256", {}).items():
            check("runtime_file:" + name, any(Path(name).resolve().is_relative_to(v) for v in runtime_roots) and digest(name) == wanted)

        def audit(event, values):
            if event.startswith("socket."):
                raise PermissionError("Network disabled in independent synthetic replay")
            if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(values[0])).resolve()
            mode, flags = values[1] if len(values) > 1 else "r", values[2] if len(values) > 2 else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and path not in {temporary, destination, Path("/dev/null")} and not path.is_relative_to(runtime_tmp):
                raise PermissionError("Unadmitted replay write: " + str(path))
            if not writing and path not in allowed and not path.is_relative_to(runtime_tmp) and not any(path.is_relative_to(v) for v in runtime_roots):
                raise PermissionError("Unadmitted replay read: " + str(path))

        sys.dont_write_bytecode = True
        sys.addaudithook(audit)
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        signal.signal(signal.SIGUSR1, stop)
        signal.signal(signal.SIGTERM, stop)
        nodes = [n for n in ast.parse(native_source.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Recurrent"]
        check("native_AST", len(nodes) == 1)
        namespace = {"nn": torch.nn, "torch": torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native_source), "exec"), namespace)
        cls, maximum_difference = namespace["Recurrent"], [0.0]
        runtime = {"torch": str(torch.__version__), "numpy": str(np.__version__), "python": sys.version,
                   "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(), "torch_interop": torch.get_num_interop_threads(),
                   "mkldnn_enabled": bool(torch.backends.mkldnn.enabled), "mkldnn_deterministic": bool(torch.backends.mkldnn.deterministic),
                   "default_dtype": str(torch.get_default_dtype()), "byteorder": sys.byteorder, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}

        def ash(array):
            return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()

        def state_hash(state):
            h = hashlib.sha256()
            for name, tensor in state.items():
                h.update(name.encode("utf-8") + str(tensor.dtype).encode("ascii") + json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
            return h.hexdigest()

        def near(name, actual, wanted, exact=False):
            a, b = np.asarray(actual), np.asarray(wanted)
            check("finite_shape:" + name, a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all())
            if a.size:
                maximum_difference[0] = max(maximum_difference[0], float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
            check("exact:" + name if exact else "near:" + name, (a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")) if exact else np.allclose(a, b, atol=ATOL, rtol=RTOL))

        risks, controls, pairing = {}, {}, {}
        nfit, nstop, ntest = p["counts"]
        test = slice(nfit + nstop, None)
        for L, H in p["shapes"]:
            deadline()
            rng = np.random.default_rng(p["data_seed"] + L)
            increments = rng.standard_normal((nfit + nstop + ntest, L - 1))
            raw = np.zeros((len(increments), L), dtype=np.float64)
            raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(L)
            check(f"raw_bound:{L}", bool(np.isfinite(raw).all()) and float(np.abs(raw).max()) <= 32)
            q = 2.0 ** -12
            K = np.rint(raw / q).astype(np.int64)
            D = np.zeros_like(K)
            D[:, 1:] = K[:, 1:] - K[:, :-1]
            check(f"integer_bounds:{L}", int(np.abs(K).max()) <= 131072 and int(np.abs(D).max()) <= 262144 and np.all(K[:, -1] == 0))
            z = math.sqrt(L) * q * (K[:, -1] - K[:, -2])
            noise = rng.standard_normal((len(K), H)).cumsum(axis=1)
            mean = .25 * z[:, None] * (np.arange(1, H + 1, dtype=np.float64) / math.sqrt(H))[None, :]
            y = noise + mean
            ym = y[:nfit].sum(axis=0, dtype=np.float64) / nfit
            ys = np.sqrt(np.maximum(np.square(y[:nfit]).sum(axis=0, dtype=np.float64) / nfit - ym * ym, 0.0))
            ys = np.where(ys > 1e-12, ys, 1.0)
            qualification = read_json(run / f"INPUT_QUALIFICATION_L{L}_H{H}.json")
            check(f"qualification_hashes:{L}", all(qualification[k] == ash(a) for k, a in (("K_sha256", K), ("D_sha256", D), ("z_sha256", z), ("noise_sha256", noise), ("target_sha256", y), ("known_mean_sha256", mean))))
            zc, yc = z[:nfit] - z[:nfit].mean(), y[:nfit] - ym
            slope = (zc[:, None] * yc).sum(axis=0) / np.square(zc).sum()
            coefficients = np.stack((ym - z[:nfit].mean() * slope, slope))
            control = read_json(run / f"control_L{L}_H{H}_a0.25.json")
            near(f"OLS_coefficients:{L}", control["coefficients"], coefficients)
            ols = np.column_stack((np.ones(ntest), z[test])) @ coefficients
            for arm, integers in (("C", K), ("D", D)):
                maximum = int(np.abs(integers[:nfit]).max())
                scale = float(1 if maximum == 0 else 1 << (maximum - 1).bit_length())
                x = (integers.astype(np.float64) / scale).astype(np.float32)
                decoded = x.astype(np.float64) * scale
                near(f"input_integer_exact:{L}:{arm}", decoded, integers.astype(np.float64), exact=True)
                recovered = decoded.astype(np.int64)
                if arm == "D":
                    check(f"inverse_sum_safe:{L}", L * 262144 < 2 ** 63)
                    recovered = np.zeros_like(K)
                    recovered[:, :-1] = -np.cumsum(decoded[:, 1:].astype(np.int64)[:, ::-1], axis=1, dtype=np.int64)[:, ::-1]
                near(f"full_K_exact:{L}:{arm}", recovered, K, exact=True)
                zz = math.sqrt(L) * q * (recovered[:, -1] - recovered[:, -2])
                near(f"signal_exact:{L}:{arm}", zz, z, exact=True)
                near(f"mean_exact:{L}:{arm}", .25 * zz[:, None] * (np.arange(1, H + 1, dtype=np.float64) / math.sqrt(H))[None, :], mean, exact=True)
                check(f"signal_mean_bitwise:{L}:{arm}", ash(zz) == ash(z) and ash(.25 * zz[:, None] * (np.arange(1, H + 1, dtype=np.float64) / math.sqrt(H))[None, :]) == ash(mean))
                iid = {"arm": arm, "K_sha256": ash(K), "D_sha256": ash(D), "x_sha256": ash(x), "z_sha256": ash(z), "noise_sha256": ash(noise), "x_scale": scale, "quantum": q}
                check(f"qualification_arm:{L}:{arm}", qualification["arms"][arm]["input_identity"] == iid and qualification["arms"][arm]["fit_max"] == maximum)
                for seed in p["training_seeds"]:
                    deadline()
                    name = f"L{L}_H{H}_a0.25_s{seed}_{arm}"
                    cp = torch.load(run / name / "latest.pt", weights_only=True, map_location="cpu")
                    ident, state = cp["identity"], cp["state"]
                    check("checkpoint_identity:" + name, ident["protocol_sha256"] == digest(pp) and ident["shape"] == [L, H] and ident["alpha"] == .25 and ident["seed"] == seed and ident["arm"] == arm and ident["input_identity"] == iid)
                    check("runtime_identity:" + name, ident["runtime"] == runtime)
                    check("frozen_metadata:" + name, all(saved[name][k] == value for k, value in units[name].items()) and units[name]["input_identity"] == iid and all(units[name][k] == ident[k] for k in ("initial_model_sha256", "permutation_plan_sha256", "target_scaler_sha256")))
                    check("target_scaler_hash:" + name, ident["target_scaler_sha256"] == hashlib.sha256(ym.tobytes() + ys.tobytes()).hexdigest())
                    near("target_mean:" + name, cp["scaler"]["y_mean"], ym, exact=True)
                    near("target_scale:" + name, cp["scaler"]["y_std"], ys, exact=True)
                    check("input_scaler:" + name, cp["scaler"]["x_mean"] == 0 and cp["scaler"]["x_std"] == scale and cp["scaler"]["anchor_mean"] == 0 and cp["scaler"]["anchor_std"] == 1)
                    torch.manual_seed(seed)
                    model = cls("lstm", H, width=32)
                    check("initial_model:" + name, state_hash(model.state_dict()) == ident["initial_model_sha256"])
                    prng, plan, epoch_hashes = np.random.default_rng(seed), hashlib.sha256(), []
                    for _ in range(20):
                        permutation = prng.permutation(nfit).astype(np.int64)
                        plan.update(permutation.tobytes(order="C"))
                        epoch_hashes.append(ash(permutation))
                    check("permutation_plan:" + name, plan.hexdigest() == ident["permutation_plan_sha256"])
                    check("epoch_permutations:" + name, [h["permutation_sha256"] for h in state["history"]] == epoch_hashes[:state["epoch"]])
                    summary = read_json(run / name / "TRAINING_SUMMARY.json")
                    check("training_summary:" + name, summary["status"] == "complete" and summary["identity"] == ident and summary["history"] == state["history"] and all(summary[k] == units[name][k] for k in ("epochs", "steps", "best_epoch")) and summary["epochs"] == state["epoch"] and summary["steps"] == state["steps"] and summary["best_epoch"] == state["best_epoch"])
                    check("selected_checkpoint:" + name, 1 <= state["best_epoch"] <= state["epoch"] <= 20 and state["best_epoch"] == min(state["history"], key=lambda h: h["raw_endpoint_stop_mse"])["epoch"] and state["steps"] == state["epoch"] * 16)
                    pairing[(L, seed, arm)] = (ident["initial_model_sha256"], ident["permutation_plan_sha256"], epoch_hashes[:state["epoch"]])
                    model.load_state_dict(state["best_model"])
                    model.eval()
                    predictions = []
                    with torch.inference_mode():
                        for offset in range(0, ntest, 128):
                            deadline()
                            xx = torch.as_tensor(x[test][offset:offset + 128, :, None])
                            predictions.append(model(xx, torch.zeros((len(xx), 1), dtype=xx.dtype)).numpy().astype(np.float64) * ys + ym)
                    prediction = np.concatenate(predictions)
                    with np.load(run / name / "SYNTHETIC_PREDICTIONS.npz", allow_pickle=False) as payload:
                        for key, value in (("native", prediction), ("ols", ols), ("known_mean", mean[test]), ("visible_z", z[test])):
                            near(key + ":" + name, payload[key], value, exact=key in {"known_mean", "visible_z"})
                            check("payload_bitwise:" + name + key, ash(payload[key]) == ash(value)) if key in {"known_mean", "visible_z"} else None
                    risks[name] = {}
                    for metric, select in (("endpoint", lambda a: a[:, -1]), ("raw_path", lambda a: a)):
                        zero, native, linear = (float(np.square(select(a)).mean()) for a in (mean[test], prediction - mean[test], ols - mean[test]))
                        check("positive_zero_risk:" + name + metric, math.isfinite(zero) and zero > 0)
                        risk = {"zero": zero, "native": native, "ols": linear, "native_recovery_fraction": 1 - native / zero, "ols_recovery_fraction": 1 - linear / zero, "native_half_recovery": native <= .5 * zero, "control_half_recovery": linear <= .5 * zero}
                        for key, value in risk.items():
                            check(key + ":" + name + metric, saved[name]["conditional_excess_risk"][metric][key] is value) if isinstance(value, bool) else near(key + ":" + name + metric, saved[name]["conditional_excess_risk"][metric][key], value)
                        risks[name][metric] = risk
                        controls[f"L{L}_H{H}:{metric}"] = bool(linear <= .5 * zero)
        pair_gates, half_gates = [], []
        for L, H in p["shapes"]:
            for seed in p["training_seeds"]:
                C, D = (f"L{L}_H{H}_a0.25_s{seed}_{a}" for a in ("C", "D"))
                pc, pd = pairing[(L, seed, "C")], pairing[(L, seed, "D")]
                check("matched_initial_and_batches:" + C, pc[:2] == pd[:2] and pc[2][:min(len(pc[2]), len(pd[2]))] == pd[2][:min(len(pc[2]), len(pd[2]))])
                for metric in ("endpoint", "raw_path"):
                    c, d = risks[C][metric]["native"], risks[D][metric]["native"]
                    pair_gates.append({"shape": [L, H], "seed": seed, "metric": metric, "cumulative": c, "difference": d, "passed": bool(c > 0 and d <= .9 * c)})
                    half_gates.append({"name": D, "metric": metric, "passed": risks[D][metric]["native_half_recovery"]})
        check("OLS_unique_gates", len(controls) == 4 and recorded["control_unique_gates"] == controls)
        for key, values in (("pair_relative_gates", pair_gates), ("diff_half_recovery_gates", half_gates)):
            check("exact_gate_count:" + key, len(values) == len(recorded[key]) == 12)
            for index, (actual, reference) in enumerate(zip(recorded[key], values)):
                for field, value in reference.items():
                    near(key + str(index) + field, actual[field], value) if field in {"cumulative", "difference"} else check(key + str(index) + field, actual[field] == value)
        groups = {"control_qualified": all(controls.values()), "all_pair_relative_gates_passed": all(v["passed"] for v in pair_gates), "all_diff_half_recovery_gates_passed": all(v["passed"] for v in half_gates)}
        check("exact_group_bools", all(recorded[k] is v for k, v in groups.items()))
        report.update(status="completed_independent_lattice_pair_replay", units=12, control_unique_gates=controls, pair_relative_gates=pair_gates, diff_half_recovery_gates=half_gates, groups=groups, maximum_numeric_difference=maximum_difference[0], risk_reconstruction=risks)
    except BaseException as error:
        report.update(status="failed_independent_lattice_pair_replay", error={"type": type(error).__name__, "message": str(error)})
        print(json.dumps({"first_replay_error": report["error"], "checks_passed": len(checks)}), file=sys.stderr, flush=True)
    report.update(checks_passed=len(checks), elapsed_seconds=time.monotonic() - started, finished_epoch=time.time(), original_stage_deadline_epoch=p["stage_deadline_epoch"])
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
    print(json.dumps({"status": report["status"], "checks": len(checks), "report": str(destination)}), flush=True)
    return 0 if report["status"].startswith("completed_") else 1


if __name__ == "__main__":
    sys.exit(main())
