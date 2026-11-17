"""Independent replay of three fixed-H60 context fits; no training or old weights."""
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
    assert not destination.exists(), "Terminal replay exists; no automatic retry"
    checks, started = [], time.monotonic()
    report = {"status": "not_complete", "checks": checks, "protocol_sha256": digest(pp),
              "verifier_sha256": digest(__file__), "ATOL": ATOL, "RTOL": RTOL,
              "scientific_gate_tolerance": 0, "training_updates": 0, "worker_functions_imported": False,
              "old_model_weights_read": False, "market_data_read": False, "Q3_read": False,
              "new_market_candidate_admitted": False, "full_research_goal_achieved": False}

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def deadline():
        if STOP or time.time() >= p["stage_deadline_epoch"] - 10:
            raise TimeoutError("Original context stage absolute deadline or external stop")

    def read_json(path):
        path = Path(path)
        check("small_json:" + path.name, path.suffix == ".json" and path.stat().st_size < 2_000_000)
        return json.loads(path.read_text())

    try:
        check("scope", p["phase"] == "formal_synthetic" and p["synthetic_training_admitted"] is True
              and all(p[k] is False for k in ("market_data_allowed", "GPU", "paid_resources")))
        check("CPU24_CUDA_off", set(os.sched_getaffinity(0)) == {24} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1")
        check("fixed_matrix", p["shapes"] == [[480, 60]] and p["generator_lookback"] == 480 and p["context_length"] == 96 and p["counts"] == [2048, 512, 4096]
              and p["training_seeds"] == [2026, 2027, 2028] and p["data_seed"] == 20261110 and p["alphas"] == [.25])
        check("fixed_recipe", p["batch_size"] == 128 and p["max_epochs"] == 20 and p["patience"] == 3 and p["learning_rate"] == .001)
        check("fixed_aggregation", p["aggregation"] == {"OLS_controls": 2, "pair_checks": 6, "minimum_pair_reduction": .1, "short_half_checks": 6, "minimum_recovery": .5})
        check("finite_deadline", math.isfinite(p["deadline_epoch"]) and math.isfinite(p["stage_deadline_epoch"]) and p["stage_deadline_epoch"] >= p["deadline_epoch"])
        deadline()
        sources = {Path(name).resolve(): wanted for name, wanted in p["source_sha256"].items()}
        prior_sources = {Path(name).resolve(): wanted for name, wanted in p["prior_read_sha256"].items()}
        check("old_inputs_small_JSON_only", all(path.suffix == ".json" and (path not in sources or sources[path] == wanted) for path, wanted in prior_sources.items()))
        sources.update(prior_sources)
        for path, wanted in sources.items():
            check("source:" + str(path), path.suffix in {".py", ".md", ".json"} and path.stat().st_size < 2_000_000 and digest(path) == wanted)
        native_source, prior_path, acceptance_path = (Path(p[k]).resolve() for k in ("native_model_source", "prior_results_path", "prior_stage_acceptance_path"))
        check("required_sources_frozen", {native_source, prior_path, acceptance_path, Path(__file__).resolve()} <= set(sources))
        prior, acceptance = read_json(prior_path), read_json(acceptance_path)
        check("accepted_full_controls", prior["status"] == "completed_fixed_lattice_input_pair" and acceptance["recipe_closed"] is True
              and acceptance["artifacts_sha256"]["formal_runs/formal/RESULTS.json"] == digest(prior_path))
        old = {u["seed"]: u for u in prior["units"] if u["shape"] == [480, 60] and u["arm"] == "D"}
        check("three_original_longD_controls", set(old) == {2026, 2027, 2028})
        control_path = Path(old[2026]["control_path"]).resolve()
        check("exact_three_old_JSON_inputs", set(prior_sources) == {prior_path, acceptance_path, control_path})
        check("control_frozen_small_JSON", control_path in sources and all(Path(u["control_path"]).resolve() == control_path and digest(control_path) == u["control_sha256"] for u in old.values()))
        frozen, recorded = read_json(run / "GLOBAL_FREEZE.json"), read_json(run / "RESULTS.json")
        check("freeze_identity", frozen["protocol_sha256"] == digest(pp) and frozen["test_scored"] is False)
        check("result_identity", recorded["protocol_sha256"] == digest(pp) and recorded["status"] == "completed_fixed_irrelevant_prefix_probe")
        check("prior_freeze_identities", all(data["prior_results_sha256"] == digest(prior_path) and data["prior_stage_acceptance_sha256"] == digest(acceptance_path) for data in (frozen, recorded)))
        check("result_scope", all(recorded[k] is False for k in ("market_data_read", "Q3_read", "new_market_candidate_admitted", "full_research_goal_achieved")))
        check("sufficiency_scope", recorded["full_history_information_equivalent"] is False and recorded["conditional_mean_sufficient"] is True)
        expected = {f"L480_H60_a0.25_s{s}_D_context96": s for s in p["training_seeds"]}
        units, saved = {u["name"]: u for u in frozen["units"]}, {u["name"]: u for u in recorded["units"]}
        check("all_three_new_units", len(frozen["units"]) == len(recorded["units"]) == len(units) == len(saved) == 3 and set(units) == set(saved) == set(expected))
        allowed = set(sources) | {pp, run / "GLOBAL_FREEZE.json", run / "RESULTS.json", run / "INPUT_QUALIFICATION.json"}
        for name, seed in expected.items():
            u = units[name]
            check("unit_identity:" + name, u["seed"] == seed and u["arm"] == "D_context96" and u["shape"] == [96, 60]
                  and u["generation_shape"] == [480, 60] and u["context_length"] == 96 and u["prior_full_unit"] == old[seed]["name"])
            check("unit_artifacts:" + name, Path(u["control_path"]).resolve() == control_path and digest(control_path) == u["control_sha256"] and digest(run / name / "latest.pt") == u["checkpoint_sha256"])
            allowed.update({run / name / "latest.pt", run / name / "TRAINING_SUMMARY.json", run / name / "SYNTHETIC_PREDICTIONS.npz"})
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
                raise PermissionError("Network disabled in synthetic context replay")
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

        nfit, nstop, ntest = p["counts"]
        test, L, H = slice(nfit + nstop, None), 480, 60
        rng = np.random.default_rng(p["data_seed"] + L)
        increments = rng.standard_normal((nfit + nstop + ntest, L - 1))
        raw = np.zeros((len(increments), L), dtype=np.float64)
        raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(L)
        check("raw_bound", bool(np.isfinite(raw).all()) and float(np.abs(raw).max()) <= 32)
        q = 2.0 ** -12
        K = np.rint(raw / q).astype(np.int64)
        D = np.zeros_like(K)
        D[:, 1:] = K[:, 1:] - K[:, :-1]
        check("integer_bounds", int(np.abs(K).max()) <= 131072 and int(np.abs(D).max()) <= 262144 and np.all(K[:, -1] == 0))
        z = math.sqrt(L) * q * (K[:, -1] - K[:, -2]).astype(np.float64)
        noise = rng.standard_normal((len(K), H)).cumsum(axis=1)
        factor = np.arange(1, H + 1, dtype=np.float64) / math.sqrt(H)
        mean = .25 * z[:, None] * factor[None, :]
        y = noise + mean
        ym = y[:nfit].sum(axis=0, dtype=np.float64) / nfit
        ys = np.sqrt(np.maximum(np.square(y[:nfit]).sum(axis=0, dtype=np.float64) / nfit - ym * ym, 0.0))
        ys = np.where(ys > 1e-12, ys, 1.0)
        maximum = int(np.abs(D[:nfit]).max())
        scale = float(1 if maximum == 0 else 1 << (maximum - 1).bit_length())
        full_x = (D.astype(np.float64) / scale).astype(np.float32)
        short_D = np.zeros((len(K), 96), dtype=np.int64)
        short_D[:, 1:] = D[:, -95:]
        x = (short_D.astype(np.float64) / scale).astype(np.float32)
        near("same_retained_FP32_tail", x[:, 1:], full_x[:, -95:], exact=True)
        decoded = x.astype(np.float64) * scale
        near("short_integer_decode", decoded, short_D.astype(np.float64), exact=True)
        recovered = np.zeros_like(short_D)
        check("inverse_int64_sum_safe", 96 * 262144 < 2 ** 63)
        recovered[:, :-1] = -np.cumsum(decoded[:, 1:].astype(np.int64)[:, ::-1], axis=1, dtype=np.int64)[:, ::-1]
        near("retained_K_suffix", recovered, K[:, -96:], exact=True)
        zz = math.sqrt(L) * q * (recovered[:, -1] - recovered[:, -2]).astype(np.float64)
        near("same_visible_signal", zz, z, exact=True)
        near("same_known_mean", .25 * zz[:, None] * factor[None, :], mean, exact=True)
        iid = {"arm": "D_context96", "K_sha256": ash(K), "D_sha256": ash(D), "x_sha256": ash(x), "z_sha256": ash(z), "noise_sha256": ash(noise), "x_scale": scale, "quantum": q,
               "K_suffix_sha256": ash(K[:, -96:]), "short_D_sha256": ash(short_D), "full_D_x_sha256": ash(full_x), "context_length": 96, "generator_lookback": 480}
        qualification = read_json(run / "INPUT_QUALIFICATION.json")
        check("qualification_identity", qualification["status"] == "passed_conditional_mean_sufficient_context_deletion" and qualification["protocol_sha256"] == digest(pp) and qualification["input_identity"] == iid and qualification["target_sha256"] == ash(y) and qualification["known_mean_sha256"] == ash(mean) and qualification["target_scaler_sha256"] == hashlib.sha256(ym.tobytes() + ys.tobytes()).hexdigest())
        check("qualification_scope", qualification["full_history_information_equivalent"] is False and qualification["conditional_mean_sufficient"] is True and qualification["training_started_before_qualification"] is False)
        check("qualification_three_splits", [(s["split"], s["count"]) for s in qualification["splits"]] == [("fit", nfit), ("stop", nstop), ("test", ntest)] and all(s[k] is True for s in qualification["splits"] for k in ("suffix_history_bitwise_exact", "z_bitwise_exact", "known_mean_bitwise_exact", "labels_bitwise_exact", "last95_actual_inputs_bitwise_exact")))
        for seed, prior_unit in old.items():
            check("same_original_data:" + str(seed), all(prior_unit["input_identity"][k] == iid[k] for k in ("K_sha256", "D_sha256", "z_sha256", "noise_sha256", "x_scale", "quantum")) and prior_unit["input_identity"]["x_sha256"] == ash(full_x))
        zc = z[:nfit] - z[:nfit].mean()
        slope = (zc[:, None] * (y[:nfit] - ym)).sum(axis=0) / np.square(zc).sum()
        coefficients = np.stack((ym - z[:nfit].mean() * slope, slope))
        near("original_OLS_coefficients", read_json(control_path)["coefficients"], coefficients)
        ols = np.column_stack((np.ones(ntest), z[test])) @ coefficients
        risks, controls = {}, {}
        for name, seed in expected.items():
            deadline()
            cp = torch.load(run / name / "latest.pt", weights_only=True, map_location="cpu")
            ident, state = cp["identity"], cp["state"]
            check("checkpoint_identity:" + name, ident["protocol_sha256"] == digest(pp) and ident["shape"] == [96, 60] and ident["generation_shape"] == [480, 60] and ident["context_length"] == 96 and ident["alpha"] == .25 and ident["seed"] == seed and ident["arm"] == "D_context96" and ident["input_identity"] == iid and ident["prior_full_unit"] == old[seed]["name"] and ident["prior_results_sha256"] == digest(prior_path))
            check("runtime:" + name, ident["runtime"] == runtime)
            check("frozen_metadata:" + name, all(saved[name][k] == v for k, v in units[name].items()) and units[name]["input_identity"] == iid and saved[name]["prior_full_conditional_excess_risk"] == old[seed]["conditional_excess_risk"])
            check("matched_three_recipe_hashes:" + name, all(ident[k] == units[name][k] == old[seed][k] for k in ("initial_model_sha256", "permutation_plan_sha256", "target_scaler_sha256")))
            check("target_scaler_hash:" + name, ident["target_scaler_sha256"] == hashlib.sha256(ym.tobytes() + ys.tobytes()).hexdigest())
            near("target_mean:" + name, cp["scaler"]["y_mean"], ym, exact=True)
            near("target_scale:" + name, cp["scaler"]["y_std"], ys, exact=True)
            check("full_D_scale_used:" + name, cp["scaler"]["x_mean"] == 0 and cp["scaler"]["x_std"] == scale and cp["scaler"]["anchor_mean"] == 0 and cp["scaler"]["anchor_std"] == 1)
            torch.manual_seed(seed)
            model = cls("lstm", H, width=32)
            check("initial_model:" + name, state_hash(model.state_dict()) == ident["initial_model_sha256"])
            prng, plan, epoch_hashes = np.random.default_rng(seed), hashlib.sha256(), []
            for _ in range(20):
                permutation = prng.permutation(nfit).astype(np.int64)
                plan.update(permutation.tobytes(order="C"))
                epoch_hashes.append(ash(permutation))
            check("permutation_plan:" + name, plan.hexdigest() == ident["permutation_plan_sha256"])
            check("actual_epoch_permutations:" + name, [h["permutation_sha256"] for h in state["history"]] == epoch_hashes[:state["epoch"]])
            summary = read_json(run / name / "TRAINING_SUMMARY.json")
            check("summary:" + name, summary["status"] == "complete" and summary["identity"] == ident and summary["history"] == state["history"] and all(summary[k] == units[name][k] for k in ("epochs", "steps", "best_epoch")) and summary["epochs"] == state["epoch"] and summary["steps"] == state["steps"] and summary["best_epoch"] == state["best_epoch"])
            check("selected_checkpoint:" + name, 1 <= state["best_epoch"] <= state["epoch"] <= 20 and state["best_epoch"] == min(state["history"], key=lambda h: h["raw_endpoint_stop_mse"])["epoch"] and state["steps"] == state["epoch"] * 16)
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
            risks[name] = {}
            for metric, select in (("endpoint", lambda a: a[:, -1]), ("raw_path", lambda a: a)):
                zero, native, linear = (float(np.square(select(a)).mean()) for a in (mean[test], prediction - mean[test], ols - mean[test]))
                check("positive_zero:" + name + metric, math.isfinite(zero) and zero > 0)
                near("original_zero:" + name + metric, zero, old[seed]["conditional_excess_risk"][metric]["zero"])
                near("original_OLS_risk:" + name + metric, linear, old[seed]["conditional_excess_risk"][metric]["ols"])
                risk = {"zero": zero, "native": native, "ols": linear, "native_recovery_fraction": 1 - native / zero, "ols_recovery_fraction": 1 - linear / zero, "native_half_recovery": native <= .5 * zero, "control_half_recovery": linear <= .5 * zero}
                for key, value in risk.items():
                    check(key + ":" + name + metric, saved[name]["conditional_excess_risk"][metric][key] is value) if isinstance(value, bool) else near(key + ":" + name + metric, saved[name]["conditional_excess_risk"][metric][key], value)
                risks[name][metric] = risk
                controls["L480_H60:" + metric] = bool(linear <= .5 * zero)
        pairs, halves = [], []
        for name, seed in expected.items():
            for metric in ("endpoint", "raw_path"):
                full, short = old[seed]["conditional_excess_risk"][metric]["native"], risks[name][metric]["native"]
                pairs.append({"seed": seed, "metric": metric, "full_context": full, "short_context": short, "passed": bool(full > 0 and short <= .9 * full)})
                halves.append({"name": name, "metric": metric, "passed": risks[name][metric]["native_half_recovery"]})
        check("two_unique_OLS_gates", len(controls) == 2 and recorded["control_unique_gates"] == controls)
        for key, values in (("pair_relative_gates", pairs), ("context_half_recovery_gates", halves)):
            check("six_gates:" + key, len(recorded[key]) == len(values) == 6)
            for index, (actual, reference) in enumerate(zip(recorded[key], values)):
                for field, value in reference.items():
                    near(key + str(index) + field, actual[field], value) if field in {"full_context", "short_context"} else check(key + str(index) + field, actual[field] == value)
        groups = {"control_qualified": all(controls.values()), "all_pair_relative_gates_passed": all(g["passed"] for g in pairs), "all_context_half_recovery_gates_passed": all(g["passed"] for g in halves)}
        check("group_bools_exact", all(recorded[k] is value for k, value in groups.items()))
        report.update(status="completed_independent_context_probe_replay", units=3, control_unique_gates=controls, pair_relative_gates=pairs, context_half_recovery_gates=halves, groups=groups, risk_reconstruction=risks, maximum_numeric_difference=maximum_difference[0])
    except BaseException as error:
        report.update(status="failed_independent_context_probe_replay", error={"type": type(error).__name__, "message": str(error)})
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
