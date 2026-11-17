"""Independent formal-only replay of the one frozen artificial calibration.

No worker import, market loader, optimizer update, or recipe selection. Run with
CPU 24, CUDA_VISIBLE_DEVICES=-1 and the original formal stage deadline.
Numerical tolerances are fixed before execution; scientific gates have no epsilon.
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


ATOL = 1e-11
RTOL = 1e-9
STOP = False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signal_stop(_number, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--repair-authorization", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    root = Path(protocol["output_root"]).resolve()
    run = Path(args.run_directory).resolve()
    assert run == root / "formal" and run.is_dir()
    runtime_tmp = (root / "runtime_tmp").resolve()
    assert runtime_tmp.is_relative_to(root)
    destination = run / "INDEPENDENT_REPLAY_R3.json"
    repair_path = Path(args.repair_authorization).resolve()
    repair = json.loads(repair_path.read_text())
    assert repair["original_formal_protocol_sha256"] == digest(protocol_path)
    assert repair["stage_deadline_epoch"] == protocol["stage_deadline_epoch"]
    assert repair["repaired_verifier_sha256"] == digest(__file__)
    assert repair["numeric_recipe_and_tolerances_unchanged"] is True
    assert not destination.exists(), "Terminal independent report exists; no rerun"
    checks = []
    started = time.monotonic()
    report = {
        "status": "not_complete", "checks": checks,
        "protocol_sha256": digest(protocol_path),
        "verifier_sha256": digest(__file__),
        "fixed_array_and_scalar_tolerance": {"absolute": ATOL, "relative": RTOL},
        "scientific_gate_tolerance": 0,
        "market_data_or_old_weights_read": False, "Q3_read": False,
        "training_updates": 0, "worker_functions_imported": False,
        "runtime_temporary_scope": str(runtime_tmp),
        "independent_market_confirmation": False, "new_market_candidate_admitted": False,
        "weak_natural_signal_proven": False, "full_research_goal_achieved": False,
    }

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def small_json(path):
        path = Path(path)
        assert path.suffix == ".json" and path.stat().st_size < 2_000_000
        return json.loads(path.read_text())

    def deadline():
        if STOP or time.time() >= protocol["stage_deadline_epoch"] - 10:
            raise TimeoutError("signal_or_original_formal_stage_absolute_deadline")

    try:
        check("formal_only", protocol["phase"] == "formal_synthetic"
              and protocol["synthetic_training_admitted"] is True)
        check("synthetic_resource_scope", protocol["market_data_allowed"] is False
              and protocol["GPU"] is False and protocol["paid_resources"] is False)
        check("CPU24", set(os.sched_getaffinity(0)) == {24})
        check("CUDA_disabled", os.environ.get("CUDA_VISIBLE_DEVICES") == "-1")
        check("frozen_counts", protocol["counts"] == [2048, 512, 4096])
        check("frozen_shapes", protocol["shapes"] == [[480, 60], [96, 8]])
        check("frozen_alpha_pair", protocol["alphas"] == [.25, 0.0])
        check("frozen_training_seeds", protocol["training_seeds"] == [2026, 2027, 2028])
        check("frozen_data_seed", protocol["data_seed"] == 20261110)
        check("frozen_recipe", protocol["batch_size"] == 128
              and protocol["max_epochs"] == 20 and protocol["patience"] == 3
              and protocol["learning_rate"] == .001)
        aggregation = protocol["aggregation"]
        check("frozen_aggregation", aggregation == {
            "recovery_fraction": .5, "control_unique_checks": 4,
            "native_positive_checks": 12, "null_no_threshold": True})
        check("original_finite_stage_deadline",
              math.isfinite(protocol["deadline_epoch"])
              and math.isfinite(protocol["stage_deadline_epoch"])
              and protocol["stage_deadline_epoch"] >= protocol["deadline_epoch"])
        report["original_training_deadline_epoch"] = protocol["deadline_epoch"]
        report["original_stage_deadline_epoch"] = protocol["stage_deadline_epoch"]
        deadline()
        sources = {}
        for name, wanted in protocol["source_sha256"].items():
            path = Path(name).resolve()
            check("source_text_type:" + path.name,
                  path.suffix in {".py", ".json", ".md"} and path.stat().st_size < 2_000_000)
            check("source_identity:" + str(path), digest(path) == wanted)
            sources[path] = wanted
        native_source = Path(protocol["native_model_source"]).resolve()
        check("native_source_hash_declared", native_source in sources)
        check("repair_identity_bound", digest(__file__) == repair["repaired_verifier_sha256"])
        sources[Path(__file__).resolve()] = digest(__file__)
        sources[repair_path] = digest(repair_path)
        report["source_sha256"] = {str(p): h for p, h in sources.items()}
        freeze_path, result_path = run / "GLOBAL_FREEZE.json", run / "RESULTS.json"
        frozen, recorded = small_json(freeze_path), small_json(result_path)
        report["global_freeze_sha256"] = digest(freeze_path)
        report["formal_results_sha256"] = digest(result_path)
        check("freeze_protocol", frozen["protocol_sha256"] == report["protocol_sha256"]
              and frozen["test_scored"] is False)
        check("formal_results_terminal", recorded["status"] == "completed_fixed_synthetic_calibration"
              and recorded["protocol_sha256"] == report["protocol_sha256"])
        check("results_scope", recorded["market_data_read"] is False
              and recorded["Q3_read"] is False
              and recorded["new_market_candidate_admitted"] is False
              and recorded["weak_natural_signal_proven"] is False
              and recorded["pure_optimization_or_capacity_cause_identified"] is False
              and recorded["full_research_goal_achieved"] is False)
        expected = {}
        for shape in protocol["shapes"]:
            for alpha in protocol["alphas"]:
                for seed in protocol["training_seeds"]:
                    name = f"L{shape[0]}_H{shape[1]}_a{alpha}_s{seed}"
                    expected[name] = (shape, alpha, seed)
        units = {u["name"]: u for u in frozen["units"]}
        compared_results = {u["name"]: u for u in recorded["units"]}
        check("exact12frozen_units", len(frozen["units"]) == len(units) == 12
              and set(units) == set(expected))
        check("exact12recorded_units", len(recorded["units"]) == len(compared_results) == 12
              and set(compared_results) == set(expected))
        allowed = {protocol_path, Path(__file__).resolve(), freeze_path, result_path} | set(sources)
        for name, (shape, alpha, seed) in expected.items():
            unit = units[name]
            control = run / f"control_L{shape[0]}_H{shape[1]}_a{alpha}.json"
            cp = run / name / "latest.pt"
            check("unit_configuration:" + name,
                  unit["shape"] == shape and unit["alpha"] == alpha and unit["seed"] == seed)
            check("control_path:" + name, Path(unit["control_path"]).resolve() == control)
            check("checkpoint_hash:" + name, digest(cp) == unit["checkpoint_sha256"])
            check("control_hash:" + name, digest(control) == unit["control_sha256"])
            allowed.update({cp, control, run / name / "TRAINING_SUMMARY.json",
                            run / name / "SYNTHETIC_PREDICTIONS.npz"})

        repo = Path(protocol["repository"]).resolve()
        extra_runtime = [Path(v).resolve() for v in repair["additional_runtime_roots"]]
        assert all(v.name == "site-packages" for v in extra_runtime)
        for runtime_file, wanted in repair["runtime_file_sha256"].items():
            runtime_path = Path(runtime_file).resolve()
            assert any(runtime_path.is_relative_to(v) for v in extra_runtime)
            assert digest(runtime_path) == wanted
        runtime_roots = extra_runtime + [Path(p).resolve() for p in (
            "/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev",
            sys.prefix, sys.base_prefix, repo / ".venv")]
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(runtime_tmp)
        os.environ["TMPDIR"] = str(runtime_tmp)
        os.environ["TEMP"] = str(runtime_tmp)
        os.environ["TMP"] = str(runtime_tmp)
        report["runtime_read_prefixes"] = [str(p) for p in runtime_roots]
        report["allowed_persistent_output"] = str(destination)

        def audit(event, values):
            if event.startswith("socket."):
                raise PermissionError("Network is not a synthetic replay input")
            if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(values[0])).resolve()
            mode, flags = values[1] if len(values) > 1 else "r", values[2] if len(values) > 2 else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing:
                report_write = path.parent == run and (path.name.startswith(".independent_replay_")
                                                        or path == destination)
                if not report_write and not path.is_relative_to(runtime_tmp):
                    raise PermissionError("Unadmitted write outside own report/runtime_tmp: " + str(path))
            elif (path not in allowed and not path.is_relative_to(runtime_tmp)
                  and not any(path.is_relative_to(p) for p in runtime_roots)):
                raise PermissionError("Unadmitted replay read: " + str(path))

        sys.dont_write_bytecode = True
        sys.addaudithook(audit)
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        signal.signal(signal.SIGUSR1, signal_stop)
        signal.signal(signal.SIGTERM, signal_stop)
        runtime = {
            "torch": str(torch.__version__), "numpy": str(np.__version__), "python": sys.version,
            "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch_threads": torch.get_num_threads(),
            "torch_interop": torch.get_num_interop_threads(),
            "mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
            "mkldnn_deterministic": bool(torch.backends.mkldnn.deterministic),
            "default_dtype": str(torch.get_default_dtype()), "byteorder": sys.byteorder,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        }
        report["runtime"] = runtime
        nodes = [x for x in ast.parse(native_source.read_text()).body
                 if isinstance(x, ast.ClassDef) and x.name == "Recurrent"]
        check("one_original_Recurrent_AST", len(nodes) == 1)
        namespace = {"nn": torch.nn, "torch": torch}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native_source), "exec"), namespace)
        cls = namespace["Recurrent"]
        array_differences = []

        def near(name, actual, wanted, exact=False):
            a, b = np.asarray(actual), np.asarray(wanted)
            check("shape:" + name, a.shape == b.shape)
            check("finite:" + name, np.isfinite(a).all() and np.isfinite(b).all())
            difference = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))) if a.size else 0.
            same = bool(np.array_equal(a, b))
            check("numeric:" + name, same if exact else np.allclose(a, b, atol=ATOL, rtol=RTOL))
            array_differences.append({"name": name, "exact": same, "max_absolute_difference": difference})

        def scalar(name, actual, wanted):
            check("scalar:" + name, math.isfinite(float(actual)) and math.isfinite(float(wanted))
                  and abs(float(actual) - float(wanted)) <= ATOL + RTOL * abs(float(wanted)))

        rebuilt = {}
        controls = {}
        nfit, nstop, ntest = protocol["counts"]
        for length, horizon in protocol["shapes"]:
            deadline()
            generator = np.random.default_rng(20261110 + length)
            increments = generator.standard_normal((nfit + nstop + ntest, length - 1))
            history = np.zeros((len(increments), length), dtype=np.float64)
            running = np.zeros(len(increments), dtype=np.float64)
            for column in range(length - 2, -1, -1):
                running = running + increments[:, column]
                history[:, column] = -running / math.sqrt(length)
            fitted = history[:nfit]
            xm = float(np.sum(fitted, dtype=np.float64) / fitted.size)
            xs = math.sqrt(max(float(np.sum(fitted * fitted, dtype=np.float64) / fitted.size) - xm * xm, 0.))
            xs = xs if xs > 1e-12 else 1.
            visible = ((history - xm) / xs).astype(np.float32)
            z = math.sqrt(length) * xs * (visible[:, -1].astype(np.float64)
                                         - visible[:, -2].astype(np.float64))
            innovations = generator.standard_normal((len(visible), horizon))
            noise = np.empty_like(innovations)
            accumulated = np.zeros(len(visible), dtype=np.float64)
            for column in range(horizon):
                accumulated = accumulated + innovations[:, column]
                noise[:, column] = accumulated
            factor = np.arange(1, horizon + 1, dtype=np.float64) / math.sqrt(horizon)
            for alpha in protocol["alphas"]:
                mean = alpha * z[:, None] * factor[None, :]
                labels = noise + mean
                ym = np.sum(labels[:nfit], axis=0, dtype=np.float64) / nfit
                ys = np.sqrt(np.maximum(np.sum(labels[:nfit] * labels[:nfit], axis=0,
                                                dtype=np.float64) / nfit - ym * ym, 0.))
                ys = np.where(ys > 1e-12, ys, 1.)
                # Centered closed-form OLS; no worker function or lstsq call.
                zmean = float(np.mean(z[:nfit]))
                centered_z = z[:nfit] - zmean
                centered_y = labels[:nfit] - ym
                denominator = float(np.dot(centered_z, centered_z))
                check(f"OLS_positive_design_L{length}_a{alpha}", denominator > 0 and math.isfinite(denominator))
                slope = centered_z @ centered_y / denominator
                coefficients = np.stack([ym - zmean * slope, slope])
                control_path = run / f"control_L{length}_H{horizon}_a{alpha}.json"
                control = small_json(control_path)
                check("control_configuration:" + control_path.name,
                      control["shape"] == [length, horizon] and control["alpha"] == alpha)
                near("OLS_coefficients:" + control_path.name, coefficients, control["coefficients"])
                tail = slice(nfit + nstop, None)
                ols = coefficients[0] + z[tail, None] * coefficients[1]
                scaler = {"x_mean": xm, "x_std": xs, "y_mean": ym.tolist(), "y_std": ys.tolist(),
                          "anchor_mean": 100., "anchor_std": 1.}
                rebuilt[(length, alpha)] = (visible[tail], z[tail], mean[tail], ym, ys, scaler, ols)
                controls[(length, alpha)] = control

        reconstructed = []
        control_gates = {}
        native_gates = []
        for name, (shape, alpha, seed) in expected.items():
            deadline()
            unit = units[name]
            cp_path = run / name / "latest.pt"
            check("checkpoint_still_frozen:" + name, digest(cp_path) == unit["checkpoint_sha256"])
            checkpoint = torch.load(cp_path, weights_only=True, map_location="cpu")
            check("checkpoint_configuration_runtime:" + name, checkpoint["identity"] == {
                "protocol_sha256": report["protocol_sha256"], "shape": shape,
                "alpha": alpha, "seed": seed, "runtime": runtime})
            x, z, known, ym, ys, scaler, ols = rebuilt[(shape[0], alpha)]
            check("scaler_fields:" + name, set(checkpoint["scaler"]) == set(scaler))
            for field in scaler:
                near("scaler:" + name + ":" + field, checkpoint["scaler"][field], scaler[field])
            state = checkpoint["state"]
            history = state["history"]
            epochs = state["epoch"]
            batches = math.ceil(nfit / protocol["batch_size"])
            check("completed_epoch_state:" + name,
                  1 <= epochs <= protocol["max_epochs"] and len(history) == epochs
                  and state["cursor"] == 0 and state["permutation"] is None
                  and state["steps"] == epochs * batches
                  and state["epoch_seen"] == 0 and state["epoch_loss_sum"] == 0.)
            losses = [h["raw_endpoint_stop_mse"] for h in history]
            check("finite_training_history:" + name,
                  all(math.isfinite(v) and v >= 0 for v in losses)
                  and all(math.isfinite(h["training_scaled_path_loss"])
                          and h["training_scaled_path_loss"] >= 0 for h in history))
            best_epoch = min(range(epochs), key=losses.__getitem__) + 1
            check("epoch_history_steps:" + name,
                  all(h["epoch"] == i + 1 and h["steps"] == (i + 1) * batches for i, h in enumerate(history)))
            check("best_epoch_and_stale:" + name,
                  state["best_epoch"] == best_epoch and state["best_loss"] == losses[best_epoch - 1]
                  and state["stale"] == epochs - best_epoch
                  and (epochs == protocol["max_epochs"] or state["stale"] >= protocol["patience"]))
            summary = small_json(run / name / "TRAINING_SUMMARY.json")
            check("training_summary:" + name, summary == {
                "status": "complete", "epochs": epochs, "steps": state["steps"],
                "best_epoch": best_epoch, "history": history, "identity": checkpoint["identity"]})
            for field, value in (("epochs", epochs), ("steps", state["steps"]), ("best_epoch", best_epoch)):
                check("freeze_progress:" + name + ":" + field, unit[field] == value)
            groups = checkpoint["optimizer"]["param_groups"]
            check("Adam_configuration:" + name,
                  len(groups) == 1 and groups[0]["lr"] == protocol["learning_rate"]
                  and groups[0]["betas"] == (.9, .999) and groups[0]["eps"] == 1e-8
                  and groups[0]["weight_decay"] == 0 and groups[0]["amsgrad"] is False)
            check("Adam_all_steps:" + name,
                  len(checkpoint["optimizer"]["state"]) == len(groups[0]["params"])
                  and all(float(s["step"]) == state["steps"] for s in checkpoint["optimizer"]["state"].values()))
            model = cls("lstm", shape[1], width=32)
            check("native_model_keys:" + name,
                  set(model.state_dict()) == set(state["best_model"]) == set(checkpoint["model"]))
            model.load_state_dict(state["best_model"], strict=True)
            check("finite_best_model:" + name, all(torch.isfinite(v).all() for v in model.state_dict().values()))
            model.eval()
            output = []
            with torch.inference_mode():
                for begin in range(0, ntest, protocol["batch_size"]):
                    deadline()
                    batch = torch.as_tensor(x[begin:begin + protocol["batch_size"], :, None])
                    values = model(batch, torch.zeros((len(batch), 1), dtype=batch.dtype))
                    output.append(values.numpy().astype(np.float64) * ys + ym)
            prediction = np.concatenate(output, axis=0)
            npz_path = run / name / "SYNTHETIC_PREDICTIONS.npz"
            with np.load(npz_path, allow_pickle=False) as saved:
                check("saved_array_keys:" + name, set(saved.files) == {"native", "ols", "known_mean", "visible_z"})
                near("native_replay:" + name, prediction, saved["native"])
                near("OLS_replay:" + name, ols, saved["ols"])
                near("known_mean:" + name, known, saved["known_mean"], exact=True)
                near("visible_z:" + name, z, saved["visible_z"], exact=True)
            record = {"name": name, "shape": shape, "alpha": alpha, "seed": seed,
                      "epochs": epochs, "steps": state["steps"], "best_epoch": best_epoch,
                      "checkpoint_sha256": unit["checkpoint_sha256"],
                      "predictions_sha256": digest(npz_path), "conditional_excess_risk": {}}
            old = compared_results[name]
            for key, value in unit.items():
                check("recorded_unit_identity:" + name + ":" + key, old[key] == value)
            for metric in ("endpoint", "raw_path"):
                def select(a):
                    return a[:, -1] if metric == "endpoint" else a

                def mean_square(a):
                    values = np.asarray(a, dtype=np.float64).reshape(-1)
                    return math.fsum(float(v) * float(v) for v in values) / len(values)

                zero = mean_square(select(known))
                native = mean_square(select(prediction - known))
                linear = mean_square(select(ols - known))
                risks = {"zero": zero, "native": native, "ols": linear}
                if alpha > 0:
                    check("positive_signal_denominator:" + name + ":" + metric, zero > 0 and math.isfinite(zero))
                    threshold = 1. - aggregation["recovery_fraction"]
                    risks.update(native_recovery_fraction=1. - native / zero,
                                 ols_recovery_fraction=1. - linear / zero,
                                 native_half_recovery=native <= threshold * zero,
                                 control_half_recovery=linear <= threshold * zero)
                    gate_key = f"L{shape[0]}_H{shape[1]}:{metric}"
                    gate = risks["control_half_recovery"]
                    check("same_control_gate_across_seeds:" + name + ":" + metric,
                          gate_key not in control_gates or control_gates[gate_key] == gate)
                    control_gates[gate_key] = gate
                    native_gates.append({"name": name, "metric": metric,
                                         "passed": risks["native_half_recovery"]})
                else:
                    check("null_signal_exact_zero:" + name + ":" + metric, zero == 0.)
                    noise = float(shape[1]) if metric == "endpoint" else (shape[1] + 1) / 2.
                    risks.update(native_energy_over_noise=native / noise, ols_energy_over_noise=linear / noise)
                wanted = old["conditional_excess_risk"][metric]
                check("recorded_risk_fields:" + name + ":" + metric, set(wanted) == set(risks))
                for key, value in risks.items():
                    label = name + ":" + metric + ":" + key
                    if isinstance(value, bool):
                        check("exact_gate:" + label, isinstance(wanted[key], bool) and wanted[key] == value)
                    else:
                        scalar(label, value, wanted[key])
                record["conditional_excess_risk"][metric] = risks
            reconstructed.append(record)
        check("four_unique_OLS_gates", len(control_gates) == aggregation["control_unique_checks"])
        check("twelve_native_positive_gates", len(native_gates) == aggregation["native_positive_checks"])
        control_ok, native_ok = all(control_gates.values()), all(g["passed"] for g in native_gates)
        if not control_ok:
            verdict = "inconclusive_positive_control_unqualified_no_neural_inability_inference"
        elif native_ok:
            verdict = "recovered_fixed_artificial_conditional_mean_by_prespecified_operation_only"
        else:
            verdict = "partial_fixed_recipe_recovery_deficit_on_this_artificial_design_only"
        report.update(status="passed_independent_formal_synthetic_reconstruction_and_checkpoint_inference",
                      reconstructed_units=reconstructed, control_unique_gates=control_gates,
                      native_positive_gates=native_gates, control_qualified=control_ok,
                      all_native_positive_gates_passed=native_ok, verdict=verdict,
                      array_comparisons=array_differences,
                      limits=["Fixed artificial testX; no natural-market mechanism or weak-signal inference.",
                              "Three training seeds share data; not three independent artificial histories.",
                              "Control failure makes neural inability inconclusive. Null energy has no pass threshold.",
                              "No training replay or proof of global exact native-network expressivity.",
                              "Runtime-only temporary files may be read/written under declared output_root/runtime_tmp; the sole persistent report is INDEPENDENT_REPLAY.json.",
                              "Python read guard and frozen source checks are not an OS sandbox."])
    except Exception as error:
        report.update(status="failed_or_incomplete_independent_formal_synthetic_replay",
                      failure_type=type(error).__name__, failure=str(error))
        print("INDEPENDENT_PRIMARY_FAILURE " + type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
    report["checks_passed"] = len(checks)
    report["elapsed_seconds"] = time.monotonic() - started
    temporary = run / (".independent_replay_" + str(os.getpid()) + ".json")
    with temporary.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    print(json.dumps({"status": report["status"], "checks_passed": len(checks), "report": str(destination)}))
    return 0 if report["status"].startswith("passed_") else 1


if __name__ == "__main__":
    sys.exit(main())
