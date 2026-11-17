"""Independent audit implementation using the same PyTorch autograd backend.

No worker import, optimizer, parameter perturbation, JVP, training, or test scoring.
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

ATOL, RTOL, STOP = 1e-11, 1e-9, False
COMPONENTS = ("init", "signal", "center", "noise", "norm_residual", "cast_residual", "output_arithmetic_residual")


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
    assert not destination.exists(), "Independent terminal report exists; no automatic retry"
    checks, started = [], time.monotonic()
    report = {"status": "not_complete", "checks": checks, "protocol_sha256": digest(pp), "verifier_sha256": digest(__file__),
              "ATOL": ATOL, "RTOL": RTOL, "training_updates": 0, "parameter_perturbations": 0,
              "worker_imported": False, "independent_Jacobian_engine": False,
              "same_PyTorch_autograd_backend": True, "old_model_weights_read": False,
              "market_data_read": False, "Q3_read": False, "new_market_candidate_admitted": False,
              "scientific_success_gate_defined": False, "full_research_goal_achieved": False}

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        checks.append(name)

    def deadline():
        if STOP or time.time() >= p["stage_deadline_epoch"] - 10:
            raise TimeoutError("Original absolute gradient stage deadline or external pause")

    def read_json(path):
        path = Path(path)
        check("small_JSON:" + path.name, path.suffix == ".json" and path.stat().st_size < 2_000_000)
        return json.loads(path.read_text())

    try:
        check("scope", p["phase"] == "formal_gradient_audit" and p["numerical_audit_admitted"] is True and p["training_updates"] == 0
              and all(p[k] is False for k in ("GPU", "paid_resources", "market_data_allowed", "Q3_allowed", "synthetic_training_admitted")))
        check("CPU24_CUDA_off", set(os.sched_getaffinity(0)) == {24} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1")
        check("fixed_design", p["shapes"] == [[480, 60]] and p["generator_lookback"] == p["context_length"] == 480 and p["counts"] == [2048, 512, 4096]
              and p["data_seed"] == 20261110 and p["training_seeds"] == [2026, 2027, 2028] and p["batch_size"] == 128 and p["alphas"] == [.25] and p["max_epochs"] == 20)
        check("fixed_numeric_rules", p["learning_rate"] == .001 and p["adam_epsilon"] == 1e-8 and p["closure_multiplier"] == 256 and p["numeric_floor"] == 1e-12)
        check("finite_deadline", math.isfinite(p["stage_deadline_epoch"]) and p["stage_deadline_epoch"] >= p["deadline_epoch"])
        deadline()
        source_map = {Path(path).resolve(): wanted for path, wanted in p["source_sha256"].items()}
        prior_map = {Path(path).resolve(): wanted for path, wanted in p["prior_read_sha256"].items()}
        check("prior_small_JSON_only", all(path.suffix == ".json" and (path not in source_map or source_map[path] == wanted) for path, wanted in prior_map.items()))
        sources = {**source_map, **prior_map}
        for path, wanted in sources.items():
            check("source:" + str(path), path.suffix in {".py", ".md", ".json"} and path.stat().st_size < 2_000_000 and digest(path) == wanted)
        native_source, prior_path = Path(p["native_model_source"]).resolve(), Path(p["prior_results_path"]).resolve()
        check("required_sources_frozen", {native_source, prior_path, Path(__file__).resolve()} <= set(sources))
        prior = read_json(prior_path)
        acceptance_path = Path(p["prior_stage_acceptance_path"]).resolve()
        check("acceptance_frozen", acceptance_path in prior_map)
        acceptance = read_json(acceptance_path)
        check("accepted_prior_results", acceptance["status"] == "accepted_completed_artificial_coordinate_probe_scientific_conjunctions_failed" and acceptance["artifacts_sha256"]["formal_runs/formal/RESULTS.json"] == digest(prior_path))
        old = {u["seed"]: u for u in prior["units"] if u["shape"] == [480, 60] and u["arm"] == "D"}
        check("three_accepted_initial_identities", prior["status"] == "completed_fixed_lattice_input_pair" and set(old) == {2026, 2027, 2028})
        recorded = read_json(run / "RESULTS.json")
        check("completed_worker", recorded["status"] == "completed_fit_only_known_mean_gradient_audit" and recorded["protocol_sha256"] == digest(pp) and len(recorded["units"]) == 3)
        check("zero_training_no_promotion", recorded["training_updates"] == 0 and recorded["candidate_admitted"] is False and recorded["scientific_success_gate_defined"] is False)
        units = {u["seed"]: u for u in recorded["units"]}
        check("all_three_seeds", set(units) == {2026, 2027, 2028})
        allowed = set(sources) | {pp, run / "RESULTS.json"}
        reports = {}
        for seed, unit in units.items():
            rp, vp = run / f"seed{seed}" / "REPORT.json", run / f"seed{seed}" / "VECTORS.npz"
            check("seed_artifact_identity:" + str(seed), Path(unit["report_path"]).resolve() == rp and Path(unit["vectors_path"]).resolve() == vp and digest(rp) == unit["report_sha256"] and digest(vp) == unit["vectors_sha256"])
            reports[seed] = read_json(rp)
            allowed.update({rp, vp})
        runtime_tmp = root / "runtime_tmp"
        runtime_tmp.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(runtime_tmp)
        for key in ("TMPDIR", "TMP", "TEMP"):
            os.environ[key] = str(runtime_tmp)
        runtime_roots = [Path(v).resolve() for v in ("/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", sys.prefix, sys.base_prefix)]
        runtime_roots += [Path(v).resolve() for v in sys.path if v and Path(v).name == "site-packages"]
        for path, wanted in p.get("runtime_file_sha256", {}).items():
            check("runtime_entry:" + path, any(Path(path).resolve().is_relative_to(v) for v in runtime_roots) and digest(path) == wanted)

        def audit(event, values):
            if event.startswith("socket."):
                raise PermissionError("No network in zero-update replay")
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
        namespace = {"torch": torch, "nn": torch.nn}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native_source), "exec"), namespace)
        maximum_difference = [0.0]

        def ash(array):
            return hashlib.sha256(np.asarray(array).tobytes(order="C")).hexdigest()

        def state_hash(model):
            h = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                h.update(name.encode() + str(tensor.dtype).encode() + json.dumps(list(tensor.shape), separators=(",", ":")).encode() + tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
            return h.hexdigest()

        def near(name, actual, wanted, exact=False):
            a, b = np.asarray(actual), np.asarray(wanted)
            check("finite_shape:" + name, a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all())
            if a.size:
                maximum_difference[0] = max(maximum_difference[0], float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
            check("exact:" + name if exact else "near:" + name, (a.dtype == b.dtype and a.tobytes(order="C") == b.tobytes(order="C")) if exact else np.allclose(a, b, atol=ATOL, rtol=RTOL))

        nfit, nstop, ntest = p["counts"]
        rng = np.random.default_rng(p["data_seed"] + 480)
        increments = rng.standard_normal((nfit + nstop + ntest, 479))
        raw = np.zeros((len(increments), 480), dtype=np.float64)
        raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(480)
        check("raw_bound", bool(np.isfinite(raw).all()) and np.abs(raw).max() <= 32)
        K = np.rint(raw / (2.0 ** -12)).astype(np.int64)
        D = np.zeros_like(K)
        D[:, 1:] = K[:, 1:] - K[:, :-1]
        z = math.sqrt(480) * (2.0 ** -12) * (K[:, -1] - K[:, -2]).astype(np.float64)
        noise = rng.standard_normal((len(K), 60)).cumsum(axis=1)
        mean = .25 * z[:, None] * (np.arange(1, 61, dtype=np.float64) / math.sqrt(60))[None, :]
        y = noise + mean
        ym = y[:nfit].sum(axis=0, dtype=np.float64) / nfit
        ys = np.sqrt(np.maximum(np.square(y[:nfit]).sum(axis=0, dtype=np.float64) / nfit - ym * ym, 0.0))
        ys = np.where(ys > 1e-12, ys, 1.0)
        maximum = int(np.abs(D[:nfit]).max())
        scale = float(1 if maximum == 0 else 1 << (maximum - 1).bit_length())
        x = (D.astype(np.float64) / scale).astype(np.float32)
        iid = {"arm": "D", "K_sha256": ash(K), "D_sha256": ash(D), "x_sha256": ash(x), "z_sha256": ash(z), "noise_sha256": ash(noise), "x_scale": scale, "quantum": 2.0 ** -12}
        target64 = (y[:nfit] - ym) / ys
        target32 = target64.astype(np.float32)
        mcont = (mean[:nfit] - ym) / ys
        mean32 = mcont.astype(np.float32)
        reconstructed = []
        for seed in p["training_seeds"]:
            deadline()
            saved = reports[seed]
            check("seed_scope:" + str(seed), saved["seed"] == seed and saved["status"] == "passed_seed_audit" and saved["protocol_sha256"] == digest(pp) and saved["mode"] == "audit" and saved["prior_full_unit"] == old[seed]["name"] and saved["training_updates"] == 0 and saved["optimizer_step_called"] is False and saved["stop_test_model_evaluations"] == 0 and saved["full_fit_references_completed"] is True)
            check("data_identity:" + str(seed), saved["input_identity"] == iid == old[seed]["input_identity"] and saved["target_scaler_sha256"] == old[seed]["target_scaler_sha256"] == hashlib.sha256(ym.tobytes() + ys.tobytes()).hexdigest())
            permutation_rng, plan = np.random.default_rng(seed), hashlib.sha256()
            first = None
            for epoch in range(20):
                permutation = permutation_rng.permutation(nfit).astype(np.int64)
                plan.update(permutation.tobytes(order="C"))
                if epoch == 0:
                    first = permutation[:128]
            check("permutation_and_batch:" + str(seed), saved["permutation_plan_sha256"] == old[seed]["permutation_plan_sha256"] == plan.hexdigest() and saved["first_batch_indices_sha256"] == ash(first))
            check("native_labels:" + str(seed), saved["first_batch_target_FP32_sha256"] == ash(target32[first]) and saved["first_batch_mean_FP32_sha256"] == ash(mean32[first]))
            torch.manual_seed(seed)
            model = namespace["Recurrent"]("lstm", 60, width=32)
            before = state_hash(model)
            check("initial_state:" + str(seed), before == saved["initial_model_sha256"] == old[seed]["initial_model_sha256"] == saved["model_state_sha256_before"])
            params, layout, cursor = tuple(model.parameters()), [], 0
            for name, tensor in model.named_parameters():
                layout.append({"name": name, "shape": list(tensor.shape), "start": cursor, "stop": cursor + tensor.numel(), "group": "head" if name.startswith("head.") else "backbone", "dtype": str(tensor.dtype)})
                cursor += tensor.numel()
            check("parameter_layout:" + str(seed), saved["param_layout"] == layout)
            masks = {"all": np.ones(cursor, dtype=bool), "head": np.zeros(cursor, dtype=bool), "backbone": np.zeros(cursor, dtype=bool)}
            for item in layout:
                masks[item["group"]][item["start"]:item["stop"]] = True

            def flat(grads):
                return np.concatenate([g.detach().cpu().contiguous().numpy().ravel().astype(np.float64) for g in grads])

            def gradients(loss):
                return flat(torch.autograd.grad(loss, params, retain_graph=True))

            def forward(indices):
                xx = torch.as_tensor(x[indices, :, None])
                return model(xx, torch.zeros((len(xx), 1), dtype=xx.dtype))

            output = forward(first)
            target = torch.as_tensor(target32[first])
            loss = (output - target).square().mean()
            cotactual = torch.autograd.grad(loss, output, retain_graph=True)[0].detach().numpy().astype(np.float64)
            gY = gradients(loss)
            gM = gradients((output - torch.as_tensor(mean32[first])).square().mean())
            f64, N = output.detach().numpy().astype(np.float64), output.numel()
            normres = target64[first] - (mean[first] / ys + noise[first] / ys - ym / ys)
            castres = target32[first].astype(np.float64) - target64[first]
            cots = {"init": 2 * f64 / N, "signal": -2 * mean[first] / (N * ys), "center": np.broadcast_to(2 * ym / (N * ys), f64.shape),
                    "noise": -2 * noise[first] / (N * ys), "norm_residual": -2 * normres / N, "cast_residual": -2 * castres / N}
            sum6 = np.zeros_like(f64)
            for key in COMPONENTS[:6]:
                sum6 += cots[key]
            cots["output_arithmetic_residual"] = cotactual - sum6
            comp = {key: flat(torch.autograd.grad(output, params, grad_outputs=torch.as_tensor(np.ascontiguousarray(cots[key]).astype(np.float32)), retain_graph=True)) for key in COMPONENTS}
            sum7 = np.zeros_like(gY)
            for key in COMPONENTS:
                sum7 += comp[key]
            closure = gY - sum7
            vectors = {"gY": gY, "gM": gM, "deltaY": -.001 * gY / (np.abs(gY) + 1e-8), "deltaM": -.001 * gM / (np.abs(gM) + 1e-8), "param_closure_residual": closure,
                       "cot_actual": cotactual, "target_norm_residual64": normres, "target_cast_residual32": castres}
            vectors.update({"component_" + k: v for k, v in comp.items()})
            vectors.update({"cot_" + k: v for k, v in cots.items()})
            references = {"batch": {}, "full_fit": {}}
            for kind in ("scaled", "raw"):
                fd = output.double()
                reference_loss = (fd - torch.as_tensor(mcont[first])).square().mean() if kind == "scaled" else (fd * torch.as_tensor(ys) + torch.as_tensor(ym) - torch.as_tensor(mean[first])).square().mean()
                references["batch"][kind] = gradients(reference_loss)
                full = np.zeros_like(gY)
                for start in range(0, nfit, 128):
                    deadline()
                    section = slice(start, start + 128)
                    fd = forward(section).double()
                    objective = (fd - torch.as_tensor(mcont[section])).square().mean() if kind == "scaled" else (fd * torch.as_tensor(ys) + torch.as_tensor(ym) - torch.as_tensor(mean[section])).square().mean()
                    full += gradients(objective) * (128 / nfit)
                references["full_fit"][kind] = full
                for scope in references:
                    vectors["reference_" + scope + "_" + kind] = references[scope][kind]
            output_norm = float(np.linalg.norm(cots["output_arithmetic_residual"]))
            output_bound = 256 * np.finfo(np.float32).eps * (float(np.linalg.norm(cotactual)) + sum(float(np.linalg.norm(cots[k])) for k in COMPONENTS[:6])) + 1e-12
            parameter_norm = float(np.linalg.norm(closure))
            parameter_bound = 256 * np.finfo(np.float32).eps * (float(np.linalg.norm(gY)) + sum(float(np.linalg.norm(comp[k])) for k in COMPONENTS)) + 1e-12
            for key, actual_norm, bound in (("output_arithmetic", output_norm, output_bound), ("parameter_closure", parameter_norm, parameter_bound)):
                q = saved["qualification"][key]
                near(key + "_norm:" + str(seed), q["residual_norm"], actual_norm)
                near(key + "_bound:" + str(seed), q["bound"], bound)
                check(key + "_qualification:" + str(seed), q["passed"] is bool(actual_norm <= bound) and actual_norm <= bound)
            for key, vector in comp.items():
                near(key + "_component_norm:" + str(seed), saved["components"][key]["norm"], np.linalg.norm(vector))
                for group, mask in masks.items():
                    item = saved["components"][key]["groups"][group]
                    for field, value in (("norm", np.linalg.norm(vector[mask])), ("dot_deltaY", np.dot(vector[mask], vectors["deltaY"][mask])), ("dot_deltaM", np.dot(vector[mask], vectors["deltaM"][mask]))):
                        near(key + group + field + str(seed), item[field], value)
            direction_records = 0
            for scope, objectives in references.items():
                for kind, gradient in objectives.items():
                    for direction in ("deltaY", "deltaM"):
                        for group, mask in masks.items():
                            g, d = gradient[mask], vectors[direction][mask]
                            dot, ng, nd = float(np.dot(g, d)), float(np.linalg.norm(g)), float(np.linalg.norm(d))
                            margin = 256 * np.finfo(np.float32).eps * ng * nd + 1e-12
                            cosine = dot / (ng * nd) if ng > 0 and nd > 0 else None
                            reference = {"dot": dot, "gradient_norm": ng, "proposal_norm": nd, "gradient_step_cosine": cosine, "predicted_decrease": -dot, "numeric_margin": margin,
                                         "sign": "descent" if dot < -margin else "ascent" if dot > margin else "unresolved_at_numeric_margin"}
                            actual = saved["directional_references"][scope][kind][direction][group]
                            for key, value in reference.items():
                                check(key + scope + kind + direction + group + str(seed), actual[key] == value) if isinstance(value, str) or value is None else near(key + scope + kind + direction + group + str(seed), actual[key], value)
                            direction_records += 1
            check("all24_direction_records:" + str(seed), direction_records == 24)
            vectorpath = run / f"seed{seed}" / "VECTORS.npz"
            check("vector_file_identity:" + str(seed), Path(saved["vectors"]["file"]).resolve() == vectorpath and saved["vectors"]["sha256"] == digest(vectorpath))
            with np.load(vectorpath, allow_pickle=False) as payload:
                check("exact_vector_keyset:" + str(seed), set(payload.files) == set(vectors) == set(saved["vectors"]["arrays"]))
                for key, value in vectors.items():
                    metadata = saved["vectors"]["arrays"][key]
                    check("vector_metadata:" + key + str(seed), metadata["shape"] == list(payload[key].shape) and metadata["dtype"] == str(payload[key].dtype) == "float64" and metadata["sha256"] == ash(payload[key]))
                    near("vector:" + key + str(seed), payload[key], value)
            after = state_hash(model)
            check("model_state_unchanged:" + str(seed), before == after == saved["model_state_sha256_after"] and all(parameter.grad is None for parameter in params))
            reconstructed.append({"seed": seed, "parameter_closure_norm": parameter_norm, "parameter_closure_bound": parameter_bound, "output_arithmetic_norm": output_norm, "output_arithmetic_bound": output_bound, "direction_records": direction_records})
        check("all_groups_qualified", recorded["all_parameter_closures_passed"] is True and recorded["all_output_arithmetic_passed"] is True and recorded["model_states_unchanged"] is True)
        report.update(status="completed_independent_gradient_audit_replay", seeds=3, direction_records=72, seed_reconstructions=reconstructed, maximum_numeric_difference=maximum_difference[0])
    except BaseException as error:
        report.update(status="failed_independent_gradient_audit_replay", error={"type": type(error).__name__, "message": str(error)})
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
