"""Zero-update fit-only known-mean/noise gradient audit at frozen initialization.

Native FP32 gradients feed an explicitly ideal float64 first-Adam formula.
No optimizer, optimizer.step, weight perturbation, JVP or non-fit forward calls.
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
COMPONENTS = ("init", "signal", "center", "noise", "norm_residual", "cast_residual", "output_arithmetic_residual")
RUNTIME_ROOTS = ["/home/ayaka/.venvs/xpatch-ls/lib/python3.13/site-packages",
                 "/home/ayaka/.venvs/xpatch-v2/lib/python3.13/site-packages",
                 "/home/ayaka/wslcode/aapl-research/.venv/lib/python3.13/site-packages"]


def stopped(_number, _frame):
    global STOP
    STOP = True


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("w") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npz(np, path, arrays):
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def deadline(p):
    if STOP or time.time() >= p["deadline_epoch"] - 5:
        raise TimeoutError("Signal or frozen audit absolute deadline")


def prepare(args):
    pp = Path(args.protocol).resolve()
    p = json.loads(pp.read_text())
    assert p["phase"] in ("preparation", "formal_gradient_audit")
    assert p["numerical_audit_admitted"] is (p["phase"] == "formal_gradient_audit")
    assert p["synthetic_training_admitted"] is False
    assert all(p[k] is False for k in ("market_data_allowed", "Q3_allowed", "GPU", "paid_resources"))
    assert set(os.sched_getaffinity(0)) == {24} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert p["counts"] == [2048, 512, 4096] and p["shapes"] == [[480, 60]]
    assert p["generator_lookback"] == p["context_length"] == 480 and p["data_seed"] == 20261110
    assert p["alphas"] == [.25] and p["training_seeds"] == [2026, 2027, 2028]
    assert p["batch_size"] == 128 and p["learning_rate"] == .001 and p["adam_epsilon"] == 1e-8
    assert p["closure_multiplier"] == 256 and p["numeric_floor"] == 1e-12 and p["max_epochs"] == 20
    assert math.isfinite(p["deadline_epoch"]) and p["started_epoch"] <= time.time()
    assert p["deadline_epoch"] > p["started_epoch"]
    deadline(p)
    source = {Path(name).resolve(): wanted for name, wanted in p["source_sha256"].items()}
    prior_files = {Path(name).resolve(): wanted for name, wanted in p["prior_read_sha256"].items()}
    assert source.get(Path(__file__).resolve()) == sha(__file__)
    for path, wanted in {**source, **prior_files}.items():
        assert path.suffix in {".py", ".md", ".json"} and path.stat().st_size < 2_000_000 and sha(path) == wanted
    native = Path(p["native_model_source"]).resolve()
    assert native in source
    prior_path = Path(p["prior_results_path"]).resolve()
    acceptance_path = Path(p["prior_stage_acceptance_path"]).resolve()
    assert prior_path in prior_files and acceptance_path in prior_files
    prior = json.loads(prior_path.read_text())
    accepted = json.loads(acceptance_path.read_text())
    assert prior["status"] == "completed_fixed_lattice_input_pair" and len(prior["units"]) == 12
    assert accepted["status"] == "accepted_completed_artificial_coordinate_probe_scientific_conjunctions_failed"
    assert accepted["artifacts_sha256"]["formal_runs/formal/RESULTS.json"] == sha(prior_path)
    units = {u["seed"]: u for u in prior["units"] if u["shape"] == [480, 60] and u["arm"] == "D"}
    assert set(units) == {2026, 2027, 2028}
    root, output = Path(p["output_root"]).resolve(), Path(args.output).resolve()
    assert output.is_relative_to(root)
    output.mkdir(parents=True, exist_ok=True)
    runtime_tmp = root / "runtime_tmp"
    runtime_tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(runtime_tmp)
    for key in ("TMPDIR", "TMP", "TEMP"):
        os.environ[key] = str(runtime_tmp)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    runtime = [Path(v).resolve() for v in RUNTIME_ROOTS +
               ["/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", sys.prefix, sys.base_prefix]]
    for name, wanted in p["runtime_file_sha256"].items():
        path = Path(name).resolve()
        assert any(path.is_relative_to(v) for v in runtime) and sha(path) == wanted
    allowed = set(source) | set(prior_files) | {pp}

    def audit(event, values):
        if event.startswith("socket."):
            raise PermissionError("No network input")
        if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode, flags = values[1], values[2]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and not path.is_relative_to(root) and path != Path("/dev/null"):
            raise PermissionError("Unadmitted audit write: " + str(path))
        if not writing and path not in allowed and not path.is_relative_to(root) and not any(path.is_relative_to(v) for v in runtime):
            raise PermissionError("Unadmitted audit read: " + str(path))

    sys.dont_write_bytecode = True
    sys.addaudithook(audit)
    signal.signal(signal.SIGUSR1, stopped)
    signal.signal(signal.SIGTERM, stopped)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    nodes = [n for n in ast.parse(native.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "Recurrent"]
    assert len(nodes) == 1
    namespace = {"nn": torch.nn, "torch": torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(native), "exec"), namespace)
    deadline(p)
    return p, output, np, torch, namespace["Recurrent"], sha(pp), units

def initial_model_sha(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode("utf-8"))
        result.update(str(tensor.dtype).encode("ascii"))
        result.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return result.hexdigest()


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


def flatten(np, gradients):
    return np.concatenate([g.detach().cpu().contiguous().numpy().astype(np.float64).reshape(-1) for g in gradients])


def layout_for(model):
    layout, offset = [], 0
    for name, parameter in model.named_parameters():
        size = parameter.numel()
        layout.append({"name": name, "shape": list(parameter.shape), "start": offset, "stop": offset + size,
                       "group": "head" if name.startswith("head.") else "backbone", "dtype": str(parameter.dtype)})
        offset += size
    assert offset == 6520
    return layout


def group_indices(np, layout):
    return {group: np.concatenate([np.arange(x["start"], x["stop"], dtype=np.int64) for x in layout
            if group == "all" or x["group"] == group]) for group in ("all", "head", "backbone")}


def directional(np, gradient, proposal, p):
    gn, dn = float(np.linalg.norm(gradient)), float(np.linalg.norm(proposal))
    dot = float(np.dot(gradient, proposal))
    margin = p["closure_multiplier"] * float(np.finfo(np.float32).eps) * gn * dn + p["numeric_floor"]
    assert all(math.isfinite(v) for v in (gn, dn, dot, margin))
    return {"dot": dot, "gradient_norm": gn, "proposal_norm": dn,
            "gradient_step_cosine": dot / (gn * dn) if gn * dn > 0 else None,
            "predicted_decrease": -dot, "numeric_margin": margin,
            "sign": "descent" if dot < -margin else "ascent" if dot > margin else "unresolved_at_numeric_margin"}


def reference_gradients(np, torch, model, params, x, mean, b, s):
    xx = torch.as_tensor(x[:, :, None])
    prediction = model(xx, torch.zeros((len(xx), 1), dtype=xx.dtype))
    mu = torch.as_tensor(mean, dtype=torch.float64)
    center, scale = torch.as_tensor(b, dtype=torch.float64), torch.as_tensor(s, dtype=torch.float64)
    scaled = (prediction.double() - (mu - center) / scale).square().mean()
    raw = (prediction.double() * scale + center - mu).square().mean()
    gs = flatten(np, torch.autograd.grad(scaled, params, retain_graph=True))
    gr = flatten(np, torch.autograd.grad(raw, params))
    return gs, gr


def analyse_seed(np, torch, cls, p, base, prior_unit, seed, full_fit):
    began = time.monotonic()
    nfit = p["counts"][0]
    mean = .25 * base["z"][:nfit, None] * base["factor"][None, :]
    noise = base["noise"][:nfit]
    y = noise + mean
    b = y.sum(axis=0, dtype=np.float64) / nfit
    s = np.sqrt(np.maximum(np.square(y).sum(axis=0, dtype=np.float64) / nfit - b * b, 0.0))
    s = np.where(s > 1e-12, s, 1.0)
    t64 = (y - b) / s
    t32 = t64.astype(np.float32)
    mcont = (mean - b) / s
    m32 = mcont.astype(np.float32)
    x = base["arms"]["D"]["x"]
    iid = {"arm": "D", "K_sha256": array_sha(base["K"]), "D_sha256": array_sha(base["D"]),
        "x_sha256": array_sha(x), "z_sha256": array_sha(base["z"]), "noise_sha256": array_sha(base["noise"]),
        "x_scale": base["arms"]["D"]["x_std"], "quantum": base["quantum"]}
    assert iid == prior_unit["input_identity"]
    assert hashlib.sha256(b.tobytes() + s.tobytes()).hexdigest() == prior_unit["target_scaler_sha256"]
    plan_rng, plan = np.random.default_rng(seed), hashlib.sha256()
    first = None
    for _ in range(20):
        permutation = plan_rng.permutation(nfit).astype(np.int64)
        if first is None:
            first = permutation[:128].copy()
        plan.update(permutation.tobytes(order="C"))
    assert plan.hexdigest() == prior_unit["permutation_plan_sha256"] and np.all(first < nfit)
    torch.manual_seed(seed)
    model = cls("lstm", 60, width=32)
    model.train()
    before = initial_model_sha(model)
    assert before == prior_unit["initial_model_sha256"]
    params = tuple(model.parameters())
    layout = layout_for(model)
    groups = group_indices(np, layout)
    xx = torch.as_tensor(x[first, :, None])
    prediction = model(xx, torch.zeros((len(xx), 1), dtype=xx.dtype))
    loss_y = (prediction - torch.as_tensor(t32[first])).square().mean()
    loss_m = (prediction - torch.as_tensor(m32[first])).square().mean()
    actual_cot = torch.autograd.grad(loss_y, prediction, retain_graph=True)[0].detach().numpy().astype(np.float64)
    gy = flatten(np, torch.autograd.grad(loss_y, params, retain_graph=True))
    gm = flatten(np, torch.autograd.grad(loss_m, params, retain_graph=True))
    f64, count = prediction.detach().numpy().astype(np.float64), len(first) * 60
    mu, eps = mean[first], noise[first]
    norm_residual = t64[first] - (mu / s + eps / s - b / s)
    cast_residual = t32[first].astype(np.float64) - t64[first]
    cots = {"init": 2 * f64 / count, "signal": -2 * mu / (count * s),
        "center": np.broadcast_to(2 * b / (count * s), f64.shape).copy(), "noise": -2 * eps / (count * s),
        "norm_residual": -2 * norm_residual / count, "cast_residual": -2 * cast_residual / count}
    summed_cot = np.zeros_like(actual_cot)
    for name in COMPONENTS[:-1]:
        summed_cot += cots[name]
    cots["output_arithmetic_residual"] = actual_cot - summed_cot
    output_residual_norm = float(np.linalg.norm(cots["output_arithmetic_residual"]))
    output_bound = p["closure_multiplier"] * float(np.finfo(np.float32).eps) * (
        float(np.linalg.norm(actual_cot)) + sum(float(np.linalg.norm(cots[name])) for name in COMPONENTS[:-1])) + p["numeric_floor"]
    output_passed = output_residual_norm <= output_bound
    vectors = {"gY": gy, "gM": gm, "cot_actual": actual_cot,
               "target_norm_residual64": norm_residual, "target_cast_residual32": cast_residual}
    components = {}
    for name in COMPONENTS:
        deadline(p)
        cot32 = np.ascontiguousarray(cots[name]).astype(np.float32)
        gradient = flatten(np, torch.autograd.grad(prediction, params, grad_outputs=torch.as_tensor(cot32), retain_graph=True))
        vectors["component_" + name], vectors["cot_" + name] = gradient, cots[name]
        components[name] = {"norm": float(np.linalg.norm(gradient))}
    sum_grads = np.zeros_like(gy)
    for name in COMPONENTS:
        sum_grads += vectors["component_" + name]
    residual = gy - sum_grads
    bound = p["closure_multiplier"] * float(np.finfo(np.float32).eps) * (
        float(np.linalg.norm(gy)) + sum(components[name]["norm"] for name in COMPONENTS)) + p["numeric_floor"]
    residual_norm = float(np.linalg.norm(residual))
    closure_passed = residual_norm <= bound
    vectors["param_closure_residual"] = residual
    for name, gradient in (("deltaY", gy), ("deltaM", gm)):
        vectors[name] = -p["learning_rate"] * gradient / (np.abs(gradient) + p["adam_epsilon"])
    report = {"seed": seed, "status": "passed_seed_audit" if closure_passed and output_passed else "failed_numerical_qualification",
        "prior_full_unit": prior_unit["name"], "input_identity": iid,
        "target_scaler_sha256": prior_unit["target_scaler_sha256"], "initial_model_sha256": before,
        "permutation_plan_sha256": plan.hexdigest(), "first_batch_indices_sha256": array_sha(first),
        "first_batch_target_FP32_sha256": array_sha(t32[first]), "first_batch_mean_FP32_sha256": array_sha(m32[first]),
        "param_layout": layout, "model_state_sha256_before": before, "components": components,
        "qualification": {"parameter_closure": {"residual_norm": residual_norm, "bound": bound, "passed": closure_passed},
            "output_arithmetic": {"residual_norm": output_residual_norm, "bound": output_bound, "passed": output_passed}},
        "full_fit_references_completed": False, "directional_references": {}, "training_updates": 0,
        "optimizer_step_called": False, "stop_test_model_evaluations": 0}
    assert all(np.isfinite(a).all() for a in vectors.values())
    if closure_passed and output_passed and full_fit:
        mu_t = torch.as_tensor(mu, dtype=torch.float64)
        b_t, s_t = torch.as_tensor(b, dtype=torch.float64), torch.as_tensor(s, dtype=torch.float64)
        scaled = (prediction.double() - (mu_t - b_t) / s_t).square().mean()
        raw = (prediction.double() * s_t + b_t - mu_t).square().mean()
        vectors["reference_batch_scaled"] = flatten(np, torch.autograd.grad(scaled, params, retain_graph=True))
        vectors["reference_batch_raw"] = flatten(np, torch.autograd.grad(raw, params))
        del prediction, loss_y, loss_m
        if full_fit:
            scaled_sum, raw_sum = np.zeros_like(gy), np.zeros_like(gy)
            for start in range(0, nfit, 128):
                deadline(p)
                end = min(start + 128, nfit)
                gs, gr = reference_gradients(np, torch, model, params, x[start:end], mean[start:end], b, s)
                scaled_sum += gs * ((end - start) / nfit)
                raw_sum += gr * ((end - start) / nfit)
            vectors["reference_full_fit_scaled"], vectors["reference_full_fit_raw"] = scaled_sum, raw_sum
            report["full_fit_references_completed"] = True
        for part in ("batch", "full_fit") if full_fit else ("batch",):
            report["directional_references"][part] = {}
            for units in ("scaled", "raw"):
                reference = vectors["reference_" + part + "_" + units]
                report["directional_references"][part][units] = {proposal: {
                    group: directional(np, reference[index], vectors[proposal][index], p)
                    for group, index in groups.items()} for proposal in ("deltaY", "deltaM")}
    for name in COMPONENTS:
        vector = vectors["component_" + name]
        components[name]["groups"] = {group: {"norm": float(np.linalg.norm(vector[index]))}
                                     for group, index in groups.items()}
        if full_fit:
            for group, index in groups.items():
                components[name]["groups"][group].update(
                    dot_deltaY=float(np.dot(vector[index], vectors["deltaY"][index])),
                    dot_deltaM=float(np.dot(vector[index], vectors["deltaM"][index])))
    report["model_state_sha256_after"] = initial_model_sha(model)
    assert report["model_state_sha256_after"] == before and all(parameter.grad is None for parameter in params)
    assert all(np.isfinite(a).all() for a in vectors.values())
    deadline(p)
    report["elapsed_seconds"] = time.monotonic() - began
    return report, vectors


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--protocol", required=True)
    cli.add_argument("--output", required=True)
    cli.add_argument("--mode", choices=("qualify", "audit"), required=True)
    cli.add_argument("--resume", action="store_true")
    args = cli.parse_args()
    p, output, np, torch, cls, protocol_hash, prior_units = prepare(args)
    assert (args.mode == "qualify") == (p["phase"] == "preparation")
    assert not (output / "RESULTS.json").exists(), "Terminal result exists; no rerun"
    began, units = time.monotonic(), []
    progress = output / "PROGRESS.json"
    try:
        if args.resume:
            saved = json.loads(progress.read_text())
            assert saved["protocol_sha256"] == protocol_hash
            units = saved["units"]
            assert [u["seed"] for u in units] == p["training_seeds"][:len(units)]
            for unit in units:
                assert Path(unit["report_path"]).resolve() == output / f"seed{unit['seed']}" / "REPORT.json"
                assert Path(unit["vectors_path"]).resolve() == output / f"seed{unit['seed']}" / "VECTORS.npz"
                assert sha(unit["report_path"]) == unit["report_sha256"] and sha(unit["vectors_path"]) == unit["vectors_sha256"]
        else:
            assert not progress.exists(), "Progress requires explicit --resume"
        base = generate(np, p, 480, 60)
        seeds = [2026] if args.mode == "qualify" else p["training_seeds"]
        completed = {unit["seed"] for unit in units}
        for seed in seeds:
            if seed in completed:
                continue
            deadline(p)
            report, vectors = analyse_seed(np, torch, cls, p, base, prior_units[seed], seed, args.mode == "audit")
            report.update(protocol_sha256=protocol_hash, mode=args.mode,
                proposal_definition="Ideal float64 bias-corrected first Adam formula from direct native FP32 gradient; not literal FP32 optimizer displacement.",
                closure_definition="Descriptive signed native-FP32 VJPs plus separately reported arithmetic residual; never substitutes component sum for direct Adam gradient.",
                numeric_margin_definition="256*eps32*norm(reference)*norm(proposal)+1e-12 is a fixed resolution policy, not a theorem or success gate.",
                limits="Fit-only local derivatives at initialization; not later-trajectory, H-dimension causation, task independence, predictive success or candidate admission.")
            directory = output / f"seed{seed}"
            directory.mkdir(exist_ok=True)
            vector_path, report_path = directory / "VECTORS.npz", directory / "REPORT.json"
            atomic_npz(np, vector_path, vectors)
            report["vectors"] = {"file": str(vector_path), "sha256": sha(vector_path),
                "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha(value)} for key, value in vectors.items()}}
            atomic_json(report_path, report)
            if report["status"] != "passed_seed_audit":
                raise ArithmeticError("Frozen parameter-closure qualification failed; no tolerance repair")
            units.append({"seed": seed, "report_path": str(report_path), "report_sha256": sha(report_path),
                          "vectors_path": str(vector_path), "vectors_sha256": sha(vector_path)})
            atomic_json(progress, {"protocol_sha256": protocol_hash, "units": units, "training_updates": 0})
            print("COMPLETE seed" + str(seed), flush=True)
        assert len(units) == len(seeds)
        reports = [json.loads(Path(u["report_path"]).read_text()) for u in units]
        result = {"status": "qualification_and_cost_complete" if args.mode == "qualify" else "completed_fit_only_known_mean_gradient_audit",
            "units": units, "protocol_sha256": protocol_hash, "elapsed_seconds": time.monotonic() - began,
            "training_updates": 0, "optimizer_step_called": False, "stop_test_model_evaluations": 0,
            "model_states_unchanged": all(r["model_state_sha256_before"] == r["model_state_sha256_after"] for r in reports),
            "all_parameter_closures_passed": all(r["qualification"]["parameter_closure"]["passed"] for r in reports),
            "all_output_arithmetic_passed": all(r["qualification"]["output_arithmetic"]["passed"] for r in reports),
            "candidate_admitted": False, "market_data_read": False, "Q3_read": False, "full_goal_achieved": False,
            "scientific_success_gate_defined": False}
        if args.mode == "qualify":
            result.update(costs=[{"seed": 2026, "batch_size": 128, "steps": 0,
                "elapsed_seconds": reports[0]["elapsed_seconds"]}], performance_scored=False,
                oracle_prediction_scored=False, numerical_qualification_passed=True)
        deadline(p)
        atomic_json(output / "RESULTS.json", result)
        print(json.dumps({"status": result["status"], "units": len(units), "training_updates": 0}), flush=True)
        return 0
    except BaseException as error:
        paused = isinstance(error, TimeoutError)
        atomic_json(output / ("PAUSE.json" if paused else "FAILURE.json"), {
            "status": "paused_completed_seed_progress_preserved" if paused else "failed_audit_completed_seeds_preserved",
            "protocol_sha256": protocol_hash, "units": units, "error": {"type": type(error).__name__, "message": str(error)},
            "deadline_epoch": p["deadline_epoch"], "training_updates": 0})
        return 75 if paused else 1


if __name__ == "__main__":
    sys.exit(main())
