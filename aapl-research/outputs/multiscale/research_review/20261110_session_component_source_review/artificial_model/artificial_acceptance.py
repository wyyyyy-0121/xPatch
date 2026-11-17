"""Execute the frozen artificial-only engineering gate. No market/data imports."""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import traceback

CONTRACT_SHA = "79ef8e05d70c4a7cc2b88f809b6be9cf8681d01dbd1898e6c84975d0ba654817"
AS_LIMIT = 6 * 1024**3
RSS_LIMIT = 3 * 1024**3
ADAM = {"lr": .001, "betas": [.9, .999], "eps": 1e-8,
        "weight_decay": 0., "amsgrad": False, "foreach": False, "fused": False}
RTOL, ATOL = 1e-9, 1e-11


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        stream.write(json_bytes(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)


def atomic_torch(torch, path, value):
    require(not path.exists(), "Refuse to overwrite immutable tensor artifact: " + path.name)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)
    fsync_directory(path.parent)
    return {"path": path.name, "sha256": digest(path), "bytes": path.stat().st_size}


def rss_bytes():
    current = None
    with open("/proc/self/status") as stream:
        for line in stream:
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
                break
    require(current is not None, "Cannot verify current RSS")
    return current, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def validate_protocol(path):
    require(path.is_absolute(), "--protocol must be absolute")
    path = path.resolve(strict=True)
    body = path.read_bytes()
    p = json.loads(body)
    home = Path(__file__).resolve().parent
    require(p.get("market_arrays_admitted") is False and p.get("GPU") is False
            and p.get("Q3_allowed") is False, "Artificial-only scope flags required")
    require(p.get("contract_sha256") == CONTRACT_SHA
            and digest(home / "ENGINEERING_CONTRACT.json") == CONTRACT_SHA,
            "Contract identity mismatch")
    for name in ("prototype_model.py", "artificial_acceptance.py"):
        require(p["code_hashes"][name] == digest(home / name), "Frozen source mismatch: " + name)
    start, deadline = p["started_epoch"], p["deadline_epoch"]
    require(type(start) in (int, float) and type(deadline) in (int, float)
            and math.isfinite(start) and math.isfinite(deadline) and deadline-start == 180
            and start <= time.time() < deadline, "Original absolute 180-second window invalid")
    require(type(p.get("seed")) is int and 0 <= p["seed"] < 2**31-1, "Fixed artificial seed required")
    require(p.get("adam") == ADAM, "Adam contract mismatch")
    require(sorted(os.sched_getaffinity(0)) == [24], "Main numerical CPU must be exactly 24")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "GPU must be hidden before torch import")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "1", name + " must be 1 before torch import")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    require(hard == resource.RLIM_INFINITY or hard >= AS_LIMIT, "Existing hard AS limit too small")
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
    output = home / "artificial_run"
    output.mkdir(exist_ok=True)
    allowed = {"SUPERVISOR.json", "MAIN_PROCESS.log"}
    require(all(x.name in allowed for x in output.iterdir()),
            "Prior main run artifacts exist; terminal results may not be overwritten")
    return p, hashlib.sha256(body).hexdigest(), output


def guard(deadline):
    require(time.time() < deadline, "Original absolute deadline reached")
    require(sorted(os.sched_getaffinity(0)) == [24], "CPU affinity drift")
    current, peak = rss_bytes()
    require(current <= RSS_LIMIT and peak <= RSS_LIMIT, "3 GiB RSS cap exceeded")


def allclose(torch, a, b, name):
    require(a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
            and torch.allclose(a, b, rtol=RTOL, atol=ATOL), name)


def equal_tree(torch, a, b, name="payload"):
    if isinstance(a, torch.Tensor):
        require(isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape
                and torch.equal(a, b), "Tensor replay mismatch: " + name)
    elif isinstance(a, dict):
        require(isinstance(b, dict) and set(a) == set(b), "Dictionary mismatch: " + name)
        for key in a:
            equal_tree(torch, a[key], b[key], name + "/" + str(key))
    elif isinstance(a, (list, tuple)):
        require(type(a) is type(b) and len(a) == len(b), "Sequence mismatch: " + name)
        for i, (left, right) in enumerate(zip(a, b)):
            equal_tree(torch, left, right, name + "/" + str(i))
    else:
        require(type(a) is type(b) and a == b, "Scalar replay mismatch: " + name)


def fixture(torch, gen, B, L, H, dtype, boundary=True):
    x = .2 * torch.randn((B, L, 1), generator=gen, dtype=dtype)
    anchor = .2 * torch.randn((B, 1), generator=gen, dtype=dtype)
    context = .2 * torch.randn((B, 17), generator=gen, dtype=dtype)
    mask = torch.zeros((B, L), dtype=torch.bool)
    if boundary:
        for i in range(B):
            mask[i, max(1, L // 2 + i % max(1, L // 3))] = True
    mu = torch.linspace(.01, .03, H, dtype=dtype)
    sigma = torch.linspace(.5, 1.5, H, dtype=dtype)
    zero = -mu / sigma
    y = zero + .2 * torch.randn((B, H), generator=gen, dtype=dtype)
    return {"x": x, "boundary": mask, "anchor": anchor, "context": context,
            "mu": mu, "sigma": sigma, "zero_scaled": zero, "target": y}


def call(model, f, flag):
    return model(f["x"], f["boundary"], f["anchor"], f["context"], reset_flag=flag)


def optimizer(torch, model):
    cfg = dict(ADAM); cfg["betas"] = tuple(cfg["betas"])
    return torch.optim.Adam(model.parameters(), **cfg)


def evaluation_copy(torch, model):
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.head.weight.copy_(torch.linspace(-.03, .04, m.head.weight.numel(),
                                          dtype=m.zero_scaled.dtype).reshape_as(m.head.weight))
        m.head.bias.copy_(torch.linspace(-.01, .02, m.horizon, dtype=m.zero_scaled.dtype))
    return m


def cloned_inputs(f, gradients=False):
    out = {k: v.detach().clone() for k, v in f.items()}
    if gradients:
        for name in ("x", "anchor", "context"):
            out[name].requires_grad_(True)
    return out


def detached_result(value):
    return {k: v.detach().clone() for k, v in value.items()}


def correctness(torch, module, gen, out, deadline):
    checks = []
    f = fixture(torch, gen, 3, 7, 8, torch.float64)
    pair = module.paired_models(8, f["zero_scaled"])
    for H, count in ((8, 7576), (60, 11892)):
        probe = module.TwoStateGRU(H, torch.zeros(H, dtype=torch.float64))
        require(sum(p.numel() for p in probe.parameters()) == count, "Parameter count")
    equal_tree(torch, pair["C"].state_dict(), pair["R"].state_dict(), "paired_initial_state")
    cp, rp = dict(pair["C"].named_parameters()), dict(pair["R"].named_parameters())
    require(all(cp[k].untyped_storage().data_ptr() != rp[k].untyped_storage().data_ptr() for k in cp),
            "C/R parameter storage alias")
    require(pair["C"].zero_scaled.untyped_storage().data_ptr()
            != pair["R"].zero_scaled.untyped_storage().data_ptr(), "C/R buffer storage alias")
    require(all(a.untyped_storage().data_ptr() != b.untyped_storage().data_ptr()
                for a, b in zip(pair["C"].cell1.parameters(), pair["C"].cell2.parameters())),
            "Within-arm cell storage alias")
    require(any(not torch.equal(a, b) for a, b in zip(pair["C"].cell1.parameters(), pair["C"].cell2.parameters())),
            "Within-arm independent initialization collapsed")
    require(not pair["C"].zero_scaled.requires_grad, "Baseline buffer has gradients")
    checks.append("paired_parameters_buffers_counts_independent_cells_no_alias")
    invalid = []
    def reject(name, mutate, flag=0):
        bad = cloned_inputs(f); mutate(bad)
        try:
            call(pair["C"], bad, flag)
        except (ValueError, TypeError):
            invalid.append(name)
        else:
            raise ValueError("Invalid input accepted: " + name)
    reject("x_shape", lambda z: z.update(x=z["x"].squeeze(-1)))
    reject("boundary_not_bool", lambda z: z.update(boundary=z["boundary"].double()))
    reject("boundary_shape", lambda z: z.update(boundary=z["boundary"][:, :-1]))
    reject("firstslot_true", lambda z: z["boundary"].__setitem__((0, 0), True))
    reject("anchor_shape", lambda z: z.update(anchor=z["anchor"].expand(-1, 2)))
    reject("context_shape", lambda z: z.update(context=z["context"][:, :-1]))
    reject("dtype_mismatch", lambda z: z.update(anchor=z["anchor"].float()))
    for field in ("x", "anchor", "context"):
        reject(field + "_nonfinite", lambda z, k=field: z[k].view(-1).__setitem__(0, float("nan")))
    reject("invalid_reset_flag", lambda z: None, flag=2)
    for label, zero in (("wrong_shape", torch.zeros(2, dtype=torch.float64)),
                        ("nonfinite", torch.full((8,), float("nan"), dtype=torch.float64))):
        try:
            module.TwoStateGRU(8, zero)
        except ValueError:
            invalid.append("zero_scaled_" + label)
        else:
            raise ValueError("Invalid baseline accepted")
    checks.append({"invalid_inputs_rejected": invalid})
    for arm, flag in (("C", 0), ("R", 1)):
        result = call(pair[arm], f, flag)
        allclose(torch, result["output"], f["zero_scaled"].expand_as(result["output"]), "Initial scaled zero")
        allclose(torch, result["output"] * f["sigma"] + f["mu"],
                 torch.zeros_like(result["output"]), "Initial physical zero")
        pair[arm].zero_grad(set_to_none=True)
        (result["output"] - f["target"]).square().mean().backward()
        require(all(p.grad is not None and torch.count_nonzero(p.grad) == 0
                    for p in list(pair[arm].cell1.parameters()) + list(pair[arm].cell2.parameters())),
                "First zero-head backbone gradients must be zero")
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in pair[arm].head.parameters()),
                "First head gradients invalid")
    checks.append("common_initial_physical_zero_first_backbone_gradients_zero")
    ev = {k: evaluation_copy(torch, m) for k, m in pair.items()}
    no = cloned_inputs(f); no["boundary"].zero_()
    records = []
    for name, fx in (("no_boundary", no), ("with_boundary", f)):
        outputs = {arm: detached_result(call(ev[arm], fx, flag)) for arm, flag in (("C", 0), ("R", 1))}
        records.append({"name": name, "x": fx["x"], "boundary": fx["boundary"],
                        "anchor": fx["anchor"], "context": fx["context"],
                        "zero_scaled": fx["zero_scaled"], "C": outputs["C"], "R": outputs["R"]})
    nf = {k: cloned_inputs(no, gradients=True) for k in ev}
    grads = {}; states = {}
    for arm, flag in (("C", 0), ("R", 1)):
        states[arm] = call(ev[arm], nf[arm], flag)
        variables = tuple(ev[arm].parameters()) + tuple(nf[arm][k] for k in ("x", "anchor", "context"))
        grads[arm] = torch.autograd.grad(states[arm]["output"].square().sum(), variables)
    for key in ("output", "h1", "h2"):
        allclose(torch, states["C"][key], states["R"][key], "No-boundary " + key)
    for i, (a, b) in enumerate(zip(grads["C"], grads["R"])):
        allclose(torch, a, b, "No-boundary gradient " + str(i))
    checks.append("no_boundary_states_outputs_all_parameter_and_floating_input_gradients_match")
    rf = cloned_inputs(f, gradients=True)
    rr = call(ev["R"], rf, 1); cc = call(ev["C"], f, 0)
    allclose(torch, rr["h1"], cc["h1"], "Common carry h1 mismatch")
    prefix = torch.zeros_like(f["boundary"])
    for i in range(len(f["x"])):
        last = int(torch.nonzero(f["boundary"][i], as_tuple=False)[-1, 0])
        prefix[i, :last] = True
        h = torch.zeros((1, 32), dtype=torch.float64)
        for j in range(last, f["x"].shape[1]):
            inp = torch.cat((f["x"][i:i+1, j], f["boundary"][i:i+1, j:j+1].double()), dim=1)
            h = ev["R"].cell2(inp, h)
        allclose(torch, rr["h2"][i:i+1], h, "Reset/suffix state mismatch")
    gh2 = torch.autograd.grad(rr["h2"].sum(), rf["x"], retain_graph=True)[0]
    gh1 = torch.autograd.grad(rr["h1"].sum(), rf["x"])[0]
    require(torch.count_nonzero(gh2.squeeze(-1)[prefix]) == 0, "Reset prefix gradient not zero")
    require(all(torch.count_nonzero(gh1[i, :, 0][prefix[i]]) > 0 for i in range(3)),
            "Common carry prefix not reachable in controlled fixture")
    perturbed = cloned_inputs(f); perturbed["x"][:, :, 0][prefix] += .3
    changed = call(ev["R"], perturbed, 1)
    allclose(torch, changed["h2"], rr["h2"].detach(), "Reset prefix invariance")
    require(all(not torch.allclose(changed["h1"][i], rr["h1"][i].detach(), rtol=RTOL, atol=ATOL)
                for i in range(3)), "Carry prefix perturbation had no measurable state effect")
    checks.append("per_sample_reset_suffix_prefix_invariance_gradient_cut_and_h1_reachability")
    reference = {"schema": 1, "config": {"horizon": 8, "input_size": 2, "width": 32,
                  "dtype": "float64", "tolerance": {"rtol": RTOL, "atol": ATOL}},
                 "model_state": copy.deepcopy(ev["C"].state_dict()), "fixtures": records}
    reference_receipt = atomic_torch(torch, out / "REFERENCE_CASES.pt", reference)
    guard(deadline)
    return checks, reference_receipt


def snapshot(torch, config, model, opt, gen, cursor, trace=None):
    return {"schema": 1, "config": copy.deepcopy(config), "model": copy.deepcopy(model.state_dict()),
            "adam": copy.deepcopy(opt.state_dict()), "torch_cpu_rng": torch.get_rng_state().clone(),
            "data_generator_rng": gen.get_state().clone(), "cursor": cursor,
            "trace": copy.deepcopy(trace)}


def train_step(torch, model, opt, gen, config):
    token = torch.rand((), dtype=torch.float64)
    f = fixture(torch, gen, config["batch"], config["length"], config["horizon"],
                getattr(torch, config["dtype"]))
    opt.zero_grad(set_to_none=True)
    pred = call(model, f, config["reset_flag"])["output"]
    loss = (pred - f["target"]).square().mean()
    require(torch.isfinite(loss), "Nonfinite artificial loss")
    loss.backward(); opt.step()
    return {"global_rng_token": token, "fixture": detached_result(f),
            "prediction": pred.detach().clone(), "loss": loss.detach().clone()}


def recovery(torch, module, p, identity, out):
    config = {"purpose": "artificial_checkpoint_replay_only", "protocol_sha256": identity,
              "contract_sha256": CONTRACT_SHA, "code_hashes": p["code_hashes"],
              "started_epoch": p["started_epoch"], "deadline_epoch": p["deadline_epoch"],
              "batch": 3, "length": 7, "horizon": 8, "dtype": "float64", "reset_flag": 1,
              "seed": p["seed"], "adam": ADAM, "input_size": 2, "hidden_width": 32,
              "context_width": 17, "head_input_size": 82, "dropout": 0,
              "initialization": "independent_cells_common_physical_zero_paired_base"}
    gen = torch.Generator(device="cpu").manual_seed(p["seed"] + 1)
    mu = torch.linspace(.01, .03, 8, dtype=torch.float64)
    sigma = torch.linspace(.5, 1.5, 8, dtype=torch.float64)
    config["artificial_scaler"] = {"mu": mu.tolist(), "sigma": sigma.tolist()}
    model = module.TwoStateGRU(8, -mu / sigma); opt = optimizer(torch, model)
    first = train_step(torch, model, opt, gen, config)
    checkpoint = snapshot(torch, config, model, opt, gen, 1, first)
    receipt = atomic_torch(torch, out / "CHECKPOINT_000001.pt", checkpoint)
    atomic_json(out / "CHECKPOINT.json", {"schema": 1, **receipt, "cursor": 1,
                "protocol_sha256": identity, "config_sha256": hashlib.sha256(json_bytes(config)).hexdigest()})
    uninterrupted_trace = train_step(torch, model, opt, gen, config)
    uninterrupted = snapshot(torch, config, model, opt, gen, 2, uninterrupted_trace)
    saved = torch.load(out / receipt["path"], map_location="cpu", weights_only=True)
    require(digest(out / receipt["path"]) == receipt["sha256"], "Checkpoint digest mismatch")
    equal_tree(torch, checkpoint, saved, "checkpoint_roundtrip")
    require(saved["config"] == config and saved["cursor"] == 1, "Checkpoint config/cursor mismatch")
    restored = module.TwoStateGRU(8, torch.zeros(8, dtype=torch.float64))
    restored.load_state_dict(saved["model"], strict=True)
    restored_opt = optimizer(torch, restored); restored_opt.load_state_dict(saved["adam"])
    restored_gen = torch.Generator(device="cpu"); restored_gen.set_state(saved["data_generator_rng"])
    torch.set_rng_state(saved["torch_cpu_rng"])
    reloaded_trace = train_step(torch, restored, restored_opt, restored_gen, config)
    reloaded = snapshot(torch, config, restored, restored_opt, restored_gen, 2, reloaded_trace)
    equal_tree(torch, uninterrupted, reloaded, "next_step_replay")
    next_receipt = atomic_torch(torch, out / "CHECKPOINT_000002_RELOADED.pt", reloaded)
    guard(p["deadline_epoch"])
    return {"checkpoint": receipt, "reloaded_next_step": next_receipt,
            "complete_payload_bitwise_equal": True, "cursor_before": 1, "cursor_after": 2,
            "external_signal_pause_resume_tested": False}


def maximum_cost(torch, module, p, identity, out):
    # End-to-end begins before model/Adam and synthetic batch construction.
    begin = time.perf_counter(); phases = {}
    gen = torch.Generator(device="cpu").manual_seed(p["seed"] + 2)
    f = fixture(torch, gen, 128, 480, 60, torch.float32)
    model = module.TwoStateGRU(60, f["zero_scaled"]); opt = optimizer(torch, model)
    phases["model_optimizer_and_artificial_batch_seconds"] = time.perf_counter() - begin
    def pre_hook(_module, _args):
        guard(p["deadline_epoch"])
    hooks = [model.cell1.register_forward_pre_hook(pre_hook), model.cell2.register_forward_pre_hook(pre_hook)]
    forward_start = time.perf_counter()
    pred = call(model, f, 1)["output"]
    loss = (pred - f["target"]).square().mean()
    loss.backward(); opt.step()
    phases["forward_backward_adam_seconds"] = time.perf_counter() - forward_start
    for handle in hooks:
        handle.remove()
    guard(p["deadline_epoch"])
    checkpoint_start = time.perf_counter()
    config = {"purpose": "artificial_maximum_shape_cost_only", "protocol_sha256": identity,
              "contract_sha256": CONTRACT_SHA, "code_hashes": p["code_hashes"],
              "started_epoch": p["started_epoch"], "deadline_epoch": p["deadline_epoch"],
              "batch": 128, "length": 480, "horizon": 60, "dtype": "float32", "reset_flag": 1,
              "input_size": 2, "hidden_width": 32, "context_width": 17, "head_input_size": 82,
              "dropout": 0, "seed": p["seed"] + 2, "adam": ADAM,
              "artificial_scaler": {"mu": f["mu"].tolist(), "sigma": f["sigma"].tolist()}}
    checkpoint = snapshot(torch, config, model, opt, gen, 1)
    receipt = atomic_torch(torch, out / "MAXIMUM_CHECKPOINT.pt", checkpoint)
    phases["complete_checkpoint_snapshot_write_fsync_and_digest_seconds"] = time.perf_counter() - checkpoint_start
    phases["end_to_end_seconds"] = time.perf_counter() - begin
    current, peak = rss_bytes(); guard(p["deadline_epoch"])
    return {"shape": [128, 480, 2, 32, 60], "dtype": "float32", "reset_flag": 1,
            "timings": phases, "includes_artificial_batch_model_Adam_forward_backward_step_checkpoint_fsync": True,
            "per_cell_resource_guard_in_timing": True, "rss_current_bytes": current,
            "rss_peak_bytes": peak, "checkpoint": receipt, "loss_finite": bool(torch.isfinite(loss)),
            "predictive_metric": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    output = None; identity = None
    try:
        p, identity, output = validate_protocol(args.protocol)
        atomic_json(output / "MAIN_STARTED.json", {"protocol_sha256": identity,
                    "started_epoch": time.time(), "deadline_epoch": p["deadline_epoch"], "status": "running"})
        def stop(signum, _frame):
            raise RuntimeError("Terminal interruption/deadline signal " + str(signum))
        for sig in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, p["deadline_epoch"] - time.time()))
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        require(torch.get_num_threads() == 1 and torch.get_num_interop_threads() == 1, "Torch thread drift")
        guard(p["deadline_epoch"])
        spec = importlib.util.spec_from_file_location("frozen_artificial_two_state", Path(__file__).with_name("prototype_model.py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        torch.manual_seed(p["seed"])
        gen = torch.Generator(device="cpu").manual_seed(p["seed"] + 3)
        checks, reference = correctness(torch, module, gen, output, p["deadline_epoch"])
        replay = recovery(torch, module, p, identity, output)
        maximum = maximum_cost(torch, module, p, identity, output)
        guard(p["deadline_epoch"])
        value = {"schema": 1, "status": "artificial_correctness_and_cost_complete",
                 "protocol_sha256": identity, "contract_sha256": CONTRACT_SHA,
                 "code_hashes": p["code_hashes"], "ended_epoch": time.time(),
                 "deadline_epoch": p["deadline_epoch"], "deadline_reset": False,
                 "CPU_affinity": sorted(os.sched_getaffinity(0)), "torch_threads": 1,
                 "torch_interop_threads": 1, "GPU": False, "RLIMIT_AS_bytes": AS_LIMIT,
                 "RSS_cap_bytes": RSS_LIMIT, "RSS_enforcement": "sampled_before_each_maxshape_cell_and_stage_plus_peak_checks",
                 "checks": checks, "reference_cases": reference, "checkpoint_replay": replay,
                 "maximum_shape": maximum, "numeric_tolerance": {"rtol": RTOL, "atol": ATOL},
                 "external_signal_pause_resume_tested": False, "market_arrays_read": False,
                 "market_candidate_admitted": False, "predictive_success": False, "research_goal_achieved": False}
        atomic_json(output / "ACCEPTANCE.json", value)
        signal.setitimer(signal.ITIMER_REAL, 0)
        print(json.dumps(value, allow_nan=False))
    except BaseException as error:
        if output is not None:
            failure = {"status": "terminal_failure_preserve_artifacts", "protocol_sha256": identity,
                       "at_epoch": time.time(), "error_type": type(error).__name__, "message": str(error),
                       "traceback": traceback.format_exc(), "automatic_retry": False,
                       "market_arrays_read": False, "external_signal_pause_resume_tested": False}
            try:
                atomic_json(output / "FAILURE.json", failure)
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    main()
