"""Frozen, synthetic-only LSTM centering checks; run behind the stage guard.

Importing this module imports only the standard library.  Torch is imported by
main only after the protocol, own source, rules and original class are checked.
No market input, archived checkpoint, NumPy, or network client is used.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import struct
import sys
import tempfile
import time


STOP_SIGNAL = None


class QualificationFailure(RuntimeError):
    pass


class DeadlineReached(RuntimeError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".json-", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_checkpoint(torch, path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                     prefix=".synthetic-", delete=False) as stream:
        temporary = Path(stream.name)
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def signal_handler(number, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = number


def check_time(deadline):
    if time.time() >= deadline:
        raise DeadlineReached("The original global stage deadline has expired")


def require(value, message):
    if not value:
        raise QualificationFailure(message)


def snapshot(module):
    return {name: value.detach().clone() for name, value in module.state_dict().items()}


def state_equal(torch, left, right):
    if isinstance(left, torch.Tensor):
        return (isinstance(right, torch.Tensor) and left.dtype == right.dtype
                and left.shape == right.shape and torch.equal(left, right))
    if isinstance(left, dict):
        return (isinstance(right, dict) and left.keys() == right.keys()
                and all(state_equal(torch, left[k], right[k]) for k in left))
    if isinstance(left, (list, tuple)):
        return (type(left) is type(right) and len(left) == len(right)
                and all(state_equal(torch, a, b) for a, b in zip(left, right)))
    return type(left) is type(right) and left == right


def canonical_payload_digest(torch, value):
    """Hash typed recursive values and raw tensor bytes, not torch.save ZIP bytes."""
    hasher = hashlib.sha256()

    def put(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            raw = bytes(tensor.reshape(-1).view(torch.uint8).tolist())
            hasher.update(b"tensor\0" + str(tensor.dtype).encode() + b"\0")
            put(tuple(tensor.shape))
            hasher.update(struct.pack("!Q", len(raw)) + raw)
        elif isinstance(item, dict):
            hasher.update(b"dict\0" + struct.pack("!Q", len(item)))
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                put(key)
                put(item[key])
        elif isinstance(item, (list, tuple)):
            hasher.update(type(item).__name__.encode() + b"\0" + struct.pack("!Q", len(item)))
            for member in item:
                put(member)
        elif isinstance(item, float):
            hasher.update(b"float\0" + struct.pack("!d", item))
        elif isinstance(item, (str, int, bool)) or item is None:
            raw = repr(item).encode("utf-8")
            hasher.update(type(item).__name__.encode() + b"\0" + struct.pack("!Q", len(raw)) + raw)
        else:
            raise QualificationFailure("Unsupported canonical payload member:" + type(item).__name__)

    put(value)
    return hasher.hexdigest()


def extract_original_class(source, rules, torch):
    require(digest(source) == rules["original_model_source_sha256"], "Original source SHA mismatch")
    tree = ast.parse(Path(source).read_text(encoding="utf-8"), filename=str(source))
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name == rules["original_class"]]
    require(len(nodes) == 1, "Expected precisely one original Recurrent class")
    isolated = ast.Module(body=[nodes[0]], type_ignores=[])
    namespace = {"torch": torch, "nn": torch.nn}
    exec(compile(isolated, str(source) + ":AST-only-Recurrent", "exec"), namespace)
    return namespace[rules["original_class"]]


def validate_protocol(args):
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    rules_path = Path(protocol["rules_path"]).resolve()
    require(digest(rules_path) == protocol["rules_sha256"], "Rules SHA mismatch")
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    require(digest(__file__) == protocol["worker_sha256"], "Worker SHA mismatch")
    require(protocol["deadline_epoch"] == rules["deadline_epoch"], "Deadline may not reset")
    require(rules["deadline_may_reset"] is False, "Deadline reset not admitted")
    require(protocol["mode"] in (args.mode, "all"), "Protocol mode mismatch")
    output = Path(args.output).resolve()
    output.relative_to(Path(protocol["output_directory"]).resolve())
    require(sorted(os.sched_getaffinity(0)) == rules["resources"]["numeric_CPU"],
            "Numerical CPU affinity mismatch")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1", "GPU must be hidden")
    require(not rules["market_arrays_or_weights_allowed"] and not rules["Q3_allowed"],
            "Synthetic-only scope mismatch")
    source = Path(rules["original_model_source"])
    require(digest(source) == rules["original_model_source_sha256"], "Original source SHA mismatch")
    require(digest(rules["source_review_decision"]) == rules["source_review_decision_sha256"],
            "Source-review decision SHA mismatch")
    check_time(rules["deadline_epoch"])
    return protocol_path, protocol, rules_path, rules, output


def make_fixture(torch, case, rules):
    dtype = getattr(torch, case["dtype"])
    batch, length, horizon = case["batch"], case["lookback"], case["horizon"]
    torch.manual_seed(rules["seed"])
    random.seed(rules["seed"])
    x = torch.randn(batch, length, 1, dtype=dtype)
    anchor = torch.randn(batch, 1, dtype=dtype)
    mu = torch.linspace(.01, .03, horizon, dtype=dtype)
    sigma = torch.linspace(.5, 1.5, horizon, dtype=dtype)
    baseline = -mu / sigma
    index = torch.arange(1, batch * horizon + 1, dtype=dtype).reshape(batch, horizon)
    target = baseline + .2 * torch.sin(index)
    return {"x": x, "anchor": anchor, "mu": mu, "sigma": sigma,
            "baseline": baseline, "target": target}


def make_arms(torch, Recurrent, case, rules, fixture):
    dtype = getattr(torch, case["dtype"])
    default = Recurrent("lstm", case["horizon"], width=rules["width"]).to(dtype=dtype)
    zero = copy.deepcopy(default)
    with torch.no_grad():
        zero.head.weight.zero_()
        zero.head.bias.copy_(fixture["baseline"])

    class Centered(torch.nn.Module):
        def __init__(self, model, baseline):
            super().__init__()
            self.learner = copy.deepcopy(model)
            self.reference = copy.deepcopy(model)
            self.reference.requires_grad_(False)
            # The scaler is checkpointed separately; it is not a trainable input.
            self.baseline_scaled = baseline.detach().clone()

        def forward(self, x, anchor):
            learner_output = self.learner(x, anchor)
            with torch.no_grad():
                reference_output = self.reference(x, anchor)
            return (learner_output - reference_output) + self.baseline_scaled

    centered = Centered(default, fixture["baseline"])
    return {"D": default, "Z": zero, "C": centered}


def trainable(module):
    return [parameter for parameter in module.parameters() if parameter.requires_grad]


def canonical_parameters(module, arm):
    model = module.learner if arm == "C" else module
    return dict(model.named_parameters())


def new_optimizer(torch, module, rules):
    return torch.optim.Adam(trainable(module), lr=rules["optimizer"]["learning_rate"],
                            weight_decay=rules["optimizer"]["weight_decay"])


def parameters_report(arms):
    return {arm: {"trainable": sum(p.numel() for p in trainable(model)),
                  "reference": (sum(p.numel() for p in model.reference.parameters())
                                if arm == "C" else 0),
                  "total_stored": sum(p.numel() for p in model.parameters())}
            for arm, model in arms.items()}


def verify_original_modules(torch, arms):
    for arm, model in arms.items():
        for name, module in model.named_modules():
            require(not isinstance(module, torch.nn.modules.batchnorm._BatchNorm),
                    arm + ": unexpected BatchNorm " + name)
            require(not isinstance(module, torch.nn.modules.dropout._DropoutNd),
                    arm + ": unexpected dropout " + name)
            if isinstance(module, torch.nn.LSTM):
                require(module.num_layers == 1 and module.dropout == 0,
                        "Original LSTM must remain single-layer dropout0")
        require(not list(model.named_buffers()), arm + ": unexpected registered buffer")


def reference_check(torch, centered, frozen, optimizer=None):
    require(state_equal(torch, centered.reference.state_dict(), frozen), "Reference state drift")
    require(all(not p.requires_grad and p.grad is None for p in centered.reference.parameters()),
            "Reference gradient or requires_grad changed")
    learner_storage = {p.untyped_storage().data_ptr() for p in centered.learner.parameters()}
    reference_storage = {p.untyped_storage().data_ptr() for p in centered.reference.parameters()}
    require(not learner_storage.intersection(reference_storage), "Reference shares writable storage")
    if optimizer is not None:
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        require(optimized == {id(p) for p in centered.learner.parameters()},
                "Optimizer must contain precisely the learner parameters")
        require(not any(id(p) in optimized for p in centered.reference.parameters()),
                "Reference entered Adam")


def compare(torch, actual, expected, tolerance, label, records):
    require(actual.shape == expected.shape, label + ": shape mismatch")
    require(bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all()),
            label + ": nonfinite value")
    error = (actual - expected).abs()
    allowance = tolerance["atol"] + tolerance["rtol"] * torch.maximum(actual.abs(), expected.abs())
    passed = bool((error <= allowance).all())
    records.append({"check": label, "passed": passed,
                    "max_absolute_difference": float(error.max()),
                    "max_normalized_difference": float((error / allowance).max())})
    require(passed, label + ": frozen tolerance failed")


def forward_rng_check(torch, model, x, anchor):
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    value = model(x, anchor)
    require(torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng,
            "Deterministic original LSTM forward consumed RNG")
    return value


def update(torch, model, fixture, optimizer):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    permutation = torch.randperm(len(fixture["x"]))
    prediction = forward_rng_check(torch, model, fixture["x"][permutation],
                                   fixture["anchor"][permutation])
    loss = (prediction - fixture["target"][permutation]).square().mean()
    require(bool(torch.isfinite(loss)), "Nonfinite synthetic loss")
    loss.backward()
    require(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in trainable(model)),
            "Missing/nonfinite trainable synthetic gradient")
    optimizer.step()
    return float(loss.detach()), permutation


def identities(torch, Recurrent, rules):
    results = []
    for case in rules["synthetic_cases"]:
        check_time(rules["deadline_epoch"])
        fixture = make_fixture(torch, case, rules)
        arms = make_arms(torch, Recurrent, case, rules, fixture)
        verify_original_modules(torch, arms)
        frozen = snapshot(arms["C"].reference)
        require(state_equal(torch, arms["D"].state_dict(), arms["C"].learner.state_dict()),
                "D/C initial trainable state differs")
        require(state_equal(torch, arms["D"].rnn.state_dict(), arms["Z"].rnn.state_dict()),
                "D/Z initial recurrent state differs")
        reference_check(torch, arms["C"], frozen)
        tolerance = rules["numeric_tolerances"][case["dtype"]]
        records = []
        baseline = fixture["baseline"].expand(case["batch"], -1)
        mode_predictions = {}
        for mode in (True, False):
            for arm, model in arms.items():
                model.train(mode)
                value = forward_rng_check(torch, model, fixture["x"], fixture["anchor"])
                mode_predictions[(arm, mode)] = value.detach().clone()
                reverse = torch.arange(case["batch"] - 1, -1, -1)
                reordered = model(fixture["x"][reverse], fixture["anchor"][reverse])[reverse]
                compare(torch, reordered, value, tolerance, arm + ":reorder:" + str(mode), records)
                halves = [model(fixture["x"][i:i+1], fixture["anchor"][i:i+1])
                          for i in range(case["batch"])]
                compare(torch, torch.cat(halves), value, tolerance,
                        arm + ":split_batch:" + str(mode), records)
                if arm in ("Z", "C"):
                    compare(torch, value, baseline, tolerance, arm + ":baseline:" + str(mode), records)
                    physical = fixture["mu"] + fixture["sigma"] * value
                    bound = 32 * torch.finfo(value.dtype).eps * max(
                        1., float(fixture["mu"].abs().max()), float(fixture["sigma"].abs().max()))
                    require(float(physical.abs().max()) <= bound, "Physical zero exceeds rounding bound")
                    records.append({"check": arm + ":physical_zero:" + str(mode), "passed": True,
                                    "maximum_absolute_value": float(physical.abs().max()),
                                    "frozen_rounding_bound": bound})
            reference_check(torch, arms["C"], frozen)
        for arm in arms:
            compare(torch, mode_predictions[(arm, True)], mode_predictions[(arm, False)],
                    tolerance, arm + ":train_eval", records)
        # A deliberately omitted baseline gives mu, not physical zero.
        omitted = fixture["mu"].expand_as(baseline)
        require(float(omitted.abs().min()) > 0, "Nonzero-mean counterexample lost")
        for cotangent_name in rules["fixed_vjp_cotangents"]:
            check_time(rules["deadline_epoch"])
            index = torch.arange(1, baseline.numel()+1, dtype=baseline.dtype).reshape_as(baseline)
            cotangent = (torch.ones_like(baseline) if cotangent_name == "ones" else
                         torch.sin(index) if cotangent_name == "sin(index+1)" else torch.cos(index))
            gradients = {}
            for arm in ("D", "C"):
                model = arms[arm]
                model.train()
                value = model(fixture["x"], fixture["anchor"])
                named = canonical_parameters(model, arm)
                gradients[arm] = dict(zip(named, torch.autograd.grad(
                    value, tuple(named.values()), grad_outputs=cotangent)))
            for name in gradients["D"]:
                compare(torch, gradients["D"][name], gradients["C"][name], tolerance,
                        "VJP:" + cotangent_name + ":" + name, records)
            reference_check(torch, arms["C"], frozen)
        initial_loss_gradients = {}
        initial_losses = {}
        for arm in ("Z", "C"):
            model = arms[arm]
            model.train()
            model.zero_grad(set_to_none=True)
            value = model(fixture["x"], fixture["anchor"])
            loss = (value - fixture["target"]).square().mean()
            loss.backward()
            initial_losses[arm] = loss.detach()
            initial_loss_gradients[arm] = {n: p.grad.detach().clone()
                                           for n, p in canonical_parameters(model, arm).items()}
        compare(torch, initial_losses["C"], initial_losses["Z"], tolerance,
                "C_Z_initial_loss", records)
        for name in ("head.weight", "head.bias"):
            compare(torch, initial_loss_gradients["C"][name], initial_loss_gradients["Z"][name],
                    tolerance, "C_Z_initial_head_gradient:" + name, records)
        zero_bone = [v for n, v in initial_loss_gradients["Z"].items() if n.startswith("rnn.")]
        centered_bone = [v for n, v in initial_loss_gradients["C"].items() if n.startswith("rnn.")]
        require(all(bool(torch.isfinite(v).all()) and int(torch.count_nonzero(v)) == 0
                    for v in zero_bone), "Z initial backbone gradient is not exactly zero")
        norm = max(float(v.abs().max()) for v in centered_bone)
        require(math.isfinite(norm) and norm > 10 * (tolerance["atol"] + tolerance["rtol"] * norm),
                "Fixed C backbone gradient witness is not numerically discriminating")
        reference_check(torch, arms["C"], frozen)
        histories = {}
        common_update_rng = torch.get_rng_state().clone()
        for arm, model in arms.items():
            torch.set_rng_state(common_update_rng)
            optimizer = new_optimizer(torch, model, rules)
            histories[arm] = []
            for _ in range(rules["short_update_count"]):
                check_time(rules["deadline_epoch"])
                loss, _ = update(torch, model, fixture, optimizer)
                histories[arm].append(loss)
                if arm == "C":
                    reference_check(torch, model, frozen, optimizer)
            if arm == "C":
                # In-memory state restore is separate from the actual process recovery test.
                payload = {"model": copy.deepcopy(model.state_dict()),
                           "optimizer": copy.deepcopy(optimizer.state_dict())}
                model.load_state_dict(copy.deepcopy(payload["model"]))
                optimizer.load_state_dict(copy.deepcopy(payload["optimizer"]))
                require(state_equal(torch, payload["model"], model.state_dict())
                        and state_equal(torch, payload["optimizer"], optimizer.state_dict()),
                        "Synthetic in-memory state roundtrip differs")
                reference_check(torch, model, frozen, optimizer)
        arms["C"].eval()
        centered_after = arms["C"](fixture["x"], fixture["anchor"]).detach()
        require(not torch.equal(centered_after, baseline), "C never departed from baseline after8updates")
        results.append({"case": case, "checks": records, "C_first_backbone_gradient_inf": norm,
                        "Z_first_backbone_exact_zero": True,
                        "reference_immutable_no_alias_no_grad_no_Adam": True,
                        "short_updates_per_arm": rules["short_update_count"],
                        "short_update_losses": histories, "parameters": parameters_report(arms),
                        "omitting_baseline_physical_output_is_mu": True,
                        "omitted_baseline_min_abs_physical_value": float(omitted.abs().min()),
                        "C_departed_from_baseline": True})
    return {"status": "passed_synthetic_identities", "cases": results,
            "actual_disk_and_process_recovery_claimed": False}


def cost(torch, Recurrent, rules):
    case = rules["maximum_shape_cost_case"]
    fixture = make_fixture(torch, case, rules)
    arms = make_arms(torch, Recurrent, case, rules, fixture)
    verify_original_modules(torch, arms)
    frozen = snapshot(arms["C"].reference)
    rows = []
    for arm, model in arms.items():
        optimizer = new_optimizer(torch, model, rules)
        elapsed = []
        for _ in range(case["trials_per_arm"]):
            check_time(rules["deadline_epoch"])
            start = time.perf_counter()
            loss, _ = update(torch, model, fixture, optimizer)
            seconds = time.perf_counter() - start
            elapsed.append(seconds)
            require(seconds <= 30, "Maximum shape step exceeded30seconds:" + arm)
            if arm == "C":
                reference_check(torch, model, frozen, optimizer)
        rows.append({"arm": arm, "trainstep_seconds": elapsed,
                     "maximum_trainstep_seconds": max(elapsed), "last_loss": loss})
    return {"status": "passed_measured_maximum_shape_cost", "case": case, "arms": rows,
            "parameters": parameters_report(arms), "C_reference_extra_forward_count_per_step": 1,
            "qualification_is_forecasting_evidence": False}


def trajectory(torch, Recurrent, rules, args, output, identities_record):
    case = rules["recovery_trajectory_case"]
    fixture = make_fixture(torch, case, rules)
    arms = make_arms(torch, Recurrent, case, rules, fixture)
    model = arms["C"]
    optimizer = new_optimizer(torch, model, rules)
    frozen = snapshot(model.reference)
    index = 0
    history = []
    last_permutation = None
    checkpoint = output / "checkpoint.pt"
    if args.resume:
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for key, value in identities_record.items():
            require(saved["identity"][key] == value, "Recovery identity mismatch:" + key)
        require(saved["case"] == case, "Recovery case differs")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        require(state_equal(torch, saved["fixture"], fixture), "Recovery fixture differs")
        require(state_equal(torch, saved["reference_initial"], frozen), "Recovery reference initial differs")
        require(state_equal(torch, saved["scaler"], {k: fixture[k] for k in ("mu", "sigma", "baseline")}),
                "Recovery scaler differs")
        index, history = saved["index"], saved["history"]
        last_permutation = saved["last_permutation"]
        model.train(saved["training_mode"])
        # Restore after model/fixture/optimizer construction and loading.
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        reference_check(torch, model, frozen, optimizer)

    def save():
        payload = {"identity": identities_record, "case": case,
                   "model": model.state_dict(), "reference_initial": frozen,
                   "scaler": {k: fixture[k] for k in ("mu", "sigma", "baseline")},
                   "fixture": fixture, "optimizer": optimizer.state_dict(),
                   "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                   "index": index, "history": history, "last_permutation": last_permutation,
                   "cursor": case["batch"] if index else 0,
                   "training_mode": model.training}
        atomic_checkpoint(torch, checkpoint, payload)
        return payload

    local_deadline = rules["deadline_epoch"]
    save()
    while index < rules["short_update_count"]:
        if STOP_SIGNAL is not None or time.time() >= min(local_deadline, rules["deadline_epoch"]):
            save()
            reason = ("signal_" + str(STOP_SIGNAL) if STOP_SIGNAL is not None else
                      "local_deadline_expired" if local_deadline < rules["deadline_epoch"] else
                      "global_deadline_expired")
            atomic_json(output / "PAUSED.json", {"status": "checkpointed", "reason": reason,
                                                "index": index, "exit_code": 75})
            return None, 75
        loss, last_permutation = update(torch, model, fixture, optimizer)
        index += 1
        history.append(loss)
        reference_check(torch, model, frozen, optimizer)
        save()
        if index == 4 and not args.resume:
            if args.pause_kind == "deadline":
                local_deadline = time.time() - 1
            elif args.pause_kind == "signal":
                print("READY_FOR_SIGNAL", flush=True)
                while STOP_SIGNAL is None and time.time() < rules["deadline_epoch"]:
                    time.sleep(.02)
    final_payload = save()
    # Read the actual saved generation and check all recursive payload members.
    reloaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(canonical_payload_digest(torch, final_payload) == canonical_payload_digest(torch, reloaded),
            "Actual final checkpoint payload roundtrip differs")
    summary = {"status": "completed_synthetic8update_trajectory", "updates": index,
            "losses": history, "reference_immutable": True,
            "parameters": parameters_report(arms),
            "canonical_payload_digest": canonical_payload_digest(torch, reloaded),
            "canonical_payload_top_level_keys": sorted(reloaded),
            "canonical_payload_includes": "all recursive keys and typed scalar/tensor raw bytes, including identity, learner/reference model, reference_initial, scaler, fixture, Adam, Torch/Python RNG, index/history/permutation/cursor/mode",
            "excluded_nonpayload_fields": ["pid", "walltime", "output_path", "local_deadline", "pause_kind"],
            "actual_final_checkpoint_reloaded_and_digest_matched": True}
    atomic_json(output / "SUMMARY.json", summary)
    return summary, 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--mode", choices=("identities", "cost", "trajectory"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-kind", choices=("signal", "deadline"))
    args = parser.parse_args()
    require(args.mode == "trajectory" or not (args.resume or args.pause_kind),
            "Resume/pause options are trajectory-only")
    protocol_path, protocol, rules_path, rules, output = validate_protocol(args)
    output.mkdir(parents=True, exist_ok=True)
    for number in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(number, signal_handler)
    import torch
    torch.set_num_threads(rules["backend"]["torch_num_threads"])
    torch.set_num_interop_threads(rules["backend"]["torch_num_interop_threads"])
    torch.use_deterministic_algorithms(rules["backend"]["deterministic_algorithms"])
    runtime = {"torch": str(torch.__version__), "mkldnn_enabled": torch.backends.mkldnn.enabled,
               "mkldnn_available": torch.backends.mkldnn.is_available(),
               "threads": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
               "CPU": sorted(os.sched_getaffinity(0)), "GPU_hidden": True}
    Recurrent = extract_original_class(Path(rules["original_model_source"]), rules, torch)
    identity = {"protocol_sha256": digest(protocol_path), "rules_sha256": digest(rules_path),
                "worker_sha256": digest(__file__),
                "original_source_sha256": rules["original_model_source_sha256"],
                "global_deadline_epoch": rules["deadline_epoch"], "runtime": runtime}
    try:
        check_time(rules["deadline_epoch"])
        if args.mode == "identities":
            result, code = identities(torch, Recurrent, rules), 0
        elif args.mode == "cost":
            result, code = cost(torch, Recurrent, rules), 0
        else:
            result, code = trajectory(torch, Recurrent, rules, args, output, identity)
        if result is not None:
            result.update({"identity": identity, "runtime": runtime,
                           "market_arrays_or_weights_read": False, "Q3_read": False,
                           "new_market_candidate_admitted": False, "full_goal_achieved": False})
            atomic_json(output / "RESULTS.json", result)
            print(json.dumps({"status": result["status"], "mode": args.mode}, allow_nan=False), flush=True)
        return code
    except Exception as error:
        atomic_json(output / "FAILURE.json", {"status": "failed_no_recipe_rescue",
                    "error_type": type(error).__name__, "error": str(error), "identity": identity,
                    "mode": args.mode, "market_arrays_or_weights_read": False, "Q3_read": False})
        raise


if __name__ == "__main__":
    sys.exit(main())
