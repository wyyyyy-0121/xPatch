"""Synthetic-only structural and actual-process recovery acceptance."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import types

import numpy as np
import torch

# A CLI invocation executes worker as __main__; use that same module, avoiding
# duplicate STOP_REQUESTED globals during interruption tests.
if hasattr(sys.modules.get("__main__"), "train_fixed"):
    w = sys.modules["__main__"]
else:
    import worker as w


def exact(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(exact(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(exact(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


class SyntheticProvider:
    """Future partitions are poison sentinels; mean/train must not touch them."""
    def __init__(self, cfg, dtype=torch.float32):
        self.cfg = cfg
        self.length = 257  # 128 + 129 exercises the single-remainder rule.
        h, lookback = cfg["horizon"], cfg["lookback"]
        generator = torch.Generator().manual_seed(314159)
        self.x = torch.randn((self.length, lookback, 1), generator=generator, dtype=dtype) * .3
        self.anchor = torch.randn((self.length, 1), generator=generator, dtype=dtype) * .2
        self.y = torch.randn((self.length, h), generator=generator, dtype=dtype) * .1
        self.scaler = {"x_mean": 0., "x_std": 1., "anchor_mean": 100., "anchor_std": 8.,
                       "y_mean": [90. + j for j in range(h)] if cfg["target"] == "future_close" else [.01 * (j + 1) for j in range(h)],
                       "y_std": [7. + j for j in range(h)] if cfg["target"] == "future_close" else [.3 + .1 * j for j in range(h)]}
        self.input_calls = []
        self.label_calls = []
        self.identity = {"synthetic": True, "seed": 314159, "rows": self.length}

    def inputs(self, part, positions):
        if part != "fit":
            raise AssertionError("Poison future input sentinel was consumed")
        self.input_calls.append((part, len(positions)))
        return self.x[positions], self.anchor[positions]

    def labels(self, part, positions):
        if part != "fit":
            raise AssertionError("Poison future label sentinel was consumed")
        self.label_calls.append((part, len(positions)))
        return self.y[positions]


def fixed_cfg(horizon=1, target="future_log_return", arm="B"):
    return next(copy.deepcopy(j) for j in w.job_matrix()
                if j["horizon"] == horizon and j["target"] == target and j["fold"] == 1 and j["arm"] == arm)


def trend_mean(provider):
    w.original_modules()
    from aapl.models import CausalEMA
    ema = CausalEMA(.1)
    total = torch.zeros((provider.cfg["lookback"], 1), dtype=torch.float64)
    for start in range(0, provider.length, 128):
        x, _ = provider.inputs("fit", np.arange(start, min(start + 128, provider.length)))
        total += ema(x).to(torch.float64).sum(0)
    assert not provider.label_calls
    return (total / provider.length).to(provider.x.dtype)


def structural_case(horizon, target, dtype):
    cfg = fixed_cfg(horizon, target)
    provider = SyntheticProvider(cfg, dtype)
    mean = trend_mean(provider)
    models = {arm: w.build_model(dict(cfg, arm=arm), provider.scaler, mean, dtype) for arm in ("A", "B", "C")}
    assert exact(models["A"].state_dict(), models["B"].state_dict())
    assert exact(models["A"].state_dict(), models["C"].state_dict())
    assert sum(p.numel() for p in models["A"].parameters()) == sum(p.numel() for p in models["B"].parameters())
    x, anchor = provider.x[:32], provider.anchor[:32]
    baseline_outputs = []
    for model in models.values():
        model.eval()
        p = model(x, anchor)
        ymean = torch.tensor(provider.scaler["y_mean"], dtype=dtype)
        ystd = torch.tensor(provider.scaler["y_std"], dtype=dtype)
        physical = p * ystd + ymean
        baseline = (anchor * provider.scaler["anchor_std"] + provider.scaler["anchor_mean"]).expand_as(p) if target == "future_close" else torch.zeros_like(p)
        assert torch.allclose(physical, baseline, rtol=1e-6, atol=1e-6)
        baseline_outputs.append(p.detach())
    assert all(torch.equal(baseline_outputs[0], value) for value in baseline_outputs[1:])

    # Original LayerNorm actual/fixed input, full 12 Adam updates. Do not repair
    # the original route, alter tolerances, or relax bitwise checks on failure.
    actual = w.build_model(dict(cfg, arm="A"), provider.scaler, mean, dtype)
    fixed = w.build_model(dict(cfg, arm="A"), provider.scaler, mean, dtype)
    fixed._route_arm = "C"
    _, networks, _ = w.original_modules()
    torch.manual_seed(cfg["seed"])
    native = networks.create(dict(cfg, arm="A"), provider.scaler).to(dtype=dtype)
    comparison_models = (actual, fixed, native)
    optimizers = [torch.optim.Adam(m.parameters(), lr=.001) for m in comparison_models]
    for step in range(12):
        positions = np.arange(0, 128) if step % 2 == 0 else np.arange(128, 257)
        for model, optimizer in zip(comparison_models, optimizers):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(provider.x[positions], provider.anchor[positions]) - provider.y[positions]).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
        assert exact({n: p.grad for n, p in actual.named_parameters()}, {n: p.grad for n, p in fixed.named_parameters()})
        assert exact({n: p.grad for n, p in actual.named_parameters()}, {n: p.grad for n, p in native.named_parameters()})
        for optimizer in optimizers:
            optimizer.step()
        assert exact(actual.state_dict(), fixed.state_dict())
        assert exact(actual.state_dict(), native.state_dict())
        assert exact(optimizers[0].state_dict(), optimizers[1].state_dict())
        assert exact(optimizers[0].state_dict(), optimizers[2].state_dict())
        actual.eval()
        fixed.eval()
        native.eval()
        assert torch.equal(actual(x, anchor), fixed(x, anchor))
        assert torch.equal(actual(x, anchor), native(x, anchor))

    # The trained B branch must recover trend dependence after the initially
    # zero output head has received matched Adam updates. This uses fit-only
    # synthetic data and a fixed 12-update schedule, not outcome-based search.
    repaired = [w.build_model(dict(cfg, arm=arm), provider.scaler, mean, dtype) for arm in ("B", "C")]
    repaired_optimizers = [torch.optim.Adam(m.parameters(), lr=.001) for m in repaired]
    for step in range(12):
        positions = np.arange(0, 128) if step % 2 == 0 else np.arange(128, 257)
        for model, optimizer in zip(repaired, repaired_optimizers):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = (model(provider.x[positions], provider.anchor[positions]) - provider.y[positions]).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
    repaired[0].eval()
    trained_trend = repaired[0].ema(x).detach().requires_grad_()
    trained_seasonal = (x - repaired[0].ema(x)).detach()
    trained_out = repaired[0].backbone(trained_seasonal, trained_trend)[:, :horizon, 0]
    trained_gradient = torch.autograd.grad(trained_out.sum(), trained_trend)[0]
    trained_derivative = float(trained_gradient.abs().sum())
    assert trained_derivative > (1e-8 if dtype == torch.float32 else 1e-12)

    # A separate nonzero head unmasks the branch dependency. Hook the actual
    # backbone inputs so seasonal preservation is checked directly.
    derivatives, seasonal_inputs = {}, {}
    for arm, model in models.items():
        model.eval()
        with torch.no_grad():
            model.backbone.fc8.weight.fill_(.1)
            model.backbone.fc8.bias.zero_()
        trend = model.ema(x).detach().requires_grad_()
        seasonal = x - model.ema(x)
        branch_trend = model._route_mean.to(x).expand_as(trend) if arm == "C" else trend
        out = model.backbone(seasonal.detach(), branch_trend)[:, :horizon, 0]
        grad = torch.autograd.grad(out.sum(), trend, allow_unused=True)[0]
        derivatives[arm] = 0. if grad is None else float(grad.abs().sum())
        captured = []
        handle = model.backbone.register_forward_pre_hook(lambda module, args: captured.append(tuple(v.detach().clone() for v in args)))
        model(x, anchor)
        handle.remove()
        seasonal_inputs[arm] = captured[0][0]
    assert derivatives["A"] == derivatives["C"] == 0.
    assert derivatives["B"] > (1e-8 if dtype == torch.float32 else 1e-12)
    assert torch.equal(seasonal_inputs["B"], seasonal_inputs["C"])
    return {"horizon": horizon, "lookback": cfg["lookback"], "target": target, "dtype": str(dtype),
            "initial_states_exact": True, "baseline_physical_passed": True,
            "original_actual_fixed_12_updates_bitwise_exact": True,
            "original_wrapper_native_12_updates_bitwise_exact": True,
            "matched_B_C_updates": 12, "trained_B_trend_derivative_sum_abs": trained_derivative,
            "trend_derivative_sum_abs": derivatives, "B_C_actual_seasonal_exact": True,
            "mean_consumed_future_or_labels": False, "passed": True}


def synthetic_identity(protocol_hash, arm):
    return hashlib.sha256(w.canonical({"protocol": protocol_hash, "case": "recovery_nativeH2", "cfg": fixed_cfg(arm=arm)})).hexdigest()


def child_case(args, protocol, protocol_hash):
    if args.case_folder is None:
        raise RuntimeError("Synthetic child needs an isolated case folder")
    folder = Path(args.case_folder).resolve()
    if not folder.is_relative_to((w.BASE / "engineering").resolve()):
        raise RuntimeError("Engineering child must stay in the synthetic engineering folder")
    cfg = fixed_cfg(arm=args.synthetic_arm)
    provider = SyntheticProvider(cfg)
    deadline = min(float(protocol["deadline_epoch"]), args.child_deadline or float(protocol["deadline_epoch"]))
    return w.train_fixed(cfg, provider, folder, synthetic_identity(protocol_hash, args.synthetic_arm), deadline,
                         epochs=12, checkpoint_every=1, pause_updates=args.pause_updates,
                         artificial_batch_delay=args.batch_delay,
                         training_deadline_seconds=args.training_deadline_seconds,
                         pause_mean_batches=args.pause_mean_batches)


def load_case(folder, protocol_hash, arm):
    return w.Checkpoints(folder, synthetic_identity(protocol_hash, arm)).load()


def same_recovery(reference, resumed):
    keys = ("state", "model", "optimizer", "numpy_rng", "torch_rng")
    return all(exact(reference[key], resumed[key]) for key in keys)


class SyntheticScoreProvider(SyntheticProvider):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.identity = dict(self.identity, synthetic_score=True)
        self.data = types.SimpleNamespace(splits={"stop": np.arange(1000, 1000 + self.length, dtype="<i8")})
        self.raw_truth = np.arange(self.length * cfg["horizon"], dtype=np.float64).reshape(self.length, cfg["horizon"]) / 137. + .123456789012345
        self.consumed = set()
        self.stop_after_first_batch = False

    def size(self, part):
        assert part == "stop"
        return self.length

    def inputs(self, part, positions):
        assert part == "stop"
        return self.x[positions], self.anchor[positions]

    def raw_labels(self, part, positions):
        assert part == "stop"
        assert not self.consumed.intersection(int(p) for p in positions), "Committed score truths were reread"
        self.consumed.update(int(p) for p in positions)
        if self.stop_after_first_batch:
            w.STOP_REQUESTED = True
            self.stop_after_first_batch = False
        return self.raw_truth[positions]

    def inverse(self, value):
        return np.asarray(value) * np.asarray(self.scaler["y_std"]) + np.asarray(self.scaler["y_mean"])


def score_recovery_case(session, arm, protocol_hash, deadline):
    cfg = fixed_cfg(arm=arm)
    provider = SyntheticScoreProvider(cfg)
    model = w.build_model(cfg, provider.scaler, torch.zeros((cfg["lookback"], 1)))
    model.eval()
    binding = {"protocol_sha256": protocol_hash, "provider": provider.identity,
               "synthetic_model": arm, "cfg": cfg}
    identity = hashlib.sha256(w.canonical(binding)).hexdigest()
    folder = session / arm / "score_resume"
    provider.stop_after_first_batch = True
    try:
        paused = w.score_batches(cfg, provider, model, folder, binding, identity, deadline)
        assert paused["status"] == "paused" and paused["cursor"] == 128
    finally:
        w.STOP_REQUESTED = False
    # Simulate npz fsync followed by a crash before the manifest/pointer commit:
    # remove only the synthetic manifest, retain the complete payload prefix,
    # and rewind the synthetic pointer to its initial zero-row generation.
    pointer = w.read(folder / "SCORE_LATEST.json")
    latest_manifest = folder / pointer["manifest"]
    latest_manifest.unlink()
    initial_manifest = folder / "score_generation_00000001.json"
    w.atomic_json(folder / "SCORE_LATEST.json", {"manifest": initial_manifest.name,
                                               "manifest_sha256": w.digest(initial_manifest)})
    completed = w.score_batches(cfg, provider, model, folder, binding, identity, deadline)
    assert completed["status"] == "completed"
    assert len(provider.consumed) == provider.length
    assert np.array_equal(completed["actual"], provider.raw_truth)
    with torch.no_grad():
        expected = provider.inverse(model(provider.x, provider.anchor).numpy())
    assert np.array_equal(completed["prediction"], expected)
    # Completed reentry must reuse persisted values and leave their hashes.
    pointer_hash = w.digest(folder / "SCORE_LATEST.json")
    repeated = w.score_batches(cfg, provider, model, folder, binding, identity, deadline)
    assert repeated["status"] == "completed" and w.digest(folder / "SCORE_LATEST.json") == pointer_hash
    wrong_identity_rejected = False
    try:
        w.score_batches(cfg, provider, model, folder, binding, "wrong-score-identity", deadline)
    except RuntimeError:
        wrong_identity_rejected = True
    assert wrong_identity_rejected
    return {"arm": arm, "pause_resume_exact": True, "raw_truth_float64_exact": True,
            "committed_truth_rows_read_once": True, "orphan_npz_adopted": True,
            "completed_reentry_unchanged": True, "wrong_score_identity_rejected": True}


def run_engineering(args, protocol, protocol_hash):
    started = time.time()
    root = w.BASE / "engineering"
    root.mkdir(parents=True, exist_ok=True)
    session = Path(tempfile.mkdtemp(prefix="acceptance_", dir=root))
    cases = []
    try:
        for horizon in (1, 3):
            for target in ("future_close", "future_log_return"):
                for dtype in (torch.float32, torch.float64):
                    if time.time() >= protocol["deadline_epoch"]:
                        raise RuntimeError("Engineering deadline expired")
                    cases.append(structural_case(horizon, target, dtype))

        def command(folder, arm, extra=()):
            return [sys.executable, str(Path(__file__).with_name("worker.py")), "--engineering",
                    "--engineering-child", "--protocol", str(Path(args.protocol).resolve()),
                    "--case-folder", str(folder), "--synthetic-arm", arm, *extra]

        def run(folder, arm, extra=()):
            remaining = float(protocol["deadline_epoch"]) - time.time()
            if remaining <= 1:
                raise RuntimeError("No engineering budget remaining")
            completed = subprocess.run(command(folder, arm, extra), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       text=True, timeout=min(60., remaining), check=False)
            w.atomic_json(folder / "PROCESS.json", {"returncode": completed.returncode,
                                                     "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:]})
            if completed.returncode != 0:
                raise RuntimeError("Synthetic subprocess failed; retain PROCESS.json")

        recovery = []
        for arm in ("A", "B", "C"):
            continuous = session / arm / "continuous"
            run(continuous, arm)
            reference = load_case(continuous, protocol_hash, arm)
            immutable = {name: w.digest(continuous / name) for name in ("LATEST.json", "DONE.json")}
            run(continuous, arm)
            assert immutable == {name: w.digest(continuous / name) for name in immutable}
            mean_pause = session / arm / "mean_resume"
            run(mean_pause, arm, ("--pause-mean-batches", "1"))
            mean_before = load_case(mean_pause, protocol_hash, arm)
            assert (mean_before["state"]["phase"] == "mean" and mean_before["state"]["mean_cursor"] == 128 and
                    mean_before["state"]["mean_seen"] == 128 and mean_before["model"] is None)
            run(mean_pause, arm)
            mean_resumed = load_case(mean_pause, protocol_hash, arm)
            assert same_recovery(reference, mean_resumed)
            assert all(mean_resumed["elapsed"][key] >= mean_before["elapsed"][key] for key in ("wall_seconds", "cpu_seconds"))
            pause = session / arm / "exit_resume"
            run(pause, arm, ("--pause-updates", "2"))
            before = load_case(pause, protocol_hash, arm)
            assert before["state"]["phase"] == "train" and before["state"]["updates"] == 2
            run(pause, arm)
            resumed = load_case(pause, protocol_hash, arm)
            assert same_recovery(reference, resumed)
            assert all(resumed["elapsed"][key] >= before["elapsed"][key] for key in ("wall_seconds", "cpu_seconds"))

            timeout_case = session / arm / "timeout_resume"
            run(timeout_case, arm, ("--training-deadline-seconds", "1", "--batch-delay", ".25"))
            timed = load_case(timeout_case, protocol_hash, arm)
            assert timed is not None and timed["state"]["phase"] == "train" and 1 <= timed["state"]["updates"] < 24
            run(timeout_case, arm)
            assert same_recovery(reference, load_case(timeout_case, protocol_hash, arm))

            signal_case = session / arm / "signal_resume"
            process = subprocess.Popen(command(signal_case, arm, ("--batch-delay", ".25")), stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, text=True)
            try:
                wait_deadline = min(time.time() + 30., float(protocol["deadline_epoch"]))
                while time.time() < wait_deadline:
                    if process.poll() is not None:
                        raise RuntimeError("Synthetic signal child ended before interruption")
                    latest = signal_case / "LATEST.json"
                    if latest.exists():
                        current = load_case(signal_case, protocol_hash, arm)
                        if current["state"]["updates"] >= 1:
                            break
                    time.sleep(.05)
                else:
                    raise RuntimeError("Synthetic signal child did not reach a checkpoint")
                process.send_signal(signal.SIGUSR1)
                stdout, stderr = process.communicate(timeout=30.)
                w.atomic_json(signal_case / "PROCESS_SIGNAL.json", {"returncode": process.returncode, "stdout": stdout[-8000:], "stderr": stderr[-8000:]})
                assert process.returncode == 0
            finally:
                if process.poll() is None:
                    process.kill()
                    process.communicate()
            assert load_case(signal_case, protocol_hash, arm)["state"]["phase"] != "completed"
            run(signal_case, arm)
            assert same_recovery(reference, load_case(signal_case, protocol_hash, arm))

            wrong_source_rejected = False
            try:
                w.Checkpoints(continuous, "wrong-source-identity").load()
            except RuntimeError:
                wrong_source_rejected = True
            assert wrong_source_rejected
            tampered = session / arm / "tampered_payload"
            shutil.copytree(continuous, tampered)
            pointer = w.read(tampered / "LATEST.json")
            manifest = w.read(tampered / pointer["manifest"])
            payload = tampered / manifest["payload"]
            with payload.open("ab") as f:
                f.write(b"intentional synthetic corruption")
            corruption_rejected = False
            try:
                load_case(tampered, protocol_hash, arm)
            except RuntimeError:
                corruption_rejected = True
            assert corruption_rejected
            recovery.append({"arm": arm, "exit_resume_exact": True, "timeout_resume_exact": True,
                             "signal_resume_exact": True, "completed_rerun_hashes_unchanged": True,
                             "mean_accumulation_resume_exact": True, "elapsed_nondecreasing": True,
                             "wrong_source_rejected": True, "corrupt_payload_rejected": True})
        score_recovery = [score_recovery_case(session, arm, protocol_hash, float(protocol["deadline_epoch"]))
                          for arm in ("A", "B", "C")]
        result = {"status": "passed", "protocol_sha256": protocol_hash, "cases": cases,
                  "recovery_by_arm": recovery,
                  "score_recovery_by_arm": score_recovery,
                  "exit_resume_exact": True, "timeout_resume_exact": True, "signal_resume_exact": True,
                  "wrong_source_rejected": True, "corrupt_payload_rejected": True,
                  "session_directory": str(session), "elapsed_seconds": time.time() - started,
                  "market_provider_constructed": False, "market_values_read": False,
                  "GPU": False, "Q3_read": False, "candidate_admitted": False}
    except Exception as error:
        result = {"status": "failed_stop_no_tolerance_rescue", "protocol_sha256": protocol_hash,
                  "cases_completed": cases, "error_type": type(error).__name__, "error": str(error),
                  "session_directory": str(session), "elapsed_seconds": time.time() - started,
                  "market_provider_constructed": False, "market_values_read": False,
                  "GPU": False, "Q3_read": False, "candidate_admitted": False}
        w.atomic_json(session / "ENGINEERING_RESULT.json", result)
        raise
    w.atomic_json(session / "ENGINEERING_RESULT.json", result)
    return result
