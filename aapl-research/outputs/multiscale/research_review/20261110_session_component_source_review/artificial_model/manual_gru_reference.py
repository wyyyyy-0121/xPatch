"""Independent artificial-only GRU gate replay; never imports the primary model."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import resource
import signal
import tempfile
import time

BASE = Path(__file__).resolve().parent
CONTRACT_SHA = "79ef8e05d70c4a7cc2b88f809b6be9cf8681d01dbd1898e6c84975d0ba654817"
AS_LIMIT = 6442450944
RSS_LIMIT = 3221225472
RTOL = 1e-9
ATOL = 1e-11
CODE_NAMES = {"prototype_model.py", "artificial_acceptance.py", "manual_gru_reference.py", "run_supervisor.py"}


def digest(body):
    return hashlib.sha256(body).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        result = hashlib.sha256()
        for block in iter(lambda: stream.read(65536), b""):
            result.update(block)
    return result.hexdigest()


def atomic_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
            name = stream.name
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # link refuses an existing final artifact while atomically exposing all bytes.
        os.link(name, path)
        os.unlink(name)
        name = None
        directory = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if name is not None:
            os.unlink(name)


def run(protocol_path):
    began = time.time()
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
    body = protocol_path.read_bytes()
    protocol = json.loads(body)
    identity = digest(body)
    start, end = protocol["started_epoch"], protocol["deadline_epoch"]
    if (type(start) not in (int, float) or type(end) not in (int, float)
            or not math.isfinite(start) or not math.isfinite(end) or end-start != 180
            or not start <= began < end-1
            or protocol.get("market_arrays_admitted") is not False
            or protocol.get("GPU") is not False or protocol.get("Q3_allowed") is not False
            or protocol.get("paid_resources", False) is not False
            or protocol["contract_sha256"] != CONTRACT_SHA
            or os.sched_getaffinity(0) != {26}):
        raise ValueError("Independent artificial protocol/resource identity invalid")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise ValueError("Numerical thread admission must precede imports")
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise ValueError("GPU visibility must be disabled")
    if file_sha(BASE/"ENGINEERING_CONTRACT.json") != CONTRACT_SHA:
        raise ValueError("Frozen engineering contract changed")
    hashes = protocol["code_hashes"]
    if set(hashes) != CODE_NAMES:
        raise ValueError("Exact reviewed code basenames required")
    for name, expected in hashes.items():
        if file_sha(BASE/name) != expected:
            raise ValueError("Frozen engineering/reviewer/supervisor code changed")
    if protocol["supervisor_sha256"] != hashes["run_supervisor.py"]:
        raise ValueError("Supervisor identity differs from frozen code")
    root = BASE/"artificial_run"
    output = root/"INDEPENDENT_MANUAL_GRU_REVIEW.json"
    if output.exists():
        raise ValueError("Do not overwrite an accepted independent artifact")

    def guard():
        if time.time() >= end-1:
            raise TimeoutError("Original artificial180s deadline reached")
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 > RSS_LIMIT:
            raise MemoryError("Original3GiB RSS ceiling reached")

    def expired(*_):
        raise TimeoutError("Original artificial180s deadline or external termination")

    signal.signal(signal.SIGALRM, expired)
    signal.signal(signal.SIGTERM, expired)
    signal.signal(signal.SIGINT, expired)
    signal.setitimer(signal.ITIMER_REAL, max(0.001, end-1-time.time()))
    try:
        # Library imports only; no source-model, acceptance, training or market imports.
        import numpy as np
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        guard()
        payload_path = root/"REFERENCE_CASES.pt"
        if not 0 < payload_path.stat().st_size <= 16*1024*1024:
            raise ValueError("Artificial reference payload size invalid")
        payload_bytes = payload_path.read_bytes()
        payload_hash = digest(payload_bytes)
        payload = torch.load(io.BytesIO(payload_bytes), map_location="cpu", weights_only=True)
        if payload.get("schema") != 1 or not isinstance(payload.get("config"), dict):
            raise ValueError("Artificial reference schema invalid")

        def floating(value, shape=None):
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != torch.float64):
                raise ValueError("Reference values must be CPU FP64 tensors")
            array = value.detach().numpy().copy()
            if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
                raise ValueError("Reference floating shape/finiteness invalid")
            return array

        state = payload["model_state"]
        keys = {cell+"."+key for cell in ("cell1", "cell2")
                for key in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")}
        keys |= {"head.weight", "head.bias", "zero_scaled"}
        if set(state) != keys:
            raise ValueError("Unexpected artificial state dictionary")
        horizon = int(state["zero_scaled"].numel())
        if horizon not in (8, 60):
            raise ValueError("Only original artificial horizon8/60 shapes admitted")
        arrays = {}
        for cell in ("cell1", "cell2"):
            for name, shape in (("weight_ih", (96, 2)), ("weight_hh", (96, 32)),
                                ("bias_ih", (96,)), ("bias_hh", (96,))):
                arrays[cell+"."+name] = floating(state[cell+"."+name], shape)
        arrays["head.weight"] = floating(state["head.weight"], (horizon, 82))
        arrays["head.bias"] = floating(state["head.bias"], (horizon,))
        zero_scaled = floating(state["zero_scaled"], (horizon,))
        if not np.any(arrays["head.weight"] != 0):
            raise ValueError("Evaluation readout must be nonzero to expose state effects")
        parameters = sum(value.size for value in arrays.values())
        if parameters != {60: 11892, 8: 7576}[horizon]:
            raise ValueError("Artificial capacity count differs from contract")
        comparisons = []

        def compare(name, expected, actual):
            a, b = np.asarray(expected), np.asarray(actual)
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Invalid manual-reference comparison: "+name)
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=ATOL, err_msg=name)
            comparisons.append({"name": name, "values": int(a.size),
                                "maxabs": float(np.max(np.abs(a-b))) if a.size else 0.})

        def sigmoid(value):
            result = np.empty_like(value)
            positive = value >= 0
            result[positive] = 1/(1+np.exp(-value[positive]))
            exp_value = np.exp(value[~positive])
            result[~positive] = exp_value/(1+exp_value)
            return result

        def cell_step(value, hidden, prefix):
            # PyTorch GRUCell gate order r,z,n; reset gates the hidden-new affine term.
            from_input = value @ arrays[prefix+".weight_ih"].T + arrays[prefix+".bias_ih"]
            from_hidden = hidden @ arrays[prefix+".weight_hh"].T + arrays[prefix+".bias_hh"]
            ir, iz, inn = np.split(from_input, 3, axis=1)
            hr, hz, hn = np.split(from_hidden, 3, axis=1)
            reset = sigmoid(ir+hr)
            update = sigmoid(iz+hz)
            candidate = np.tanh(inn+reset*hn)
            return (1-update)*candidate+update*hidden

        fixtures = payload["fixtures"]
        if not isinstance(fixtures, list) or [row.get("name") for row in fixtures] != ["no_boundary", "with_boundary"]:
            raise ValueError("Only two fixed artificial reference fixtures admitted")
        for fixture in fixtures:
            guard()
            name = fixture["name"]
            x = floating(fixture["x"])
            if x.ndim != 3 or x.shape[2] != 1 or not 1 <= x.shape[0] <= 128 or not 2 <= x.shape[1] <= 480:
                raise ValueError("Artificial input dimensions outside fixed maximum")
            batch, length, _ = x.shape
            bound_tensor = fixture["boundary"]
            if (not isinstance(bound_tensor, torch.Tensor) or bound_tensor.device.type != "cpu"
                    or bound_tensor.dtype != torch.bool or tuple(bound_tensor.shape) != (batch, length)):
                raise ValueError("Artificial boundary must be CPU bool[B,L]")
            boundary = bound_tensor.numpy().copy()
            if boundary[:, 0].any() or (name == "no_boundary" and boundary.any()) or (name == "with_boundary" and not boundary.any()):
                raise ValueError("Fixture boundary identity invalid")
            anchor = floating(fixture["anchor"], (batch, 1))
            context = floating(fixture["context"], (batch, 17))
            compare(name+"/zero_scaled", zero_scaled, floating(fixture["zero_scaled"], (horizon,)))
            expected = {}
            for arm, flag in (("C", 0), ("R", 1)):
                h1, h2 = np.zeros((batch, 32)), np.zeros((batch, 32))
                for step in range(length):
                    guard()
                    b = boundary[:, step:step+1].astype(np.float64)
                    value = np.concatenate((x[:, step], b), axis=1)
                    h2 = h2*(1-flag*b)
                    h1 = cell_step(value, h1, "cell1")
                    h2 = cell_step(value, h2, "cell2")
                head_input = np.concatenate((h1, h2, anchor, context), axis=1)
                prediction = zero_scaled+head_input @ arrays["head.weight"].T+arrays["head.bias"]
                expected[arm] = {"h1": h1, "h2": h2, "output": prediction}
                for key, value in expected[arm].items():
                    compare(name+"/"+arm+"/"+key, value, floating(fixture[arm][key], value.shape))
            compare(name+"/full_carry_branch_identical", expected["C"]["h1"], expected["R"]["h1"])
            if name == "no_boundary":
                for key in ("output", "h1", "h2"):
                    compare(name+"/paired_"+key, expected["C"][key], expected["R"][key])
            else:
                for row in range(batch):
                    guard()
                    selected = np.flatnonzero(boundary[row])
                    first = int(selected[-1]) if len(selected) else 0
                    suffix_hidden = np.zeros((1, 32))
                    for step in range(first, length):
                        guard()
                        value = np.concatenate((x[row:row+1, step], boundary[row:row+1, step:step+1].astype(np.float64)), axis=1)
                        suffix_hidden = cell_step(value, suffix_hidden, "cell2")
                    compare(name+"/suffix_second_state_row"+str(row), suffix_hidden, expected["R"]["h2"][row:row+1])
        guard()
        if file_sha(payload_path) != payload_hash or protocol_path.read_bytes() != body:
            raise ValueError("Reference payload or original protocol changed during replay")
        for name, expected_hash in hashes.items():
            if file_sha(BASE/name) != expected_hash:
                raise ValueError("Frozen code changed during replay")
        result = {"kind": "independent_artificial_manual_GRU_reference_v1",
                  "status": "manual_gate_states_outputs_and_suffix_reference_passed",
                  "protocol_sha256": identity, "contract_sha256": CONTRACT_SHA,
                  "reviewer_sha256": file_sha(Path(__file__)),
                  "reference_payload_sha256": payload_hash,
                  "started_epoch": began, "ended_epoch": time.time(), "deadline_epoch": end,
                  "seconds": time.time()-began, "CPU_affinity": sorted(os.sched_getaffinity(0)),
                  "threads": torch.get_num_threads(), "RLIMIT_AS_bytes": AS_LIMIT,
                  "maximum_RSS_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                  "horizon": horizon, "parameters": parameters, "fixtures": [x["name"] for x in fixtures],
                  "rtol": RTOL, "atol": ATOL, "comparisons": comparisons,
                  "scalar_comparison_count": sum(x["values"] for x in comparisons),
                  "maxabs_numeric_difference": max(x["maxabs"] for x in comparisons),
                  "primary_model_imported": False, "primary_forward_called": False,
                  "same_torch_GRU_backend_used": False,
                  "independent_numpy_gate_equations": True,
                  "artificial_only": True, "market_arrays_read": False, "HTTP_calls": 0,
                  "Q3_read": False, "GPU_used": False,
                  "scientific_training_success_claimed": False,
                  "external_signal_recovery_tested": False,
                  "research_goal_achieved": False}
        atomic_new_json(output, result)
        signal.setitimer(signal.ITIMER_REAL, 0)
        print(json.dumps({key: result[key] for key in ("status", "seconds", "scalar_comparison_count", "maxabs_numeric_difference")}))
    except Exception as error:
        signal.setitimer(signal.ITIMER_REAL, 0)
        failure = root/("INDEPENDENT_FAILED_"+str(os.getpid())+".json")
        atomic_new_json(failure, {"protocol_sha256": identity, "status": "failed",
                                "error_type": type(error).__name__, "reason": str(error),
                                "ended_epoch": time.time(), "original_deadline_epoch": end,
                                "artificial_only": True, "market_or_training_success_claimed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    run(parser.parse_args().protocol)
