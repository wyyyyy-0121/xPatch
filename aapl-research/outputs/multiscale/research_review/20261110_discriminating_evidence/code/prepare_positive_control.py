"""Bounded synthetic preparation controller; no experiment runs on import.

Run only with a separately reviewed, hash-frozen preparation protocol.  This
controller launches the worker for cost/identity qualification and a real
external-signal recovery comparison.  It never launches the formal 12 fits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import sys
import tempfile
import time


REQUESTED_STOP = False


class PreparationFailure(Exception):
    pass


class DeadlineExpired(PreparationFailure):
    pass


def request_stop(_number, _frame):
    global REQUESTED_STOP
    REQUESTED_STOP = True


def deadline_alarm(_number, _frame):
    raise DeadlineExpired("Original absolute preparation deadline reached")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf8", dir=path.parent,
                                         prefix=".preparation-", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def bounded_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise PreparationFailure("Preparation output must stay below output_root")
    return path


class Controller:
    def __init__(self, protocol_path):
        self.protocol_path = protocol_path.resolve()
        self.protocol = json.loads(self.protocol_path.read_text())
        self.protocol_sha256 = sha(self.protocol_path)
        protocol = self.protocol
        if protocol.get("phase") != "preparation":
            raise PreparationFailure("Only preparation phase is supported")
        if any(protocol.get(key) is not False for key in ("market_data_allowed", "GPU", "paid_resources")):
            raise PreparationFailure("Synthetic preparation requires market/GPU/payment disabled")
        if protocol.get("shapes") != [[480, 60], [96, 8]]:
            raise PreparationFailure("Expected the two original shapes in fixed order")
        if protocol.get("counts") != [2048, 512, 4096]:
            raise PreparationFailure("Expected the proposed fixed synthetic counts")
        for key, expected in (("max_epochs", 20), ("patience", 3),
                              ("learning_rate", 0.001), ("batch_size", 128)):
            if protocol.get(key) != expected:
                raise PreparationFailure("Unexpected frozen native recipe field: " + key)
        self.deadline = float(protocol["deadline_epoch"])
        self.root = Path(protocol["output_root"]).resolve()
        if not self.root.is_dir():
            raise PreparationFailure("Protocol output_root must already exist")
        self.worker = Path(__file__).resolve().with_name("positive_control_worker.py")
        self.own_source = Path(__file__).resolve()
        frozen = {Path(name).resolve(): digest for name, digest in protocol["source_sha256"].items()}
        for source in (self.worker, self.own_source):
            if source not in frozen or sha(source) != frozen[source]:
                raise PreparationFailure("Controller and worker must both be hash-frozen")
        for source, digest in frozen.items():
            if sha(source) != digest:
                raise PreparationFailure("Frozen source changed: " + str(source))
        self.directory = bounded_path(self.root, protocol.get("preparation_relative_directory", "preparation"))
        self.result_path = self.root / "PREPARATION_RESULT.json"
        if self.directory.exists() or self.result_path.exists():
            raise PreparationFailure("Preparation output exists; automatic retry is forbidden")
        self.directory.mkdir()
        self.temporary_directory = bounded_path(self.root, "runtime_tmp")
        self.temporary_directory.mkdir(exist_ok=True)
        self.env = dict(os.environ)
        self.env.update({"CUDA_VISIBLE_DEVICES": "-1", "OMP_NUM_THREADS": "1",
                         "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                         "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
                         "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                         "TMPDIR": str(self.temporary_directory),
                         "TMP": str(self.temporary_directory), "TEMP": str(self.temporary_directory)})
        os.environ.update({key: self.env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTHONDONTWRITEBYTECODE",
            "TMPDIR", "TMP", "TEMP")})
        sys.dont_write_bytecode = True
        tempfile.tempdir = str(self.temporary_directory)
        os.sched_setaffinity(0, {24})
        self.report = {"status": "preparation_running", "protocol": str(self.protocol_path),
                       "protocol_sha256": self.protocol_sha256,
                       "controller_sha256": sha(self.own_source), "worker_sha256": sha(self.worker),
                       "original_deadline_epoch": self.deadline, "deadline_reset": False,
                       "started_epoch": time.time(), "CPU_affinity": sorted(os.sched_getaffinity(0)),
                       "environment": {key: self.env[key] for key in (
                           "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TMPDIR")},
                       "children": [], "formal_training_started": False,
                       "market_arrays_or_historical_weights_read": False,
                       "new_market_candidate_admitted": False, "full_goal_achieved": False}
        self.child = None
        self.persist()

    def persist(self):
        atomic_json(self.result_path, self.report)

    def admit_next_action(self):
        if REQUESTED_STOP:
            raise PreparationFailure("Controller externally asked to pause; no subsequent stage")
        if time.time() >= self.deadline - 15:
            raise DeadlineExpired("Less than 15 seconds remain; no subsequent stage")
        if sha(self.protocol_path) != self.protocol_sha256:
            raise PreparationFailure("Original protocol changed")

    def signal_child(self, event, number, reason):
        if self.child is None or self.child.poll() is not None:
            return
        os.killpg(self.child.pid, number)
        event.setdefault("signals", []).append({"signal": number, "reason": reason, "at_epoch": time.time()})
        self.persist()

    def run_child(self, name, output, mode, *, resume=False, signal_at_ready=False):
        self.admit_next_action()
        command = [sys.executable, "-u", str(self.worker), "--protocol", str(self.protocol_path),
                   "--output", str(output), "--mode", mode]
        if mode == "trajectory":
            command.extend(["--stop-after-updates", "8"])
        if resume:
            command.append("--resume")
        if signal_at_ready:
            command.extend(["--ready-after-updates", "2"])
        event = {"name": name, "command": command, "output": str(output),
                 "started_epoch": time.time(), "pid": None, "exit_code": None,
                 "externally_paused_after_ready": False, "deadline_signal_sent": False,
                 "deadline_killed": False}
        self.report["children"].append(event)
        self.persist()
        log_path = self.directory / (name + ".log")
        selector = selectors.DefaultSelector()
        pending = b""
        deadline_or_stop = False
        try:
            with log_path.open("xb", buffering=0) as log:
                self.child = subprocess.Popen(command, cwd=self.root, env=self.env,
                                              stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                              stderr=subprocess.STDOUT, start_new_session=True)
                event["pid"] = self.child.pid
                self.persist()  # Persist PID before waiting or reading child output.
                os.set_blocking(self.child.stdout.fileno(), False)
                selector.register(self.child.stdout, selectors.EVENT_READ)
                while True:
                    now = time.time()
                    if (REQUESTED_STOP or now >= self.deadline - 15) and not deadline_or_stop:
                        deadline_or_stop = True
                        event["deadline_signal_sent"] = now >= self.deadline - 15
                        self.signal_child(event, signal.SIGUSR1, "deadline_minus15_or_parent_stop")
                    if now >= self.deadline and self.child.poll() is None:
                        event["deadline_killed"] = True
                        self.signal_child(event, signal.SIGKILL, "original_absolute_deadline")
                    for key, _mask in selector.select(timeout=max(0.0, min(0.2, self.deadline - now))):
                        try:
                            chunk = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        log.write(chunk)
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            if signal_at_ready and line.strip() == b"READY_FOR_EXTERNAL_SIGNAL":
                                if event["externally_paused_after_ready"]:
                                    raise PreparationFailure("Duplicate readiness marker")
                                event["externally_paused_after_ready"] = True
                                event["ready_observed_epoch"] = time.time()
                                self.signal_child(event, signal.SIGUSR1, "external_pause_at_committed_step2")
                        if len(pending) > 1048576:
                            raise PreparationFailure("Unbounded child output line")
                    if self.child.poll() is not None and not selector.get_map():
                        break
                event["exit_code"] = self.child.wait(timeout=5)
                event["finished_epoch"] = time.time()
                event["log_sha256"] = sha(log_path)
                self.persist()
            if deadline_or_stop or time.time() >= self.deadline - 15:
                raise DeadlineExpired("Preparation interrupted; no subsequent stages")
            expected = 75 if signal_at_ready else 0
            if event["exit_code"] != expected:
                raise PreparationFailure(name + " returned " + str(event["exit_code"]))
            if signal_at_ready and not event["externally_paused_after_ready"]:
                raise PreparationFailure("Child exited without an observed external pause")
            return event
        finally:
            selector.close()
            if self.child is not None:
                if self.child.poll() is None:
                    self.signal_child(event, signal.SIGKILL, "controller_failure_cleanup")
                event["exit_code"] = self.child.wait(timeout=5)
                event.setdefault("finished_epoch", time.time())
                if self.child.stdout is not None:
                    self.child.stdout.close()
                self.child = None
                self.persist()

    def compare_payloads(self, first, second):
        self.admit_next_action()
        # Deadline interruption also covers import/deserialization/comparison.
        previous_alarm_handler = signal.signal(signal.SIGALRM, deadline_alarm)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, self.deadline - time.time()))
        try:
            import torch  # Only numerical import in this controller; no model helpers.
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            first_payload = torch.load(first, weights_only=True, map_location="cpu")
            second_payload = torch.load(second, weights_only=True, map_location="cpu")
            counts = {"tensors": 0, "tensor_bytes": 0, "scalar_leaves": 0,
                      "dicts": 0, "lists": 0, "tuples": 0}

            def same(a, b, location):
                self.admit_next_action()
                if type(a) is not type(b):
                    raise PreparationFailure("Payload type differs: " + location)
                if torch.is_tensor(a):
                    if (a.dtype != b.dtype or a.shape != b.shape or a.stride() != b.stride()
                            or a.layout != b.layout or a.device != b.device
                            or a.requires_grad != b.requires_grad or a.layout != torch.strided):
                        raise PreparationFailure("Tensor metadata differs: " + location)
                    def raw(value):
                        materialized = value.detach().cpu().resolve_conj().resolve_neg().contiguous().reshape(-1)
                        return bytes(materialized.view(torch.uint8).tolist())
                    av, bv = raw(a), raw(b)
                    if av != bv:
                        raise PreparationFailure("Tensor raw bytes differ: " + location)
                    counts["tensors"] += 1
                    counts["tensor_bytes"] += len(av)
                elif isinstance(a, dict):
                    ak = sorted((type(k).__name__, repr(k)) for k in a)
                    bk = sorted((type(k).__name__, repr(k)) for k in b)
                    if ak != bk:
                        raise PreparationFailure("Dict keys differ: " + location)
                    counts["dicts"] += 1
                    for key in a:
                        same(a[key], b[key], location + "[" + repr(key) + "]")
                elif isinstance(a, (list, tuple)):
                    if len(a) != len(b):
                        raise PreparationFailure("Sequence length differs: " + location)
                    counts["lists" if isinstance(a, list) else "tuples"] += 1
                    for index, (aa, bb) in enumerate(zip(a, b)):
                        same(aa, bb, location + "[" + str(index) + "]")
                elif isinstance(a, float):
                    if struct.pack("!d", a) != struct.pack("!d", b):
                        raise PreparationFailure("Float bytes differ: " + location)
                    counts["scalar_leaves"] += 1
                elif a is None or isinstance(a, (str, int, bool, bytes)):
                    if a != b:
                        raise PreparationFailure("Scalar differs: " + location)
                    counts["scalar_leaves"] += 1
                else:
                    raise PreparationFailure("Unsupported payload type at " + location + ": " + str(type(a)))

            same(first_payload, second_payload, "payload")
            if first_payload["state"]["steps"] != 8 or first_payload["state"]["epoch"] != 0:
                raise PreparationFailure("Comparison must end at eight updates before any stop scoring")
            if first_payload["state"]["history"]:
                raise PreparationFailure("Preparation unexpectedly scored an epoch")
            return {"status": "passed_exact_complete_payload_leaf_comparison", **counts,
                    "first_checkpoint": str(first), "second_checkpoint": str(second),
                    "torch_version": str(torch.__version__), "CPU_affinity": sorted(os.sched_getaffinity(0)),
                    "weights_only": True, "updates": 8, "epoch": 0,
                    "compared_file_hashes_instead_of_payloads": False}
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_alarm_handler)

    def execute(self):
        try:
            self.run_child("qualify", self.directory / "qualify", "qualify")
            qualification = json.loads((self.directory / "qualify" / "RESULTS.json").read_text())
            costs = qualification.get("costs", [])
            if (qualification.get("status") != "qualification_and_cost_complete"
                    or len(costs) != 2 or any(case.get("steps") != 8 for case in costs)
                    or qualification.get("performance_scored") is not False
                    or qualification.get("oracle_prediction_scored") is not False):
                raise PreparationFailure("Qualification receipt is incomplete or scored performance")
            self.report["qualification_cost"] = qualification
            self.persist()
            continuous = self.directory / "continuous"
            recovered = self.directory / "signal_resume"
            self.run_child("continuous8", continuous, "trajectory")
            self.run_child("external_signal_at2", recovered, "trajectory", signal_at_ready=True)
            pause_path = recovered / "PAUSE.json"
            pause = json.loads(pause_path.read_text())
            if (pause.get("status") != "paused_checkpoint_preserved"
                    or pause.get("protocol_sha256") != self.protocol_sha256
                    or pause.get("original_deadline_epoch") != self.deadline):
                raise PreparationFailure("External-signal pause receipt changed identity/deadline")
            self.report["external_pause"] = {"status": pause["status"], "receipt_sha256": sha(pause_path),
                                              "new_process_resume_required": True}
            self.persist()
            self.run_child("explicit_new_process_resume_to8", recovered, "trajectory", resume=True)
            self.report["recovery_comparison"] = self.compare_payloads(
                continuous / "unit" / "latest.pt", recovered / "unit" / "latest.pt")
            self.report["status"] = "passed_actual_synthetic_preparation_cost_and_external_signal_recovery"
            self.report["preparation_accepted"] = True
            self.report["formal_training_admitted"] = False
            self.report["finished_epoch"] = time.time()
            self.persist()
            return 0
        except BaseException as exc:
            self.report["status"] = ("stopped_original_deadline_checkpoint_preserved"
                                     if isinstance(exc, DeadlineExpired) else "failed_preparation_no_retry")
            self.report["preparation_accepted"] = False
            self.report["formal_training_admitted"] = False
            self.report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            self.report["finished_epoch"] = time.time()
            self.persist()
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    args = parser.parse_args()
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    controller = Controller(args.protocol)
    return controller.execute()


if __name__ == "__main__":
    sys.exit(main())
