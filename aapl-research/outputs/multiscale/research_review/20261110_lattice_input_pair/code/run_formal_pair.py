"""Supervise the one frozen synthetic study and its independent replay."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

STOP = False


def stopped(_number, _frame):
    global STOP
    STOP = True


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    path = Path(args.protocol).resolve()
    p = json.loads(path.read_text())
    assert p["phase"] == "formal_synthetic" and p["synthetic_training_admitted"]
    assert not p["market_data_allowed"] and not p["GPU"] and not p["paid_resources"]
    assert p["stage_deadline_epoch"] - p["deadline_epoch"] == 120
    for name, wanted in p["source_sha256"].items():
        assert digest(name) == wanted
    root = Path(p["output_root"]).resolve()
    record = root / "FORMAL_EXECUTION.json"
    assert root.is_dir() and not record.exists() and not (root / "formal").exists()
    tmp = root / "runtime_tmp"
    tmp.mkdir()
    os.sched_setaffinity(0, {24})
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    signal.signal(signal.SIGUSR1, stopped)
    signal.signal(signal.SIGTERM, stopped)
    report = {"status": "running", "protocol_sha256": digest(path), "children": [],
              "deadline_epoch": p["deadline_epoch"], "stage_deadline_epoch": p["stage_deadline_epoch"],
              "deadline_reset": False, "CPU_affinity": [24], "market_data": False, "GPU": False}

    def save():
        with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False) as f:
            temporary = Path(f.name)
            json.dump(report, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, record)

    def run(name, argv, deadline):
        if STOP or time.time() >= deadline - 15:
            raise TimeoutError("No time for next frozen stage")
        child = None
        event = {"name": name, "command": argv, "pid": None, "exit_code": None,
                 "started_epoch": time.time(), "signals": []}
        report["children"].append(event)
        save()
        try:
            with (root / (name + ".log")).open("xb", buffering=0) as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env,
                                         cwd=root, start_new_session=True, stdin=subprocess.DEVNULL)
                event["pid"] = child.pid
                save()
                pause_at = None
                while child.poll() is None:
                    now = time.time()
                    if (STOP or now >= deadline - 15) and pause_at is None:
                        os.killpg(child.pid, signal.SIGUSR1)
                        pause_at = now
                        event["signals"].append({"signal": "SIGUSR1", "at": now})
                        save()
                    if now >= deadline or (pause_at is not None and now >= pause_at + 15):
                        os.killpg(child.pid, signal.SIGKILL)
                        event["signals"].append({"signal": "SIGKILL", "at": now})
                        break
                    time.sleep(.2)
                event["exit_code"] = child.wait(timeout=5)
                event["finished_epoch"] = time.time()
                save()
                if event["exit_code"] != 0:
                    raise RuntimeError(name + " terminal exit " + str(event["exit_code"]))
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                event["exit_code"] = child.wait(timeout=5)
                save()

    save()
    try:
        worker = Path(__file__).with_name("pair_worker.py")
        verifier = Path(__file__).with_name("verify_lattice_pair.py")
        run("training", [sys.executable, "-u", str(worker), "--protocol", str(path),
                         "--output", str(root / "formal"), "--mode", "train"], p["deadline_epoch"])
        run("independent_replay", [sys.executable, "-u", str(verifier), "--protocol", str(path),
                                   "--run-directory", str(root / "formal")], p["stage_deadline_epoch"])
        report["status"] = "completed_synthetic_training_and_independent_replay"
    except BaseException as error:
        report["status"] = "terminal_incomplete_checkpoints_and_failures_retained"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["finished_epoch"] = time.time()
    report["full_research_goal_achieved"] = False
    report["new_market_candidate_admitted"] = False
    save()
    print(json.dumps({"status": report["status"], "record": str(record)}), flush=True)
    return 0 if report["status"].startswith("completed_") else 1


if __name__ == "__main__":
    sys.exit(main())
