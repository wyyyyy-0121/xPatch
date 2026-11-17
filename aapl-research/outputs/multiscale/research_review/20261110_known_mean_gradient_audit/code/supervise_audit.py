"""Run the frozen zero-update qualification or audit, then independent replay."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

STOP = False


def stopped(_number, _frame):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    pp = Path(args.protocol).resolve()
    p = json.loads(pp.read_text())
    root = Path(p["output_root"]).resolve()
    assert root.is_dir() and not (root / "EXECUTION.json").exists()
    assert p["phase"] in {"preparation", "formal_gradient_audit"}
    assert not any(p[k] for k in ("GPU", "paid_resources", "market_data_allowed", "Q3_allowed", "synthetic_training_admitted"))
    assert p["training_updates"] == 0
    assert p["numerical_audit_admitted"] is (p["phase"] == "formal_gradient_audit")
    assert p["deadline_epoch"] - p["started_epoch"] == (600 if p["phase"] == "preparation" else 780)
    if p["phase"] == "formal_gradient_audit":
        assert p["stage_deadline_epoch"] - p["deadline_epoch"] == 120
    for name, wanted in p["source_sha256"].items():
        assert digest(name) == wanted, name
    os.sched_setaffinity(0, {24})
    tmp = root / "runtime_tmp"
    tmp.mkdir()
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp))
    signal.signal(signal.SIGUSR1, stopped)
    signal.signal(signal.SIGTERM, stopped)
    report = {"status": "running", "protocol_sha256": digest(pp), "supervisor_pid": os.getpid(),
              "phase": p["phase"], "children": [], "deadline_epoch": p["deadline_epoch"],
              "stage_deadline_epoch": p.get("stage_deadline_epoch"), "deadline_reset": False,
              "training_updates": 0, "market_data_read": False, "Q3_read": False,
              "new_market_candidate_admitted": False, "full_goal_achieved": False,
              "started_epoch": time.time(), "CPU_affinity": [24]}

    def save():
        target = root / "EXECUTION.json"
        temporary = root / (".execution_" + str(os.getpid()) + ".json")
        with temporary.open("w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def run(name, argv, deadline):
        if STOP or time.time() >= deadline - 15:
            raise TimeoutError("No time for next frozen action")
        event = {"name": name, "pid": None, "exit_code": None,
                 "started_epoch": time.time(), "command": argv, "signals": []}
        report["children"].append(event)
        child = None
        save()
        try:
            with (root / (name + ".log")).open("xb", buffering=0) as log:
                child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=root, env=env, start_new_session=True,
                                         stdin=subprocess.DEVNULL)
                event["pid"] = child.pid
                save()
                pause_at = None
                while child.poll() is None:
                    now = time.time()
                    if (STOP or now >= deadline - 15) and pause_at is None:
                        os.killpg(child.pid, signal.SIGUSR1)
                        pause_at = now
                        event["signals"].append({"number": signal.SIGUSR1, "at": now})
                        save()
                    if now >= deadline or (pause_at is not None and now >= pause_at + 15):
                        os.killpg(child.pid, signal.SIGKILL)
                        event["signals"].append({"number": signal.SIGKILL, "at": now})
                        break
                    time.sleep(.2)
                event["exit_code"] = child.wait(timeout=5)
                event["finished_epoch"] = time.time()
                save()
                if event["exit_code"] != 0:
                    raise RuntimeError(name + " exited " + str(event["exit_code"]))
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                event["exit_code"] = child.wait(timeout=5)
                save()

    save()
    try:
        worker = Path(__file__).with_name("gradient_worker.py")
        if p["phase"] == "preparation":
            run("qualification", [sys.executable, "-u", str(worker), "--protocol", str(pp),
                                  "--output", str(root / "qualify"), "--mode", "qualify"], p["deadline_epoch"])
            q = json.loads((root / "qualify" / "RESULTS.json").read_text())
            assert q["status"] == "qualification_and_cost_complete"
            assert q["performance_scored"] is False and q["oracle_prediction_scored"] is False
            assert len(q["costs"]) == 1
            report["qualification"] = q
            report["status"] = "completed_frozen_gradient_qualification"
        else:
            run("audit", [sys.executable, "-u", str(worker), "--protocol", str(pp),
                          "--output", str(root / "formal"), "--mode", "audit"], p["deadline_epoch"])
            verifier = Path(__file__).with_name("verify_gradient_audit.py")
            run("independent_replay", [sys.executable, "-u", str(verifier), "--protocol", str(pp),
                                       "--run-directory", str(root / "formal")], p["stage_deadline_epoch"])
            report["status"] = "completed_zero_update_gradient_audit_and_replay"
    except BaseException as error:
        report.update(status="terminal_incomplete_committed_progress_retained",
                      error={"type": type(error).__name__, "message": str(error)})
    report["finished_epoch"] = time.time()
    save()
    print(json.dumps({"status": report["status"], "record": str(root / "EXECUTION.json")}), flush=True)
    return 0 if report["status"].startswith("completed_") else 1


if __name__ == "__main__":
    sys.exit(main())
