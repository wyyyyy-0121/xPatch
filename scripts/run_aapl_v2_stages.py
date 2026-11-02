"""Run accepted post-training stages in order, with an on-disk status and exclusive lock.

This launcher does not modify the signed experiment. It can wait for an already
running validation PID, then requires independent acceptance before final freeze.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_status(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def pid_identity(pid):
    try:
        # starttime after the parenthesized executable field; a reused PID is
        # not the validation process we were instructed to wait for.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/aapl_v2/main")
    parser.add_argument("--wait-validation-pid", type=int)
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "pipeline_status.json"
        state = {"started_at_utc": utc_now(), "supervisor_pid": os.getpid(), "state": "starting",
                 "order": ["validate", "verify", "freeze", "confirm", "report", "verify"], "stages": []}
        write_status(state_path, state)
        if args.wait_validation_pid:
            pid = args.wait_validation_pid
            original = pid_identity(pid)
            state.update(state="waiting_for_validation", validation_pid=pid,
                         validation_process_starttime=original)
            write_status(state_path, state)
            while original is not None and pid_identity(pid) == original:
                time.sleep(10)
            state["validation_process_exited_at_utc"] = utc_now()
            # Completeness and integrity are checked by the independent verify
            # command; disappearance of a PID alone is never treated as success.
        else:
            state["order"] = ["validate", "verify", "freeze", "confirm", "report", "verify"]
        stages = ["verify", "freeze", "confirm", "report", "verify"] if args.wait_validation_pid else state["order"]
        environment = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
        for i, stage in enumerate(stages):
            entry = {"stage": stage, "started_at_utc": utc_now(), "state": "running"}
            state.update(state="running", current_stage=stage)
            state["stages"].append(entry)
            write_status(state_path, state)
            log = output / f"pipeline_{i+1:02}_{stage}.log"
            command = [sys.executable, "-m", "finance_v2.run", "--stage", stage, "--output", str(output)]
            with log.open("a") as stream:
                completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            entry.update(exit_code=completed.returncode, completed_at_utc=utc_now(), log=str(log),
                         state="passed" if completed.returncode == 0 else "failed")
            if completed.returncode:
                state.update(state="failed", failed_stage=stage, completed_at_utc=utc_now())
                write_status(state_path, state)
                return completed.returncode
            write_status(state_path, state)
        frozen = json.loads((output / "frozen.json").read_text())
        acceptance = json.loads((output / "acceptance.json").read_text())
        state.update(state="completed", completed_at_utc=utc_now(),
                     selected=frozen["selected"], independent_acceptance=acceptance["passed"],
                     research_result="no_candidate_passed" if not any(frozen["selected"].values()) else
                                     "historical_confirmation_completed; prospective_confirmation_still_required",
                     transaction_cost_research="not opened by this launcher")
        write_status(state_path, state)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
