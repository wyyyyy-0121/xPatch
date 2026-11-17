"""One frozen local analytic stage: real OS pause/resume, main, independent replay."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from runtime_common import (BASE, MEMORY_BYTES, atomic_json, load_protocol, memory_limit,
                            peak_rss_bytes, read_json, require, set_environment, sha, utc_now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    set_environment()
    protocol, identity = load_protocol(args.protocol)
    memory_limit()
    output = Path(protocol["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "FINAL_RUN_RECEIPT.json").exists(), "Do not replace terminal supervisor receipt")
    require(time.time() < protocol["deadline_epoch"] - 60, "No main budget left; absolute deadline is unchanged")
    begin = time.monotonic()
    processes = []
    events = []
    peak_child_rss = 0
    active = None
    received = []
    def handler(number, frame):
        received.append(number)
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGUSR1, handler)

    def launch(name, script, arguments):
        nonlocal active
        require(not received and time.time() < protocol["deadline_epoch"] - 2, "Supervisor stop/deadline")
        log = output / (name + ".log")
        require(not log.exists(), "Child log identity is immutable")
        with log.open("xb") as stream:
            active = subprocess.Popen([sys.executable, "-B", str(BASE / script), "--protocol", str(args.protocol.resolve()),
                                       *arguments], cwd=BASE, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = {"name": name, "pid": active.pid, "script": script, "started_epoch": time.time()}
        processes.append(record)
        return active, record

    def poll_child(process, record, *, ready_path=None):
        nonlocal peak_child_rss
        while process.poll() is None:
            now = time.time()
            status_path = Path(f"/proc/{process.pid}/status")
            try:
                status = status_path.read_text()
                rss = next((int(row.split()[1]) * 1024 for row in status.splitlines() if row.startswith("VmRSS:")), 0)
                peak_child_rss = max(peak_child_rss, rss)
                if rss > MEMORY_BYTES:
                    process.send_signal(signal.SIGTERM)
                    raise MemoryError("Child exceeded 512 MiB RSS")
                affinity = os.sched_getaffinity(process.pid)
                require(affinity.issubset({24, 26}), "Child escaped analysis CPU union")
            except ProcessLookupError:
                pass
            if ready_path is not None and ready_path.exists():
                ready = read_json(ready_path)
                require(ready["pid"] == process.pid and ready["protocol_sha256"] == identity and
                        ready["deadline_epoch"] == protocol["deadline_epoch"], "Signal handshake identity mismatch")
                event = {"sender_pid": os.getpid(), "recipient_pid": process.pid, "signal": int(signal.SIGUSR1),
                         "sent_epoch": time.time(), "ready": ready}
                os.kill(process.pid, signal.SIGUSR1)
                events.append(event)
                atomic_json(output / "EXTERNAL_SIGNAL_SENT.json", event, exclusive=True)
                ready_path = None
            if received or now >= protocol["deadline_epoch"] - 2:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=max(0.01, min(1.5, protocol["deadline_epoch"] - time.time())))
                except subprocess.TimeoutExpired:
                    process.kill()
                raise InterruptedError("Supervisor stopped child at unchanged absolute deadline")
            time.sleep(0.01)
        record.update(exit_code=process.returncode, ended_epoch=time.time())
        return process.returncode

    def run(name, script, arguments, expected=0, ready_path=None):
        process, record = launch(name, script, arguments)
        code = poll_child(process, record, ready_path=ready_path)
        require(code == expected, f"{name} exit {code}; expected {expected}")
        return record

    try:
        lifecycle_path = output / "LIFECYCLE_ACCEPTANCE.json"
        require(not lifecycle_path.exists(), "This controller runs the actual lifecycle once")
        common = ["--mode", "toy"]
        continuous = run("toy_continuous", "main_worker.py", common + ["--output-name", "toy_continuous"])
        paused = run("toy_pause", "main_worker.py", common + ["--output-name", "toy_resumed", "--signal-handshake"],
                     expected=75, ready_path=output / "toy_resumed" / "SIGNAL_READY.json")
        paused_path = output / "toy_resumed" / "CHECKPOINT.json"
        before = read_json(paused_path)
        digest_before = sha(paused_path)
        require(before["status"] == "paused" and len(before["completed_units"]) == 1 and
                any(e["signal"] == signal.SIGUSR1 and e["pid"] == paused["pid"] for e in before["signal_history"]),
                "External signal did not save the committed first unit")
        time.sleep(0.02)
        require(sha(paused_path) == digest_before, "Paused checkpoint was changed without worker")
        resumed = run("toy_resume", "main_worker.py", common + ["--output-name", "toy_resumed"])
        final = read_json(paused_path)
        uninterrupted = read_json(output / "toy_continuous" / "CHECKPOINT.json")
        require(resumed["pid"] != paused["pid"] and final["pid_history"] == [paused["pid"], resumed["pid"]],
                "Resume did not use a new process")
        fields = ("protocol_sha256", "deadline_epoch", "runtime", "mode", "completed_units", "records",
                  "input_qualifications", "operator_qualifications", "outcome")
        require(all(final[k] == uninterrupted[k] for k in fields), "Continuous/resumed scientific state differs")
        require(final["completed_units"][:1] == before["completed_units"] and final["records"][:12] == before["records"],
                "Previously committed unit lost on resume")
        lifecycle = {"passed": True, "protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                     "real_external_OS_signal": True, "external_sender_pid": os.getpid(),
                     "paused_pid": paused["pid"], "resumed_pid": resumed["pid"],
                     "paused_checkpoint_sha256": digest_before, "paused_checkpoint_unchanged": True,
                     "completed_units": len(final["completed_units"]), "records_bitwise_JSON_equal": len(final["records"]),
                     "original_committed_units_preserved": True, "scientific_fields_equal": list(fields), "signal_events": events}
        atomic_json(lifecycle_path, lifecycle, exclusive=True)
        run("main", "main_worker.py", ["--mode", "analytic", "--output-name", "main"])
        # Independent verifier is a separate CPU26 process and uses the same
        # frozen deadline including the 60-second reserve, never a reset timer.
        run("independent_replay", "independent_replay.py", [])
        summary, independent = read_json(output / "SUMMARY.json"), read_json(output / "INDEPENDENT_REPLAY.json")
        require(independent["status"] == "passed" and independent["protocol_sha256"] == identity and
                independent["decision"] == summary["decision"], "Independent acceptance differs")
        supported = summary["decision"]["supports_fixed_artificial_dynamic_estimation_cost"]
        decision = {"status": "complete_fixed_artificial_analytic_comparison", **summary["decision"],
                    "protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                    "exact_explanatory_continuation_closed": not supported,
                    "next_action": ("review_new_real_fit_only_identification_evidence_availability" if supported else
                                    "close_this_exact_artificial_dynamic_cost_continuation_no_recipe_rescue"),
                    "Q3_read": False, "GPU": False, "market_arrays_read": False, "training_updates": 0,
                    "numeric_backend_independence": "separate_methods_shared_numpy_RNG_and_runtime_only"}
        atomic_json(output / "DECISION.json", decision, exclusive=True)
        receipt = {"status": "completed", "protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                   "ended_epoch": time.time(), "total_seconds_since_freeze": time.time() - protocol["started_epoch"],
                   "supervisor_elapsed_seconds": time.monotonic() - begin, "processes": processes,
                   "child_peak_observed_rss_bytes": peak_child_rss, "supervisor_peak_rss_bytes": peak_rss_bytes(),
                   "analysis_cpu_union": [24, 26], "numeric_threads_each": 1,
                   "all_processes_terminal": all(p["exit_code"] in (0, 75) for p in processes),
                   "lifecycle_passed": True, "independent_replay_passed": True, "decision": decision}
        require(receipt["ended_epoch"] <= protocol["deadline_epoch"], "Stage completed after original deadline")
        atomic_json(output / "FINAL_RUN_RECEIPT.json", receipt, exclusive=True)
        print({"status": "completed", "support": supported, "seconds": receipt["total_seconds_since_freeze"]}, flush=True)
        return 0
    except BaseException as error:
        if active is not None and active.poll() is None:
            active.send_signal(signal.SIGTERM)
            try:
                active.wait(timeout=1)
            except subprocess.TimeoutExpired:
                active.kill(); active.wait()
        for record in processes:
            if "exit_code" not in record and active is not None and record["pid"] == active.pid:
                record["exit_code"] = active.poll()
        atomic_json(output / f"SUPERVISOR_FAILURE_{os.getpid()}.json", {"status": "failed_preserved",
                    "error": repr(error), "traceback": traceback.format_exc(), "processes": processes,
                    "protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                    "total_seconds_since_freeze": time.time() - protocol["started_epoch"]}, exclusive=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
