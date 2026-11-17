"""Wait for verified full evidence, then create documents for visual review.

This service does not train, resume, stop or signal any experiment process.
It never marks the research goal complete or publishes anything externally.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/home/ayaka/wslcode/aapl-research")
OUT = ROOT / "outputs/multiscale"
TOOLS = Path(__file__).resolve().parent
RUN = OUT / "remaining_continuations/20261108T031022909698Z"
EVIDENCE = OUT / "completed_core_evidence/20261108_approved_continuation"
PARENT = OUT / "private_working_manuscripts/20261107T211544188288Z"
CONTROL = OUT / "completed_manuscript_runs/20261108_full_core"
DESTINATION = OUT / "completed_manuscripts/20261108_full_core"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def state(phase, **extra):
    save(CONTROL / "STATE.json", {"phase": phase, "at_utc": datetime.now(timezone.utc).isoformat(),
                                 "training_started": False, "signals_sent": False, **extra})


def command(arguments, label, timeout=300):
    state(label, command=[str(x) for x in arguments])
    result = subprocess.run([str(x) for x in arguments], cwd=ROOT, capture_output=True, text=True,
                            timeout=timeout, env=os.environ)
    (CONTROL / (label + ".log")).write_text(result.stdout + result.stderr, encoding="utf8")
    if result.returncode:
        raise RuntimeError(label + " failed; see " + str(CONTROL / (label + ".log")))


def resource_note():
    run_state = load(RUN / "STATE.json")
    assert run_state["phase"] == "completed" and all(j["phase"] == "completed" for j in run_state["jobs"].values())
    approval = load(RUN / "APPROVAL.json")
    manifest = load(run_state["manifest"])
    assert digest(run_state["manifest"]) == run_state["manifest_sha256"]
    # The controller allows 35,999 seconds within the approved 36,000-second envelope.
    budget_started_epoch = run_state["deadline_epoch"] - (run_state["approved_maximum_wall_seconds"] - 1)
    completed_write_epoch = (RUN / "STATE.json").stat().st_mtime
    assert 0 <= completed_write_epoch - budget_started_epoch <= approval["maximum_wall_seconds"]
    events = []
    with (RUN / "OBSERVED_TRANSITIONS.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            events.append(json.loads(line))
    maximum_running = max(sum(j["phase"] == "running" for j in e["jobs"].values()) for e in events)
    assert maximum_running <= 6
    note = {
        "status": "completed_core_compute_scope_and_elapsed_limitations_documented",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run": str(RUN), "approval_sha256": digest(RUN / "APPROVAL.json"),
        "manifest_sha256": run_state["manifest_sha256"], "final_state_sha256": digest(RUN / "STATE.json"),
        "approved_maximum_wall_seconds": approval["maximum_wall_seconds"],
        "configured_maximum_training_processes": manifest["maximum_workers"],
        "logical_cpus_per_training_process": manifest["threads_per_worker"],
        "training_slots": manifest["slots"], "analysis_and_control_cpus": manifest["control_cores"],
        "gpu_training": manifest["gpu_training"], "new_seeds": manifest["new_seeds"],
        "new_configurations": manifest["new_configurations"], "jobs_completed": len(run_state["jobs"]),
        "original_live_states_resumed": sum(j.get("mode") == "live_memory" for j in run_state["jobs"].values()),
        "original_unstarted_folds": sum(j["original_process"] is None for j in manifest["jobs"]),
        "maximum_simultaneously_running_jobs_observed": maximum_running,
        "observer_events_checked": len(events), "observer_events_sha256": digest(RUN / "OBSERVED_TRANSITIONS.jsonl"),
        "budget_start_epoch_from_deadline": budget_started_epoch,
        "completed_state_file_modified_epoch": completed_write_epoch,
        "continuation_wall_envelope_seconds": completed_write_epoch - budget_started_epoch,
        "continuation_last_heartbeat_elapsed_boottime_seconds": run_state["heartbeat_boottime"] -
             (run_state["deadline_boottime"] - (run_state["approved_maximum_wall_seconds"] - 1)),
        "scope": "Only the approved continuation; not total historical study duration or CPU time.",
        "exact_total_study_cpu_seconds": None, "exact_per_model_cpu_seconds": None,
        "elapsed_limitations": [
            "Some original monotonic elapsed receipts include a requested night suspension.",
            "The seven resumed original fold timers also include a later original-budget freeze.",
            "The waiting H5-return fold includes additional time until the H60-return slot was available.",
            "The compiler subtracts only the known night envelope, not every later suspension.",
            "Neither raw sums nor the compiler night-adjusted sums measure active CPU computation.",
            "Parallel stage durations overlap and are not the total study wall duration.",
            "The service file mtime is a recorded completion envelope, not an optimizer event timestamp.",
            "Process observations are sampled; CPU increments are lower bounds with missing endpoints."
        ], "training_started_by_this_service": False, "Q3_values_read": False
    }
    save(DESTINATION / "COMPUTE_ACCOUNTING_NOTE.json", note)
    (DESTINATION / "COMPUTE_ACCOUNTING_NOTE.md").write_text(
        "# Compute accounting for the completed core study\n\n"
        "The separately approved continuation finished the fifteen missing original folds. It preserved "
        "the original seed, model configurations, splits, epoch cap and stopping rules. The configured limit "
        "was six training processes with two logical CPUs each, no GPU training, no new seeds and no new "
        "configurations. Analysis and control shared CPUs 24 and 26.\n\n"
        f"The recorded continuation wall envelope was approximately {note['continuation_wall_envelope_seconds']/3600:.3f} "
        "hours, within the separately approved maximum of ten hours. This is the continuation's wall envelope, "
        "not total study wall time or CPU time.\n\n"
        "Original elapsed receipts remain unchanged. Some include a requested night pause; seven also include "
        "a later budget freeze, and the waiting H5-return worker includes a CPU-slot wait. The result compiler "
        "only subtracts the night pause. Its adjusted totals therefore must not be described as actual active "
        "compute. Raw stage envelopes overlap under parallel execution. Neither these totals nor surviving "
        "process CPU counters provide a complete per-model or total-study CPU cost.\n\n"
        "The JSON note binds this explanation to the approval, manifest, final state and sampled observations. "
        "The historical 85-fold budget endpoint and its copied checkpoints are retained separately.\n",
        encoding="utf8")
    return note


def main():
    assert os.sched_getaffinity(0) == {24, 26}
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert not CONTROL.exists(), "Inspect an existing manuscript run instead of duplicating it"
    CONTROL.mkdir(parents=True)
    state("waiting_for_verified_all100_evidence", evidence=str(EVIDENCE), run=str(RUN))
    while not (EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json").exists():
        if (EVIDENCE / "PIPELINE_ERROR.json").exists():
            raise RuntimeError("Full evidence pipeline needs attention; no complete manuscript generated")
        phase = load(RUN / "STATE.json")["phase"]
        if phase == "paused_preserved":
            state("training_paused_no_complete_manuscript", preserved_training_state=str(RUN / "STATE.json"))
            return
        time.sleep(10)
    receipt = load(EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json")
    assert receipt["status"] == "all100_evidence_ready_for_final_manuscript"
    assert receipt["completed_folds"] == 100 and receipt["complete_tasks"] == 20
    python = sys.executable
    command([python, TOOLS / "build_complete_manuscript.py", "--evidence", EVIDENCE,
             "--parent", PARENT, "--output", DESTINATION], "build_complete_manuscript")
    command([python, TOOLS / "plot_complete_task_errors.py", "--metrics", DESTINATION / "all_task_results/METRICS_POOLED.csv",
             "--output", DESTINATION / "figures"], "plot_all_twenty_tasks")
    command(["pandoc", "--from=markdown+tex_math_single_backslash+autolink_bare_uris", "--to=docx",
             "--standalone", "--resource-path=" + str(DESTINATION),
             "--output", DESTINATION / "MANUSCRIPT_RAW.docx", DESTINATION / "MANUSCRIPT_EN.md"], "create_raw_word")
    command([python, TOOLS / "format_complete_manuscript.py", DESTINATION / "MANUSCRIPT_RAW.docx",
             DESTINATION / "MANUSCRIPT_EN.docx"], "format_complete_word")
    command(["libreoffice", "-env:UserInstallation=file:///tmp/aapl_complete_manuscript_lo_20261108",
             "--headless", "--convert-to", "pdf", "--outdir", DESTINATION, DESTINATION / "MANUSCRIPT_EN.docx"], "render_complete_pdf")
    command([python, TOOLS / "verify_complete_manuscript.py", DESTINATION], "verify_complete_document")
    resource = resource_note()
    (DESTINATION / "previews").mkdir()
    command(["pdftoppm", "-png", "-scale-to", "1600", DESTINATION / "MANUSCRIPT_EN.pdf",
             DESTINATION / "previews/page"], "render_page_previews")
    sources = DESTINATION / "analysis_sources"
    sources.mkdir()
    for path in TOOLS.glob("*.py"):
        shutil.copy2(path, sources / path.name)
    doc = load(DESTINATION / "DOCUMENT_CONTENT_VERIFICATION.json")
    state("complete_manuscript_generated_pending_visual_and_release_audit", destination=str(DESTINATION),
          completed_folds=100, complete_tasks=20, pdf_pages=doc["checks"]["pdf_pages"],
          continuation_wall_envelope_seconds=resource["continuation_wall_envelope_seconds"],
          document_verification_sha256=digest(DESTINATION / "DOCUMENT_CONTENT_VERIFICATION.json"),
          goal_marked_complete=False, externally_published=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if CONTROL.exists():
            save(CONTROL / "ERROR.json", {"error": type(exc).__name__ + ": " + str(exc),
                 "at_utc": datetime.now(timezone.utc).isoformat(), "training_started": False, "signals_sent": False})
        raise
