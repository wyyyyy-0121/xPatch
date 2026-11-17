"""Check the final package CLI after complete evidence and supplement exist.

This one-shot dependency job never trains, signals workers, changes a budget,
submits a document, or marks the goal complete.
"""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/ayaka/wslcode/aapl-research")
BASE = ROOT / "outputs/multiscale"
TOOLS = Path(__file__).resolve().parent
RUN = BASE / "remaining_continuations/20261108T040346048516Z-parallel8"
EVIDENCE = BASE / "completed_core_evidence/20261108_parallel8_continuation"
SUPPLEMENT = BASE / "completed_manuscripts/20261108_parallel8_full_core"
SUPPLEMENT_CONTROL = BASE / "completed_manuscript_runs/20261108_parallel8_full_core"
CONTROL = BASE / "completed_package_example_runs/20261108_parallel8_full_core"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(phase, **extra):
    value = {"phase": phase, "at_utc": datetime.now(timezone.utc).isoformat(),
             "training_started": False, "signals_sent": False, "Q3_values_read": False, **extra}
    path = CONTROL / "STATE.json"
    pending = CONTROL / "STATE.pending.json"
    pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")
    pending.replace(path)


def main():
    assert os.sched_getaffinity(0) == {24, 26} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert not CONTROL.exists(), "Inspect the existing example job instead of duplicating it"
    CONTROL.mkdir(parents=True)
    state("waiting_for_complete_evidence_and_supplement", run=str(RUN), evidence=str(EVIDENCE))
    while not (SUPPLEMENT / "DOCUMENT_CONTENT_VERIFICATION.json").exists():
        if (EVIDENCE / "PIPELINE_ERROR.json").exists() or (SUPPLEMENT_CONTROL / "ERROR.json").exists():
            raise RuntimeError("Evidence or supplement needs attention; no final example claimed")
        if read(RUN / "STATE.json")["phase"] == "paused_preserved":
            state("training_paused_no_final_example_claim", training_state=str(RUN / "STATE.json"))
            return
        time.sleep(10)
    full = read(EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json")
    assert full["completed_folds"] == 100 and full["complete_tasks"] == 20 and not full["Q3_values_read"]
    folder = SUPPLEMENT / "package_example"
    assert not folder.exists(), "Preserve any existing example and inspect its receipt"
    arguments = [sys.executable, str(TOOLS / "prepare_complete_package_example.py"),
                 "--evidence", str(EVIDENCE), "--manuscript", str(SUPPLEMENT)]
    state("checking_complete_package_cli_example", command=arguments)
    result = subprocess.run(arguments, cwd=ROOT, capture_output=True, text=True, env=os.environ, timeout=300)
    (CONTROL / "CLI_EXAMPLE.log").write_text(result.stdout + result.stderr, encoding="utf8")
    if result.returncode:
        raise RuntimeError("Complete-package CLI example failed; inspect CLI_EXAMPLE.log")
    receipt = read(folder / "EXAMPLE_VERIFICATION.json")
    assert receipt["core_comparison_complete"] and receipt["CLI_matches_checked_API_exactly"]
    assert receipt["selected_fold"] == 5 and not receipt["new_training_started"] and not receipt["Q3_values_read"]
    assert sha(folder / "H60_RETURN_PAST_480_BARS.npz") == receipt["input_sha256"]
    assert sha(folder / "H60_RETURN_XPATCH_FORECAST.json") == receipt["forecast_sha256"]
    state("complete_package_cli_example_verified", example=str(folder),
          example_receipt_sha256=sha(folder / "EXAMPLE_VERIFICATION.json"),
          script_sha256=sha(__file__), goal_marked_complete=False)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if CONTROL.exists():
            state("example_check_needs_attention", error=type(exc).__name__ + ": " + str(exc))
        raise
