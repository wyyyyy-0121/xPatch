"""Wait for complete checked evidence, then create the user's two-column paper.

This service only prepares documents for visual review. It cannot train, signal
workers, extend the compute budget, publish externally or complete the goal.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import shutil
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
OUTPUT = BASE / "double_column_manuscripts/20261108_parallel8_full_core"
CONTROL = BASE / "double_column_manuscript_runs/20261108_parallel8_full_core"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state(phase, **extra):
    value = {"phase": phase, "at_utc": datetime.now(timezone.utc).isoformat(),
             "training_started": False, "signals_sent": False, **extra}
    (CONTROL / "STATE.json").write_text(json.dumps(value, indent=2)+"\n")


def command(arguments, name):
    state(name, command=[str(x) for x in arguments])
    result = subprocess.run([str(x) for x in arguments], cwd=ROOT, capture_output=True, text=True,
                            timeout=600, env=os.environ)
    (CONTROL / (name+".log")).write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(name+" failed; see "+str(CONTROL/(name+".log")))


def main():
    assert os.sched_getaffinity(0) == {24, 26} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert not CONTROL.exists(), "Inspect the existing compact-paper run before repeating work"
    CONTROL.mkdir(parents=True)
    state("waiting_for_complete_evidence_and_supplement", run=str(RUN), supplement=str(SUPPLEMENT),
          author="Wang Yang", email_completion="author_will_fill")
    while not (SUPPLEMENT / "DOCUMENT_CONTENT_VERIFICATION.json").exists():
        if (EVIDENCE / "PIPELINE_ERROR.json").exists() or (SUPPLEMENT_CONTROL / "ERROR.json").exists():
            raise RuntimeError("Evidence or supplementary report needs attention")
        if read(RUN / "STATE.json")["phase"] == "paused_preserved":
            state("training_paused_no_complete_paper_claim", training_state=str(RUN/"STATE.json"))
            return
        time.sleep(10)
    evidence = read(EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json")
    assert evidence["completed_folds"] == 100 and evidence["complete_tasks"] == 20
    python = sys.executable
    command([python, TOOLS/"build_short_paper.py", "--results", EVIDENCE/"results", "--evidence", EVIDENCE,
             "--supplement", SUPPLEMENT, "--output", OUTPUT], "build_main_paper")
    command([python, TOOLS/"plot_short_paper.py", OUTPUT], "plot_all_twenty_tasks")
    command(["pandoc", "--from=markdown+tex_math_single_backslash+autolink_bare_uris", "--to=docx", "--standalone",
             "--resource-path="+str(OUTPUT), "--output", OUTPUT/"MAIN_PAPER_RAW.docx", OUTPUT/"MAIN_PAPER_EN.md"], "create_raw_word")
    command([python, TOOLS/"format_short_paper.py", OUTPUT/"MAIN_PAPER_RAW.docx", OUTPUT/"MAIN_PAPER_EN.docx"], "format_double_columns")
    command(["libreoffice", "-env:UserInstallation=file:///tmp/aapl_short_paper_complete_parallel8_20261108",
             "--headless", "--convert-to", "pdf", "--outdir", OUTPUT, OUTPUT/"MAIN_PAPER_EN.docx"], "render_main_pdf")
    command([python, TOOLS/"verify_short_paper.py", OUTPUT], "verify_main_content_and_page_count")
    (OUTPUT/"previews").mkdir()
    command(["pdftoppm", "-png", "-scale-to", "1600", OUTPUT/"MAIN_PAPER_EN.pdf", OUTPUT/"previews/page"], "render_all_main_pages")
    source = OUTPUT/"analysis_sources"
    source.mkdir()
    for name in ("build_short_paper.py", "plot_short_paper.py", "format_short_paper.py", "verify_short_paper.py", "finish_short_paper_parallel8.py"):
        shutil.copy2(TOOLS/name, source/name)
    verification = read(OUTPUT/"MAIN_PAPER_CONTENT_VERIFICATION.json")
    state("double_column_paper_generated_pending_visual_review", output=str(OUTPUT), supplement=str(SUPPLEMENT),
          pdf_pages=verification["pdf_pages"], completed_folds=100, complete_tasks=20,
          author="Wang Yang", email_completion="author_will_fill",
          main_content_verification_sha256=sha(OUTPUT/"MAIN_PAPER_CONTENT_VERIFICATION.json"),
          externally_submitted=False, goal_marked_complete=False)
    (BASE/"LATEST_DOUBLE_COLUMN_PAPER.txt").write_text(str(OUTPUT)+"\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if CONTROL.exists():
            (CONTROL/"ERROR.json").write_text(json.dumps({"error":type(exc).__name__+": "+str(exc),
                "at_utc":datetime.now(timezone.utc).isoformat(), "training_started":False, "signals_sent":False}, indent=2)+"\n")
        raise
