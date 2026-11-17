"""Create and check one past-only CLI example for the new complete package."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from multiscale.matrix_executor import read_bars
from multiscale.inference import _cutoff_ns

ROOT = Path("/home/ayaka/wslcode/aapl-research")
ADMISSION = ROOT / "outputs/multiscale/admission/alpaca_20261107T052228573242Z"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.sched_getaffinity(0) == {24, 26}
    evidence, manuscript = args.evidence.resolve(), args.manuscript.resolve()
    release = load(evidence / "ALL100_EVIDENCE_RECEIPT.json")
    assert release["completed_folds"] == 100 and release["complete_tasks"] == 20
    package = evidence / "inference_package"
    registry = load(package / "REGISTRY.json")
    assert registry["core_comparison_complete"] and registry["five_fold_tasks"] == 20
    audit_path = evidence / "package_audit/PACKAGE_AUDIT.json"
    assert digest(audit_path) == release["artifacts"]["package_audit/PACKAGE_AUDIT.json"]
    audit = load(audit_path)
    selected = next(r["result"] for r in audit["results"] if r["task"] == "minute:60:future_log_return")
    expected = selected["example"]
    assert expected["selected_fold"] == 5 and expected["core_comparison_complete"]
    decision = expected["decision_ns"]
    assert decision < _cutoff_ns()
    admission = load(ADMISSION / "admission.json")
    assert digest(ADMISSION / "bars_minute.npz") == admission["files"]["bars_minute.npz"]
    bars = read_bars(ADMISSION, admission, "minute")
    origin = int(np.searchsorted(bars.end_ns, decision))
    assert int(bars.end_ns[origin]) == decision and origin >= 479
    part = slice(origin - 479, origin + 1)
    assert bool(np.all(bars.observed[part]))
    folder = manuscript / "package_example"
    folder.mkdir(exist_ok=False)
    history = folder / "H60_RETURN_PAST_480_BARS.npz"
    np.savez_compressed(history, frequency=np.asarray("minute"), start_ns=bars.start_ns[part],
                        end_ns=bars.end_ns[part], close=bars.close[part], observed=bars.observed[part],
                        session_date=bars.session_date[part])
    identity = "saved-original-fold5-past-only-input:" + digest(history)
    command = [sys.executable, "-m", "multiscale.study_package", "forecast", "--package", str(package),
               "--registry-sha256", digest(package / "REGISTRY.json"), "--bars", str(history),
               "--profile", "minute", "--horizon", "60", "--target", "future_log_return",
               "--model", "xpatch", "--as-of-ns", str(decision), "--history-identity", identity, "--device", "cpu"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=120, env=os.environ)
    (folder / "CLI_STDERR.txt").write_text(result.stderr, encoding="utf8")
    if result.returncode:
        raise RuntimeError("Full-package CLI example failed: " + result.stderr[-2000:])
    actual = json.loads(result.stdout)
    assert actual["core_comparison_complete"] and actual["selected_fold"] == 5 and actual["independent_test"] is False
    assert actual["decision_ns"] == decision
    a, b = np.asarray(actual["prediction"]), np.asarray(expected["prediction"])
    assert a.shape == b.shape == (60,) and np.isfinite(a).all()
    assert np.array_equal(a, b)
    output = folder / "H60_RETURN_XPATCH_FORECAST.json"
    output.write_text(json.dumps(actual, indent=2, allow_nan=False) + "\n", encoding="utf8")
    (folder / "RUN_EXAMPLE.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n" + "cd " + shlex.quote(str(ROOT)) + "\n" +
        "export CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2\n" +
        "taskset -c 24,26 " + shlex.join(command) + "\n", encoding="utf8")
    receipt = {"status": "complete_package_past_only_cli_forecast_verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "script_sha256": digest(__file__), "package_registry_sha256": digest(package / "REGISTRY.json"),
               "source_package_audit_sha256": digest(audit_path), "input_sha256": digest(history),
               "forecast_sha256": digest(output), "input_bars": 480, "output_path_steps": 60,
               "decision_ns": decision, "selected_fold": 5, "core_comparison_complete": True,
               "fresh_cli_process": True, "CLI_matches_checked_API_exactly": True,
               "max_absolute_difference": float(np.max(np.abs(a-b))), "saved_target_arrays_opened": False,
               "new_training_started": False, "Q3_values_read": False,
               "scope": "Historical inference example at one original validation origin; not an investment recommendation or future-data evaluation."}
    (folder / "EXAMPLE_VERIFICATION.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
