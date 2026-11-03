"""Require matching performance evidence before starting an RV30 long run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(run_name, report_path):
    report = read(report_path)
    freeze = read(ROOT / "delivery/rv30_freeze.json")
    if report["base_signature"] != freeze["signature"]:
        raise ValueError("Performance evidence belongs to another research signature")
    if not report["throughput_measurement_passed"]:
        raise ValueError("Representative throughput measurement has not passed")
    for path, expected in report["evidence_files"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Performance evidence changed: " + path)
    setup = read(ROOT / report["benchmark_directory"] / "setup.json")
    summary = read(ROOT / report["benchmark_directory"] / "summary.json")
    if not summary["passed"] or setup["scope"] != "pre2023_throughput_only_no_formal_scores":
        raise ValueError("Invalid isolated throughput scope or result")
    if max(setup["partition"]["evaluation_dates"]) >= "2023-01-01":
        raise ValueError("Performance data extends into the outer evaluation")
    for path, expected in setup["source_files"].items():
        if digest(ROOT / path) != expected:
            raise ValueError("Measured fitting implementation changed: " + path)
    if setup["script_sha256"] != digest(ROOT / "scripts/benchmark_rv30_runtime.py"):
        raise ValueError("Benchmark implementation changed since measurement")
    if run_name not in report["qualified_existing_runs"]:
        if not report["new_experiment_ready"]:
            raise ValueError("New experiments are blocked pending: " + "; ".join(report["required_before_new_experiment"]))
    else:
        directory = ROOT / "artifacts/rv30" / freeze["signature"] / run_name
        existing = read(directory / "setup.json")
        if existing["signature"] != freeze["signature"] or existing["config"] != freeze["payload"]["config"]:
            raise ValueError("Resume is not the measured existing experiment")
    if not report.get("search_duration_scenarios"):
        raise ValueError("A search duration estimate is required before launch")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--performance-report", default="delivery/performance.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        report = check(args.run_name, ROOT / args.performance_report)
    except (OSError, KeyError, ValueError, TypeError) as error:
        print("Performance gate failed: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps({"performance_gate": "passed", "run_name": args.run_name,
        "scope": "resume_existing_signed_search" if args.run_name in report["qualified_existing_runs"] else "new_experiment",
        "note": "The finance CLI also verifies the complete freeze, dependencies, and exclusive driver lock."}), flush=True)
    if args.check_only:
        return 0
    return subprocess.call([sys.executable, "-m", "finance", "train", "--run-name", args.run_name], cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
