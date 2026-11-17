"""Freeze one reviewed 180-second synthetic-only window; never starts workers.

This script is source-only preparation until a human/root runs it after source
review. It imports no NumPy and reads no market/model payload.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from runtime_common import (BASE, MAIN_FILES, SPEC_FILES, MEMORY_BYTES, atomic_json, canonical_digest,
                            read_json, require, runtime_identity, set_environment, sha, utc_now)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--additional-source", type=Path, action="append", default=[])
    args = parser.parse_args()
    set_environment()
    target = BASE / "EXECUTION_PROTOCOL.json"
    require(not target.exists(), "Never replace a frozen protocol or reset its deadline")
    review_path = args.review_receipt.resolve()
    require(review_path.is_relative_to(BASE) and review_path.suffix == ".json", "Review receipt must be local small JSON")
    review = read_json(review_path)
    require(review.get("status") in ("approved_for_fixed_analytic_execution", "accepted_source_review_for_execution"),
            "An independent source review accepting this fixed execution is required")
    files = [BASE / name for name in MAIN_FILES + SPEC_FILES]
    for raw in args.additional_source:
        source = raw.resolve()
        require(source.is_relative_to(BASE) and source.suffix == ".py", "Extra source must be a local Python file")
        files.append(source)
    require(len({str(path) for path in files}) == len(files), "Duplicate frozen source")
    inputs = {str(path.resolve()): sha(path) for path in files}
    reviewed = review.get("source_sha256", {})
    require(all(reviewed.get(path) == identity for path, identity in inputs.items()),
            "Review must bind every final main/design/additional-source identity")
    require(str(review_path) not in inputs, "Review cannot be one of the reviewed sources")
    inputs[str(review_path)] = sha(review_path)
    spec = read_json(BASE / "DESIGN_SPEC.json")
    require(spec["status"] == "prepared_design_only_not_numerically_executed" and
            spec["execution_admitted"] is False and spec["risk"]["expected_per_h_estimator_arm_rows"] == 816,
            "Prepared design does not match the fixed analytic specification")
    runtime = runtime_identity()
    now = time.time()
    protocol = {"schema": "origin_noise_execution_v1", "execution_admitted": True,
                "created_utc": utc_now(), "started_epoch": now, "deadline_epoch": now + 180,
                "maximum_total_wall_seconds": 180, "final_reserve_seconds": 60,
                "deadline_may_reset": False, "resume_requires_same_protocol_sha256": True,
                "main_cpu": 24, "independent_cpu": 26, "analysis_cpu_union": [24, 26], "numeric_threads": 1,
                "memory_hard_limit_bytes": MEMORY_BYTES,
                "memory_enforcement": "512MiB_RLIMIT_AS_stricter_than_RSS_plus_supervisor_RSS_guard",
                "GPU": False, "paid_resources": False, "market_payloads_admitted": False,
                "Q3_read_allowed": False, "training_or_inference_admitted": False,
                "source_review": str(review_path), "inputs": inputs, "runtime": runtime,
                "design_spec_sha256": inputs[str(BASE / "DESIGN_SPEC.json")],
                "output_root": str(BASE / "run"), "lifecycle_required_before_main": True,
                "independent_replay_invoked_by_these_programs": True,
                "scientific_recipe_id": canonical_digest({"spec": inputs[str(BASE / "DESIGN_SPEC.json")],
                                                            "core": inputs[str(BASE / "analytic_core.py")]})}
    atomic_json(target, protocol, exclusive=True)
    print({"protocol": str(target), "sha256": sha(target), "deadline_epoch": protocol["deadline_epoch"],
           "worker_started": False}, flush=True)


if __name__ == "__main__":
    main()
