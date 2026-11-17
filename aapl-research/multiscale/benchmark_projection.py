"""Make a clearly nonbinding runtime illustration from synthetic measurements.

The projection excludes actual-source I/O and several matrix costs. It is not a
resource authorization, a convergence claim, or a cloud-speed estimate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import numpy as np

from .bars import file_digest
from .matrix_gate import _representative_benchmark
from .spec import OUT


NEURAL = ("mlp", "gru", "lstm", "xpatch")


def project(plan, benchmark):
    if (plan.get("status") != "draft_blocked_not_runnable" or plan.get("frozen") is not False or
            plan.get("market_price_arrays_opened") is not False or plan.get("experiment_scored") is not False):
        raise ValueError("Expected an unscored candidate matrix plan")
    if (benchmark.get("status") != "completed_engineering_only" or benchmark.get("formal_experiment") is not False or
            benchmark.get("market_data_read") is not False or benchmark.get("all_shapes_measured") is not True):
        raise ValueError("Expected a complete synthetic engineering benchmark")
    grouped = defaultdict(list)
    for record in benchmark["records"]:
        grouped[(record["profile"], record["model"])].append(record)
    estimates = {}
    for quantile, label in ((.5, "median"), (.9, "p90")):
        training = defaultdict(float)
        prediction = defaultdict(float)
        for job in plan["jobs"]:
            for model in NEURAL:
                records = grouped[(job["profile"], model)]
                if not records:
                    raise ValueError("A candidate shape is missing from the synthetic benchmark")
                per_update = [r["training_seconds_including_scaler_stop_and_checkpoints"] / r["steps"] for r in records]
                per_validation = [(r["prediction_seconds"] + r["compressed_storage_and_hash_seconds"]) /
                                  r["validation_origins"] for r in records]
                training[job["profile"]] += math.ceil(job["fit_origins"] / 128) * 20 * float(np.quantile(per_update, quantile))
                prediction[job["profile"]] += job["validation_origins"] * float(np.quantile(per_validation, quantile))
        estimates[label] = {"neural_training_seconds_by_profile": dict(training),
                            "neural_prediction_and_synthetic_storage_seconds_by_profile": dict(prediction),
                            "neural_training_seconds": sum(training.values()),
                            "neural_prediction_and_synthetic_storage_seconds": sum(prediction.values()),
                            "with_30_percent_arithmetic_margin_seconds": 1.3 * (sum(training.values()) + sum(prediction.values()))}
    result = {"status": "illustration_synthetic_only_not_budget_admission", "frozen": False,
            "independent_test": False, "experiment_scored": False, "market_data_read": False,
            "candidate_folds": len(plan["jobs"]), "models_per_fold": len(NEURAL), "seed_count": 1,
            "max_epochs_assumed": 20, "batch_size_assumed": 128,
            "method": "For each candidate fold and neural model, multiply ceil(fit origins / 128) * 20 by the measured per-update duration for its profile; add measured synthetic prediction and compressed-write duration scaled by validation origins. Multiply the subtotal by 1.3 as a planning illustration.",
            "quantiles": estimates,
            "limitations": ["All work uses synthetic prices on one local laptop; the estimate does not predict a cloud GPU.",
                            "Minute H60 measurements are applied to H5/H15/H30 as well as H60; size scaling and output-head costs are approximate.",
                            "Early stopping is ignored; multiplying an entire one-epoch measurement also repeats setup and scaler costs that may not repeat in a real run.",
                            "The estimate excludes real-source reads and preparation, baseline and Ridge computations, orchestration, scoring, verification, failed attempts, shared-horizon experiments and extra seeds.",
                            "Synthetic price variability and compression may not represent actual market data, and a 30% multiplier does not prove sufficient headroom.",
                            "Some candidate folds are blocked or have zero fit origins; an admissible long-history source and revised folds change these counts.",
                            "Quarterly, annual, daily and hourly shapes were measured once per model, not repeatedly.",
                            "This is neither a guaranteed upper nor lower bound and cannot authorize training or spending."]}
    if not benchmark.get("source_sha256_at_start") or not benchmark.get("source_sha256_at_end"):
        result["limitations"].append("The measurement did not record source hashes at launch; a later snapshot is not runtime attestation.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    args = parser.parse_args()
    plan_path, benchmark_path = args.plan.resolve(), args.benchmark.resolve()
    if (not plan_path.is_relative_to((OUT / "matrix_plans").resolve()) or
            not benchmark_path.is_relative_to((OUT / "synthetic_matrix_benchmark").resolve())):
        raise ValueError("Projection inputs must be in the isolated multiscale workspace")
    plan = json.loads(plan_path.read_text(encoding="utf8"))
    benchmark = json.loads(benchmark_path.read_text(encoding="utf8"))
    evidence = {"path": str(benchmark_path), "sha256": file_digest(benchmark_path)}
    if not _representative_benchmark(evidence, OUT, benchmark.get("device"), require_runtime_source=False):
        raise ValueError("Synthetic benchmark does not meet the stated size and duration checks")
    result = project(plan, benchmark)
    result.update(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=file_digest(plan_path), benchmark_sha256=file_digest(benchmark_path),
                  measured_device=benchmark["device"])
    directory = OUT / "benchmark_projections"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    hours = {name: round(data["with_30_percent_arithmetic_margin_seconds"] / 3600, 2)
             for name, data in result["quantiles"].items()}
    print(json.dumps({"status": result["status"], "path": str(path), "illustrative_hours": hours}))


if __name__ == "__main__":
    main()
