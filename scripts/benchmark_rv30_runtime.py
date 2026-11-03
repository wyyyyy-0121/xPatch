"""Isolated, pre-2023 throughput evidence; never reads formal prediction scores.

Run only with the formal driver stopped. This does not amend the signed fit code.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=False)
    from finance.io import read_json, sha, write_json
    from finance.runtime import configure, configure_torch, exclusive_run
    configure(4, list(range(22)))
    with exclusive_run(ROOT / "artifacts/rv30/driver.lock"):
        # Acquire the driver lock before importing numerical libraries or CUDA.
        import numpy as np
        import torch
        from finance.cache import load_data
        from finance.features import FitState, FeatureStore
        from finance import models
        from finance.splits import partition
        configure_torch()
        frozen = read_json(ROOT / "delivery/rv30_freeze.json")
        config = frozen["payload"]["config"]
        data = load_data(ROOT / frozen["payload"]["cache_directory"], frozen["payload"]["cache_identity"])
        days = [str(day) for day in data.days if day < "2023-01-01"]
        split = partition("pre2023_throughput", days[:-20], days[-20:])
        assert max(split.evaluation_dates) < "2023-01-01"
        setup_start = time.monotonic()
        state = FitState.fit(data, split.fit_dates, config["seasonality"])
        store = FeatureStore(data, config)
        origins = data.origins(split.fit_dates, stride=config["sampling"]["training_stride_minutes"])
        calibration = data.origins(split.calibration_dates, stride=config["sampling"]["calibration_stride_minutes"])
        evaluation = data.origins(split.evaluation_dates, stride=config["sampling"]["inner_validation_stride_minutes"])
        trials = read_json(ROOT / "configs/rv30_search.json")["models"]["xpatch"]
        setup = {"created_utc": datetime.now(timezone.utc).isoformat(), "base_signature": frozen["signature"],
            "scope": "pre2023_throughput_only_no_formal_scores", "partition": split.dictionary(),
            "training_origins": len(origins), "calibration_origins": len(calibration), "evaluation_origins": len(evaluation),
            "fit_state_setup_seconds": time.monotonic() - setup_start,
            "source_files": {p: sha(ROOT / p) for p in ("finance/models.py", "finance/runtime.py", "layers/dema.py", "layers/ema.py")},
            "script_sha256": sha(__file__), "cpu_limit": 22, "benchmark_epoch_count": 1,
            "note": "All eligible pre-2023 samples; one full epoch is a throughput test, not HPO validation."}
        write_json(output / "setup.json", setup)
        cases = []
        # Fixed model dimensions isolate decomposition and thread overhead.
        template = next(t["params"] for t in trials if t["params"]["seq_len"] == 512 and t["params"]["batch_size"] == 128)
        for panel in ("aapl", "market"):
            for length in (64, 512):
                for mode in ("ema", "dema"):
                    cases.append((panel, length, mode, 4))
        cases.extend([("market", 512, "dema", 18), ("market", 512, "dema", 4)])
        records = []
        for index, (panel, length, mode, threads) in enumerate(cases):
            os.sched_setaffinity(0, list(range(threads)))
            torch.set_num_threads(threads)
            params = {**template, "seq_len": length, "ma_type": mode}
            identity = f"{index:02d}_{panel}_{mode}_l{length}_t{threads}"
            timings = defaultdict(float)
            original_inputs, original_labels = models.sequence_inputs, models.sequence_labels
            def timed_input(*a, **kw):
                start = time.monotonic()
                value = original_inputs(*a, **kw)
                timings["sequence_inputs_seconds"] += time.monotonic() - start
                return value
            def timed_labels(*a, **kw):
                start = time.monotonic()
                value = original_labels(*a, **kw)
                timings["sequence_labels_seconds"] += time.monotonic() - start
                return value
            models.sequence_inputs, models.sequence_labels = timed_input, timed_labels
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            try:
                model, fitting = models.fit_neural(data, store, state, split, origins, params, panel,
                    config, 2026, output / identity, max_epochs=1, fixed_epochs=1)
                torch.cuda.synchronize()
                timings["fit_one_epoch_seconds"] = time.monotonic() - start
                for label, selected in (("calibration", calibration), ("evaluation", evaluation)):
                    start = time.monotonic()
                    values = models.raw_predictions("xpatch", model, data, store, state, selected, params, panel, config, "cuda")
                    torch.cuda.synchronize()
                    if not np.isfinite(values).all():
                        raise ValueError("Nonfinite benchmark output")
                    timings[label + "_seconds"] = time.monotonic() - start
                row = {"case": identity, "panel": panel, "ma_type": mode, "seq_len": length, "threads": threads,
                    "params": params, "fitting": fitting, "timings": dict(timings),
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(), "passed": True}
                records.append(row)
                write_json(output / "progress.json", {"cases": records, "total_cases": len(cases)})
                print(json.dumps({"case": identity, "timings": dict(timings)}), flush=True)
            finally:
                models.sequence_inputs, models.sequence_labels = original_inputs, original_labels
            del model
            gc.collect()
            torch.cuda.empty_cache()
        # Profile a fixed full-size batch without changing the persisted models.
        os.sched_setaffinity(0, list(range(4)))
        torch.set_num_threads(4)
        profiles = []
        for mode in ("ema", "dema"):
            params = {**template, "seq_len": 512, "ma_type": mode}
            models.seed_torch(2026)
            model = models.native_xpatch(params, 3, "cuda")
            selected = origins[-128:]
            x = torch.as_tensor(models.sequence_inputs(data, state, selected, 512, params["representation"], "market", config), device="cuda")
            y = torch.as_tensor(models.sequence_labels(data, state, selected, params["representation"], "market", config), device="cuda")
            # Warm up, then record CPU launch overhead and CUDA operator activity.
            model(x).square().mean().backward()
            torch.cuda.synchronize()
            model.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                loss = torch.nn.functional.mse_loss(model(x), y)
                loss.backward()
                torch.cuda.synchronize()
            events = profile.key_averages()
            entries = [{"name": e.key, "count": e.count, "self_cpu_us": e.self_cpu_time_total,
                "self_device_us": e.self_device_time_total} for e in events]
            entries.sort(key=lambda r: r["self_cpu_us"], reverse=True)
            profiles.append({"mode": mode, "operators": entries,
                "cpu_total_us": sum(e.self_cpu_time_total for e in events),
                "device_total_us": sum(e.self_device_time_total for e in events)})
            write_json(output / "operator_profile.json", profiles)
            del model, x, y, loss, profile
            gc.collect()
            torch.cuda.empty_cache()
        summary = {"passed": len(records) == len(cases) and all(r["passed"] for r in records),
            "scope": setup["scope"], "base_signature": frozen["signature"], "cases": records,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "limitations": ["One full epoch per case; no validation early stopping or full search benchmark.",
                "Pre-2023 training history is smaller than late outer folds; extrapolation must scale sample counts.",
                "This evidence alone does not authorize a changed training implementation or search space."]}
        write_json(output / "summary.json", summary)
        print("Benchmark completed", flush=True)


if __name__ == "__main__":
    main()
