"""Resource and real-weight acceptance using dates strictly before first outer validation."""
import argparse
import copy
from pathlib import Path
import time

import numpy as np
import torch

from .data import load_market_data, quarterly_splits, training_dates
from .io import ROOT, read_json, write_json
from .models import NEURAL, Pretrained, verify_model_lock
from .training import fit_baseline, setup_runtime, synchronize, train_neural


def benchmark(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config)
    verify_model_lock(config)
    device = setup_runtime(config)
    split = quarterly_splits(config)[0]
    partition = training_dates(split["train_dates"], config)
    record = {"scope": "runtime and implementation acceptance; only dates before 2023Q1, no outer evaluation labels scored",
              "device": device, "results": {}}
    train_count_by_fold = []
    inference_count = 0
    for feature in config["feature_sets"]:
        data = load_market_data(ROOT / config["data"], feature_set=feature,
                                start=config["periods"]["development_start"],
                                end=split["train_dates"][-1])
        origins = data.samples(partition["early_fit"], config["origin_stride_minutes"])
        scale = data.scaler(partition["early_fit"])
        for name in sorted(NEURAL):
            started = time.perf_counter()
            run = train_neural(name, data, origins, None, scale, config, 2026,
                               output / feature / name, fixed_epochs=1, phase="benchmark")
            record["results"][f"{feature}/{name}"] = {"training_samples": len(origins), **run,
                                                            "wall_seconds": time.perf_counter() - started}
            write_json(output / "benchmark.json", record)
        # A fixed short tree fit permits a linear upper projection by trees and
        # training samples; this is an estimate, not a promised wall-clock bound.
        short = copy.deepcopy(config)
        short["training"]["lightgbm"]["n_estimators"] = 10
        directory = output / feature / "lightgbm"
        directory.mkdir(parents=True, exist_ok=True)
        run = fit_baseline("lightgbm", data, origins, scale, short, directory)
        record["results"][f"{feature}/lightgbm"] = {"training_samples": len(origins), "benchmark_trees": 10, **run}
        write_json(output / "benchmark.json", record)
        for name in ("kronos_small", "chronos2"):
            predictor = Pretrained(name, config, device)
            settings = config["pretrained"]["kronos" if name == "kronos_small" else "chronos2"]
            count = settings["batch_size"]
            selected = origins[-count:]
            warmup = predictor.predict(data, selected, 2026)
            synchronize(device)
            started = time.perf_counter()
            prediction = predictor.predict(data, selected, 2026)
            synchronize(device)
            elapsed = time.perf_counter() - started
            np.testing.assert_allclose(warmup, prediction, rtol=1e-6, atol=1e-9)
            # Alter every future row after the largest observed origin. A real
            # adapter must produce the same output without reading those rows.
            changed = copy.copy(data)
            changed.bars = data.bars.copy()
            changed.bars.iloc[selected[-1] + 1:, :4] *= 3
            changed.features = data.features.copy()
            changed.features[selected[-1] + 1:] = 123.0
            changed.returns = data.returns.copy()
            changed.returns[selected[-1] + 1:] = .5
            future_changed = predictor.predict(changed, selected, 2026)
            np.testing.assert_array_equal(prediction, future_changed)
            record["results"][f"{feature}/{name}"] = {"inference_samples": len(selected),
                "inference_seconds": elapsed, "ms_per_sample": elapsed / len(selected) * 1000,
                "finite_cumulative_predictions": bool(np.isfinite(prediction).all()),
                "repeatable_fixed_seed_and_batch": True, "future_rows_cannot_affect_forecast": True}
            write_json(output / "benchmark.json", record)
            del predictor
            if device == "cuda":
                torch.cuda.empty_cache()
    # Counts use calendar/window eligibility only, not outcome values.
    data = load_market_data(ROOT / config["data"], start=config["periods"]["development_start"],
                            end=config["periods"]["historical_confirmation_end"])
    for split in quarterly_splits(config):
        dates = training_dates(split["train_dates"], config)
        train_count_by_fold.append(sum(len(data.samples(dates[k], config["origin_stride_minutes"]))
                                       for k in ("early_fit", "final_fit")))
        inference_count += len(data.samples(dates["calibration"], config["origin_stride_minutes"])) + len(
            data.samples(split["eval_dates"], config["origin_stride_minutes"]))
    projected_neural = 0.0
    projected_trees = 0.0
    projected_pretrained = 0.0
    for feature in config["feature_sets"]:
        for name in NEURAL:
            r = record["results"][f"{feature}/{name}"]
            projected_neural += r["training_seconds"] / r["training_samples"] * sum(train_count_by_fold) * len(
                config["seeds"]) * config["training"]["max_epochs"]
        r = record["results"][f"{feature}/lightgbm"]
        projected_trees += r["training_seconds"] / r["training_samples"] * sum(train_count_by_fold) / 2 * (
            config["training"]["lightgbm"]["n_estimators"] / 10)
        for name in ("kronos_small", "chronos2"):
            r = record["results"][f"{feature}/{name}"]
            projected_pretrained += r["ms_per_sample"] / 1000 * inference_count * (len(config["seeds"]) if name == "kronos_small" else 1)
    total = projected_neural + projected_trees + projected_pretrained
    record["projection"] = {"neural_max_epoch_hours": projected_neural / 3600,
                            "lightgbm_linear_estimate_hours": projected_trees / 3600,
                            "pretrained_inference_hours": projected_pretrained / 3600,
                            "total_estimate_hours": total / 3600,
                            "budget_hours": config["runtime"]["max_validation_runtime_hours"],
                            "origin_stride_minutes": config["origin_stride_minutes"],
                            "limitation": "extrapolation from pre-validation data; excludes setup/verification and may under/overestimate"}
    write_json(output / "benchmark.json", record)
    print(record["projection"], flush=True)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aapl_v2.json")
    parser.add_argument("--output", default="artifacts/aapl_v2/benchmark")
    args = parser.parse_args()
    benchmark(read_json(args.config), args.output)
