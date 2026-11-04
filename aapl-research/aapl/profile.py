"""G3 bounded end-to-end pilot; only 2021-2022 labels, no model selection scores."""
import time
PROCESS_START = time.perf_counter()

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from .common import ROOT, digest, runtime, sync, utc_now, write_json
from .data import Market, development_splits
from .models import chronos_pipeline, chronos_predict, neural
from .training import Trainer, neural_predict, source_identity
from .resources import ResourceMonitor
from . import baselines


def config_for(name, lookback=240):
    return dict(model=name, lookback=lookback, patch=16 if name == "xpatch" else 8,
                stride=8 if name == "xpatch" else 4, alpha=0.7, revin=True,
                layers=3, d_model=64, dropout=0.2, loss="cumulative12_mse", lr=1e-4 if name.startswith("chronos") else 0.001,
                weight_decay=0.01, batch_size=16 if name.startswith("chronos") else 64,
                rank=16, point="native_median", representation="minute_return",
                alpha_ridge=0.0001, num_leaves=31, max_depth=-1, min_child_samples=100,
                learning_rate=0.01, reg_lambda=0, colsample_bytree=1.0, n_estimators=1000)


def timed(fn):
    sync()
    t = time.perf_counter()
    value = fn()
    sync()
    return value, time.perf_counter() - t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["ridge", "lightgbm", "xpatch", "patchtst", "chronos2_frozen", "chronos2_lora"])
    parser.add_argument("--mode", choices=["small", "maximum"], required=True)
    parser.add_argument("--sustain-seconds", type=float, default=0)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--tag", default="")
    parser.add_argument('--optimized', action='store_true')
    args = parser.parse_args()
    device = runtime()
    name = f"{args.model}_{args.mode}{args.tag}"
    directory = ROOT / "outputs/performance" / name
    directory.mkdir(parents=True, exist_ok=False)
    record = {"model": args.model, "mode": args.mode, "started_at_utc": utc_now(),
              'implementation_sha256': source_identity(), 'implementation_plan_sha256': digest(ROOT / 'IMPLEMENTATION_PLAN.md'),
              "source_label_scope": "2021-2022 only", "outer_scores_computed": False,
              "phases_seconds": {}, "config": config_for(args.model, args.lookback)}
    cfg = record["config"]
    if args.optimized:
        from .common import read_json
        cfg.update(read_json(ROOT / 'outputs/optimization_admission.json')['adopted'].get(args.model, {}))
    with ResourceMonitor() as monitor:
        try:
            market, seconds = timed(Market.load)
            record["phases_seconds"]["read_validate_transform"] = seconds
            dates = sorted(d for d in set(market.dates) if d <= "2022-12-30")
            fit_dates = dates[:20] if args.mode == "small" else dates[:-21]
            val_dates = dates[21:24] if args.mode == "small" else dates[-20:]
            train, val = market.select(fit_dates), market.select(val_dates)
            scale = market.scale(fit_dates)
            max_count = len(market.select(development_splits(market)['final_fit_dates']))
            if args.mode == "maximum":
                # Performance-only repeated training origins emulate the planned largest N without later labels.
                train = np.resize(train, max_count)
            record.update(training_samples=len(train), unique_source_training_samples=len(set(train)),
                          validation_samples=len(val), max_planned_training_samples=max_count,
                          duplicated_origins_for_performance_only=args.mode == "maximum")
            monitor.check()
            if args.model in ("ridge", "lightgbm"):
                if args.model == "ridge":
                    cfg["alpha"] = cfg["alpha_ridge"]
                    fn = lambda: baselines.fit_ridge(market, train, cfg, scale)
                else:
                    # Full tree cap, no early stop: upper cost envelope rather than pilot accuracy selection.
                    fn = lambda: baselines.fit_lightgbm(market, train, cfg, scale)
                runs = 4 if args.mode == "small" else 1
                costs = []
                for j in range(runs):
                    model, seconds = timed(fn)
                    costs.append(seconds)
                    monitor.check()
                record["fit_wall_seconds"] = costs
                pred, seconds = timed(lambda: baselines.predict(model, market, val, cfg, scale))
                record["phases_seconds"]["validation_inference"] = seconds
                t = time.perf_counter()
                if args.model == "ridge":
                    np.savez(directory / "ridge.npz", coef=model.coef_, intercept=model.intercept_)
                    with np.load(directory / "ridge.npz") as saved:
                        recovered = (baselines.design(market, val, cfg, scale) @ saved["coef"] + saved["intercept"]) * scale["target_std"]
                else:
                    import lightgbm as lgb
                    model.booster_.save_model(str(directory / "lightgbm.txt"))
                    recovered = lgb.Booster(model_file=str(directory / "lightgbm.txt")).predict(
                        baselines.design(market, val, cfg, scale), num_threads=4) * scale["target_std"]
                np.testing.assert_allclose(pred, recovered, rtol=1e-10, atol=1e-12)
                record["phases_seconds"]["save_reload_verify"] = time.perf_counter() - t
                record["saved_model_prediction_verified"] = True
                record["total_trees"] = cfg["n_estimators"] if args.model == "lightgbm" else None
            else:
                if args.model.startswith("chronos"):
                    (pipe, weight), seconds = timed(lambda: chronos_pipeline(args.model.endswith("lora"), rank=16, device=device))
                    record["weight_identity"] = weight
                    model = pipe.model
                    predict = lambda ids: chronos_predict(pipe, market, ids, cfg, batch_size=32)
                else:
                    model, seconds = timed(lambda: neural(cfg).to(device))
                    predict = lambda ids: neural_predict(model, cfg, market, ids, scale, device)
                record["phases_seconds"]["model_load"] = seconds
                if args.model != "chronos2_frozen":
                    trainer = Trainer(model, cfg, market, train, scale, device, 2026,
                                      900 if args.model == "chronos2_lora" else 27,
                                      lora=args.model == "chronos2_lora")
                    costs = []
                    train_started = time.perf_counter()
                    if args.model == "chronos2_lora":
                        for stop in (100, 300, 900):
                            _, seconds = timed(lambda: trainer.train_updates(stop))
                            costs.append({"cumulative_updates": stop, "increment_wall_seconds": seconds})
                            monitor.check()
                            trainer.save(directory / "latest.pt")
                    else:
                        minimum_epochs = 4 if args.mode == "small" else 1
                        while len(costs) < minimum_epochs or time.perf_counter() - train_started < args.sustain_seconds:
                            _, seconds = timed(trainer.train_epoch)
                            costs.append({"epoch": trainer.epoch, "epoch_wall_seconds": seconds})
                            monitor.check()
                            if len(costs) % 5 == 0:
                                write_json(directory / "progress.json", {"epoch": trainer.epoch,
                                           "elapsed_train_seconds": time.perf_counter() - train_started,
                                           "last_epoch_seconds": seconds})
                    record["training_runs"] = costs
                    record["training_wall_seconds"] = time.perf_counter() - train_started
                    record["updates"] = trainer.updates
                    record["batch_wall_p50_seconds"] = float(np.median(trainer.batch_seconds))
                    record["batch_wall_p95_seconds"] = float(np.quantile(trainer.batch_seconds, 0.95))
                    # Save/reload includes optimizer/RNG, with a signature and payload hash.
                    _, seconds = timed(lambda: trainer.save(directory / "latest.pt"))
                    record["phases_seconds"]["save_checkpoint"] = seconds
                    _, seconds = timed(lambda: trainer.restore(directory / "latest.pt"))
                    record["phases_seconds"]["reload_checkpoint"] = seconds
                # Three warm measurements after the first evaluation, without computing predictive MSE.
                inference = []
                repeats = 4 if args.mode == "small" else 1
                for repeat in range(repeats):
                    prediction, seconds = timed(lambda: predict(val))
                    inference.append(seconds)
                    monitor.check()
                record["validation_inference_wall_seconds"] = inference
                np.savez_compressed(directory / "predictions_training_dates_only.npz", origins=val, prediction=prediction)
                record["prediction_finite"] = bool(np.isfinite(prediction).all())
                if args.model != "chronos2_frozen":
                    before = predict(val[:64])
                    trainer.restore(directory / "latest.pt")
                    np.testing.assert_array_equal(before, predict(val[:64]))
                    record["saved_model_prediction_verified"] = True
                else:
                    record["saved_model_prediction_verified"] = "immutable_checkpoint_sha256_verified"
                record["torch_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
                record["torch_peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            monitor.check()
            record["passed"] = True
        except Exception as exc:
            record.update(passed=False, error=repr(exc))
            raise
        finally:
            record.update(completed_at_utc=utc_now(), end_to_end_wall_seconds=time.perf_counter() - PROCESS_START,
                          resources=monitor.summary())
            write_json(directory / "result.json", record)
            write_json(directory / "resource_samples.json", monitor.rows)
            print(json.dumps({k: record.get(k) for k in ("model", "mode", "passed", "error", "training_samples", "end_to_end_wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
