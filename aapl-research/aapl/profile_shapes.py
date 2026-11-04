"""Time all declared supervised trial shapes, without scoring or ranking predictions."""
import gc
import time

import numpy as np
import torch

from .common import ROOT, runtime, utc_now, write_json, read_json
from .data import Market
from .models import neural
from .search_space import generate
from .training import Trainer, neural_predict
from .resources import ResourceMonitor


def main():
    started = time.perf_counter()
    device = runtime()
    market = Market.load()
    days = sorted(set(market.dates))[:24]
    fit, val = market.select(days[:20]), market.select(days[21:24])
    scale = market.scale(days[:20])
    destination = ROOT / "outputs/shape_profiles"
    destination.mkdir(parents=True, exist_ok=False)
    records = {}
    with ResourceMonitor() as monitor:
        for name in ("xpatch", "patchtst"):
            records[name] = []
            options = read_json(ROOT / 'outputs/optimization_admission.json')['adopted'][name]
            trials = [{**c, **options} for c in generate()[name]]
            # Include the exact maximum pressure-test geometry as a scale calibration anchor.
            from .profile import config_for
            trials = trials + [{**config_for(name), **options, "trial_id": "pressure_reference"}]
            for config in trials:
                runtime()
                t = time.perf_counter()
                model = neural(config).to(device)
                load_seconds = time.perf_counter() - t
                trainer = Trainer(model, config, market, fit, scale, device, 2026, 27)
                epochs = []
                for _ in range(4):
                    t = time.perf_counter()
                    trainer.train_epoch()
                    epochs.append(time.perf_counter() - t)
                    monitor.check()
                inf = []
                for _ in range(3):
                    t = time.perf_counter()
                    prediction = neural_predict(model, config, market, val, scale, device)
                    inf.append(time.perf_counter() - t)
                t = time.perf_counter()
                path = destination / (name + "_" + config["trial_id"] + ".pt")
                trainer.save(path)
                saved = time.perf_counter() - t
                t = time.perf_counter()
                trainer.restore(path)
                reloaded = time.perf_counter() - t
                row = {"trial": config, "training_samples": len(fit), "validation_samples": len(val),
                       "first_epoch_seconds": epochs[0], "warm_epoch_seconds": epochs[1:],
                       "inference_seconds": inf, "load_seconds": load_seconds,
                       "save_seconds": saved, "reload_seconds": reloaded,
                       "checkpoint_bytes": path.stat().st_size, "prediction_finite": bool(np.isfinite(prediction).all())}
                records[name].append(row)
                write_json(ROOT / "outputs/shape_profiles_progress.json", {"records": records, "last_model": name,
                           "last_trial": config["trial_id"], "elapsed_seconds": time.perf_counter() - started})
                del trainer, model
                gc.collect()
                torch.cuda.empty_cache()
        monitor.check()
        write_json(ROOT / "outputs/shape_profiles.json", {"passed": True, "completed_at_utc": utc_now(),
                   "wall_seconds": time.perf_counter() - started, "records": records, "resources": monitor.summary(),
                   "scope": "20 training days; all declared geometries; no validation loss or HPO ranking computed"})
    print("All supervised shape timing complete", time.perf_counter() - started, flush=True)


if __name__ == "__main__":
    main()
