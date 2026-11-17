"""Pre-2023 numeric-equivalence and cost tests; no development-block selection."""
import gc
import time

import numpy as np
import torch

from .common import ROOT, runtime, utc_now, write_json, sync
from .data import Market
from .models import neural, chronos_pipeline, chronos_predict
from .profile import config_for
from .resources import ResourceMonitor
from .training import Trainer, neural_predict


def exercise(name, options, market, fit, val, scale):
    device = runtime()
    cfg = {**config_for(name), **options}
    t = time.perf_counter()
    if name == 'chronos2_lora':
        pipe, _ = chronos_pipeline(True, 16, device)
        model = pipe.model
        predict = lambda: chronos_predict(pipe, market, val, cfg)
    else:
        model = neural(cfg).to(device)
        predict = lambda: neural_predict(model, cfg, market, val, scale, device)
    trainer = Trainer(model, cfg, market, fit, scale, device, 2026,
                      900 if name == 'chronos2_lora' else 27, lora=name == 'chronos2_lora')
    costs, losses = [], []
    for j in range(4):
        sync()
        tick = time.perf_counter()
        if name == 'chronos2_lora':
            value = np.mean(trainer.train_updates((j + 1) * 25))
        else:
            value = trainer.train_epoch()
        sync()
        costs.append(time.perf_counter() - tick)
        losses.append(float(value))
    pred = predict()
    result = {'options': options, 'first_unit_seconds': costs[0], 'warm_unit_seconds': costs[1:],
              'total_seconds': time.perf_counter() - t, 'training_losses': losses,
              'training_period_validation_mse': float(np.mean((pred - market.y(val)) ** 2)),
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30}
    del trainer, model
    if name == 'chronos2_lora':
        del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return result, pred


def main():
    started = time.perf_counter()
    runtime()
    market = Market.load()
    days = sorted(set(market.dates))[:24]
    fit, val = market.select(days[:20]), market.select(days[21:24])
    scale = market.scale(days[:20])
    rows, adopted = {}, {}
    # Fixed before testing, in original log-return units.
    tolerance = {'max_relative_mse_change': 0.001, 'prediction_rtol': 0.0001, 'prediction_atol': 1e-7}
    with ResourceMonitor() as monitor:
        for name in ('xpatch', 'patchtst', 'chronos2_lora'):
            reference, prediction = exercise(name, {}, market, fit, val, scale)
            reference.update(equivalent=True, reference=True)
            rows[name] = [reference]
            best = reference
            options = [dict(device_cache=True), dict(device_cache=True, optimizer_fused=True),
                       dict(device_cache=True, optimizer_fused=True, precision='tf32'),
                       dict(device_cache=True, optimizer_fused=True, precision='bf16')]
            if name != 'chronos2_lora':
                options.append(dict(device_cache=True, sample_fraction=1 / 3))
            for option in options:
                try:
                    row, value = exercise(name, option, market, fit, val, scale)
                    rel = abs(row['training_period_validation_mse'] / reference['training_period_validation_mse'] - 1)
                    row.update(relative_mse_change=rel,
                               max_prediction_abs_difference=float(np.max(np.abs(value - prediction))),
                               equivalent=bool(rel < tolerance['max_relative_mse_change'] and
                                   np.allclose(value, prediction, rtol=tolerance['prediction_rtol'], atol=tolerance['prediction_atol'])))
                    if row['equivalent'] and np.median(row['warm_unit_seconds']) < np.median(best['warm_unit_seconds']):
                        best = row
                except Exception as exc:
                    row = {'options': option, 'equivalent': False, 'error': repr(exc)}
                    gc.collect()
                    torch.cuda.empty_cache()
                rows[name].append(row)
                monitor.check()
                write_json(ROOT / 'outputs/optimization_progress.json', {'records': rows, 'last_model': name})
            adopted[name] = best['options']
        monitor.check()
        report = {'passed': True, 'completed_at_utc': utc_now(), 'wall_seconds': time.perf_counter() - started,
                  'tolerance_declared_before_tests': tolerance, 'records': rows, 'adopted': adopted,
                  'resources': monitor.summary(), 'source_label_scope': '2021 training-period small cases only',
                  'sample_fraction_policy': 'Only allowed if the same numeric-equivalence gate passes; otherwise all origins retained',
                  'compile_policy': 'Separate bounded cold/warm test required before adoption',
                  'development_or_holdout_scores_read': False}
        write_json(ROOT / 'outputs/optimization_admission.json', report)
    print({'passed': True, 'adopted': adopted, 'wall_seconds': report['wall_seconds']}, flush=True)


if __name__ == '__main__':
    main()
