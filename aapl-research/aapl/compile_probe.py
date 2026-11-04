"""One bounded compiler experiment in an isolated process, on 2021 data."""
import time

import numpy as np

from .common import ROOT, read_json, runtime, utc_now, write_json
from .data import Market
from .optimize import exercise


def main():
    runtime()
    market = Market.load()
    days = sorted(set(market.dates))[:24]
    fit, val = market.select(days[:20]), market.select(days[21:24])
    scale = market.scale(days[:20])
    adopted = read_json(ROOT / 'outputs/optimization_admission.json')['adopted']
    rows = {}
    for name in ('xpatch', 'patchtst'):
        reference, pred = exercise(name, adopted[name], market, fit, val, scale)
        try:
            compiled, result = exercise(name, {**adopted[name], 'compile': True}, market, fit, val, scale)
            mse_change = abs(compiled['training_period_validation_mse'] / reference['training_period_validation_mse'] - 1)
            same = bool(mse_change < 0.001 and np.allclose(result, pred, rtol=0.0001, atol=1e-7))
            rows[name] = {'reference': reference, 'compiled': compiled, 'equivalent': same,
                          'relative_mse_change': mse_change, 'max_abs_prediction_change': float(np.max(np.abs(result - pred))),
                          'four_epoch_net_gain': compiled['total_seconds'] < reference['total_seconds']}
        except Exception as exc:
            rows[name] = {'reference': reference, 'error': repr(exc), 'equivalent': False,
                          'four_epoch_net_gain': False}
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        write_json(ROOT / 'outputs/compile_probe_progress.json', {'records': rows})
    write_json(ROOT / 'outputs/compile_probe.json', {'completed': True, 'records': rows, 'completed_at_utc': utc_now()})


if __name__ == '__main__':
    main()
