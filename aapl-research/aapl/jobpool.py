"""Bounded process workers with explicit CPU ownership and reusable development data."""
import gc
import os
from pathlib import Path
import time

import numpy as np

from .common import ROOT, digest, identity, read_json, runtime, utc_now, write_json
from .data import Market
from .models import chronos_pipeline, chronos_predict
from .resources import ResourceMonitor
from .runner import execute_spec, neural_job, baseline_job
from .training import neural_predict, source_identity
from .accounting import charge

MARKET = None


def initialize(slot_queue):
    slot = slot_queue.get()
    os.environ['AAPL_CPU_THREADS'] = '2'
    os.sched_setaffinity(0, {2 * slot, 2 * slot + 1})
    runtime()


def development_market():
    global MARKET
    if MARKET is None:
        MARKET = Market.load()
    return MARKET


def search_work(spec):
    market = development_market()
    with charge(Path(spec['directory']), spec['config']['model']), ResourceMonitor() as monitor:
        value = execute_spec(spec, market)
        monitor.check()
        write_json(Path(spec['directory']) / 'resources.json', monitor.summary())
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return value


def predict_work(spec):
    """Fit to a fixed budget; write predictions without consulting evaluation outcomes."""
    directory = Path(spec['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    signature = identity(spec)
    path, receipt_path = directory / 'predictions.npz', directory / 'receipt.json'
    if receipt_path.exists():
        old = read_json(receipt_path)
        if old['specification_identity'] != signature or digest(path) != old['prediction_sha256']:
            raise ValueError('Completed prediction job identity changed')
        return old
    try:
        return _predict_work(spec, signature, directory, path, receipt_path)
    except Exception as exc:
        write_json(directory / 'failure.json', {'failed_at_utc': utc_now(), 'spec': spec,
                   'error': repr(exc), 'artifacts_preserved': True})
        raise


def _predict_work(spec, signature, directory, path, receipt_path):
    import torch
    device = runtime(spec['seed'])
    market = development_market()
    cfg = spec['config']
    name = cfg['model']
    hold = pipe = None
    with charge(directory, name), ResourceMonitor() as monitor:
        if name in ('xpatch', 'patchtst', 'chronos2_lora'):
            _, predict, hold = neural_job(cfg, spec['seed'], spec['fit_dates'], spec['calibration_dates'],
                                         spec['budget'], directory, market, name == 'chronos2_lora', refit=True)
            if name == 'chronos2_lora':
                for_market = lambda data, origins: chronos_predict(hold['pipe'], data, origins, cfg)
            else:
                for_market = lambda data, origins: neural_predict(hold['model'], cfg, data, origins,
                                                                    hold['trainer'].scale, device)
        elif name == 'chronos2_frozen':
            pipe, _ = chronos_pipeline(device=device)
            predict = lambda origins: chronos_predict(pipe, market, origins, cfg)
            for_market = lambda data, origins: chronos_predict(pipe, data, origins, cfg)
        else:
            _, predict = baseline_job(cfg, spec['seed'], spec['fit_dates'], spec['calibration_dates'],
                                      directory, market, fixed_trees=spec['budget'])
            for_market = predict.for_market
        calibration = market.select(spec['calibration_dates'])
        residual = market.y(calibration) - predict(calibration)
        calibration_file = directory / 'calibration.json'
        write_json(calibration_file, {'levels': [0.1, 0.5, 0.9], 'dates': spec['calibration_dates'],
                   'additive_residual_quantiles': np.quantile(residual, [0.1, 0.5, 0.9]).tolist()})
        target = Market.load_holdout(spec['unseal_authorization']) if spec['stage'] == 'holdout' else market
        origins = target.select(spec['evaluation_dates'])
        prediction = for_market(target, origins)
        np.savez_compressed(path, origins=origins, bar_start_ns=target.bars.index[origins].as_unit('ns').asi8,
                            prediction=prediction)
        monitor.check()
        receipt = {'completed_at_utc': utc_now(), 'specification_identity': signature,
                   'protocol_sha256': spec['protocol_sha256'], 'stage': spec['stage'],
                   'prediction_sha256': digest(path), 'calibration_sha256': digest(calibration_file),
                   'implementation_sha256': source_identity(), 'config': cfg, 'seed': spec['seed'],
                   'evaluation_target_read_for_scoring': False, 'resources': monitor.summary()}
        write_json(receipt_path, receipt)
    del hold, pipe, predict, for_market
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def cpu_search(name, trials, split, directory):
    from .runner import run_inner
    threads, cpus = (8, set(range(6, 14))) if name == 'lightgbm' else (2, {14, 15})
    os.environ['AAPL_CPU_THREADS'] = str(threads)
    os.sched_setaffinity(0, cpus)
    runtime()
    with ResourceMonitor() as monitor:
        result = run_inner(name, trials, split, development_market(), Path(directory))
        monitor.check()
        write_json(Path(directory) / 'search_resources.json', monitor.summary())
    return result
