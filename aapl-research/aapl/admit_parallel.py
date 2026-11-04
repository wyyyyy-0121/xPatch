"""Exercise three independent maximum-geometry jobs before admitting shared GPU work."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time

import numpy as np

from .common import ROOT, read_json, runtime, utc_now, write_json
from .data import Market, development_splits
from .jobpool import initialize
from .models import neural, chronos_pipeline, chronos_predict
from .profile import config_for
from .resources import ResourceMonitor
from .training import Trainer, neural_predict


def worker(name, slot):
    device = runtime(2026 + slot)
    market = Market.load()
    days = sorted(d for d in set(market.dates) if d <= '2022-12-30')
    source = market.select(days[:-21])
    maximum = len(market.select(development_splits(market)['final_fit_dates']))
    train = np.resize(source, maximum)
    val = market.select(days[-20:])
    scale = market.scale(days[:-21])
    options = read_json(ROOT / 'outputs/optimization_admission.json')['adopted'].get(name, {})
    cfg = {**config_for(name), **options}
    started = time.perf_counter()
    started_epoch = time.time()
    with ResourceMonitor() as monitor:
        if name.startswith('chronos'):
            pipe, _ = chronos_pipeline(name == 'chronos2_lora', rank=16, device=device)
            model = pipe.model
            predict = lambda: chronos_predict(pipe, market, val, cfg)
        else:
            model = neural(cfg).to(device)
            predict = lambda: neural_predict(model, cfg, market, val, scale, device)
        if name != 'chronos2_frozen':
            trainer = Trainer(model, cfg, market, train, scale, device, 2026 + slot,
                              900 if name == 'chronos2_lora' else 27, lora=name == 'chronos2_lora')
            if name == 'chronos2_lora':
                trainer.train_updates(100)
            else:
                trainer.train_epoch()
        prediction = predict()
        monitor.check()
        row = {'model': name, 'slot': slot, 'passed': bool(np.isfinite(prediction).all()),
               'started_epoch': started_epoch, 'ended_epoch': time.time(),
               'wall_seconds': time.perf_counter() - started, 'maximum_samples': maximum,
               'resources': monitor.summary()}
    write_json(ROOT / f'outputs/parallel/{name}_{slot}.json', row)
    return row


def main():
    runtime()
    context = mp.get_context('spawn')
    results, adopted = {}, {}
    for name in ('xpatch', 'patchtst', 'chronos2_frozen', 'chronos2_lora'):
        queue = context.Queue()
        for slot in range(3):
            queue.put(slot)
        try:
            with ResourceMonitor() as group:
                with ProcessPoolExecutor(max_workers=3, mp_context=context, initializer=initialize, initargs=(queue,)) as pool:
                    rows = list(pool.map(worker, [name] * 3, range(3)))
                group.check()
                overlap = max(r['started_epoch'] for r in rows) < min(r['ended_epoch'] for r in rows)
                results[name] = {'passed': overlap and all(r['passed'] and r['resources']['passed'] for r in rows),
                                 'three_jobs_overlapped': overlap,
                                 'workers': rows, 'group': group.summary()}
            adopted[name] = 3 if results[name]['passed'] else 1
        except Exception as exc:
            results[name] = {'passed': False, 'error': repr(exc), 'fallback': 'single_job_already_sustained_admitted'}
            adopted[name] = 1
        write_json(ROOT / 'outputs/parallel_admission_progress.json', {'records': results, 'adopted_concurrency': adopted})
    report = {'passed': True, 'completed_at_utc': utc_now(), 'records': results,
              'adopted_concurrency': adopted, 'source_label_scope': 'pre2023 only; no research model scores',
              'fallback_policy': 'A failed three-job resource test retains the already admitted single-job path'}
    write_json(ROOT / 'outputs/parallel_admission.json', report)
    print({'passed': True, 'adopted_concurrency': adopted}, flush=True)


if __name__ == '__main__':
    main()
