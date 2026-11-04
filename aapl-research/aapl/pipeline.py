"""Execute the user implementation plan after all admission gates have passed."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
from pathlib import Path
import signal
import threading
import time

from .common import ROOT, digest, identity, read_json, runtime, utc_now, write_json
from .data import Market
from .jobpool import initialize, search_work, predict_work, cpu_search
from .runner import run_inner, require_lock
from .accounting import BudgetGuard

FORMAL = ROOT / 'outputs/formal'
GPU_NAMES = ('xpatch', 'patchtst', 'chronos2_frozen', 'chronos2_lora')


def progress(stage, **details):
    write_json(FORMAL / 'progress.json', {'stage': stage, 'updated_at_utc': utc_now(), **details})


def gpu_pool(context, n):
    queue = context.Queue()
    for slot in range(n):
        queue.put(slot)
    return ProcessPoolExecutor(max_workers=n, mp_context=context, initializer=initialize, initargs=(queue,))


def cpu_prediction(spec):
    name = spec['config']['model']
    os.environ['AAPL_CPU_THREADS'] = '8' if name == 'lightgbm' else '2'
    os.sched_setaffinity(0, set(range(6, 14)) if name == 'lightgbm' else {14, 15})
    runtime(spec['seed'])
    return predict_work(spec)


def prediction_specs(protocol, split, stage, selected):
    rows = []
    for name in selected:
        winner = read_json(FORMAL / 'search' / name / 'selection.json')[0]
        seeds = [2026] if name in ('ridge', 'chronos2_frozen') else [2026, 2027, 2028]
        for seed in seeds:
            row = {'config': winner['trial'], 'seed': seed, 'budget': winner['final_budget'], 'stage': stage,
                   'protocol_sha256': digest(ROOT / 'protocol_locked.json'),
                   'fit_dates': split['confirmation']['fit_dates'] if stage == 'confirmation' else split['final_fit_dates'],
                   'calibration_dates': split['confirmation']['calibration_dates'] if stage == 'confirmation' else split['calibration_dates'],
                   'evaluation_dates': split['confirmation']['evaluation_dates'] if stage == 'confirmation' else split['holdout_dates_calendar_only'],
                   'directory': str(FORMAL / stage / name / f'seed{seed}')}
            if stage == 'holdout':
                decision_path = ROOT / 'outputs/development_decision.json'
                row['unseal_authorization'] = identity({'protocol_sha256': digest(ROOT / 'protocol_locked.json'),
                    'decision_sha256': digest(decision_path), 'holdout_csv_sha256': read_json(ROOT / 'holdout_manifest.json')['csv_sha256']})
            rows.append(row)
            if stage == 'confirmation' and name == 'xpatch':
                alt = dict(row)
                alt['config'] = {**winner['trial'], 'loss': 'step_mse' if winner['trial']['loss'] == 'cumulative12_mse' else 'cumulative12_mse'}
                alt['directory'] = str(FORMAL / stage / 'xpatch_loss_ablation' / f'seed{seed}')
                rows.append(alt)
            if stage == 'confirmation' and name == 'chronos2_lora' and seed == 2026:
                alt = dict(row)
                alt['config'] = {**winner['trial'], 'model': 'chronos2_frozen'}
                alt['directory'] = str(FORMAL / stage / 'chronos2_matched_frozen' / 'seed2026')
                rows.append(alt)
    return rows


def run_predictions(protocol, split, stage, selected):
    context = mp.get_context('spawn')
    specs = prediction_specs(protocol, split, stage, selected)
    progress(stage + '_predictions', scheduled=len(specs))
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as cpu:
        cpufutures = [cpu.submit(cpu_prediction, s) for s in specs if s['config']['model'] in ('ridge', 'lightgbm')]
        for name in GPU_NAMES:
            jobs = [s for s in specs if s['config']['model'] == name]
            if not jobs:
                continue
            concurrency = protocol['execution']['gpu_concurrency'][name]
            with gpu_pool(context, concurrency) as gpu:
                list(gpu.map(predict_work, jobs))
        for future in cpufutures:
            future.result()


def run_search(protocol, split, market):
    context = mp.get_context('spawn')
    trials = read_json(ROOT / protocol['trial_list_file'])
    with ProcessPoolExecutor(max_workers=2, mp_context=context) as cpu:
        pending = [cpu.submit(cpu_search, name, trials[name], split, str(FORMAL / 'search' / name))
                   for name in ('ridge', 'lightgbm')]
        for name in GPU_NAMES:
            progress('search', model=name, initial_configurations=len(trials[name]))
            with gpu_pool(context, protocol['execution']['gpu_concurrency'][name]) as gpu:
                run_inner(name, trials[name], split, market, FORMAL / 'search' / name,
                          dispatch=lambda specs: list(gpu.map(search_work, specs)),
                          execution=protocol['execution']['model_options'].get(name, {}))
        for future in pending:
            future.result()
    write_json(FORMAL / 'search_complete.json', {'completed_at_utc': utc_now(),
               'protocol_sha256': digest(ROOT / 'protocol_locked.json'),
               'selections': {name: digest(FORMAL / 'search' / name / 'selection.json') for name in trials}})


def run(stage='all'):
    runtime()
    protocol = require_lock()
    FORMAL.mkdir(parents=True, exist_ok=True)
    state_file = FORMAL / 'run.json'
    if state_file.exists():
        if read_json(state_file)['protocol_sha256'] != digest(ROOT / 'protocol_locked.json'):
            raise ValueError('Existing formal run belongs to a different protocol')
    else:
        write_json(state_file, {'started_epoch': time.time(), 'started_at_utc': utc_now(),
                   'protocol_sha256': digest(ROOT / 'protocol_locked.json'), 'pid': os.getpid()})
    with BudgetGuard():
        return _run_admitted(stage, protocol)


def _run_admitted(stage, protocol):
    # Only the manager uses these two cores. Worker initializers own their disjoint sets.
    os.sched_setaffinity(0, {16, 17})
    split = read_json(ROOT / 'outputs/development_splits.json')
    split['inner'] = [i for i in split['inner'] if i['name'] in protocol['development']['admitted_validation_blocks']]
    market = Market.load(protocol)
    from . import evaluate
    try:
        if stage in ('search', 'all'):
            run_search(protocol, split, market)
        if stage in ('confirm', 'all'):
            if not (FORMAL / 'search_complete.json').exists():
                raise RuntimeError('All planned searches must finish before D-block confirmation')
            run_predictions(protocol, split, 'confirmation', [n for n in protocol['models_primary'] if n != 'zero'])
            decision = evaluate.development(protocol, split, market)
            progress('development_decision', enter_holdout=decision['enter_holdout'], selected=decision['selected_for_holdout'])
        if stage in ('final', 'all'):
            decision = read_json(ROOT / 'outputs/development_decision.json')
            if decision['enter_holdout']:
                run_predictions(protocol, split, 'holdout', decision['selected_for_holdout'])
            else:
                progress('completed_development_negative_holdout_remains_sealed')
        if stage in ('evaluate', 'all'):
            evaluate.final(protocol, split, market)
            from .verify_results import verify
            verify()
            from .accounting import usage
            write_json(FORMAL / 'usage.json', usage())
            from .reporting import write_results
            write_results()
            from .final_audit import audit
            audit(completion=True)
            progress('complete', result_file='outputs/formal/final_summary.json')
    except Exception as exc:
        write_json(FORMAL / 'execution_failure.json', {'failed_at_utc': utc_now(), 'stage': stage,
                   'error': repr(exc), 'protocol_sha256': digest(ROOT / 'protocol_locked.json'),
                   'artifacts_preserved': True})
        progress('failed', error=repr(exc))
        raise
