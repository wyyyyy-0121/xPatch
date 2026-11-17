"""One preregistered CLV information screen; explicit CLI execution only."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import resource
import signal
import sys
import tempfile
import time
import numpy as np

import screen_core as core

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[3]
OLD = BASE.parent/'20261109_information_audit'
PRIOR = ROOT/'outputs/multiscale/controlled_followup/20261108_v2'
TASKS = (('minute', 60, 480), ('hourly', 8, 96))
BATCH = 2048
STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno()); temporary = f.name
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('wb', dir=path.parent, delete=False) as f:
        np.savez_compressed(f, **arrays)
        f.flush(); os.fsync(f.fileno()); temporary = f.name
    os.replace(temporary, path)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec); spec.loader.exec_module(obj)
    return obj


def stop(*_):
    global STOP
    STOP = True


def main():
    protocol_file = BASE/'MARKET_PROTOCOL.json'
    protocol = read(protocol_file); identity = sha(protocol_file)
    output = BASE/'screen'; output.mkdir(exist_ok=True)
    state_file = output/'RUN_STATE.json'
    def guard():
        if STOP or time.time() >= protocol['deadline_epoch']:
            raise InterruptedError('Same frozen CLV deadline or stop request; saved batches retained')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, stop)
    started = time.monotonic()
    try:
        guard()
        if os.sched_getaffinity(0) != {24}:
            raise ValueError('CLV worker must use only logical CPU24')
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            if os.environ.get(key) != '1':
                raise ValueError('Numeric runtime must use one thread: '+key)
        if (protocol['GPU'] is not False or protocol['planned_fits'] != 12
                or protocol['market_arrays_admitted'] is not True):
            raise ValueError('Unexpected preregistered design')
        for file, expected in protocol['dependencies'].items():
            guard()
            if sha(file) != expected:
                raise ValueError('Frozen screen dependency changed: '+file)
        admission_receipt = read(BASE/'MARKET_ADMISSION.json')
        if (admission_receipt['protocol_sha256'] != identity or admission_receipt['admitted'] is not True):
            raise ValueError('No matching explicit market execution admission')
        atomic(state_file, dict(status='running', protocol_sha256=identity, pid=os.getpid(),
                               deadline_epoch=protocol['deadline_epoch'], GPU=False))
        # Source caches must already be complete. Never rebuild them in a screen.
        from ohlc_source import read_cache
        source_receipt = read(BASE/'source_cache/OHLC_SOURCE_RECEIPT.json')
        ohlc = read_cache(BASE/'source_cache', source_receipt['identity'], guard)
        old_source = module(OLD/'code/volume_source.py', 'frozen_old_volume_source')
        cache = OLD/'screen_20261109_v1/cache'
        if not (cache/'VOLUME_SOURCE_RECEIPT.json').is_file():
            raise ValueError('Only the pre-existing qualified volume cache is allowed')
        volume = old_source.load_volume(cache, guard)
        context = module(ROOT/'outputs/multiscale/mechanism_repair/20261109_v3b/code/predictor.py',
                         'original_context_dictionary')
        sys.path.insert(0, str(PRIOR/'code/reference_source'))
        from multiscale.features import CloseWindows
        from multiscale.bars import Bars, targets
        data_protocol = read(PRIOR/'protocol/DATA.json')
        admission = read(Path(data_protocol['admission'])/'admission.json')
        states = []
        for profile, h, lookback in TASKS:
            frequency = 'minute' if profile == 'minute' else 'hour'
            data, extra = volume[frequency], ohlc[frequency]
            for key in old_source.BAR_FIELDS:
                if not np.array_equal(data[key], extra[key]):
                    raise ValueError('OHLC and volume caches differ on original '+key)
            bars = Bars(frequency, **{k:data[k] for k in old_source.BAR_FIELDS})
            windows = CloseWindows(bars, lookback, admission['split_actions'])
            task = f'{profile}_h{h}_future_log_return'
            meta = read(PRIOR/'cache'/task/'TASK.json')
            abnormal = data['abnormal']; finite = np.isfinite(abnormal)
            cumulative = np.r_[0., np.cumsum(np.where(finite, abnormal, 0.))]
            bad = np.r_[0, np.cumsum(~finite)]
            lag_column = context.feature_names(h, lookback).index(f'return_lag_{h}')
            columns = {'A':lookback+18, 'B':lookback+20, 'C':lookback+21}
            def available(o):
                return bad[o+1]-bad[o-h+1] == 0
            def batch(o, with_target=True):
                guard()
                history, anchor = windows.arrays(o)
                old = context.features(history, bars.end_ns[o], h)
                u = (cumulative[o+1]-cumulative[o-h+1])/h
                x = np.column_stack((history, anchor, old, u, old[:, lag_column]*u, extra['clv'][o]))
                if x.shape != (len(o), columns['C']) or not np.isfinite(x).all():
                    raise ValueError('Unexpected CLV dictionary')
                return (x, targets(bars, o, h, 'future_log_return')) if with_target else x
            for fold in (1, 2):
                guard(); folder = output/task/f'fold_{fold}'; folder.mkdir(parents=True, exist_ok=True)
                original = meta['folds'][str(fold)]
                with np.load(original['split_file'], allow_pickle=False) as f:
                    fit = f['fit'].copy()
                if (fit.dtype != np.dtype('int64') or fit.ndim != 1 or not len(fit)
                        or len(fit) != original['counts']['fit'] or (np.diff(fit) <= 0).any()
                        or fit[0] < lookback-1 or fit[-1]+h >= len(bars.close)
                        or hashlib.sha256(fit.astype('<i8').tobytes()).hexdigest() != original['split_sha256']['fit']):
                    raise ValueError('Original fit identity mismatch')
                boundary, first, prefix, tail = core.partition_fit(fit, bars.end_ns, h, available)
                if len(prefix) < 100 or len(tail) < 20:
                    raise ValueError('Insufficient fixed common origins')
                partition = dict(protocol_sha256=identity, original_fit_count=len(fit),
                    boundary_before_mask=boundary, first_tail_origin_before_mask=first,
                    prefix_count=len(prefix), tail_count=len(tail),
                    prefix_sha256=hashlib.sha256(prefix.astype('<i8').tobytes()).hexdigest(),
                    tail_sha256=hashlib.sha256(tail.astype('<i8').tobytes()).hexdigest(),
                    prefix_last_label_end_ns=int(bars.end_ns[prefix[-1]+h]),
                    original_tail_first_decision_ns=int(bars.end_ns[first]),
                    all_arms_share_origins=True, columns=columns)
                atomic(folder/'PARTITION.json', partition)
                stats = core.empty_stats(columns['C'], h); cursor = 0
                checkpoint = folder/'stats.npz'
                if checkpoint.exists():
                    with np.load(checkpoint, allow_pickle=False) as f:
                        if str(f['protocol_sha256']) != identity:
                            raise ValueError('Checkpoint protocol identity mismatch')
                        stats = {k:f[k].copy() for k in ('n','sx','sy','xx','xy')}
                        cursor = int(f['cursor'])
                    stats['n'] = int(stats['n'])
                    if stats['n'] != cursor or not 0 <= cursor <= len(prefix):
                        raise ValueError('Invalid completed-prefix cursor')
                begin = time.monotonic()
                for i in range(cursor, len(prefix), BATCH):
                    guard(); x, y = batch(prefix[i:i+BATCH]); core.update(stats, x, y)
                    cursor = i+len(x)
                    atomic_npz(checkpoint, **stats, cursor=cursor, protocol_sha256=identity)
                    atomic(folder/'FIT_STATE.json', dict(cursor=cursor, total=len(prefix), protocol_sha256=identity))
                    if not (output/'COST_ADMISSION.json').exists():
                        elapsed = time.monotonic()-begin
                        projection = elapsed*len(prefix)/len(x)*4*2+300
                        peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024
                        passed = elapsed < 60 and peak_rss <= 3*1024**3 and projection <= protocol['deadline_epoch']-time.time()
                        atomic(output/'COST_ADMISSION.json', dict(passed=passed, first_batch_seconds=elapsed,
                             peak_RSS_bytes=peak_rss, maximum_worker_RSS_bytes=3*1024**3,
                             projected_seconds_with_margin_and_reserve=projection, protocol_sha256=identity))
                        if not passed:
                            raise InterruptedError('Fixed maximum-shape cost gate failed')
                    if protocol.get('external_signal_pause_after_first_saved_batch') is True and not (output/'PAUSE_REQUEST_READY.json').exists():
                        atomic(output/'PAUSE_REQUEST_READY.json', dict(protocol_sha256=identity,
                            checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint), cursor=cursor,
                            pid=os.getpid(), deadline_epoch=protocol['deadline_epoch']))
                        wait_until = time.monotonic()+10
                        while time.monotonic() < wait_until:
                            guard(); time.sleep(.02)
                        raise RuntimeError('Lifecycle controller did not deliver its required pause signal')
                if not read(output/'COST_ADMISSION.json')['passed']:
                    raise InterruptedError('Prior cost gate was not admitted')
                qualification = core.clv_prefix_qualification(stats,
                    (batch(prefix[i:i+BATCH], False) for i in range(0, len(prefix), BATCH)))
                atomic(folder/'FEATURE_QUALIFICATION.json', dict(qualification, protocol_sha256=identity))
                if not qualification['passed']:
                    atomic(output/'RESULTS.json', dict(status='closed_feature_qualification_failed',
                        task=task, fold=fold, continued=False, fitted_models=0, protocol_sha256=identity))
                    atomic(state_file, dict(status='closed_feature_qualification_failed', protocol_sha256=identity))
                    return
                # Keep only a frozen callable-free description; reconstruct batches later.
                states.append((profile, h, lookback, fold, folder, stats, prefix.copy(), tail.copy()))
        # No tail scoring or Ridge solving until all four prefix gates pass.
        # Freeze all twelve finite, numerically qualified predictors before scoring any tail.
        for profile, h, lookback, fold, folder, stats, prefix, tail in states:
            guard()
            columns = {'A':lookback+18, 'B':lookback+20, 'C':lookback+21}
            models = {name:core.solve(stats, p) for name,p in columns.items()}
            for name, model in models.items():
                atomic_npz(folder/(name+'.npz'), **model, protocol_sha256=identity)
        atomic(output/'FROZEN_MODELS.json', dict(protocol_sha256=identity,
            artifacts={str(folder/(name+'.npz')):sha(folder/(name+'.npz'))
                       for _,_,_,_,folder,_,_,_ in states for name in ('A','B','C')}))
        result_rows = []
        for profile, h, lookback, fold, folder, stats, prefix, tail in states:
            guard(); frequency = 'minute' if profile == 'minute' else 'hour'
            data, extra = volume[frequency], ohlc[frequency]
            bars = Bars(frequency, **{k:data[k] for k in old_source.BAR_FIELDS})
            windows = CloseWindows(bars, lookback, admission['split_actions'])
            abnormal = data['abnormal']; finite = np.isfinite(abnormal)
            cumulative = np.r_[0., np.cumsum(np.where(finite, abnormal, 0.))]
            lag_column = context.feature_names(h, lookback).index(f'return_lag_{h}')
            columns = {'A':lookback+18, 'B':lookback+20, 'C':lookback+21}
            models = {}
            for name in columns:
                with np.load(folder/(name+'.npz'), allow_pickle=False) as f:
                    if str(f['protocol_sha256']) != identity:
                        raise ValueError('Frozen model identity changed')
                    models[name] = {k:f[k].copy() for k in ('coef','x_mean','x_std','y_mean')}
            sums = {name:dict(endpoint_sse=0., path_sse=0.) for name in ('A','B','C','zero','prefix_mean')}
            n = 0
            for i in range(0, len(tail), BATCH):
                guard(); o = tail[i:i+BATCH]; history, anchor = windows.arrays(o)
                old = context.features(history, bars.end_ns[o], h)
                u = (cumulative[o+1]-cumulative[o-h+1])/h
                x = np.column_stack((history, anchor, old, u, old[:,lag_column]*u, extra['clv'][o]))
                y = targets(bars, o, h, 'future_log_return'); n += len(o)
                for name in sums:
                    pred = core.predict(models[name], x) if name in models else (0. if name == 'zero' else stats['sy']/stats['n'])
                    error = pred-y
                    sums[name]['endpoint_sse'] += float(np.sum(error[:, -1]**2))
                    sums[name]['path_sse'] += float(np.sum(error**2))
            metrics = {k:dict(v, endpoint_mse=v['endpoint_sse']/n, path_mse=v['path_sse']/(n*h)) for k,v in sums.items()}
            if (not all(np.isfinite(v) and v >= 0 for row in metrics.values() for v in row.values())
                    or metrics['zero']['endpoint_mse'] <= 0):
                raise ValueError('Nonfinite unrounded screen metrics')
            checks = core.state_checks(metrics)
            row = dict(task=f'{profile}_h{h}_future_log_return', fold=fold, tail_origins=n,
                       metrics=metrics, checks=checks, passed=all(checks.values()),
                       protocol_sha256=identity)
            atomic(folder/'RESULT.json', row); result_rows.append(row)
        task_gates = {}
        for profile, h, _ in TASKS:
            task = f'{profile}_h{h}_future_log_return'
            selected = [r for r in result_rows if r['task'] == task]
            gain = 1-sum(r['metrics']['C']['endpoint_mse'] for r in selected)/sum(r['metrics']['zero']['endpoint_mse'] for r in selected)
            task_gates[task] = dict(equal_fold_MSE_gain_over_zero=gain, at_least_one_percent=bool(gain >= .01))
        passed = all(r['passed'] for r in result_rows) and all(r['at_least_one_percent'] for r in task_gates.values())
        result = dict(status='completed', rows=result_rows, task_gates=task_gates, continued=passed,
                      passed_states=sum(r['passed'] for r in result_rows), fitted_models=12,
                      protocol_sha256=identity, elapsed_seconds=time.monotonic()-started,
                      independent_confirmation=False, Q3_read=False, GPU=False,
                      scope='exposed historical fit-tail fixed Ridge information screen only')
        atomic(output/'RESULTS.json', result)
        atomic(state_file, dict(status='completed', protocol_sha256=identity, continuation_gate_passed=passed,
                               elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        atomic(state_file, dict(status='paused' if isinstance(error, InterruptedError) else 'failed',
            protocol_sha256=identity, error_type=type(error).__name__, reason=str(error),
            elapsed_seconds=time.monotonic()-started, same_deadline_epoch=protocol['deadline_epoch']))
        raise


if __name__ == '__main__':
    if sys.argv[1:] == ['--synthetic']:
        atomic(BASE/'SYNTHETIC_SCREEN_ACCEPTANCE.json', core.synthetic_acceptance())
    elif sys.argv[1:] == ['--run']:
        main()
    else:
        raise SystemExit('Use --synthetic or --run; real runs require frozen explicit admission')
