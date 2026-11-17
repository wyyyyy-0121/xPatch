"""Independent saved-output arithmetic and direct-centered Ridge replay.

Reuses the qualified source adapter and NumPy solve backend, but does not import
the main scoring/control code. Does not independently replay neural forwards.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from common_adapter import access
from market_reader import MarketInputs

BASE = Path(__file__).resolve().parent
SEEDS = (2026, 2027, 2028)
REFS = ('zero', 'Q_mean_raw', 'Q_mean_shrink', 'E_Ridge_raw', 'E_Ridge_shrink') + tuple(
    f'C_seed{s}_{mode}' for s in SEEDS for mode in ('raw', 'shrink')) + ('C_seedmean_raw', 'C_seedmean_shrink')
CAND = 'R_seedmean_shrink'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args(); began = time.time(); body = args.protocol.read_bytes(); p = json.loads(body)
    if sorted(os.sched_getaffinity(0)) != [24, 26]: raise ValueError('Analysis affinity')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or any(os.environ.get(n) != '1' for n in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')):
        raise ValueError('Thread/GPU environment')
    inputs = MarketInputs(args.protocol); out = Path(p['output_directory'])/'postprocessing'
    for ref in p['locks']:
        if sha(ref['path']) != ref['sha256']: raise ValueError('Frozen source changed')
    destination = out/'INDEPENDENT_REVIEW.json'
    if destination.exists(): raise ValueError('No automatic repeated review')
    models = json.loads((out/'GLOBAL_MODELS_FROZEN.json').read_text())
    alphas = json.loads((out/'GLOBAL_ALPHAS_FROZEN.json').read_text())
    results = json.loads((out/'RESULTS.json').read_text())
    protocol_sha = hashlib.sha256(body).hexdigest()
    if any(v['protocol_sha256'] != protocol_sha for v in (models, alphas, results)):
        raise ValueError('One original execution protocol required')
    expected_ids = {s.state_uid+'/'+tag for s in inputs.states
                    for tag in tuple(f'{arm}:s{seed}' for arm in ('C', 'R') for seed in SEEDS)+('E_Ridge', 'mean')}
    if set(models['files']) != expected_ids or set(models['manifest']) != expected_ids:
        raise ValueError('Exact 32 actual files and manifest model IDs required')
    for mid, row in models['files'].items():
        if sha(row['path']) != row['sha256'] or row['sha256'] != models['manifest'][mid]['model_sha256']:
            raise ValueError('Actual model file / phase manifest checksum binding')
    if [r['task'] for r in results['primary_results']] != [
            'minute_h60_future_log_return', 'hourly_h8_future_log_return']:
        raise ValueError('Exactly both original primaries required')
    adapters = [inputs.adapter(s.task, s.fold) for s in inputs.states]
    for a in adapters: a.fit_scaler(access(a.state, 'Q', 'fit_scaler')); inputs.guard()
    independent_ridge = {}; checks = 0; maximum_difference = 0.

    def close(actual, expected, name):
        nonlocal checks, maximum_difference
        if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
            raise AssertionError(name+': shape/nonfinite')
        delta = float(np.max(np.abs(actual-expected))) if actual.size else 0.
        maximum_difference = max(maximum_difference, delta); checks += actual.size
        if not np.allclose(actual, expected, rtol=1e-8, atol=1e-10): raise AssertionError(name)

    # Separate centered-X covariance, with two streaming passes and bounded memory.
    for a in adapters:
        q = a.roles['Q']; width = 4*a.L+18; sx = np.zeros(width); sy = np.zeros(a.H)
        for begin in range(0, len(q), 128):
            inputs.guard(); o = q[begin:begin+128]
            sx += a.ridge_features(access(a.state, 'Q', 'inputs'), o).sum(axis=0)
            sy += a.labels(access(a.state, 'Q', 'labels'), o).sum(axis=0)
        mx = sx/len(q); my = sy/len(q); gram = np.zeros((width, width)); cross = np.zeros((width, a.H))
        for begin in range(0, len(q), 128):
            inputs.guard(); o = q[begin:begin+128]
            centered = a.ridge_features(access(a.state, 'Q', 'inputs'), o)-mx
            centered_y = a.labels(access(a.state, 'Q', 'labels'), o)-my
            gram += centered.T@centered; cross += centered.T@centered_y
        gram /= len(q); cross /= len(q)
        sd = np.sqrt(np.maximum(np.diag(gram), 0.)); sd = np.where(sd > 1e-12, sd, 1.)
        normalized = gram/sd[:, None]/sd[None, :]; normalized = (normalized+normalized.T)/2
        coef = np.linalg.solve(normalized+.1*np.eye(width), cross/sd[:, None])
        mid = a.state.state_uid+'/E_Ridge'
        with np.load(models['files'][mid]['path'], allow_pickle=False) as saved:
            close(saved['coef'], coef, 'direct-centered coefficient')
            close(saved['x_mean'], mx, 'independent column mean')
            close(saved['x_std'], sd, 'independent column standard deviation')
            close(saved['y_mean'], my, 'independent target mean')
        independent_ridge[a.state.state_uid] = (mx, sd, my, coef)
    inputs.gate.freeze_models(models['manifest'])
    # Recompute calibration alpha from preserved C_cal raw paths and raw labels.
    for a in adapters:
        for mid in inputs.gate.model_ids(a.state):
            inputs.guard(); tag = mid.split('/')[-1]
            path = out/a.state.state_uid/('Ccal_'+tag.replace(':', '_')+'.npz')
            with np.load(path, allow_pickle=False) as saved:
                origins = saved['origins']; pred = saved['raw_prediction']
            if not np.array_equal(origins, a.roles['C_cal']): raise AssertionError('Calibration origin binding')
            if pred.dtype != np.float64 or pred.shape != (len(origins), a.H) or not np.isfinite(pred).all():
                raise AssertionError('Calibration raw prediction dtype/shape/finiteness')
            y = a.labels(access(a.state, 'C_cal', 'labels'), origins)
            numerator = np.dot(pred[:, -1], y[:, -1]); denominator = np.dot(pred[:, -1], pred[:, -1])
            if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 0:
                raise AssertionError('Finite nonnegative alpha denominator and finite numerator required')
            alpha = min(1., max(0., float(numerator/denominator))) if denominator != 0 else 0.
            close(np.asarray([alphas['manifest'][mid]['alpha']]), np.asarray([alpha]), 'Independent alpha')
            if tag in ('E_Ridge', 'mean'):
                mx, sd, my, coef = independent_ridge[a.state.state_uid]
                for begin in range(0, len(origins), 128):
                    inputs.guard(); o = origins[begin:begin+128]
                    expected = (my+((a.ridge_features(access(a.state, 'C_cal', 'inputs'), o)-mx)/sd)@coef
                                if tag == 'E_Ridge' else np.broadcast_to(my, (len(o), a.H)))
                    close(pred[begin:begin+len(o)], expected, 'Independent Ccal control path')
    inputs.gate.freeze_alphas(alphas['manifest']); states = []
    for a in adapters:
        inputs.guard()
        with np.load(out/a.state.state_uid/'T_PREDICTIONS_AND_LOSSES.npz', allow_pickle=False) as z:
            data = {name: z[name] for name in z.files}
        if not np.array_equal(data['origins'], a.roles['T']): raise AssertionError('T origin binding')
        y = a.labels(access(a.state, 'T', 'labels'), a.roles['T']); close(data['truth'], y, 'Independent T label replay')
        losses = {'zero': {'endpoint': y[:, -1]**2, 'path': (y*y).mean(axis=1)}}
        for mid in inputs.gate.model_ids(a.state):
            tag = mid.split('/')[-1]; name = 'Q_mean' if tag == 'mean' else tag.replace(':s', '_seed')
            raw = data['raw_'+name]; alpha = alphas['manifest'][mid]['alpha']
            for mode, prediction in (('raw', raw), ('shrink', alpha*raw)):
                error = (prediction-y)**2
                losses[name+'_'+mode] = {'endpoint': error[:, -1], 'path': error.mean(axis=1)}
            if tag in ('E_Ridge', 'mean'):
                mx, sd, my, coef = independent_ridge[a.state.state_uid]
                for begin in range(0, len(a.roles['T']), 128):
                    inputs.guard(); o = a.roles['T'][begin:begin+128]
                    expected = (my+((a.ridge_features(access(a.state, 'T', 'inputs'), o)-mx)/sd)@coef
                                if tag == 'E_Ridge' else np.broadcast_to(my, (len(o), a.H)))
                    close(raw[begin:begin+len(o)], expected, 'Independent T control path')
        for arm in ('C', 'R'):
            for mode in ('raw', 'shrink'):
                losses[f'{arm}_seedmean_{mode}'] = {metric:
                    sum(losses[f'{arm}_seed{s}_{mode}'][metric] for s in SEEDS)/3
                    for metric in ('endpoint', 'path')}
        for name, row in losses.items():
            for metric, loss in row.items(): close(data['loss_'+name+'_'+metric], loss, 'Saved loss replay')
        sessions = inputs.bars('minute' if a.state.task.startswith('minute') else 'hour')['session_date']
        dates = sessions[a.roles['T']]; grid = np.unique(sessions[(sessions >= dates[0]) & (sessions <= dates[-1])])
        pos = np.searchsorted(grid, dates); names = tuple(losses)
        states.append({'task': a.state.task, 'fold': a.state.fold, 'losses': losses, 'names': names,
                       'counts': np.bincount(pos, minlength=len(grid)),
                       'sums': np.column_stack([np.bincount(pos, weights=losses[n]['endpoint'], minlength=len(grid)) for n in names])})
    passed = []
    for primary in results['primary_results']:
        task = primary['task']; selected = [s for s in states if s['task'] == task]; names = selected[0]['names']
        pooled = {n: {m: sum(s['losses'][n][m].sum() for s in selected) for m in ('endpoint', 'path')} for n in names}
        n = sum(int(s['counts'].sum()) for s in selected)
        zero = pooled['zero']['endpoint']; gain = float(100*(1-pooled[CAND]['endpoint']/zero)) if zero > 0 else None
        if gain is None:
            if primary['gain_overzero_pct'] is not None: raise AssertionError('Zero denominator qualification')
        else: close(np.asarray([primary['gain_overzero_pct']]), np.asarray([gain]), 'Pooled gain')
        gates = {'gain_at_least_1pct': bool(gain is not None and gain >= 1)}
        for s in selected: gates[f'fold{s["fold"]}_positive'] = bool(s['losses'][CAND]['endpoint'].sum() < s['losses']['zero']['endpoint'].sum())
        for seed in SEEDS: gates[f'R_seed{seed}_shrink_overzero'] = bool(pooled[f'R_seed{seed}_shrink']['endpoint'] < zero)
        for name in tuple(f'R_seed{s}_shrink' for s in SEEDS)+(CAND,):
            for ref in REFS:
                gates[name+'_endpoint_below_'+ref] = bool(pooled[name]['endpoint'] < pooled[ref]['endpoint'])
                gates[name+'_path_not_above_'+ref] = bool(pooled[name]['path'] <= pooled[ref]['path'])
        for name, row in pooled.items():
            for metric, value in row.items(): close(np.asarray([primary['pooled_MSE'][name][metric]]), np.asarray([value/n]), 'Pooled MSE')
        for block in ((1, 5) if task.startswith('minute') else (2, 5)):
            rng = np.random.Generator(np.random.PCG64(2026)); sample = np.zeros((5000, len(names))); counts = np.zeros(5000)
            valid = all(len(s['counts']) > block for s in selected)
            if valid:
                for s in selected:
                    size = len(s['counts'])
                    for begin in range(0, 5000, 100):
                        starts = rng.integers(0, size-block+1, (100, (size+block-1)//block))
                        index = (starts[:, :, None]+np.arange(block)).reshape(100, -1)[:, :size]
                        sample[begin:begin+100] += s['sums'][index].sum(axis=1)
                        counts[begin:begin+100] += s['counts'][index].sum(axis=1)
                valid = bool(np.all(counts > 0)); sample /= np.where(counts > 0, counts, 1)[:, None]
            for ref in REFS:
                old = next(r for r in primary['endpoint_intervals'] if r['block'] == block and r['reference'] == ref)
                delta = sample[:, names.index(ref)]-sample[:, names.index(CAND)]
                varying = bool(valid and np.isfinite(delta).all() and np.any(delta != delta[0]))
                ci = np.quantile(delta, [.025, .975], method='linear') if varying else None
                if ci is not None: close(np.asarray(old['difference_ci95']), ci, 'Independent endpoint interval')
                elif old.get('difference_ci95') is not None: raise AssertionError('Degenerate interval qualification')
                gates[f'CI_block{block}_vs_'+ref] = bool(ci is not None and ci[0] > 0)
        if gates != primary['gates'] or bool(all(gates.values())) != primary['passed']:
            raise AssertionError('Independent full scientific gate conjunction')
        passed.append(bool(all(gates.values()))); inputs.guard()
    if all(passed) != results['screen_conjunction_passed']: raise AssertionError('Two primary conjunction')
    report = {'status': 'PASS', 'protocol_sha256': protocol_sha, 'scalar_comparisons': int(checks),
              'maximum_absolute_numeric_difference': maximum_difference, 'screen_passed': all(passed),
              'direct_centered_Ridge_reconstructions': 4, 'same_source_adapter_and_NumPy_solve_backend': True,
              'neural_forwards_independently_replayed': False, 'source_files_hash_verified': True,
              'began_epoch': began, 'finished_epoch': time.time(), 'deadline_epoch': p['deadline_epoch'],
              'research_goal_achieved': False, 'Q3_read': False, 'GPU': False}
    inputs.guard()
    temporary = destination.with_suffix('.json.tmp')
    with temporary.open('x') as s: json.dump(report, s, indent=2, allow_nan=False); s.flush(); os.fsync(s.fileno())
    temporary.replace(destination); print(json.dumps(report))


if __name__ == '__main__': main()
