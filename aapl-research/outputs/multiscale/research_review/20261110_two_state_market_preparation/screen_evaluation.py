"""Fixed historical-development screen arithmetic; no files, fitting or selection.

Seeds are averaged as losses, never as predictions or independent market draws.
"""
from __future__ import annotations
import numpy as np

SEEDS = (2026, 2027, 2028)
REFERENCES = ('zero', 'Q_mean_raw', 'Q_mean_shrink', 'E_Ridge_raw', 'E_Ridge_shrink') + tuple(
    f'C_seed{s}_{mode}' for s in SEEDS for mode in ('raw', 'shrink')) + ('C_seedmean_raw', 'C_seedmean_shrink')
MODEL_NAMES = ('Q_mean', 'E_Ridge') + tuple(f'{a}_seed{s}' for a in ('C', 'R') for s in SEEDS)
CANDIDATE = 'R_seedmean_shrink'
TASK_BLOCKS = {'minute_h60_future_log_return': (1, 5), 'hourly_h8_future_log_return': (2, 5)}


def require(value, message):
    if not bool(value):
        raise ValueError(message)


def origin_losses(truth, raw_predictions, alphas):
    require(isinstance(truth, np.ndarray) and truth.dtype == np.float64 and truth.ndim == 2
            and truth.shape[0] > 0 and truth.shape[1] > 0 and np.isfinite(truth).all(), 'Raw target paths')
    require(set(raw_predictions) == set(MODEL_NAMES) and set(alphas) == set(MODEL_NAMES),
            'Exact eight models/alphas required')
    losses = {}

    def add(name, p):
        error = (p-truth)**2
        endpoint = error[:, -1].copy(); path = error.mean(axis=1, dtype=np.float64)
        require(np.isfinite(endpoint).all() and np.isfinite(path).all(), 'Nonfinite loss')
        losses[name] = {'endpoint': endpoint, 'path': path}

    add('zero', np.zeros_like(truth))
    for name in MODEL_NAMES:
        p = raw_predictions[name]; a = alphas[name]
        require(isinstance(p, np.ndarray) and p.dtype == np.float64 and p.shape == truth.shape
                and np.isfinite(p).all(), 'Raw model path: '+name)
        require(type(a) in (int, float) and np.isfinite(a) and 0 <= a <= 1, 'Frozen alpha: '+name)
        add(name+'_raw', p); add(name+'_shrink', a*p)
    for arm in ('C', 'R'):
        for mode in ('raw', 'shrink'):
            losses[f'{arm}_seedmean_{mode}'] = {
                metric: np.mean(np.stack([losses[f'{arm}_seed{s}_{mode}'][metric] for s in SEEDS]),
                                axis=0, dtype=np.float64) for metric in ('endpoint', 'path')}
    return losses


def day_groups(all_session_dates, origins, losses):
    require(isinstance(all_session_dates, np.ndarray) and all_session_dates.dtype.kind == 'U'
            and all_session_dates.ndim == 1 and np.all(all_session_dates[1:] >= all_session_dates[:-1]),
            'Original ordered cached session dates')
    require(isinstance(origins, np.ndarray) and origins.dtype == np.int64 and origins.ndim == 1
            and len(origins) > 0 and np.all(origins[1:] > origins[:-1])
            and origins[0] >= 0 and origins[-1] < len(all_session_dates), 'Original ordered T origins')
    dates = all_session_dates[origins]
    grid = np.unique(all_session_dates[(all_session_dates >= dates[0]) & (all_session_dates <= dates[-1])])
    names = tuple(losses); positions = np.searchsorted(grid, dates)
    require(set(REFERENCES) <= set(names) and CANDIDATE in names, 'All frozen screen references')
    for row in losses.values():
        require(set(row) == {'endpoint', 'path'} and all(
            a.dtype == np.float64 and a.shape == (len(origins),) and np.isfinite(a).all()
            and np.all(a >= 0) for a in row.values()), 'Aligned nonnegative per-origin losses')
    return {'names': names, 'session_grid': grid, 'counts': np.bincount(positions, minlength=len(grid)),
            'sums': np.column_stack([np.bincount(positions, weights=losses[n]['endpoint'],
                                               minlength=len(grid)) for n in names])}


def paired_intervals(groups, block):
    require(len(groups) == 2 and groups[0]['names'] == groups[1]['names'], 'Two ordered fold strata')
    names = groups[0]['names']; rows = []
    if any(len(g['counts']) <= block for g in groups):
        return [{'reference': r, 'block': block, 'status': 'insufficient_groups',
                 'supports_improvement': False} for r in REFERENCES]
    rng = np.random.Generator(np.random.PCG64(2026))
    samples = np.zeros((5000, len(names)), dtype=np.float64); counts = np.zeros(5000)
    total = np.zeros(len(names), dtype=np.float64); n_origins = 0
    for g in groups:
        n = len(g['counts']); total += g['sums'].sum(axis=0); n_origins += int(g['counts'].sum())
        for start in range(0, 5000, 100):
            begins = rng.integers(0, n-block+1, size=(min(100, 5000-start), (n+block-1)//block))
            indices = (begins[:, :, None]+np.arange(block)).reshape(len(begins), -1)[:, :n]
            samples[start:start+len(begins)] += g['sums'][indices].sum(axis=1)
            counts[start:start+len(begins)] += g['counts'][indices].sum(axis=1)
    if n_origins <= 0 or not np.all(counts > 0) or not np.isfinite(samples).all():
        return [{'reference': r, 'block': block, 'status': 'undefined_sample_count_or_loss',
                 'supports_improvement': False} for r in REFERENCES]
    point = total/n_origins; samples /= counts[:, None]
    i = names.index(CANDIDATE)
    for reference in REFERENCES:
        j = names.index(reference); delta = samples[:, j]-samples[:, i]
        finite = bool(np.isfinite(delta).all()); varying = bool(finite and np.any(delta != delta[0]))
        ci = np.quantile(delta, [.025, .975], method='linear').tolist() if varying else None
        relative_valid = bool(point[j] > 0 and np.all(samples[:, j] > 0))
        relative = 100*(1-samples[:, i]/samples[:, j]) if relative_valid else None
        relative_point = float(100*(1-point[i]/point[j])) if relative_valid else None
        relative_finite = bool(relative_valid and np.isfinite(relative).all()
                               and np.isfinite(relative_point))
        rows.append({'reference': reference, 'block': block, 'repetitions': 5000,
                     'status': 'descriptive' if varying else ('degenerate' if finite else 'nonfinite'),
                     'mse_reference_minus_candidate': float(point[j]-point[i]), 'difference_ci95': ci,
                     'relative_improvement_pct': relative_point if relative_finite else None,
                     'relative_ci95_pct': np.quantile(relative, [.025, .975], method='linear').tolist()
                        if relative_finite and varying else None,
                     'relative_status': 'descriptive' if relative_finite else 'NA_nonpositive_denominator_or_nonfinite',
                     'supports_improvement': bool(ci is not None and ci[0] > 0),
                     'scope': 'nominal exposed historical development; not independent confirmation'})
    return rows


def aggregate_primary(states):
    require(len(states) == 2 and [s['fold'] for s in states] == [1, 2]
            and len({s['state_uid'] for s in states}) == 2
            and len({s['task'] for s in states}) == 1, 'Exact two state-bound ordered folds')
    task = states[0]['task']; require(task in TASK_BLOCKS, 'Original primary task')
    names = tuple(states[0]['losses'])
    require(all(tuple(s['losses']) == names for s in states), 'Identical ordered loss definitions')
    sums = {n: {m: sum(float(s['losses'][n][m].sum(dtype=np.float64)) for s in states)
                for m in ('endpoint', 'path')} for n in names}
    zero = sums['zero']['endpoint']; candidate = sums[CANDIDATE]['endpoint']
    gain = float(100*(1-candidate/zero)) if np.isfinite(zero) and zero > 0 else None
    gates = {'gain_at_least_1pct': bool(gain is not None and np.isfinite(gain) and gain >= 1)}
    for s in states:
        gates[f'fold{s["fold"]}_positive'] = bool(
            s['losses'][CANDIDATE]['endpoint'].sum() < s['losses']['zero']['endpoint'].sum())
    for seed in SEEDS:
        name = f'R_seed{seed}_shrink'
        gates[name+'_overzero'] = bool(sums[name]['endpoint'] < zero)
    for n in tuple(f'R_seed{s}_shrink' for s in SEEDS)+(CANDIDATE,):
        for r in REFERENCES:
            gates[n+'_endpoint_below_'+r] = bool(sums[n]['endpoint'] < sums[r]['endpoint'])
            gates[n+'_path_not_above_'+r] = bool(sums[n]['path'] <= sums[r]['path'])
    groups = [day_groups(s['session_date'], s['origins'], s['losses']) for s in states]
    intervals = []
    for block in TASK_BLOCKS[task]:
        rows = paired_intervals(groups, block); intervals.extend(rows)
        for row in rows:
            gates[f'CI_block{block}_vs_'+row['reference']] = row['supports_improvement']
    require(all(np.isfinite(v) and v >= 0 for row in sums.values() for v in row.values()),
            'Nonfinite pooled losses')
    n_origins = sum(len(s['origins']) for s in states)
    return {'task': task, 'T_origin_occurrences': n_origins, 'gain_overzero_pct': gain,
            'pooled_MSE': {n: {m: v/n_origins for m, v in row.items()} for n, row in sums.items()},
            'gates': gates, 'passed': bool(all(gates.values())), 'endpoint_intervals': intervals,
            'screen_only': True, 'method_success': False, 'independent_confirmation': False}
