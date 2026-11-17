"""Paired development analysis; no fitting or recipe selection in this file."""
from common import *
from datetime import datetime, timezone
import csv

SEEDS = (2026, 2027, 2028)
HORIZONS = (5, 15, 30, 60)


def load_prediction(folder, origins):
    done = read(folder / 'DONE.json')
    assert done['status'] == 'completed'
    assert sha(folder / 'prediction.npz') == done['prediction_sha256']
    with np.load(folder / 'prediction.npz', allow_pickle=False) as a:
        assert np.array_equal(a['origins'], origins)
        p = a['prediction'].astype('float64')
    assert p.shape == (len(origins), 60) and np.isfinite(p).all()
    return p, done


def measures(p, y):
    sq = (p-y)**2
    out = {'origins': len(y), 'path_mse': float(sq.mean()),
           'path_mae': float(np.abs(p-y).mean()),
           'per_step_mse': sq.mean(0).tolist()}
    for h in HORIZONS:
        mse, zero = sq[:, h-1].mean(), (y[:, h-1]**2).mean()
        out[f'h{h}_mse'] = float(mse)
        out[f'h{h}_mae'] = float(np.abs(p[:, h-1]-y[:, h-1]).mean())
        out[f'h{h}_improvement_pct'] = float(100*(1-mse/zero))
    return out


def bootstrap(groups, names, primary, block):
    """Resample full non-circular session blocks independently within each fold.

    The same block draws apply to every model. Seeds have already been averaged
    in per-origin loss. Zero-origin sessions remain on each full session grid.
    """
    rng = np.random.default_rng(2026)
    repetitions = 5000
    sampled = np.zeros((repetitions, len(names)), dtype='float64')
    sampled_n = np.zeros(repetitions)
    point = np.zeros(len(names)); point_n = 0
    offsets = np.arange(block)
    for g in groups:
        n = len(g['counts'])
        assert n >= block
        point += g['sums'].sum(0); point_n += g['counts'].sum()
        for first in range(0, repetitions, 100):
            take = min(100, repetitions-first)
            starts = rng.integers(0, n-block+1, size=(take, (n+block-1)//block))
            idx = (starts[:, :, None]+offsets).reshape(take, -1)[:, :n]
            sampled[first:first+take] += g['sums'][idx].sum(1)
            sampled_n[first:first+take] += g['counts'][idx].sum(1)
    assert (sampled_n > 0).all()
    point /= point_n
    sampled /= sampled_n[:, None]
    pi = names.index(primary); result = []
    for j, name in enumerate(names):
        if j == pi:
            continue
        delta = sampled[:, j]-sampled[:, pi]
        rel = 100*delta/sampled[:, j]
        lo, hi = np.quantile(delta, [.025, .975])
        result.append({'primary': primary, 'reference': name,
            'block_sessions': block, 'repetitions': repetitions,
            'positive_favors_primary': True,
            'mse_reference_minus_primary': float(point[j]-point[pi]),
            'difference_ci95': [float(lo), float(hi)],
            'relative_improvement_pct': float(100*(1-point[pi]/point[j])),
            'relative_ci95_pct': np.quantile(rel, [.025, .975]).tolist(),
            'supports_improvement': bool(lo > 0),
            'scope': 'descriptive historical development; not multiplicity adjusted'})
    return result


def aggregate():
    assert (BASE/'audit/TRAINING_COMPLETE.json').exists()
    decision = read(BASE/'protocol/STAGE_C_DECISION.json')
    kinds = decision['formal_kinds']
    primary = decision['primary']
    names = ['zero']+kinds+['linear_same_features', 'lstm_constant_shrink']
    bars, _, folds, _ = bars_and_folds()
    cache = cache_arrays()
    out = BASE/'analysis'; out.mkdir(exist_ok=True)
    rows, ensembles, grouped, training = [], [], [], []
    fold_means, state_rows, session_info = [], [], []
    for f in folds:
        fold = f['fold']; origins = f['validation']; rr = rows_for(origins)
        y = np.asarray(cache['actual'][rr])
        endpoints = {'zero': y[:, -1]**2}
        rows.append({'fold': fold, 'kind': 'zero', 'seed': 0, **measures(np.zeros_like(y), y)})
        threshold = float(np.median(cache['volatility'][rows_for(f['inner']['fit'])]))
        high = np.asarray(cache['volatility'][rr]) > threshold
        for kind in kinds+['linear_same_features', 'lstm_constant_shrink']:
            seedlist = [0] if kind == 'linear_same_features' else SEEDS
            psum = np.zeros_like(y); lsum = np.zeros(len(y))
            for seed in seedlist:
                if kind == 'linear_same_features':
                    folder = BASE/'controls'/f'linear_f{fold}'
                elif kind == 'lstm_constant_shrink':
                    folder = BASE/'controls'/f'shrink_lstm_f{fold}_s{seed}'
                else:
                    folder = BASE/'runs'/f'formal_{kind}_f{fold}_s{seed}'
                p, done = load_prediction(folder, origins)
                psum += p
                lsum += (p[:, -1]-y[:, -1])**2
                rows.append({'fold': fold, 'kind': kind, 'seed': seed, **measures(p, y)})
                if kind in kinds:
                    training.append({'fold': fold, 'kind': kind, 'seed': seed,
                        'reused_original': done.get('reused_original', False),
                        'best_epoch': done['best_epoch'],
                        'epochs': done.get('epochs', len(done.get('history', []))),
                        'training_wall_seconds': done.get('training_wall_seconds', 0),
                        'training_cpu_seconds': done.get('training_cpu_seconds', 0),
                        'last_invocation_seconds': done.get('total_last_invocation_seconds', 0),
                        'prediction_sha256': done['prediction_sha256'],
                        'checkpoint_sha256': done['checkpoint_sha256']})
            endpoints[kind] = lsum/len(seedlist)
            ensembles.append({'fold': fold, 'kind': kind,
                              'seed_count': len(seedlist), **measures(psum/len(seedlist), y)})
        for name in names:
            loss = endpoints[name]
            mse = float(loss.mean()); zero = float(endpoints['zero'].mean())
            fold_means.append({'fold': fold, 'kind': name, 'origins': len(origins),
                              'mse': mse, 'improvement_pct': 100*(1-mse/zero)})
            for state, mask in [('low', ~high), ('high', high)]:
                if mask.any():
                    state_rows.append({'fold': fold, 'kind': name, 'state': state,
                        'origins': int(mask.sum()), 'mse': float(loss[mask].mean()),
                        'zero_mse': float(endpoints['zero'][mask].mean()),
                        'threshold_from_inner_fit': threshold})
        lo, hi = origins.min(), origins.max()
        sessions = np.unique(bars.session_date[lo:hi+1])
        si = np.searchsorted(sessions, bars.session_date[origins])
        counts = np.bincount(si, minlength=len(sessions)).astype('int64')
        matrix = np.column_stack([np.bincount(si, weights=endpoints[n], minlength=len(sessions)) for n in names])
        grouped.append({'counts': counts, 'sums': matrix})
        session_info.append({'fold': fold, 'sessions': len(sessions),
            'zero_origin_sessions': int((counts == 0).sum()),
            'first_validation_session': str(sessions[0]), 'last_validation_session': str(sessions[-1]),
            'fit_first': str(bars.session_date[f['inner']['fit'][0]]),
            'fit_last': str(bars.session_date[f['inner']['fit'][-1]]),
            'stop_first': str(bars.session_date[f['inner']['stop'][0]]),
            'stop_last': str(bars.session_date[f['inner']['stop'][-1]]),
            'fit_origins': len(f['inner']['fit']), 'stop_origins': len(f['inner']['stop']),
            'validation_origins': len(origins)})
        np.savez_compressed(out/f'paired_losses_fold{fold}.npz', origins=origins,
            session_index=si, sessions=sessions, counts=counts, session_loss_sums=matrix,
            **{f'loss_{n}': endpoints[n] for n in names})
    def pooled(records, kind, key):
        a = [r for r in records if r['kind'] == kind]
        return float(sum(r['origins']*r[key] for r in a)/sum(r['origins'] for r in a))
    pool = []
    for name in names:
        mse = pooled(rows, name, 'h60_mse'); zero = pooled(rows, 'zero', 'h60_mse')
        seed_scores = {}
        for seed in SEEDS:
            a = [r for r in rows if r['kind'] == name and r['seed'] == seed]
            if a:
                sm = sum(r['origins']*r['h60_mse'] for r in a)/sum(r['origins'] for r in a)
                seed_scores[str(seed)] = 100*(1-sm/zero)
        item = {'kind': name, 'h60_mse': mse, 'h60_improvement_pct': 100*(1-mse/zero),
            'seed_improvements_pct': seed_scores,
            'positive_folds': sum(r['improvement_pct'] > 0 for r in fold_means if r['kind'] == name),
            'ensemble_h60_mse': pooled(ensembles, name, 'h60_mse') if name != 'zero' else zero,
            'ensemble_h60_improvement_pct': 100*(1-pooled(ensembles, name, 'h60_mse')/zero) if name != 'zero' else 0.}
        for metric in ['path_mse', 'path_mae']+[f'h{h}_{m}' for h in HORIZONS for m in ('mse','mae')]:
            item[metric] = pooled(rows, name, metric)
        item['per_step_mse'] = np.average(
            np.array([r['per_step_mse'] for r in rows if r['kind']==name]), axis=0,
            weights=[r['origins'] for r in rows if r['kind']==name]).tolist()
        pool.append(item)
    intervals = bootstrap(grouped, names, primary, 1)+bootstrap(grouped, names, primary, 5)
    p = next(r for r in pool if r['kind'] == primary)
    comparators = ['zero', 'lstm', 'linear_same_features', 'lstm_constant_shrink']
    ablations = ['short_plain', 'short_const'] if primary == 'short_state' else ['trend','short','fixed_mix','mix_noshrink']
    all_ci = lambda refs: all(r['supports_improvement'] for r in intervals if r['reference'] in refs)
    checks = {'minimum_one_percent': p['h60_improvement_pct'] >= 1.,
        'all_three_seeds_positive': len(p['seed_improvements_pct'])==3 and all(x>0 for x in p['seed_improvements_pct'].values()),
        'at_least_four_positive_folds': p['positive_folds'] >= 4,
        'beats_all_controls_at_point': all(p['h60_mse'] < next(r['h60_mse'] for r in pool if r['kind']==c) for c in comparators),
        'control_intervals_both_block_lengths_support': all_ci(comparators),
        'module_intervals_both_block_lengths_support': all_ci(ablations)}
    result = {'status': 'complete_development_analysis', 'at_utc': datetime.now(timezone.utc).isoformat(),
        'primary': primary, 'two_stream_gate_passed': decision['two_stream_gate_passed'],
        'origins': sum(g['counts'].sum() for g in grouped), 'session_grid': session_info,
        'pooled': pool, 'fold_means': fold_means, 'fold_seed_metrics': rows,
        'ensemble_metrics_separate': ensembles, 'state_metrics': state_rows,
        'paired_intervals': intervals, 'performance_checks': checks,
        'performance_goal_passed': all(checks.values()),
        'training_records': training, 'seeds_are_not_market_replicates': True,
        'intervals_descriptive_not_multiplicity_adjusted': True,
        'independent_confirmation': False, 'Q3_values_read': False,
        'decision_sha256': sha(BASE/'protocol/STAGE_C_DECISION.json')}
    atomic(out/'RESULTS.json', result)
    flat = [{k:v for k,v in r.items() if not isinstance(v, (list,dict))} for r in rows]
    for filename, records in [('FOLD_SEED_METRICS.csv', flat), ('FOLD_MEANS.csv', fold_means), ('TRAINING_COSTS.csv', training)]:
        with (out/filename).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
    print(json.dumps({'primary': primary, 'performance_checks': checks,
        'pooled_improvements': {r['kind']:r['h60_improvement_pct'] for r in pool}}), flush=True)


if __name__ == '__main__':
    aggregate()
