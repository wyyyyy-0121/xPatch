"""Frozen complete-matrix evaluation; pure loss/statistics functions permit synthetic audits.

The CLI refuses partial training and keeps the original deadline. Importing this
module performs no market loading, fitting, label construction, or scoring.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import sys
import time
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / 'code'))
import experiment as train

SEEDS = (2026, 2027, 2028)
HORIZONS = (5, 15, 30, 60)
MODES = ('last', 'last_shrink', 'selected', 'selected_shrink')
METHODS = tuple(arm+'_'+mode for arm in ('tied', 'untied') for mode in MODES)
CANDIDATE = 'tied_selected_shrink'
META = ('origins', 'session_date', 'day_grid')
V3B = BASE.parents[1] / 'mechanism_repair/20261109_v3b'
STOP = False


def guard():
    if STOP or time.time() >= train.DEADLINE:
        raise InterruptedError('Original deadline or signal; retain completed fold losses')


def interrupted(*_):
    global STOP
    STOP = True


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def expected_prediction_keys():
    return {'origins'} | {mode+'_h'+str(h) for mode in MODES for h in HORIZONS}


def complete_training_manifest(rules):
    """Validate all 30 receipts before opening any new outer target array."""
    assert rules['admitted'] and rules['fixed_epochs'] in (6, 20)
    assert len(rules['jobs']) == 30
    assert rules['sources'] == train.training_sources()
    assert rules['plan_sha256'] == train.sha(BASE/'PLAN_ZH.md')
    assert rules['horizons'] == list(HORIZONS) and rules['seeds'] == list(SEEDS)
    assert rules['folds'] == [1,2,3,4,5] and rules['all_cells_required'] is True
    expected = {(a, f, s) for a in ('tied', 'untied') for f in range(1, 6) for s in SEEDS}
    assert {(j['arm'], j['fold'], j['seed']) for j in rules['jobs']} == expected
    rule_hash = train.sha(BASE/'protocol/RULES.json')
    inputs = {}; receipts = {}; paired_steps = {}
    meta = train.read(train.PRIOR/'cache/minute_h60_future_log_return/TASK.json')
    import torch
    for job in rules['jobs']:
        guard()
        folder = BASE/'runs'/job['id']; path = folder/'DONE.json'
        done = train.read(path)
        identity = canonical_hash({'job': job, 'rule': rule_hash})
        assert done['status'] == 'completed' and done['job'] == job
        assert done['identity'] == identity and done['rules_sha256'] == rule_hash
        assert done['fixed_epochs'] == rules['fixed_epochs']
        assert done['Q3_values_read'] is False and done['independent_confirmation'] is False
        assert done['prediction_sha256'] == train.sha(folder/'predictions.npz')
        assert set(done['alphas']) == {'last','selected'}
        for mode in ('last','selected'):
            assert set(done['alphas'][mode]) == {str(h) for h in HORIZONS}
            assert all(0. <= float(v) <= 1. for v in done['alphas'][mode].values())
        for name, digest in done['checkpoint_hashes'].items():
            assert train.sha(folder/name) == digest
            saved = torch.load(folder/name, map_location='cpu', weights_only=True)
            assert saved['identity'] == identity and saved['rules_sha256'] == rule_hash
            assert saved['job'] == job and saved['scaler'] == done['scaler']
            if name == 'last.pt':
                assert saved['epoch'] == rules['fixed_epochs']
            elif name == 'best.pt':
                assert saved['epoch'] == done['best_epoch']
            inputs[str(folder/name)] = digest
        assert set(done['checkpoint_hashes']) == {'initial.pt', 'last.pt', 'best.pt'}
        split = meta['folds'][str(job['fold'])]['split_file']
        assert train.sha(split) == rules['partitions'][str(job['fold'])]['split_npz_sha256']
        with np.load(split, allow_pickle=False) as a:
            expected_origins = a['validation'].copy()
        with np.load(folder/'predictions.npz', allow_pickle=False) as a:
            assert set(a.files) == expected_prediction_keys() and np.array_equal(a['origins'], expected_origins)
            for h in HORIZONS:
                for mode in ('last','selected'):
                    raw = a[mode+'_h'+str(h)]
                    shrunk = a[mode+'_shrink_h'+str(h)]
                    assert raw.shape == shrunk.shape == (len(expected_origins),h)
                    assert np.isfinite(raw).all() and np.isfinite(shrunk).all()
                    alpha = float(done['alphas'][mode][str(h)])
                    assert np.array_equal(shrunk, raw*alpha), (job['id'],mode,h,'shrink mismatch')
        assert [r['epoch'] for r in done['history']] == list(range(1, rules['fixed_epochs']+1))
        assert done['history'][-1]['steps'] == done['steps']
        key = (job['fold'], job['seed'])
        paired_steps.setdefault(key, []).append(done['steps'])
        receipts[job['id']] = done
        inputs[str(path)] = train.sha(path)
        inputs[str(folder/'predictions.npz')] = done['prediction_sha256']
    assert all(len(v) == 2 and v[0] == v[1] for v in paired_steps.values())
    return inputs, receipts


def freeze_evaluation():
    guard(); rules = train.freeze()
    inputs, receipts = complete_training_manifest(rules)
    meta = train.read(train.PRIOR/'cache/minute_h60_future_log_return/TASK.json')
    for fold in range(1, 6):
        rec = meta['folds'][str(fold)]
        assert train.sha(rec['targets']) == rec['targets_sha256']
        assert train.sha(rec['scaler_file']) == rec['scaler_sha256']
        control = V3B/'analysis/paired_losses/minute_h60_future_log_return'/f'fold_{fold}.npz'
        for path in (Path(rec['targets']), Path(rec['split_file']), Path(rec['scaler_file']), control):
            inputs[str(path)] = train.sha(path)
    value = {'status': 'all_30_bundles_fixed_before_new_outer_scores',
        'rules_sha256': train.sha(BASE/'protocol/RULES.json'),
        'evaluator_sha256': train.sha(__file__), 'input_hashes': inputs,
        'candidate': CANDIDATE, 'horizons': list(HORIZONS), 'seeds': list(SEEDS),
        'blocks': [1, 5], 'repetitions': 5000, 'statistics_seed': 2026,
        'legacy_h60_columns': 33,
        'mean_control': 'common fit-only H60 target means; each H separately shrunk on its own inner endpoint',
        'seed_aggregation': 'per-origin arithmetic mean of three seed losses; prediction-ensemble losses separate',
        'primary_mechanism': 'H60 tied_last raw versus untied_last raw',
        'mechanism_gate': 'all three seed gains, >=4/5 positive paired folds, both block intervals positive',
        'candidate_gate': 'all three seed gains over zero, >=4/5 positive folds over zero, >=1% over zero, both block intervals positive versus every listed control',
        'scope': 'exposed historical development, nominal intervals, no multiplicity-adjusted confirmation',
        'deadline_epoch': train.DEADLINE, 'Q3_values_read': False,
        'independent_confirmation': False}
    path = BASE/'protocol/EVALUATION_FREEZE.json'
    if path.exists():
        old = train.read(path)
        assert {k:v for k,v in old.items() if k != 'at_utc'} == value
    else:
        train.atomic(path, dict(value, at_utc=train.now()))
    return train.read(path), receipts


def mean_control(mean_path, stop_actual, horizon):
    """Only fit means and inner labels; no outer data in calibration."""
    path = np.asarray(mean_path, dtype=np.float64)[:horizon]
    y = np.asarray(stop_actual, dtype=np.float64)
    assert path.shape == (horizon,) and y.ndim == 2 and y.shape[1] >= horizon
    p = np.full(len(y), path[-1]); denom = float(p@p)
    alpha = float(np.clip(p@y[:, horizon-1]/denom, 0., 1.)) if denom else 0.
    return path, alpha


def fold_losses(predictions, actual, mean_path, mean_alpha, legacy=None):
    """Pure arithmetic; returns endpoint/path losses, absolute errors and ensemble losses."""
    y = np.asarray(actual, dtype=np.float64); n, h = y.shape
    losses = {'zero': y[:, -1]**2}; paths = {'zero': np.mean(y*y, axis=1)}
    absolute = {'zero': abs(y[:, -1])}
    for name, p in (('fit_mean_raw', np.broadcast_to(mean_path, y.shape)),
                    ('fit_mean_shrink', np.broadcast_to(np.asarray(mean_path)*mean_alpha, y.shape))):
        e = np.asarray(p, dtype=np.float64)-y
        losses[name] = e[:, -1]**2; paths[name] = np.mean(e*e, axis=1); absolute[name] = abs(e[:, -1])
    for method in METHODS:
        arrays = [np.asarray(predictions[(method, seed)], dtype=np.float64) for seed in SEEDS]
        assert all(p.shape == (n,h) and np.isfinite(p).all() for p in arrays)
        errs = [p-y for p in arrays]
        for seed, e in zip(SEEDS, errs):
            name = method+'_seed_'+str(seed)
            losses[name] = e[:, -1]**2; paths[name] = np.mean(e*e, axis=1); absolute[name] = abs(e[:, -1])
        losses[method] = np.mean([e[:, -1]**2 for e in errs], axis=0)
        paths[method] = np.mean([np.mean(e*e, axis=1) for e in errs], axis=0)
        absolute[method] = np.mean([abs(e[:, -1]) for e in errs], axis=0)
        e = np.mean(arrays, axis=0)-y; name = method+'_prediction_ensemble'
        losses[name] = e[:, -1]**2; paths[name] = np.mean(e*e, axis=1); absolute[name] = abs(e[:, -1])
    if legacy is not None:
        assert len(legacy) == 33 and 'zero' in legacy
        assert np.array_equal(legacy['zero'], losses['zero'])
        for name, value in legacy.items():
            value = np.asarray(value)
            assert value.shape == (n,) and np.isfinite(value).all() and (value >= 0).all()
            losses['legacy_v3b_'+name] = value
    assert all(v.shape == (n,) and np.isfinite(v).all() and (v >= 0).all() for v in losses.values())
    return losses, paths, absolute


def grouped_days(dates, grid, losses):
    dates = np.asarray(dates); grid = np.asarray(grid)
    assert len(dates) and len(grid) and np.all(grid[1:] > grid[:-1]) and np.all(dates[1:] >= dates[:-1])
    positions = np.searchsorted(grid, dates)
    assert (positions < len(grid)).all() and np.array_equal(grid[positions], dates)
    return {'names': list(losses), 'counts': np.bincount(positions, minlength=len(grid)),
        'sums': np.column_stack([np.bincount(positions, weights=v, minlength=len(grid)) for v in losses.values()])}


def block_samples(groups, block, repetitions=5000, check=None):
    names = groups[0]['names']; rng = np.random.default_rng(2026)
    if any(len(g['counts']) <= block for g in groups):
        return None
    values = np.zeros((repetitions, len(names))); counts = np.zeros(repetitions)
    total = np.zeros(len(names)); n_total = 0
    for group in groups:
        assert group['names'] == names
        n = len(group['counts']); total += group['sums'].sum(0); n_total += group['counts'].sum()
        for start in range(0, repetitions, 100):
            if check:
                check()
            r = min(100, repetitions-start)
            begins = rng.integers(0, n-block+1, size=(r, (n+block-1)//block))
            ix = (begins[:, :, None]+np.arange(block)).reshape(r, -1)[:, :n]
            values[start:start+r] += group['sums'][ix].sum(1)
            counts[start:start+r] += group['counts'][ix].sum(1)
    assert (counts > 0).all()
    return total/n_total, values/counts[:, None]


def intervals(groups, pairs, block, repetitions=5000, check=None):
    sampled = block_samples(groups, block, repetitions, check)
    rows = []
    for primary, reference in pairs:
        row = {'primary': primary, 'reference': reference, 'block': block,
            'repetitions': repetitions, 'supports_improvement': False}
        if sampled is None:
            rows.append(dict(row, status='insufficient_groups')); continue
        point, samples = sampled; names = groups[0]['names']
        i,j = names.index(primary), names.index(reference); delta = samples[:,j]-samples[:,i]
        varying = bool(np.any(delta != delta[0]))
        ci = np.quantile(delta, [.025,.975]).tolist() if varying else None
        valid_denominator = point[j] > 0 and bool((samples[:,j] > 0).all())
        relative = 100*(1-samples[:,i]/samples[:,j]) if valid_denominator else None
        rows.append(dict(row, status='descriptive' if varying else 'degenerate',
            mse_reference_minus_primary=float(point[j]-point[i]), difference_ci95=ci,
            relative_improvement_pct=float(100*(1-point[i]/point[j])) if point[j] > 0 else None,
            relative_ci95_pct=np.quantile(relative,[.025,.975]).tolist() if varying and relative is not None else None,
            supports_improvement=bool(ci and ci[0] > 0),
            scope='exposed development; nominal and not independent or multiplicity-adjusted'))
    return rows


def summarize(folds, horizon, repetitions=5000, check=None):
    names = list(folds[0]['losses']); n = sum(len(f['origins']) for f in folds)
    assert len(folds) == 5 and all(list(f['losses']) == names for f in folds)
    totals = {name: sum(float(f['losses'][name].sum()) for f in folds) for name in names}
    byfold = {name: [float(f['losses'][name].mean()) for f in folds] for name in names}
    summary = {name: {'origins': n, 'endpoint_mse': value/n,
        'improvement_pct_over_zero': float(100*(1-value/totals['zero'])) if totals['zero'] else None,
        'fold_mse': byfold[name], 'positive_folds_over_zero': sum(a<b for a,b in zip(byfold[name],byfold['zero']))}
        for name,value in totals.items()}
    controls = ['zero','fit_mean_raw','fit_mean_shrink']+['untied_'+mode for mode in MODES]
    if horizon == 60:
        controls += [name for name in names if name.startswith('legacy_v3b_')]
        assert len(controls) == 40
    pairs = [('tied_last','untied_last')]+[(CANDIDATE,c) for c in controls]
    groups = [grouped_days(f['session_date'], f['day_grid'], f['losses']) for f in folds]
    ci = [row for block in (1,5) for row in intervals(groups,pairs,block,repetitions,check)]
    def direction(primary, reference):
        seed_gains = {str(seed): sum(float(f['losses'][primary+'_seed_'+str(seed)].sum()) for f in folds) <
            sum(float(f['losses'][reference+'_seed_'+str(seed)].sum()) if reference in METHODS else float(f['losses'][reference].sum()) for f in folds)
            for seed in SEEDS}
        return {'seed_positive': seed_gains,
            'positive_folds': sum(a<b for a,b in zip(byfold[primary],byfold[reference])),
            'both_block_intervals_support_improvement': all(r['supports_improvement'] for r in ci if r['primary']==primary and r['reference']==reference)}
    mech = direction('tied_last','untied_last')
    mech['passed'] = all(mech['seed_positive'].values()) and mech['positive_folds'] >= 4 and mech['both_block_intervals_support_improvement']
    candidate_direction = {c: direction(CANDIDATE,c) for c in controls}
    zero = candidate_direction['zero']
    checks = {'three_positive_seeds_over_zero': all(zero['seed_positive'].values()),
        'four_positive_folds_over_zero': zero['positive_folds'] >= 4,
        'minimum_one_percent_over_zero': summary[CANDIDATE]['improvement_pct_over_zero'] is not None and summary[CANDIDATE]['improvement_pct_over_zero'] >= 1.,
        'both_block_intervals_against_every_control': all(v['both_block_intervals_support_improvement'] for v in candidate_direction.values())}
    return {'horizon': horizon, 'summary': summary, 'mechanism': mech,
        'candidate_control_directions': candidate_direction, 'candidate_checks': checks,
        'candidate_performance_gate_passed': all(checks.values()), 'intervals': ci,
        'primary_for_study': horizon==60, 'independent_confirmation': False}


def evaluate_fold(fold, evaluation):
    guard()
    assert evaluation['evaluator_sha256'] == train.sha(__file__)
    for name, digest in evaluation['input_hashes'].items():
        assert train.sha(name) == digest
    data = train.data_for(fold)
    vi = data.splits['validation']; rec = data.rec
    with np.load(rec['targets'],allow_pickle=False) as a:
        assert np.array_equal(a['origins'],vi)
        actual = a['actual'].copy()
    assert actual.shape == (len(vi),60) and np.isfinite(actual).all()
    legacy_path = V3B/'analysis/paired_losses/minute_h60_future_log_return'/f'fold_{fold}.npz'
    with np.load(legacy_path,allow_pickle=False) as a:
        assert np.array_equal(a['origins'],vi)
        dates=a['session_date'].copy(); grid=a['day_grid'].copy()
        legacy={k:a[k].copy() for k in a.files if k not in META}
    assert len(legacy)==33 and np.array_equal(legacy['zero'],actual[:,-1]**2)
    assert np.array_equal(dates,data.bars.session_date[vi]) and str(dates[-1]) <= '2026-06-30'
    expected_grid=np.unique(data.bars.session_date[(data.bars.session_date>=dates[0])&(data.bars.session_date<=dates[-1])])
    assert np.array_equal(grid,expected_grid)
    from multiscale.bars import targets
    inner=targets(data.bars,data.splits['stop'],60,'future_log_return')
    predictions={}
    for arm in ('tied','untied'):
        for seed in SEEDS:
            guard(); path=BASE/'runs'/f'{arm}_f{fold}_s{seed}'/'predictions.npz'
            with np.load(path,allow_pickle=False) as a:
                assert set(a.files)==expected_prediction_keys() and np.array_equal(a['origins'],vi)
                for h in HORIZONS:
                    for mode in MODES:
                        predictions[(h,arm+'_'+mode,seed)]=a[mode+'_h'+str(h)].copy()
    receipts=[]
    for h in HORIZONS:
        guard(); mean,alpha=mean_control(data.scaler['y_mean'],inner,h)
        pred={(method,seed):predictions[(h,method,seed)] for method in METHODS for seed in SEEDS}
        losses,paths,absolute=fold_losses(pred,actual[:,:h],mean,alpha,legacy if h==60 else None)
        task=f'minute_h{h}_future_log_return'
        p=BASE/'analysis/paired_losses'/task/f'fold_{fold}.npz'
        q=BASE/'analysis/path_losses'/task/f'fold_{fold}.npz'
        r=BASE/'analysis/absolute_errors'/task/f'fold_{fold}.npz'
        train.npz(p,origins=vi,session_date=dates,day_grid=grid,**losses)
        train.npz(q,origins=vi,**paths); train.npz(r,origins=vi,**absolute)
        metric_rows=[{'method':name,'endpoint_mse':float(losses[name].mean()),
            'path_mse':float(paths[name].mean()),'endpoint_mae':float(absolute[name].mean())} for name in paths]
        receipt={'status':'completed','horizon':h,'fold':fold,'origins':len(vi),'metrics':metric_rows,
            'mean_control_alpha':alpha,'loss_names':list(losses),'paired_loss_sha256':train.sha(p),
            'path_loss_sha256':train.sha(q),'absolute_error_sha256':train.sha(r),
            'evaluation_freeze_sha256':train.sha(BASE/'protocol/EVALUATION_FREEZE.json'),
            'Q3_values_read':False,'at_utc':train.now()}
        train.atomic(BASE/'analysis/folds'/task/f'fold_{fold}.json',receipt);receipts.append(receipt)
    return receipts


def statistics(evaluation):
    tasks={}; all_rows=[]
    for h in HORIZONS:
        folds=[];task=f'minute_h{h}_future_log_return'
        for fold in range(1,6):
            guard(); receipt=train.read(BASE/'analysis/folds'/task/f'fold_{fold}.json')
            assert receipt['evaluation_freeze_sha256']==train.sha(BASE/'protocol/EVALUATION_FREEZE.json')
            path=BASE/'analysis/paired_losses'/task/f'fold_{fold}.npz'
            assert train.sha(path)==receipt['paired_loss_sha256']
            for field,subdir in (('path_loss_sha256','path_losses'),('absolute_error_sha256','absolute_errors')):
                assert train.sha(BASE/'analysis'/subdir/task/f'fold_{fold}.npz')==receipt[field]
            with np.load(path,allow_pickle=False) as a:
                folds.append({'origins':a['origins'].copy(),'session_date':a['session_date'].copy(),
                    'day_grid':a['day_grid'].copy(),'losses':{k:a[k].copy() for k in a.files if k not in META}})
            all_rows.append(receipt)
        tasks[task]=summarize(folds,h,check=guard)
    result={'status':'complete_30_bundle_five_fold_three_seed_development_analysis','tasks':tasks,
        'candidate':CANDIDATE,'primary_task':'minute_h60_future_log_return',
        'primary_mechanism_gate_passed':tasks['minute_h60_future_log_return']['mechanism']['passed'],
        'primary_candidate_performance_gate_passed':tasks['minute_h60_future_log_return']['candidate_performance_gate_passed'],
        'rules_sha256':train.sha(BASE/'protocol/RULES.json'),
        'evaluation_freeze_sha256':train.sha(BASE/'protocol/EVALUATION_FREEZE.json'),
        'Q3_values_read':False,'independent_confirmation':False,
        'research_goal_achieved':False,'hourly_primary_goal_completed_by_this_study':False,'at_utc':train.now()}
    train.atomic(BASE/'analysis/STATS.json',result)
    train.atomic(BASE/'analysis/RESULTS.json',{'status':'completed','fold_horizon_cells':20,
        'bundles':30,'metrics':all_rows,'statistics_sha256':train.sha(BASE/'analysis/STATS.json'),
        'Q3_values_read':False,'independent_confirmation':False})
    train.atomic(BASE/'analysis/ANALYSIS_COMPLETE.json',{'status':'completed',
        'statistics_sha256':train.sha(BASE/'analysis/STATS.json'),
        'results_sha256':train.sha(BASE/'analysis/RESULTS.json'),
        'evaluation_freeze_sha256':train.sha(BASE/'protocol/EVALUATION_FREEZE.json'),
        'research_goal_achieved':False,'Q3_values_read':False,'at_utc':train.now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('freeze','run'),default='run')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and len(os.sched_getaffinity(0))<=2
    for sig in (signal.SIGUSR1,signal.SIGTERM,signal.SIGINT):signal.signal(sig,interrupted)
    try:
        evaluation,_=freeze_evaluation()
        if args.stage=='freeze':return
        for fold in range(1,6):
            # Fold outputs are idempotent and overwritten atomically if partial.
            evaluate_fold(fold,evaluation)
        statistics(evaluation)
    except InterruptedError as exc:
        train.atomic(BASE/'audit/ANALYSIS_PAUSED.json',{'status':'paused','reason':str(exc),
            'completed_folds_retained':True,'deadline_epoch':train.DEADLINE,'at_utc':train.now()})


if __name__=='__main__':main()
