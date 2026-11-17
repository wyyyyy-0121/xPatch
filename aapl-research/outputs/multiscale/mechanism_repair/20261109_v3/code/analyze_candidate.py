"""Evaluate the one fixed repair recipe on all twenty original forecasting tasks."""
from study import *
import csv
from mechanism_stats import day_groups, paired_intervals
from multiscale.matrix_executor import read_bars


def prediction(path, expected_hash, origins, name='prediction'):
    assert sha(path) == expected_hash
    with np.load(path, allow_pickle=False) as a:
        assert np.array_equal(origins, a['origins']); p = a[name]
    assert np.isfinite(p).all()
    return p


def main():
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    manifest = read(BASE/'protocol/CANDIDATE_JOBS.json'); assert manifest['admitted']
    assert read(BASE/'audit/CANDIDATE_COMPLETE.json')['jobs'] == 130
    assert read(BASE/'audit/CANDIDATE_CALIBRATION_COMPLETE.json')['jobs'] == 130
    assert read(BASE/'audit/CROSS_SCALE_CONTROLS_COMPLETE.json')['folds'] == 100
    assert read(BASE/'audit/ORIGINAL_SHRINK_COMPLETE.json')['folds'] == 100
    assert source_hashes() == manifest['training_hashes']
    jobs = manifest['jobs']; model = manifest['architecture']
    rules = read(BASE/'protocol/CANDIDATE_RULES.json'); primary = set(rules['primary_return_tasks'])
    rows = []; pooled = []; folds = []; intervals = []; gates = []; overlaps = []
    for task in sorted(read(PRIOR/'protocol/DATA.json')['task_files']):
        meta = read(PRIOR/'cache'/task/'TASK.json'); target = meta['target']; profile = meta['profile']
        bars = read_bars(SOURCE, read(SOURCE/'admission.json'), meta['frequency'])
        assert bars.session_date[-1] <= '2026-06-30'
        groups = []; totals = {}; n = 0; seed_totals = {}; fold_losses = {}; path_sums = {}; mae_sums = {}
        baseline_name = 'zero' if target == 'future_log_return' else 'persistence'
        for fold in range(1, 6):
            rec = meta['folds'][str(fold)]
            assert sha(rec['targets']) == rec['targets_sha256']
            with np.load(rec['targets'], allow_pickle=False) as a:
                origins = a['origins']; y = a['actual']; anchor = a['anchor']
            baseline = np.zeros_like(y) if target == 'future_log_return' else np.broadcast_to(anchor[:, None], y.shape)
            losses = {baseline_name: np.square(baseline[:, -1]-y[:, -1])}
            fold_jobs = [j for j in jobs if task_key(j['profile'], j['horizon'], j['target']) == task and j['fold'] == fold]
            assert len(fold_jobs) == (3 if task in primary or task == 'hourly_h1_future_close' else 1)
            per_mode = {name: [] for name in ('candidate', 'trained', 'selected', 'coupled')}
            for cfg in fold_jobs:
                out = BASE/'runs'/cfg['id']; done = read(out/'DONE.json')
                assert done['status'] == 'completed' and done['job'] == cfg
                assert sha(out/'best_trained.pt') == done['checkpoint_sha256']
                trained = prediction(out/'predictions.npz', done['prediction_sha256'], origins, 'trained')
                coupled = prediction(out/'predictions.npz', done['prediction_sha256'], origins, 'coupled')
                cal_dir = BASE/'calibration_candidate'/cfg['id']; cal = read(cal_dir/'DONE.json')
                assert cal['source_checkpoint_sha256'] == done['checkpoint_sha256']
                candidate = prediction(cal_dir/'predictions.npz', cal['prediction_sha256'], origins)
                preds = {'candidate': candidate, 'trained': trained,
                         'selected': baseline if done['selected_baseline'] else trained, 'coupled': coupled}
                for name, pred in preds.items():
                    error = np.square(pred[:, -1]-y[:, -1]); per_mode[name].append(error)
                    metric = metrics(pred, y, anchor, target)
                    rows.append({'task': task, 'model': model, 'mode': name, 'fold': fold,
                        'seed': cfg['seed'], **metric, 'shrink_alpha': cal['alpha'], 'epochs': done['epochs'],
                        'best_epoch': done['trained_best_epoch'], 'selected_baseline': done['selected_baseline'],
                        'parameters': done['parameters']})
                    seed_totals.setdefault(name, {}).setdefault(cfg['seed'], 0.)
                    seed_totals[name][cfg['seed']] += error.sum()
                    path_sums[name] = path_sums.get(name, 0.)+metric['path_mse']*len(origins)/len(fold_jobs)
                    mae_sums[name] = mae_sums.get(name, 0.)+metric['endpoint_mae']*len(origins)/len(fold_jobs)
            losses.update({name: np.mean(errors, axis=0) for name, errors in per_mode.items()})
            cdir = BASE/'controls_cross_scale'/task/f'fold_{fold}'
            ridge = read(cdir/'DONE.json'); original = read(cdir/'ORIGINAL_SHRINK.json')
            assert original['model'] == model
            for name in ('history', 'history_anchor', 'history_selected', 'history_anchor_selected'):
                p = prediction(cdir/'predictions.npz', ridge['prediction_sha256'], origins, name)
                losses['ridge_'+name] = np.square(p[:, -1]-y[:, -1])
            for name, field in [('original', 'original'), ('original_shrink', 'shrink')]:
                p = prediction(cdir/'original_predictions.npz', original['prediction_sha256'], origins, field)
                losses[name] = np.square(p[:, -1]-y[:, -1])
            for name, loss in losses.items():
                totals[name] = totals.get(name, 0.)+loss.sum()
                fold_losses.setdefault(name, []).append(float(loss.mean()))
                folds.append({'task': task, 'model': model, 'method': name, 'fold': fold,
                    'origins': len(origins), 'endpoint_mse': float(loss.mean()),
                    'baseline_mse': float(losses[baseline_name].mean()),
                    'improvement_pct': float(100*(1-loss.mean()/losses[baseline_name].mean()))})
            n += len(origins); groups.append(day_groups(bars, origins, losses))
            npz(BASE/'analysis/candidate_paired_losses'/task/f'fold_{fold}.npz', origins=origins,
                session_date=bars.session_date[origins], **{'loss_'+name: value for name, value in losses.items()})
        for name, total in totals.items():
            pooled.append({'task': task, 'model': model, 'method': name, 'origins': n,
                'endpoint_mse': float(total/n), 'baseline_mse': float(totals[baseline_name]/n),
                'improvement_pct': float(100*(1-total/totals[baseline_name])),
                'positive_folds': sum(a < b for a, b in zip(fold_losses[name], fold_losses[baseline_name])),
                'seed_improvements_pct': {str(s): float(100*(1-value/totals[baseline_name]))
                                          for s, value in seed_totals.get(name, {}).items()},
                'seed_count': len(seed_totals.get(name, {})),
                'path_mse': float(path_sums[name]/n) if name in path_sums else None,
                'endpoint_mae': float(mae_sums[name]/n) if name in mae_sums else None})
        controls = [baseline_name, 'original', 'original_shrink', 'ridge_history', 'ridge_history_anchor',
                    'ridge_history_selected', 'ridge_history_anchor_selected']
        pairs = [('candidate', ref) for ref in controls]+[('trained', 'original'), ('selected', baseline_name)]
        blocks = {'annual': [12, 24], 'quarterly': [3, 6], 'monthly': [21, 42], 'minute': [1, 5], 'hourly': [2, 5]}[profile]
        interval_groups = groups
        if profile in ('annual', 'quarterly', 'monthly'):
            interval_groups = [{'names': groups[0]['names'], 'counts': np.concatenate([g['counts'] for g in groups]),
                                'sums': np.concatenate([g['sums'] for g in groups])}]
        local = []
        for block in blocks:
            local.extend(paired_intervals(interval_groups, pairs, block, task))
        for row in local:
            row['model'] = model
        intervals.extend(local)
        summary = next(r for r in pooled if r['task'] == task and r['method'] == 'candidate')
        relevant = [r for r in local if r['primary'] == 'candidate' and r['reference'] in controls]
        checks = {'minimum_one_percent': summary['improvement_pct'] >= 1.,
                  'all_three_seeds_positive': summary['seed_count'] == 3 and
                      all(value > 0 for value in summary['seed_improvements_pct'].values()),
                  'four_positive_folds': summary['positive_folds'] >= 4,
                  'all_control_intervals_both_blocks': len(relevant) == 2*len(controls) and
                      all(r['supports_improvement'] for r in relevant)}
        if task in primary:
            gates.append({'task': task, 'checks': checks, 'performance_gate_passed': all(checks.values())})
        if task in primary or task == 'hourly_h1_future_close':
            # Every reused candidate cell must retain its exact prior calibration.
            for cfg in [j for j in jobs if task_key(j['profile'], j['horizon'], j['target']) == task]:
                old = read(BASE/'calibration'/cfg['id']/'DONE.json'); new = read(BASE/'calibration_candidate'/cfg['id']/'DONE.json')
                with np.load(BASE/'calibration'/cfg['id']/'predictions.npz', allow_pickle=False) as a:
                    p = a['prediction']
                with np.load(BASE/'calibration_candidate'/cfg['id']/'predictions.npz', allow_pickle=False) as a:
                    q = a['prediction']
                exact = old['alpha'] == new['alpha'] and np.array_equal(p, q)
                assert exact; overlaps.append({'job': cfg['id'], 'stage_c_calibration_exact': exact})
        print(json.dumps({'candidate_analyzed': task, 'improvement_pct': summary['improvement_pct']}), flush=True)
    result = {'status': 'fixed_candidate_twenty_task_analysis_complete', 'model': model,
        'candidate_cells': len(jobs), 'task_count': 20, 'pooled': pooled, 'rows': rows, 'folds': folds,
        'intervals': intervals, 'primary_performance_gates': gates,
        'primary_tasks_both_pass': len(gates) == 2 and all(g['performance_gate_passed'] for g in gates),
        'exact_stage_c_overlap': overlaps, 'Q3_values_read': False,
        'independent_confirmation': False, 'method_success_claimed': False, 'at_utc': now()}
    atomic(BASE/'analysis/CANDIDATE_RESULTS.json', result)
    for filename, data in [('CANDIDATE_POOLED.csv', pooled), ('CANDIDATE_ROWS.csv', rows),
                           ('CANDIDATE_FOLDS.csv', folds), ('CANDIDATE_INTERVALS.csv', intervals)]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (BASE/'analysis'/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    print(json.dumps({'task_count': 20, 'both_primary_gates_pass': result['primary_tasks_both_pass'],
                      'primary_gates': gates}), flush=True)


if __name__ == '__main__':
    main()
