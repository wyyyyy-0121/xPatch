"""Read the entire frozen continuation, preserving controls and all seed losses."""
from study import *
import csv, torch
from mechanism_stats import day_groups, paired_intervals, interaction
from multiscale.matrix_executor import read_bars

MODES = ('trained', 'selected', 'coupled', 'shrink')


def load_prediction(path, expected_sha, origins, name):
    assert sha(path) == expected_sha
    with np.load(path, allow_pickle=False) as a:
        assert np.array_equal(a['origins'], origins)
        value = a[name]
    assert np.isfinite(value).all()
    return value


def registered_pairs(target):
    if target == 'future_log_return':
        return [('noanchor_random', 'anchor_random'), ('noanchor_baseline', 'anchor_baseline'),
                ('anchor_baseline', 'anchor_random'), ('noanchor_baseline', 'noanchor_random')]
    return [('residual_random', 'price_random'), ('residual_baseline', 'price_baseline'),
            ('price_baseline', 'price_random'), ('residual_baseline', 'residual_random')]


def main():
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    manifest = read(BASE/'protocol/REPLICATION_JOBS.json')
    complete = read(BASE/'audit/REPLICATION_COMPLETE.json')
    assert complete['jobs'] == len(manifest['jobs'])
    assert read(BASE/'audit/CALIBRATION_COMPLETE.json')['jobs'] == complete['jobs']
    assert read(BASE/'audit/INFORMATION_CONTROLS_COMPLETE.json')['folds'] == 15
    assert manifest['training_hashes'] == source_hashes()
    rows = []; pooled = []; intervals = []; interactions = []; contrasts = []; gates = []; restorations = []
    for group in manifest['groups']:
        if not group['admitted']:
            continue
        task, model = group['task'], group['model']
        jobs = [j for j in manifest['jobs'] if task_key(j['profile'], j['horizon'], j['target']) == task
                and j['model'] == model]
        meta = read(PRIOR/'cache'/task/'TASK.json'); target = meta['target']; profile = meta['profile']
        assert len(jobs) == (15 if model == 'tiny' else 60)
        variants = list(dict.fromkeys(j['variant'] for j in jobs))
        original = 'anchor_random' if target == 'future_log_return' else 'price_random'
        baseline_name = 'zero' if target == 'future_log_return' else 'persistence'
        bars = read_bars(SOURCE, read(SOURCE/'admission.json'), meta['frequency'])
        assert bars.session_date[-1] <= '2026-06-30'
        groups = []; totals = {}; seed_totals = {}; fold_losses = {}; n = 0; base_total = 0.
        for fold in range(1, 6):
            rec = meta['folds'][str(fold)]
            assert sha(rec['targets']) == rec['targets_sha256']
            with np.load(rec['targets'], allow_pickle=False) as a:
                origins = a['origins']; truth = a['actual']; anchor = a['anchor']
            baseline = np.zeros_like(truth) if target == 'future_log_return' else \
                np.broadcast_to(anchor[:, None], truth.shape)
            losses = {baseline_name: np.square(baseline[:, -1]-truth[:, -1])}
            n += len(origins); base_total += losses[baseline_name].sum()
            for variant in variants:
                per_mode = {mode: [] for mode in MODES}
                for seed in (2026, 2027, 2028):
                    cfg = next(j for j in jobs if j['fold'] == fold and j['seed'] == seed and j['variant'] == variant)
                    folder = BASE/'runs'/cfg['id']; done = read(folder/'DONE.json')
                    assert done['status'] == 'completed' and done['job'] == cfg
                    assert sha(folder/'best_trained.pt') == done['checkpoint_sha256']
                    if seed == 2026 and variant == original:
                        old_rec = rec['models'][model]
                        assert sha(old_rec['checkpoint']) == old_rec['checkpoint_sha256']
                        a = torch.load(folder/'best_trained.pt', map_location='cpu', weights_only=True)['model']
                        b = torch.load(old_rec['checkpoint'], map_location='cpu', weights_only=True)['model']
                        previous = read(old_rec['summary'])
                        verified = a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a) and \
                            done['epochs'] == previous['epochs'] and done['trained_best_epoch'] == previous['best_epoch']
                        restorations.append({'job': cfg['id'], 'model_and_stopping_exact': verified})
                        assert verified, 'Full-fold original recipe restoration discrepancy: '+cfg['id']
                    trained = load_prediction(folder/'predictions.npz', done['prediction_sha256'], origins, 'trained')
                    coupled = load_prediction(folder/'predictions.npz', done['prediction_sha256'], origins, 'coupled')
                    cal = read(BASE/'calibration'/cfg['id']/'DONE.json')
                    shrink = load_prediction(BASE/'calibration'/cfg['id']/'predictions.npz',
                                             cal['prediction_sha256'], origins, 'prediction')
                    assert cal['source_checkpoint_sha256'] == done['checkpoint_sha256']
                    predictions = {'trained': trained, 'selected': baseline if done['selected_baseline'] else trained,
                                   'coupled': coupled, 'shrink': shrink}
                    for mode, prediction in predictions.items():
                        values = metrics(prediction, truth, anchor, target)
                        if mode != 'shrink':
                            assert np.isclose(values['endpoint_mse'], done['metrics'][mode]['endpoint_mse'], rtol=1e-12)
                        name = variant+'__'+mode; error = np.square(prediction[:, -1]-truth[:, -1])
                        per_mode[mode].append(error)
                        seed_totals.setdefault(name, {}).setdefault(seed, 0.)
                        seed_totals[name][seed] += error.sum()
                        rows.append({'task': task, 'model': model, 'variant': variant, 'mode': mode,
                            'fold': fold, 'seed': seed, **values, 'epochs': done['epochs'],
                            'best_epoch': done['trained_best_epoch'], 'selected_baseline': done['selected_baseline'],
                            'shrink_alpha': cal['alpha'], 'parameters': done['parameters']})
                for mode in MODES:
                    name = variant+'__'+mode
                    losses[name] = np.mean(per_mode[mode], axis=0)
            controls_dir = BASE/'controls'/task/f'fold_{fold}'
            controls = read(controls_dir/'DONE.json')
            for name in ('history', 'history_anchor', 'history_selected', 'history_anchor_selected'):
                pred = load_prediction(controls_dir/'predictions.npz', controls['prediction_sha256'], origins, name)
                losses['ridge_'+name] = np.square(pred[:, -1]-truth[:, -1])
            old = rec['models'][model]
            old_pred = load_prediction(old['prediction'], old['prediction_sha256'], origins, 'prediction')
            losses['old_original_seed2026'] = np.square(old_pred[:, -1]-truth[:, -1])
            for name, error in losses.items():
                totals[name] = totals.get(name, 0.)+error.sum()
                fold_losses.setdefault(name, []).append(float(error.mean()))
            groups.append(day_groups(bars, origins, losses))
            npz(BASE/'analysis/paired_losses'/task/model/f'fold_{fold}.npz', origins=origins,
                session_date=bars.session_date[origins], **{'loss_'+name: error for name, error in losses.items()})
        for name, total in totals.items():
            pooled.append({'task': task, 'model': model, 'method': name, 'origins': n,
                           'endpoint_mse': float(total/n), 'baseline_mse': float(base_total/n),
                           'improvement_pct': float(100*(1-total/base_total)),
                           'positive_folds': sum(a < b for a, b in zip(fold_losses[name], fold_losses[baseline_name])),
                           'seed_improvements_pct': {str(s): float(100*(1-v/base_total))
                                                     for s, v in seed_totals.get(name, {}).items()}})
        pairs = []
        for primary, reference in registered_pairs(target):
            for mode in MODES:
                a, b = primary+'__'+mode, reference+'__'+mode
                if a in totals and b in totals:
                    pairs.append((a, b))
        control_names = [baseline_name, original+'__trained', original+'__shrink',
                         'ridge_history', 'ridge_history_anchor', 'ridge_history_selected',
                         'ridge_history_anchor_selected', 'old_original_seed2026']
        for variant in variants:
            for mode in MODES:
                a = variant+'__'+mode
                pairs.extend((a, b) for b in control_names if a != b)
        pairs = list(dict.fromkeys(pairs)); group_intervals = []
        blocks = [1, 5] if profile == 'minute' else [2, 5]
        for block in blocks:
            current = paired_intervals(groups, pairs, block, task)
            for row in current:
                row['model'] = model
            intervals.extend(current); group_intervals.extend(current)
            if model != 'tiny':
                v0, v1, v2, v3 = ('anchor_random', 'noanchor_random', 'anchor_baseline', 'noanchor_baseline') \
                    if target == 'future_log_return' else ('price_random', 'residual_random', 'price_baseline', 'residual_baseline')
                for mode in MODES:
                    row = interaction(groups, {v3+'__'+mode: 1., v1+'__'+mode: -1.,
                                               v2+'__'+mode: -1., v0+'__'+mode: 1.}, block, task, mode)
                    row['model'] = model; interactions.append(row)
        for primary, reference in registered_pairs(target):
            for mode in MODES:
                a, b = primary+'__'+mode, reference+'__'+mode
                if a not in totals or b not in totals:
                    continue
                relevant = [r for r in group_intervals if r['primary'] == a and r['reference'] == b]
                seed_deltas = {str(s): float(seed_totals[b][s]-seed_totals[a][s]) for s in (2026, 2027, 2028)}
                positive_folds = sum(x < y for x, y in zip(fold_losses[a], fold_losses[b]))
                checks = {'both_block_intervals_support': len(relevant) == 2 and all(r['supports_improvement'] for r in relevant),
                          'three_seed_directions_positive': all(v > 0 for v in seed_deltas.values()),
                          'four_positive_fold_directions': positive_folds >= 4}
                contrasts.append({'task': task, 'model': model, 'primary': primary, 'reference': reference,
                    'mode': mode, 'pooled_improvement_over_reference_pct': float(100*(1-totals[a]/totals[b])),
                    'seed_loss_reductions': seed_deltas, 'positive_folds': positive_folds, 'checks': checks,
                    'replicated_development_mechanism_supported': all(checks.values()),
                    'independent_confirmation': False})
        for variant in variants:
            for mode in MODES:
                method = variant+'__'+mode
                summary = next(r for r in pooled if r['task'] == task and r['model'] == model and r['method'] == method)
                refs = [baseline_name, original+'__trained', original+'__shrink', 'ridge_history', 'ridge_history_anchor']
                relevant = [r for r in group_intervals if r['primary'] == method and r['reference'] in refs
                            and r['primary'] != r['reference']]
                required = 2*sum(reference != method for reference in refs)
                checks = {'candidate_differs_from_original_control': variant != original,
                          'minimum_one_percent': summary['improvement_pct'] >= 1.,
                          'all_three_seeds_positive': len(summary['seed_improvements_pct']) == 3 and
                              all(v > 0 for v in summary['seed_improvements_pct'].values()),
                          'four_positive_folds': summary['positive_folds'] >= 4,
                          'all_control_intervals_both_blocks': len(relevant) == required and
                              all(r['supports_improvement'] for r in relevant)}
                gates.append({'task': task, 'model': model, 'variant': variant, 'mode': mode,
                              'scope': 'primary return gate' if target == 'future_log_return' else 'diagnostic price gate',
                              'checks': checks, 'performance_gate_passed': all(checks.values()),
                              'independent_confirmation': False})
        print(json.dumps({'replication_analyzed': task, 'model': model}), flush=True)
    result = {'status': 'admitted_replication_analysis_complete', 'at_utc': now(),
              'admitted_cells': complete['jobs'], 'admission_groups': manifest['groups'],
              'rows': rows, 'pooled': pooled, 'contrasts': contrasts, 'intervals': intervals,
              'interactions': interactions, 'performance_gates': gates, 'original_restorations': restorations,
              'Q3_values_read': False, 'independent_confirmation': False, 'method_success_claimed': False}
    atomic(BASE/'analysis/REPLICATION_RESULTS.json', result)
    for filename, data in [('REPLICATION_ROWS.csv', rows), ('REPLICATION_POOLED.csv', pooled),
                           ('REPLICATION_CONTRASTS.csv', contrasts), ('REPLICATION_INTERVALS.csv', intervals)]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (BASE/'analysis'/filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(data)
    print(json.dumps({'analyzed_cells': complete['jobs'], 'supported_trained_contrasts':
          [r for r in contrasts if r['mode'] == 'trained' and r['replicated_development_mechanism_supported']],
          'passing_performance_gates': [r for r in gates if r['performance_gate_passed']]}), flush=True)


if __name__ == '__main__':
    main()
