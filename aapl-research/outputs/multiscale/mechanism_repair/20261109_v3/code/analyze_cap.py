"""Evaluate the fixed longer-cap diagnostic alongside initialized price controls."""
from study import *
from mechanism_stats import day_groups, paired_intervals
from multiscale.matrix_executor import read_bars


def main():
    assert len(os.sched_getaffinity(0)) <= 2
    manifest = read(BASE/'protocol/CAP_JOBS.json')
    assert manifest['admitted'] and read(BASE/'audit/CAP_COMPLETE.json')['jobs'] == 30
    task = 'hourly_h1_future_close'; meta = read(PRIOR/'cache'/task/'TASK.json')
    bars = read_bars(SOURCE, read(SOURCE/'admission.json'), 'hour')
    rows = []; summaries = []; intervals = []
    for model in ('lstm', 'xpatch'):
        groups = []; totals = {}; baseline_total = 0.; n = 0
        for fold in range(1, 6):
            rec = meta['folds'][str(fold)]
            with np.load(rec['targets'], allow_pickle=False) as a:
                origins = a['origins']; actual = a['actual'][:, -1]; anchor = a['anchor']
            base = np.square(actual-anchor); losses = {'persistence': base}
            n += len(origins); baseline_total += base.sum()
            values = {name: [] for name in ('default20', 'default60', 'price_baseline20', 'residual_baseline20')}
            for seed in (2026, 2027, 2028):
                source_cfg = make_job('hourly', 1, 'future_close', model, 'price_random', fold, seed)
                cap_id = source_cfg['id']+'_cap60'; cap_out = BASE/'runs'/cap_id
                check = read(cap_out/'PREFIX_COPY.json')
                assert check['exact_model_optimizer_rng_cursor_history']
                comparison = read(cap_out/'CAP_COMPARISON.json'); rows.append(comparison)
                identifiers = {'default20': source_cfg['id'], 'default60': cap_id,
                    'price_baseline20': make_job('hourly', 1, 'future_close', model, 'price_baseline', fold, seed)['id'],
                    'residual_baseline20': make_job('hourly', 1, 'future_close', model, 'residual_baseline', fold, seed)['id']}
                for name, job_id in identifiers.items():
                    out = BASE/'runs'/job_id; done = read(out/'DONE.json')
                    assert sha(out/'predictions.npz') == done['prediction_sha256']
                    with np.load(out/'predictions.npz', allow_pickle=False) as a:
                        assert np.array_equal(origins, a['origins']); p = a['trained'][:, -1]
                    values[name].append(np.square(p-actual))
            losses.update({name: np.mean(value, axis=0) for name, value in values.items()})
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.)+value.sum()
            groups.append(day_groups(bars, origins, losses))
            npz(BASE/'analysis/cap_paired_losses'/model/f'fold_{fold}.npz', origins=origins,
                session_date=bars.session_date[origins], **{'loss_'+k: v for k, v in losses.items()})
        for name, value in totals.items():
            summaries.append({'model': model, 'method': name, 'origins': n, 'endpoint_mse': float(value/n),
                              'improvement_over_persistence_pct': float(100*(1-value/baseline_total))})
        pairs = [('default60', 'default20'), ('price_baseline20', 'default60'),
                 ('residual_baseline20', 'default60'), ('default60', 'persistence')]
        for block in (2, 5):
            for row in paired_intervals(groups, pairs, block, task):
                row['model'] = model; intervals.append(row)
    receipt = {'status': 'completed', 'comparisons': rows, 'pooled': summaries, 'intervals': intervals,
               'cap_bound_original_cells': sum(row['original_was_cap_bound'] for row in rows),
               'extended_cells_with_new_updates': sum(row['new_updates'] > 0 for row in rows),
               'independent_confirmation': False, 'Q3_values_read': False, 'at_utc': now()}
    atomic(BASE/'analysis/CAP_RESULTS.json', receipt)
    print(json.dumps({'cap_bound': receipt['cap_bound_original_cells'], 'pooled': summaries}), flush=True)


if __name__ == '__main__':
    main()
