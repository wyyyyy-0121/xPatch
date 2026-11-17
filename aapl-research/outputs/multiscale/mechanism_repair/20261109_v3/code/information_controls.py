"""Fit registered linear controls on exactly the admitted historical inputs.

Only inner-fit labels estimate coefficients; only the purged inner stop tail
chooses regularization and the optional persistence fallback. No outer score
chooses a recipe. Raw nominal price tasks are represented as dollar changes
around the known current close, including in the linear control.
"""
from study import *
import time
from multiscale.features import CloseWindows
from multiscale.matrix_executor import read_bars
from multiscale.bars import targets

TASKS = [('minute', 60, 'future_log_return'),
         ('hourly', 8, 'future_log_return'), ('hourly', 1, 'future_close')]
ALPHAS = [.1, 1., 10.]


def main():
    assert len(os.sched_getaffinity(0)) <= 2
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    recipe = {'tasks': TASKS, 'alpha_grid': ALPHAS,
              'normal_equations': 'centered, fit-only per-column variance scaling; gram + alpha I',
              'inputs': 'original past log-relative close window, optionally observed current-price anchor',
              'target': 'return path; nominal price path minus known current price',
              'selection': 'inner raw endpoint MSE, ties retain first alpha; optional exact baseline fallback',
              'trained_on': 'inner fit only', 'Q3_values_read': False}
    manifest = BASE/'protocol/INFORMATION_CONTROLS.json'
    if manifest.exists():
        assert read(manifest)['recipe'] == json.loads(json.dumps(recipe))
        assert read(manifest)['source_sha256'] == sha(__file__)
    else:
        atomic(manifest, {'recipe': recipe, 'source_sha256': sha(__file__), 'at_utc': now(),
                          'new_control_outer_scores_seen': False})
    deadline = read(BASE/'protocol/BUDGET.json')['deadline_epoch']
    admission = read(SOURCE/'admission.json')
    for profile, h, target in TASKS:
        task = task_key(profile, h, target)
        meta = read(PRIOR/'cache'/task/'TASK.json')
        bars = read_bars(SOURCE, admission, meta['frequency'])
        assert bars.session_date[-1] <= '2026-06-30'
        windows = CloseWindows(bars, meta['lookback'], admission['split_actions'])
        for fn, rec in meta['folds'].items():
            out = BASE/'controls'/task/f'fold_{fn}'
            if (out/'DONE.json').exists():
                continue
            if time.time() >= deadline:
                atomic(BASE/'audit/CONTROLS_PAUSED.json', {'at_utc': now(), 'task': task,
                       'fold': fn, 'deadline_epoch': deadline, 'completed_folds_preserved': True})
                return
            with np.load(rec['split_file'], allow_pickle=False) as a:
                parts = {k: a[k] for k in ('fit', 'stop', 'validation')}
            fit, stop, val = (parts[k] for k in ('fit', 'stop', 'validation'))
            assert bars.end_ns[fit[-1]+h] < bars.end_ns[stop[0]]
            assert bars.end_ns[stop[-1]+h] < bars.end_ns[val[0]]
            d = meta['lookback']+1
            gram_raw = np.zeros((d, d)); rhs_raw = np.zeros((d, h))
            sx = np.zeros(d); sy = np.zeros(h)

            def batch(idx):
                x, anchor = windows.arrays(idx)
                x = np.column_stack((x, anchor[:, 0]))
                y = targets(bars, idx, h, target)
                if target == 'future_close':
                    y = y-bars.close[idx, None]
                return x, y

            for i in range(0, len(fit), 2048):
                x, y = batch(fit[i:i+2048])
                gram_raw += x.T@x; rhs_raw += x.T@y
                sx += x.sum(0); sy += y.sum(0)
            mx = sx/len(fit); my = sy/len(fit)
            sd = np.sqrt(np.maximum(np.diag(gram_raw)/len(fit)-mx*mx, 0))
            sd = np.where(sd > 1e-12, sd, 1.)
            gram = (gram_raw-len(fit)*np.outer(mx, mx))/sd[:, None]/sd[None, :]
            rhs = (rhs_raw-len(fit)*np.outer(mx, my))/sd[:, None]
            outputs = {}; records = {}
            baseline_stop = targets(bars, stop, h, target)[:, -1]
            if target == 'future_close':
                baseline_stop = baseline_stop-bars.close[stop]
            baseline_stop_mse = float(np.mean(baseline_stop**2))
            truth = targets(bars, val, h, target)
            baseline = np.zeros_like(truth) if target == 'future_log_return' else \
                np.broadcast_to(bars.close[val, None], truth.shape)
            for name, cols in [('history', np.arange(d-1)), ('history_anchor', np.arange(d))]:
                best = None; scores = []
                for alpha in ALPHAS:
                    coef = np.linalg.solve(gram[np.ix_(cols, cols)]+alpha*np.eye(len(cols)), rhs[cols])
                    error = 0.
                    for i in range(0, len(stop), 2048):
                        x, y = batch(stop[i:i+2048])
                        pred = ((x[:, cols]-mx[cols])/sd[cols])@coef+my
                        error += float(np.square(pred[:, -1]-y[:, -1]).sum())
                    score = error/len(stop)
                    scores.append({'alpha': alpha, 'inner_endpoint_mse': score})
                    if best is None or score < best[0]:
                        best = score, alpha, coef
                score, alpha, coef = best
                prediction = []
                for i in range(0, len(val), 2048):
                    idx = val[i:i+2048]; x, _ = batch(idx)
                    pred = ((x[:, cols]-mx[cols])/sd[cols])@coef+my
                    if target == 'future_close':
                        pred += bars.close[idx, None]
                    prediction.append(pred)
                pred = np.concatenate(prediction)
                selected_base = baseline_stop_mse <= score
                outputs[name] = pred
                outputs[name+'_selected'] = baseline if selected_base else pred
                npz(out/(name+'_model.npz'), coef=coef, x_mean=mx[cols], x_std=sd[cols],
                    y_mean=my, columns=cols)
                records[name] = {'alpha': alpha, 'inner_candidates': scores,
                    'selected_baseline': selected_base, 'baseline_stop_mse': baseline_stop_mse,
                    'trained_metrics': metrics(pred, truth, bars.close[val], target),
                    'selected_metrics': metrics(outputs[name+'_selected'], truth, bars.close[val], target)}
            npz(out/'predictions.npz', origins=val, **outputs)
            atomic(out/'DONE.json', {'status': 'completed', 'task': task, 'fold': int(fn),
                'records': records, 'recipe_sha256': sha(manifest),
                'prediction_sha256': sha(out/'predictions.npz'),
                'Q3_values_read': False, 'outer_labels_used_in_fitting': False, 'at_utc': now()})
            print(json.dumps({'control_complete': task, 'fold': fn}), flush=True)
    atomic(BASE/'audit/INFORMATION_CONTROLS_COMPLETE.json', {'status': 'completed',
           'tasks': len(TASKS), 'folds': 15, 'at_utc': now(), 'Q3_values_read': False})


if __name__ == '__main__':
    main()
