"""Scalar inner-tail shrinkage of the unchanged original expert on every task."""
from study import *
import torch, time
from multiscale.features import CloseWindows, Scaler
from multiscale.models import make_model
from multiscale.matrix_executor import read_bars
from multiscale.bars import targets


def main():
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    selected = read(BASE/'protocol/CANDIDATE_JOBS.json')
    assert selected['admitted']; model = selected['architecture']
    recipe = {'model': model, 'source_sha256': sha(__file__),
              'candidate_manifest_sha256': sha(BASE/'protocol/CANDIDATE_JOBS.json'),
              'alpha': 'clip(sum(q*r)/sum(q*q),0,1) from inner-stop endpoint only',
              'q_r': 'prediction/truth minus exact zero or current close',
              'outer_labels_used_for_calibration': False, 'Q3_values_read': False}
    manifest = BASE/'protocol/ORIGINAL_SHRINK_CONTROLS.json'
    if manifest.exists():
        assert read(manifest)['recipe'] == recipe
    else:
        atomic(manifest, {'recipe': recipe, 'at_utc': now()})
    deadline = selected['deadline_epoch']; admission = read(SOURCE/'admission.json'); bars_cache = {}
    for task in sorted(read(PRIOR/'protocol/DATA.json')['task_files']):
        meta = read(PRIOR/'cache'/task/'TASK.json'); frequency = meta['frequency']
        if frequency not in bars_cache:
            bars_cache[frequency] = read_bars(SOURCE, admission, frequency)
        bars = bars_cache[frequency]; assert bars.session_date[-1] <= '2026-06-30'
        windows = CloseWindows(bars, meta['lookback'], admission['split_actions'])
        for fold, rec in meta['folds'].items():
            out = BASE/'controls_cross_scale'/task/f'fold_{fold}'
            if (out/'ORIGINAL_SHRINK.json').exists():
                continue
            if time.time() >= deadline:
                atomic(BASE/'audit/ORIGINAL_SHRINK_PAUSED.json', {'task': task, 'fold': fold,
                       'at_utc': now(), 'deadline_epoch': deadline})
                return
            expert = rec['models'][model]
            assert sha(expert['checkpoint']) == expert['checkpoint_sha256']
            ck = torch.load(expert['checkpoint'], map_location='cpu', weights_only=True)
            assert ck['identity'] == expert['identity']
            net = make_model(model, meta['lookback'], meta['horizon']); net.load_state_dict(ck['model']); net.eval()
            scaler = Scaler(**ck['scaler'])
            with np.load(rec['split_file'], allow_pickle=False) as a:
                stop = a['stop']; val = a['validation']
            numerator = denominator = baseline_total = 0.
            with torch.inference_mode():
                for i in range(0, len(stop), 128):
                    idx = stop[i:i+128]; x, anchor = windows.arrays(idx); x, anchor = scaler.transform_x(x, anchor)
                    raw = scaler.invert_y(net(torch.from_numpy(x[:, :, None]), torch.from_numpy(anchor)).numpy())[:, -1]
                    actual = targets(bars, idx, meta['horizon'], meta['target'])[:, -1]
                    baseline = np.zeros(len(idx)) if meta['target'] == 'future_log_return' else bars.close[idx]
                    q = raw-baseline; r = actual-baseline
                    numerator += float(np.dot(q, r)); denominator += float(np.dot(q, q)); baseline_total += float(np.dot(r, r))
            alpha = float(np.clip(numerator/denominator, 0, 1)) if denominator else 0.
            assert sha(expert['prediction']) == expert['prediction_sha256']
            with np.load(expert['prediction'], allow_pickle=False) as a:
                assert np.array_equal(val, a['origins']); prediction = a['prediction']
            baseline = np.zeros_like(prediction) if meta['target'] == 'future_log_return' else \
                np.broadcast_to(bars.close[val, None], prediction.shape)
            shrunk = baseline+alpha*(prediction-baseline)
            npz(out/'original_predictions.npz', origins=val, original=prediction, shrink=shrunk)
            atomic(out/'ORIGINAL_SHRINK.json', {'status': 'completed', 'model': model, 'task': task, 'fold': int(fold),
                'alpha': alpha, 'baseline_stop_mse': baseline_total/len(stop),
                'inner_shrunk_mse': (baseline_total-2*alpha*numerator+alpha*alpha*denominator)/len(stop),
                'source_checkpoint_sha256': expert['checkpoint_sha256'],
                'source_prediction_sha256': expert['prediction_sha256'], 'recipe_sha256': sha(manifest),
                'prediction_sha256': sha(out/'original_predictions.npz'),
                'Q3_values_read': False, 'outer_labels_used_for_calibration': False, 'at_utc': now()})
        print(json.dumps({'original_shrink_complete': task, 'model': model}), flush=True)
    atomic(BASE/'audit/ORIGINAL_SHRINK_COMPLETE.json', {'status': 'completed', 'folds': 100,
           'model': model, 'Q3_values_read': False, 'at_utc': now()})


if __name__ == '__main__':
    main()
