"""Fixed final-fold/seed-2026 examples for every original task, no score selection."""
from study import *
import torch
from portable_predictor import Predictor
from multiscale.matrix_executor import read_bars
from multiscale.features import CloseWindows


def main():
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    jobs = read(BASE/'protocol/CANDIDATE_JOBS.json')['jobs']
    assert read(BASE/'audit/CANDIDATE_COMPLETE.json')['jobs'] == len(jobs)
    records = []; admission = read(SOURCE/'admission.json'); bars_cache = {}
    for cfg in jobs:
        if cfg['fold'] != 5 or cfg['seed'] != 2026:
            continue
        task = task_key(cfg['profile'], cfg['horizon'], cfg['target'])
        rec = read(PRIOR/'cache'/task/'TASK.json')['folds']['5']
        with np.load(rec['split_file'], allow_pickle=False) as a:
            origins = a['validation'][:2]
        if cfg['frequency'] not in bars_cache:
            bars_cache[cfg['frequency']] = read_bars(SOURCE, admission, cfg['frequency'])
        bars = bars_cache[cfg['frequency']]; assert bars.session_date[-1] <= '2026-06-30'
        windows = CloseWindows(bars, cfg['lookback'], admission['split_actions'])
        x, anchor = windows.arrays(origins); closes = np.exp(x)*anchor
        cal = read(BASE/'calibration_candidate'/cfg['id']/'DONE.json')
        model = Predictor(BASE/'runs'/cfg['id']/'best_trained.pt')
        pred = model.predict(closes, shrink_alpha=cal['alpha'])
        reference = model.predict_prepared(x, anchor[:, 0], shrink_alpha=cal['alpha'])
        assert np.allclose(pred, reference, rtol=1e-6, atol=1e-9)
        npz(BASE/'delivery'/f'{task}.npz', completed_closes=closes, expected=pred,
            origins=origins, shrink_alpha=np.asarray(cal['alpha']))
        records.append({'task': task, 'job': cfg['id'], 'seed': 2026, 'fold': 5,
                        'shrink_alpha': cal['alpha'], 'archive_checkpoint': f'models/{cfg["id"]}/best_trained.pt',
                        'file': f'delivery/{task}.npz', 'example_sha256': sha(BASE/'delivery'/f'{task}.npz')})
    assert len(records) == 20
    atomic(BASE/'delivery/EXAMPLES.json', {'examples': records, 'selection': 'all tasks, final fold, seed2026, first two origins',
           'targets_opened': False, 'Q3_values_read': False, 'at_utc': now()})
    print(json.dumps({'examples': 20, 'target_files_opened': False}), flush=True)


if __name__ == '__main__':
    main()
