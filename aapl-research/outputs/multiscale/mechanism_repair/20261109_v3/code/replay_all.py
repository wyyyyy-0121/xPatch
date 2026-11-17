"""Replay archived functions independently of the training data loader, no labels."""
from study import *
import torch, time, argparse
from portable_predictor import Predictor
from multiscale.features import CloseWindows
from multiscale.matrix_executor import read_bars


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--screening-only', action='store_true')
    parser.add_argument('--available-only', action='store_true')
    args = parser.parse_args()
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    jobs = {j['id']: j for j in read(BASE/'protocol/JOBS.json')['jobs']}
    if not args.screening_only:
        jobs.update({j['id']: j for j in read(BASE/'protocol/REPLICATION_JOBS.json')['jobs']})
        if (BASE/'protocol/CANDIDATE_JOBS.json').exists() and read(BASE/'protocol/CANDIDATE_JOBS.json')['admitted']:
            jobs.update({j['id']: j for j in read(BASE/'protocol/CANDIDATE_JOBS.json')['jobs']})
    if not args.screening_only and (BASE/'protocol/CAP_JOBS.json').exists() and read(BASE/'protocol/CAP_JOBS.json')['admitted']:
        jobs.update({j['cfg']['id']: j['cfg'] for j in read(BASE/'protocol/CAP_JOBS.json')['jobs']})
    receipt_path = BASE/'audit/INPUT_ONLY_REPLAY.json'
    previous = read(receipt_path) if receipt_path.exists() else {'records': []}
    prior_records = {r['job']: r for r in previous['records'] if r['passed']}
    records = []; bars_cache = {}; admission = read(SOURCE/'admission.json')
    for cfg in jobs.values():
        folder = BASE/'runs'/cfg['id']
        if not (folder/'DONE.json').exists() and args.available_only:
            continue
        done = read(folder/'DONE.json')
        assert sha(folder/'best_trained.pt') == done['checkpoint_sha256']
        assert sha(folder/'predictions.npz') == done['prediction_sha256']
        if cfg['id'] in prior_records and prior_records[cfg['id']]['checkpoint_sha256'] == done['checkpoint_sha256']:
            records.append(prior_records[cfg['id']]); continue
        task = task_key(cfg['profile'], cfg['horizon'], cfg['target'])
        rec = read(PRIOR/'cache'/task/'TASK.json')['folds'][str(cfg['fold'])]
        with np.load(rec['split_file'], allow_pickle=False) as a:
            origins = a['validation']
        with np.load(folder/'predictions.npz', allow_pickle=False) as a:
            assert np.array_equal(a['origins'], origins)
            expected = {name: a[name] for name in ('trained', 'coupled')}
        bars = bars_cache.setdefault(cfg['frequency'], None)
        if bars is None:
            bars = read_bars(SOURCE, admission, cfg['frequency']); bars_cache[cfg['frequency']] = bars
        assert bars.session_date[-1] <= '2026-06-30'
        windows = CloseWindows(bars, cfg['lookback'], admission['split_actions'])
        if cfg['model'] == 'tiny':
            cached_origins = np.load(PRIOR/'cache'/cfg['profile']/'origins.npy', mmap_mode='r')
            cached_x = np.load(PRIOR/'cache'/cfg['profile']/'x.npy', mmap_mode='r')
        trained = Predictor(folder/'best_trained.pt')
        if done['shadow_best_epoch'] not in (0, done['trained_best_epoch']):
            coupled = Predictor(folder/'coupled_best.pt')
        else:
            coupled = trained
        calibrated = None
        calibration_folder = BASE/'calibration_candidate'/cfg['id']
        if not (calibration_folder/'DONE.json').exists():
            calibration_folder = BASE/'calibration'/cfg['id']
        if (calibration_folder/'DONE.json').exists():
            calibrated = read(calibration_folder/'DONE.json')
            assert sha(calibration_folder/'predictions.npz') == calibrated['prediction_sha256']
            with np.load(calibration_folder/'predictions.npz', allow_pickle=False) as a:
                assert np.array_equal(origins, a['origins']); expected['shrink'] = a['prediction']
        exact = {name: True for name in expected}; maximum = {name: 0. for name in expected}
        for start in range(0, len(origins), 128):
            idx = origins[start:start+128]
            if cfg['model'] == 'tiny':
                positions = np.searchsorted(cached_origins, idx)
                assert np.array_equal(cached_origins[positions], idx)
                x = np.asarray(cached_x[positions], dtype='float32')
            else:
                x = windows.arrays(idx)[0]
            anchor = bars.close[idx]
            p = trained.predict_prepared(x, anchor)
            q = coupled.predict_prepared(x, anchor, baseline_only=done['shadow_best_epoch'] == 0)
            predictions = {'trained': p, 'coupled': q}
            if calibrated is not None:
                baseline = np.zeros_like(p) if cfg['target'] == 'future_log_return' else np.broadcast_to(anchor[:, None], p.shape)
                predictions['shrink'] = baseline+calibrated['alpha']*(p-baseline)
            for mode, value in predictions.items():
                reference = expected[mode][start:start+len(idx)]
                exact[mode] &= np.array_equal(value, reference)
                maximum[mode] = max(maximum[mode], float(np.max(np.abs(value-reference))))
        assert all(exact.values()), (cfg['id'], maximum)
        records.append({'job': cfg['id'], 'passed': True, 'origins': len(origins), 'path_steps': cfg['horizon'],
                        'exact_modes': exact, 'maximum_absolute_error': maximum,
                        'checkpoint_sha256': done['checkpoint_sha256'], 'prediction_sha256': done['prediction_sha256']})
        atomic(receipt_path, {'status': 'running', 'records': records, 'targets_opened': False,
                             'Q3_values_read': False, 'pid': os.getpid(), 'at_utc': now()})
        if len(records) % 10 == 0:
            print(json.dumps({'replayed': len(records), 'total': len(jobs)}), flush=True)
    atomic(receipt_path, {'status': 'passed_screening_subset' if args.screening_only else
                         'passed_available_subset' if args.available_only else 'passed',
                         'scope': 'screening_only' if args.screening_only else
                         'available_completed_cells' if args.available_only else 'all_registered_cells',
                         'records': records, 'predictors': len(records),
                         'targets_opened': False, 'Q3_values_read': False, 'all_exact': True,
                         'origin_predictor_pairs': sum(r['origins'] for r in records), 'at_utc': now()})
    print(json.dumps({'status': 'passed', 'replayed': len(records)}), flush=True)


if __name__ == '__main__':
    main()
