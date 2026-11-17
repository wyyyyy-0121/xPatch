"""Export small predictors and replay their complete validation arrays from history only."""
from common import *
import torch, shutil, time
from models import create
from portable_predictor import Predictor
from multiscale.features import CloseWindows


def main():
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert len(os.sched_getaffinity(0)) <= 2 and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    decision = read(BASE/'protocol/STAGE_C_DECISION.json')
    kinds = [k for k in decision['formal_kinds'] if k != 'lstm']
    bars, _, folds, admission = bars_and_folds()
    windows = CloseWindows(bars, 480, admission['split_actions'])
    destination = BASE/'delivery/models'; destination.mkdir(parents=True, exist_ok=True)
    records = []
    for kind in kinds:
        for f in folds:
            origins = f['validation']; fold = f['fold']
            for seed in (2026, 2027, 2028):
                name = f'formal_{kind}_f{fold}_s{seed}'
                root = BASE/'runs'/name; done = read(root/'DONE.json')
                assert sha(root/'best.pt') == done['checkpoint_sha256']
                saved = torch.load(root/'best.pt', map_location='cpu', weights_only=True)
                model = create(kind, -np.asarray(saved['scaler']['y_mean'])/np.asarray(saved['scaler']['y_std']))
                model.load_state_dict(saved['model']); model.eval()
                out = destination/name; out.mkdir(exist_ok=True)
                example = (torch.zeros(2, 4), torch.zeros(2, 480), torch.zeros(2, 4))
                traced = torch.jit.trace(model, example, check_trace=True)
                torch.jit.save(traced, str(out/'network.pt'))
                atomic(out/'preprocess.json', {k: saved[k] for k in (
                    'cfg', 'scaler', 'feature_stats', 'vol_threshold', 'epoch',
                    'trainable_parameters', 'total_parameters', 'identity')})
                predictor = Predictor(out)
                # This verifier never loads a target file or cache/actual.npy.
                with np.load(root/'prediction.npz', allow_pickle=False) as a:
                    assert np.array_equal(a['origins'], origins)
                    expected = a['prediction']
                start = time.time(); exact = True; maximum = 0.; api_maximum = 0.
                for i in range(0, len(origins), 512):
                    idx = origins[i:i+512]; x, anchor = windows.arrays(idx)
                    p = predictor.predict_log_relative(x)
                    e = expected[i:i+len(idx)]
                    maximum = max(maximum, float(np.max(np.abs(p-e))))
                    exact = exact and np.array_equal(p, e)
                    assert np.allclose(p, e, rtol=1e-6, atol=1e-10), (name, i, maximum)
                    if i == 0 or i+512 >= len(origins):
                        positive_closes = np.exp(x)*anchor
                        q = predictor.predict(positive_closes)
                        api_maximum = max(api_maximum, float(np.max(np.abs(q-e))))
                        assert np.allclose(q, e, rtol=1e-5, atol=1e-9), (name, 'close_api', api_maximum)
                invalid = [np.ones(479), np.ones(481), np.zeros(480), -np.ones(480), np.full(480, np.nan), np.full(480, np.inf)]
                for bad in invalid:
                    try: predictor.predict(bad)
                    except ValueError: pass
                    else: raise AssertionError('Invalid input accepted')
                record = {'job': name, 'origins': len(origins), 'all_60_steps': True,
                    'exact_array_match': exact, 'max_absolute_difference': maximum,
                    'positive_close_api_sample_max_difference': api_maximum,
                    'input_rejections_passed': True, 'target_files_read': False,
                    'network_sha256': sha(out/'network.pt'), 'preprocess_sha256': sha(out/'preprocess.json'),
                    'source_checkpoint_sha256': done['checkpoint_sha256'], 'wall_seconds': time.time()-start}
                records.append(record)
                atomic(BASE/'audit/PREDICTOR_REPLAY_PROGRESS.json', {'completed': len(records),
                    'planned': len(kinds)*15, 'last': record})
                print(json.dumps({'replayed': name, 'exact': exact, 'max_abs': maximum}), flush=True)
    shutil.copy2(BASE/'code/portable_predictor.py', BASE/'delivery/predictor.py')
    # Preselected demonstration: the final chronological fold, seed 2026; no score selection.
    folder = destination/f'formal_{decision["primary"]}_f5_s2026'
    x, a = windows.arrays(folds[-1]['validation'][:2])
    close = np.exp(x)*a
    np.savez_compressed(BASE/'delivery/example.npz', closes=close, expected=Predictor(folder).predict(close))
    atomic(BASE/'audit/PREDICTOR_REPLAY.json', {'status': 'passed', 'records': records,
        'predictors': len(records), 'total_origin_predictor_pairs': sum(r['origins'] for r in records),
        'all_steps_verified': 60, 'no_target_files_read': True, 'Q3_values_read': False,
        'all_exact': all(r['exact_array_match'] for r in records),
        'portable_api': '480 positive, finite completed closes on a consistent as-of split basis',
        'research_status_unchanged_by_engineering_acceptance': True})


if __name__ == '__main__':
    main()
