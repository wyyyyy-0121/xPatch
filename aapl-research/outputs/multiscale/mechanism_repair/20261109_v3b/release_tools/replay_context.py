"""Portable input-only replay of all forty frozen deterministic context functions."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    rules = read(root/'protocol/RULES.json')
    rules_hash = sha(root/'protocol/RULES.json')
    assert rules['Q3_values_read'] is False and rules['independent_confirmation'] is False
    assert rules['candidate'] == 'D_joint_shrink'
    assert sha(root/'PLAN_ZH.md') == rules['plan_sha256']
    for name, expected in rules['sources'].items():
        assert sha(root/'code'/name) == expected, ('source hash mismatch', name)
    spec = importlib.util.spec_from_file_location('portable_context_predictor', root/'code/predictor.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    counts = {'predictors': 0, 'forecast_modes': 0, 'input_rows': 0, 'resumable_fit_states': 0,
              'streaming_comparisons': 0, 'outer_examples': 0}
    maximum_stream_difference = 0.
    maximum_outer_difference = 0.
    for task in rules['tasks']:
        horizon = int(task.split('_h')[1].split('_')[0])
        expected_lookback = 480 if task.startswith('minute_') else 96
        for fold in range(1, 6):
            folder = root/'fits'/task/f'fold_{fold}'
            done = read(folder/'INNER_DONE.json')
            fit_done = read(folder/'FIT_DONE.json')
            assert done['status'] == fit_done['status'] == 'completed'
            assert done['task'] == task and done['fold'] == fold
            assert done['rules_sha256'] == rules_hash
            assert done['source_hashes'] == rules['sources']
            assert done['seed'] is None and done['solver_deterministic'] is True
            assert done['outer_labels_constructed'] is False
            assert sha(folder/'FIT_ACCUM.npz') == done['sufficient_statistics_sha256'] == fit_done['sufficient_statistics_sha256']
            with np.load(folder/'FIT_ACCUM.npz', allow_pickle=False) as stats:
                assert int(stats['count']) == int(stats['cursor']) == done['fit_count'] == fit_done['fit_count']
                assert str(stats['identity']) == done['identity'] == fit_done['identity']
                assert stats['sx'].shape == (17,) and stats['xx'].shape == (17, 17)
                assert stats['sy'].shape == (horizon,) and stats['xy'].shape == (17, horizon)
                assert all(np.isfinite(stats[key]).all() for key in ('sx', 'sy', 'xx', 'xy'))
            counts['resumable_fit_states'] += 1
            for cell, columns in rules['cells'].items():
                model_file = folder/(cell+'.npz')
                assert sha(model_file) == done['model_hashes'][cell+'.npz']
                model = module.Predictor(model_file)
                assert model.horizon == horizon
                assert np.array_equal(model.columns, columns)
                assert model.alpha == done['inner_metrics'][cell]['alpha']
                with np.load(folder/(cell+'_EXAMPLE.npz'), allow_pickle=False) as sample:
                    history = sample['history']; timestamps = sample['completion_ns']
                    origins = sample['origins']
                    assert history.shape == (min(128, done['stop_count']), expected_lookback)
                    assert timestamps.shape == origins.shape == (len(history),)
                    assert np.all(np.diff(origins) > 0) and np.all(np.diff(timestamps) > 0)
                    assert np.allclose(history[:, -1], 0., rtol=0., atol=1e-12)
                    raw = model.predict(history, timestamps)
                    shrunk = model.predict(history, timestamps, shrunk=True)
                    assert np.array_equal(raw, sample['expected_raw']), (task, fold, cell, 'raw')
                    assert np.array_equal(shrunk, sample['expected_shrink']), (task, fold, cell, 'shrink')
                    assert np.array_equal(shrunk, model.alpha*raw)
                    stream = np.concatenate([model.predict(history[i:i+1], timestamps[i:i+1]) for i in range(min(8, len(history)))])
                    difference = float(np.max(np.abs(stream-raw[:len(stream)])))
                    maximum_stream_difference = max(maximum_stream_difference, difference)
                    assert np.allclose(stream, raw[:len(stream)], rtol=1e-12, atol=1e-15)
                    counts['streaming_comparisons'] += 1
                    counts['input_rows'] += len(history)
                outer_sample = root/'delivery'/task/f'fold_{fold}'/(cell+'.npz')
                if outer_sample.exists():
                    with np.load(outer_sample, allow_pickle=False) as sample:
                        raw = model.predict(sample['history'], sample['completion_ns'])
                        shrunk = model.predict(sample['history'], sample['completion_ns'], shrunk=True)
                        for actual, expected in ((raw, sample['expected_raw']), (shrunk, sample['expected_shrink'])):
                            maximum_outer_difference = max(maximum_outer_difference, float(np.max(np.abs(actual-expected))))
                            assert np.allclose(actual, expected, rtol=1e-12, atol=1e-15)
                    counts['outer_examples'] += 1
                counts['predictors'] += 1
                counts['forecast_modes'] += 2
    assert counts['predictors'] == 40 and counts['forecast_modes'] == 80
    assert counts['resumable_fit_states'] == 10
    print(json.dumps({'status': 'passed', **counts, 'canonical_batch_forecasts_bitwise_equal': True,
        'source_hashes_verified': True, 'fit_states_verified': True, 'target_values_constructed': False,
        'streaming_tolerance': {'rtol': 1e-12, 'atol': 1e-15},
        'maximum_stream_difference': maximum_stream_difference,
        'maximum_outer_difference': maximum_outer_difference,
        'outer_example_tolerance_note': 'Outer examples were sliced from a 2048-row inference batch; inner examples use the canonical 128-row batch and must remain bitwise equal',
        'training_or_calibration_performed': False, 'Q3_values_read': False,
        'scope': 'Portable engineering acceptance of archived functions; not independent market validation'}), flush=True)


if __name__ == '__main__':
    main()
