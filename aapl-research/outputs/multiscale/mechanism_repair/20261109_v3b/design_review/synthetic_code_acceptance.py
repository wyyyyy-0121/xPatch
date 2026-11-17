"""Synthetic-only acceptance of the proposed v3b implementation, before scoring.

No admitted bars, market labels, inner-stop labels or holdout files are loaded.
The fit function is exercised with a synthetic Data double and synthetic labels.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(STUDY / 'code'))
import predictor as p

spec = importlib.util.spec_from_file_location('v3b_experiment_review', STUDY / 'code/experiment.py')
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started = time.time()
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    checks = {}
    rng = np.random.default_rng(731)

    for horizon, lookback in ((60, 480), (8, 96)):
        increments = rng.normal(size=(5, lookback - 1)) * .001
        hist = np.c_[np.zeros(5), np.cumsum(increments, axis=1)]
        hist -= hist[:, -1:]
        local = pd.DatetimeIndex(['2026-01-05 10:30', '2026-07-06 10:30',
                                 '2026-01-09 16:00', '2026-07-10 16:00',
                                 '2026-11-27 13:00']).tz_localize('America/New_York')
        stamps = local.tz_convert('UTC').as_unit('ns').asi8
        features = p.features(hist, stamps, horizon)
        lags = [1, max(2, horizon // 4), max(3, horizon // 2), horizon,
                2*horizon, 4*horizon, lookback-1]
        expected_return = np.stack([hist[:, -1]-hist[:, -1-k] for k in lags], axis=1)
        assert np.array_equal(features[:, :7], expected_return)
        for col, k in ((7, horizon), (8, 4*horizon)):
            assert np.allclose(features[:, col], np.sqrt(np.mean(increments[:, -k:]**2, axis=1)), atol=1e-17)
        assert np.array_equal(features[0, 9:], features[1, 9:])
        assert np.array_equal(features[2, 9:], features[3, 9:])
        assert np.array_equal(features[0, 13:], [1., 0., 0., 0.])
        assert np.array_equal(features[2, 13:], np.zeros(4))
        # 13:00 uses the same fixed 390-minute denominator, even on a half-day.
        assert np.isclose(features[4, 9], np.sin(2*np.pi*210/390))
        for bad in ('2026-01-05 09:30', '2026-01-05 16:01', '2026-01-10 10:30'):
            stamp = pd.Timestamp(bad, tz='America/New_York').tz_convert('UTC').value
            try:
                p.features(hist[:1], [stamp], horizon)
            except ValueError:
                pass
            else:
                raise AssertionError('Off-session completion accepted')
        checks[f'feature_H{horizon}'] = 'exact lag signs and indices; trailing RMS; DST; Friday reference; fixed half-day denominator; invalid-clock rejection passed'

    x = rng.normal(size=(4110, 17)) + rng.normal(size=(17,))
    x[:, 16] = 1.
    y = rng.normal(size=(len(x), 8)) + 3.
    stats = {'count': np.asarray(len(x)), 'sx': x.sum(0), 'sy': y.sum(0),
             'xx': x.T@x, 'xy': x.T@y}
    for name, cols in e.CELLS.items():
        model = e.solve(stats, cols)
        mx = x.mean(0); sd = x.std(0); sd[sd <= 1e-12] = 1.
        z = ((x-mx)/sd)[:, cols]
        ref = np.linalg.solve(z.T@z/len(x)+.1*np.eye(len(cols)), z.T@(y-y.mean(0))/len(x)) if len(cols) else np.empty((0, 8))
        assert np.allclose(model['coef'], ref, atol=1e-12, rtol=1e-11)
        assert np.allclose(model['y_mean'], y.mean(0), atol=1e-13)
    checks['ridge'] = 'All four cells match direct fit-centered/standardized XTX/n + .1I reference; unpenalized target intercept; constant feature safe'

    # Replace all data-facing dependencies. The only targets below are generated
    # synthetic random-walk increments, with explicit disjoint index partitions.
    dates = pd.bdate_range('2025-01-06', periods=20)
    clock = np.concatenate([pd.date_range(f'{d:%Y-%m-%d} 09:31', periods=390, freq='min', tz='America/New_York').tz_convert('UTC').as_unit('ns').asi8 for d in dates])
    close = np.exp(np.cumsum(rng.normal(0., .001, len(clock)))) * 100.
    bars = SimpleNamespace(close=close, end_ns=clock)
    fit_idx = np.arange(479, 4589); stop_idx = np.arange(4660, 4730)
    val_idx = np.arange(4800, 4850)
    target_parts = []

    class Windows:
        def arrays(self, origins):
            prices = np.log(close[np.asarray(origins)[:, None] + np.arange(-479, 1)])
            return prices-prices[:, -1:], close[np.asarray(origins), None]

    class SyntheticData:
        def __init__(self, cfg):
            self.bars = bars; self.windows = Windows()
            self.splits = {'fit': fit_idx, 'stop': stop_idx, 'validation': val_idx}
            self.rec = {'split_file': split_file}

    def synthetic_targets(_, origins, horizon, mode):
        origins = np.asarray(origins)
        if np.isin(origins, val_idx).any():
            raise AssertionError('fit() requested outer synthetic target')
        target_parts.append('fit' if np.isin(origins, fit_idx).all() else 'stop')
        return np.log(close[origins[:, None]+np.arange(1, horizon+1)]/close[origins, None])

    original_sources = {name: digest(STUDY/'code'/name) for name in ('predictor.py', 'experiment.py')}
    with tempfile.TemporaryDirectory(prefix='v3b_synthetic_acceptance_') as temp:
        temp = Path(temp)
        # Exercise the actual freeze function against a fully synthetic identity
        # fixture. split_sha256 deliberately has the real array-hash dictionary
        # schema, so comparing it to an NPZ file hash cannot pass this check.
        e.BASE = temp/'freeze_fixture'
        e.V3 = temp/'preceding_study'
        e.PRIOR = temp/'preceding_data'
        e.SOURCE = temp/'admission'
        (e.BASE/'code').mkdir(parents=True)
        for name in original_sources:
            shutil.copyfile(STUDY/'code'/name, e.BASE/'code'/name)
        shutil.copyfile(STUDY/'PLAN_ZH.md', e.BASE/'PLAN_ZH.md')
        frozen_prior_sources = {'synthetic': 'identity-only-fixture'}
        e.source_hashes = lambda: frozen_prior_sources
        e.atomic(e.V3/'protocol/BUDGET.json', {'deadline_epoch': e.DEADLINE})
        e.atomic(e.V3/'protocol/CODE_FREEZE.json', {'hashes': frozen_prior_sources})
        e.atomic(e.SOURCE/'admission.json', {'synthetic_only': True})
        task_hashes = {}
        for task in ('minute_h60_future_log_return', 'hourly_h8_future_log_return'):
            task_folder = e.PRIOR/'cache'/task
            task_folder.mkdir(parents=True)
            folds = {}
            for fold in range(1, 6):
                split = {key: np.arange(i*10, i*10+5, dtype=np.int64)
                         for i, key in enumerate(('train', 'fit', 'stop', 'validation'))}
                split_path = task_folder/f'fold_{fold}.npz'
                np.savez(split_path, **split)
                hashes = {key: hashlib.sha256(np.asarray(value, dtype='<i8').tobytes()).hexdigest()
                          for key, value in split.items()}
                scaler_path = task_folder/f'scaler_{fold}.json'
                e.atomic(scaler_path, {'split_sha256': hashes})
                folds[str(fold)] = {'split_file': str(split_path), 'split_sha256': hashes,
                    'scaler_file': str(scaler_path), 'scaler_sha256': digest(scaler_path),
                    'targets_sha256': 'synthetic-target-path-not-opened'}
            task_path = task_folder/'TASK.json'
            e.atomic(task_path, {'folds': folds})
            task_hashes[task] = digest(task_path)
        e.atomic(e.PRIOR/'protocol/DATA.json', {'task_files': task_hashes})
        rules = e.freeze()
        assert rules['sources'] == original_sources
        assert e.freeze() == rules
        assert len(rules['original_purged_partitions']) == 2
        split_to_corrupt = e.PRIOR/'cache/minute_h60_future_log_return/fold_1.npz'
        original_npz = split_to_corrupt.read_bytes()
        np.savez(split_to_corrupt, **{key: value+1 for key, value in split.items()})
        try:
            e.freeze()
        except AssertionError:
            pass
        else:
            raise AssertionError('freeze accepted changed partition indices')
        split_to_corrupt.write_bytes(original_npz)
        assert e.freeze() == rules
        checks['actual_freeze'] = 'Real freeze() passes array-hash dictionary fixture, records separate NPZ file hash, reopens unchanged rules and rejects changed partition indices; no target file exists or is read'

        split_file = temp/'synthetic_splits.npz'
        np.savez(split_file, fit=fit_idx, stop=stop_idx, validation=val_idx)
        e.Data = SyntheticData
        e.targets = synthetic_targets
        e.make_job = lambda profile, horizon, target, model, variant, fold: {
            'profile': profile, 'horizon': horizon, 'target': target, 'fold': fold}
        e.DEADLINE = time.time()+30.

        def synthetic_freeze():
            path = e.BASE/'protocol/RULES.json'
            if not path.exists():
                e.atomic(path, {'batch_size': 2048, 'synthetic_data_only': True})
            return e.read(path)

        e.freeze = synthetic_freeze
        for label in ('continuous', 'resumed'):
            e.BASE = temp/label
            (e.BASE/'code').mkdir(parents=True)
            for name in original_sources:
                shutil.copyfile(STUDY/'code'/name, e.BASE/'code'/name)
            e.STOP = False
            if label == 'resumed':
                paused = e.fit('minute', 60, 1, stop_after_batches=1)
                assert paused['status'] == 'paused' and paused['cursor'] == 2048
                assert target_parts[-1] == 'fit'
                assert not (e.BASE/'fits/minute_h60_future_log_return/fold_1/INNER_DONE.json').exists()
            result = e.fit('minute', 60, 1)
            assert result['status'] == 'completed'
            assert result['outer_labels_constructed'] is False
            # Loading a completed fit verifies hashes and leaves artifacts alone.
            again = e.fit('minute', 60, 1)
            assert again == result
        relative = Path('fits/minute_h60_future_log_return/fold_1')
        for name in original_sources:
            assert digest(STUDY/'code'/name) == original_sources[name]
        for filename in ['FIT_ACCUM.npz'] + [name+'.npz' for name in e.CELLS]:
            with np.load(temp/'continuous'/relative/filename, allow_pickle=False) as left, np.load(temp/'resumed'/relative/filename, allow_pickle=False) as right:
                for key in left.files:
                    # Identity includes a different experiment base's rules file
                    # bytes, which are actually equal; all numeric state is exact.
                    assert np.array_equal(left[key], right[key]), (filename, key)
        for name in e.CELLS:
            model = p.Predictor(temp/'resumed'/relative/(name+'.npz'))
            hist, _ = Windows().arrays(stop_idx[:8])
            pred = model.predict(hist, clock[stop_idx[:8]])
            assert np.array_equal(model.predict(hist, clock[stop_idx[:8]], shrunk=True), model.alpha*pred)
    checks['checkpoint_and_reload'] = 'Actual fit() on synthetic Data: pause after 2048 samples, restore, and continuous run yield bit-identical sufficient statistics/models; completed-artifact reload verifies hashes; outer-target reads rejected'
    checks['no_real_data'] = 'Synthetic Data/targets replaced before any fit call; no admitted market arrays, real stop/outer labels, or Q3 values read'
    return {'status': 'passed_synthetic_code_acceptance', 'checks': checks,
            'reviewed_source_sha256': original_sources,
            'elapsed_seconds': time.time()-started, 'GPU': False,
            'affinity': sorted(os.sched_getaffinity(0)),
            'formal_stop_scores_computed': False, 'formal_outer_scores_computed': False,
            'Q3_values_read': False}


if __name__ == '__main__':
    result = main()
    (HERE/'SYNTHETIC_ACCEPTANCE.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
