import os
import sys
import json
import time
import hashlib
import argparse
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run_started = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--protocol-sha256', required=True)
    parser.add_argument('--deadline', type=float, required=True)
    args = parser.parse_args()
    assert sha(args.protocol) == args.protocol_sha256
    rules = json.loads(args.protocol.read_text())
    assert rules['runtime_limit_seconds'] == 300
    assert time.time() < args.deadline <= min(rules['parent_market_deadline_epoch'], run_started + rules['runtime_limit_seconds'])
    assert rules['predictors'] == ['D', 'Z', 'C', 'zero', 'prefix_update_mean', 'same_information_Ridge']
    assert rules['state_order'] == ['minute_f1', 'minute_f2', 'hourly_f1', 'hourly_f2']
    assert os.sched_getaffinity(0) == {24}
    for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        assert os.environ[name] == '1'
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    output = Path(rules['output'])
    assert not output.exists()
    parent = rules['input_results']
    assert sha(Path(parent['path'])) == parent['sha256']
    original = json.loads(Path(parent['path']).read_text())
    assert not original['decision']['continue_to_formal_development_review']
    assert [x['state'] for x in rules['inputs']] == rules['state_order']
    assert len(rules['inputs']) == len({x['path'] for x in rules['inputs']}) == 4
    for item in rules['inputs']:
        assert sha(Path(item['path'])) == item['sha256']
    import numpy as np

    checks = []

    def close(label, a, b, *, score=False):
        a, b = np.asarray(a), np.asarray(b)
        difference = np.abs(a - b)
        kind = 'score_check' if score else 'identity_numeric'
        bound = rules[kind + '_atol'] + rules[kind + '_rtol'] * np.maximum(np.abs(a), np.abs(b))
        assert np.isfinite(a).all() and np.isfinite(b).all()
        assert np.all(difference <= bound), label
        checks.append({'label': label, 'max_abs': float(difference.max()), 'passed': True})

    states = []
    for item in rules['inputs']:
        assert time.time() < args.deadline
        with np.load(item['path'], allow_pickle=False) as archive:
            expected = set(rules['predictors']) | {'labels', 'origins'}
            assert set(archive.files) == expected
            y = archive['labels']
            assert y.dtype == np.float64 and y.ndim == 2 and np.isfinite(y).all()
            origins = archive['origins']
            assert origins.shape == (len(y),) and origins.dtype == np.int64
            assert np.all(np.diff(origins) > 0)
            old = next(x for x in original['results'] if x['state'] == item['state'])
            assert y.shape == (old['count'], old['horizon'])
            my = y.mean(axis=0, dtype=np.float64)
            yc = y - my
            zero_mse = np.mean(y * y, axis=0, dtype=np.float64)
            predictors = {}
            for name in rules['predictors']:
                pred = archive[name]
                assert pred.shape == y.shape and pred.dtype == np.float64 and np.isfinite(pred).all()
                mp = pred.mean(axis=0, dtype=np.float64)
                pc = pred - mp
                varp = np.mean(pc * pc, axis=0, dtype=np.float64)
                cov = np.mean(yc * pc, axis=0, dtype=np.float64)
                mse = np.mean((pred - y) ** 2, axis=0, dtype=np.float64)
                level = 2 * my * mp - mp ** 2
                dynamic = 2 * cov - varp
                delta = zero_mse - mse
                close(item['state'] + ':' + name + ':identity', delta, level + dynamic)
                summaries = {}
                for group, take in [('endpoint', lambda a: float(a[-1])), ('path', lambda a: float(a.mean(dtype=np.float64)))]:
                    base = take(zero_mse)
                    assert base > 0
                    summaries[group] = {'zero_MSE': base, 'MSE': take(mse), 'MSE_reduction': take(delta), 'level_contribution': take(level), 'dynamic_contribution': take(dynamic), 'total_gain_percent': 100 * take(delta) / base, 'level_gain_percentage_points': 100 * take(level) / base, 'dynamic_gain_percentage_points': 100 * take(dynamic) / base}
                    close(item['state'] + ':' + name + ':' + group + ':saved_MSE', take(mse), old['predictors'][name][group + '_MSE'], score=True)
                    close(item['state'] + ':' + name + ':' + group + ':summary_identity', take(delta), take(level) + take(dynamic))
                if name == 'prefix_update_mean':
                    close(item['state'] + ':constant_mean:variance', varp, np.zeros_like(varp))
                    close(item['state'] + ':constant_mean:dynamic', dynamic, np.zeros_like(dynamic))
                    assert np.array_equal(pred, np.broadcast_to(pred[0], pred.shape))
                if name == 'zero':
                    assert np.count_nonzero(pred) == 0
                    close(item['state'] + ':zero:contributions', level + dynamic, np.zeros_like(level))
                predictors[name] = {'per_horizon': {'mean_y': my.tolist(), 'mean_prediction': mp.tolist(), 'prediction_variance': varp.tolist(), 'covariance_y_prediction': cov.tolist(), 'zero_MSE': zero_mse.tolist(), 'MSE': mse.tolist(), 'MSE_reduction': delta.tolist(), 'level_contribution': level.tolist(), 'dynamic_contribution': dynamic.tolist()}, 'summary': summaries}
            states.append({'state': item['state'], 'count': len(y), 'horizon': y.shape[1], 'predictors': predictors})
        assert sha(Path(item['path'])) == item['sha256']
    assert time.time() < args.deadline
    result = {'status': 'completed_descriptive_per_horizon_accounting', 'protocol_sha256': args.protocol_sha256, 'script_sha256': sha(Path(__file__)), 'deadline_epoch': args.deadline, 'parent_market_deadline_epoch': rules['parent_market_deadline_epoch'], 'deadline_reset': False, 'inputs': rules['inputs'], 'checks_passed': len(checks), 'checks': checks, 'states': states, 'new_training_updates': 0, 'new_model_inference': 0, 'new_candidate_admitted': False, 'C_recipe_remains_closed': True, 'independent_market_confirmation': False, 'Q3_read': False, 'full_goal_achieved': False, 'interpretation': rules['interpretation']}
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps({'status': result['status'], 'checks_passed': len(checks)}))


if __name__ == '__main__':
    main()
