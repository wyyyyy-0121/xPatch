"""Independent reconstruction from this stage's own artificial fixture only.

Does not import the worker, adapter, original feature functions, or Torch.
No historical market cache or historical checkpoint is admitted here.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--scaler', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    stage = Path(__file__).resolve().parents[1]
    rules = json.loads((stage / 'PREPARATION_RULES.json').read_text())
    assert time.time() < rules['deadline_epoch']
    assert set(os.sched_getaffinity(0)) == {24}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
    for path in (args.fixture, args.scaler, args.output):
        path.resolve().relative_to(stage)
    assert args.fixture.suffix == '.npz'
    import numpy as np

    with np.load(args.fixture, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    scalar_data = json.loads(args.scaler.read_text())
    scaler = scalar_data.get('scaler', scalar_data)
    lookback, horizon = 480, 60
    close, ends = data['close'], data['end_ns']
    fit = data['fit']
    assert fit.dtype == np.dtype('int64') and len(fit) == 1675
    boundary = (4 * len(fit)) // 5
    tail = fit[boundary:]
    prefix0 = fit[:boundary]
    prefix = prefix0[ends[prefix0 + horizon] < ends[tail[0]]]
    assert len(prefix) == 1280 and len(tail) == 335
    assert np.array_equal(prefix, data['prefix'])
    assert np.array_equal(tail, data['tail'])
    assert scaler['fitted_origin_count'] == len(prefix)
    logs = np.log(close)

    def raw(origins):
        positions = origins[:, None] + np.arange(1 - lookback, 1)
        history = logs[positions] - logs[origins, None]
        anchor = close[origins, None]
        labels = np.log(close[origins[:, None] + np.arange(1, horizon + 1)] / anchor)
        return history, anchor, labels

    x, anchor, y = raw(prefix)
    x_mean = float(x.sum(dtype=np.float64) / x.size)
    a_mean = float(anchor.sum(dtype=np.float64) / len(prefix))
    y_mean = y.sum(axis=0, dtype=np.float64) / len(prefix)
    x_std = float(np.sqrt(max(float(np.square(x).sum(dtype=np.float64) / x.size) - x_mean**2, 0)))
    a_std = float(np.sqrt(max(float(np.square(anchor).sum(dtype=np.float64) / len(prefix)) - a_mean**2, 0)))
    y_std = np.sqrt(np.maximum(np.square(y).sum(axis=0, dtype=np.float64) / len(prefix) - y_mean**2, 0))
    x_std = x_std if x_std > 1e-12 else 1.0
    a_std = a_std if a_std > 1e-12 else 1.0
    y_std = np.where(y_std > 1e-12, y_std, 1.0)
    checks = []

    def compare(name, actual, expected):
        a, e = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        assert a.shape == e.shape, name
        assert np.isfinite(a).all() and np.isfinite(e).all(), name
        error = np.abs(a - e)
        bound = 1e-10 + 1e-8 * np.maximum(np.abs(a), np.abs(e))
        passed = bool(np.all(error <= bound))
        checks.append({'name': name, 'passed': passed,
                       'maximum_absolute_difference': float(np.max(error)),
                       'maximum_normalized_difference': float(np.max(error / bound))})
        assert passed, name

    for key, expected in [('x_mean', x_mean), ('x_std', x_std), ('anchor_mean', a_mean),
                          ('anchor_std', a_std), ('y_mean', y_mean), ('y_std', y_std)]:
        compare('prefix_only_scaler:' + key, scaler[key], expected)
    # Preserve the original FP32 input boundary before this new Ridge control's
    # FP64 column moments; this is the exact information given to the networks.
    design = np.column_stack((((x - x_mean) / x_std).astype(np.float32),
                              ((anchor - a_mean) / a_std).astype(np.float32))).astype(np.float64)
    mean = design.mean(axis=0)
    second = np.square(design).mean(axis=0)
    variance = second - np.square(mean)
    allowance = 1e-10 * np.maximum(1.0, np.maximum(np.abs(second), np.square(mean)))
    assert np.all(variance >= -allowance)
    sd = np.sqrt(np.maximum(variance, 0))
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = (design - mean) / sd
    system = z.T @ z / len(prefix) + 0.1 * np.eye(z.shape[1])
    rhs = z.T @ (y - y_mean) / len(prefix)
    beta = np.linalg.solve(system, rhs)
    residual = np.linalg.norm(system @ beta - rhs, ord=np.inf)
    denominator = np.linalg.norm(system, ord=np.inf) * np.linalg.norm(beta, ord=np.inf) + np.linalg.norm(rhs, ord=np.inf)
    backward = float(residual / denominator) if denominator else (0.0 if residual == 0 else float('inf'))
    assert backward <= 1e-10
    compare('ridge_column_mean', data['ridge_x_mean'], mean)
    compare('ridge_column_std', data['ridge_x_std'], sd)
    compare('ridge_raw_y_intercept', data['ridge_y_mean'], y_mean)
    compare('ridge_coefficients', data['ridge_coefficients'], beta)
    tx, ta, _ = raw(tail)
    td = np.column_stack((((tx - x_mean) / x_std).astype(np.float32),
                           ((ta - a_mean) / a_std).astype(np.float32))).astype(np.float64)
    prediction = ((td - mean) / sd) @ beta + y_mean
    compare('ridge_tail_predictions', data['ridge_tail_predictions'], prediction)
    assert time.time() < rules['deadline_epoch']
    report = {'status': 'passed_independent_raw_synthetic_native_scaler_and_direct_Ridge_reconstruction',
              'at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'checks': checks, 'checks_passed': len(checks),
              'normal_equation_backward_error': backward,
              'prefix_count': len(prefix), 'tail_count': len(tail),
              'fixture_sha256': hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
              'scaler_sha256': hashlib.sha256(args.scaler.read_bytes()).hexdigest(),
              'verifier_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'worker_or_feature_helpers_imported': False, 'Torch_imported': False,
              'market_arrays_or_archived_weights_read': False, 'Q3_read': False,
              'predictive_success_claimed': False}
    temporary = args.output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, args.output)
    print(json.dumps({'status': report['status'], 'checks_passed': len(checks)}))


if __name__ == '__main__':
    main()
