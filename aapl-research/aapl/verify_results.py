"""Independently recompute selected validation and final MSE from persisted predictions."""
import math

import numpy as np

from .common import ROOT, digest, read_json, utc_now, write_json
from .data import Market
from .evaluate import seeds_for
from .runner import require_lock

FORMAL = ROOT / 'outputs/formal'


def scalar_mse(prediction, target):
    # Independent scalar summation, rather than reusing evaluator aggregation.
    return math.fsum((float(p) - float(y)) ** 2 for p, y in zip(prediction, target)) / len(target)


def verify():
    protocol = require_lock()
    market = Market.load(protocol)
    split = read_json(ROOT / 'outputs/development_splits.json')
    split['inner'] = [i for i in split['inner'] if i['name'] in protocol['development']['admitted_validation_blocks']]
    summary = read_json(FORMAL / 'final_summary.json')
    decision = read_json(ROOT / 'outputs/development_decision.json')
    trials = read_json(ROOT / protocol['trial_list_file'])
    checks, coverage = {}, {}
    for name, configs in trials.items():
        root = FORMAL / 'search' / name
        ranking = read_json(root / 'ranking_rung0.json')
        coverage[name] = {'started': len(ranking), 'declared': len(configs), 'rungs': []}
        checks[name + '_complete_initial_search'] = {r['trial']['trial_id'] for r in ranking} == {t['trial_id'] for t in configs}
        if name in ('xpatch', 'patchtst', 'chronos2_lora'):
            counts = [len(configs), min(len(configs), 4 if name == 'chronos2_lora' else 11), min(len(configs), 2 if name == 'chronos2_lora' else 4)]
            previous = None
            for rung, expected in enumerate(counts):
                ranked = read_json(root / f'ranking_rung{rung}.json')
                checks[f'{name}_rung{rung}_count_and_order'] = len(ranked) == expected and ranked == sorted(ranked, key=lambda x: (x['mse'], x['trial']['trial_id']))
                if previous is not None:
                    checks[f'{name}_rung{rung}_promotion_rule'] = {r['trial']['trial_id'] for r in ranked} == {r['trial']['trial_id'] for r in previous[:expected]}
                previous = ranked
                coverage[name]['rungs'].append(len(ranked))
        selected = read_json(root / 'selection.json')[0]
        totals, sizes = [], []
        for seed in seeds_for(name):
            total, count = 0.0, 0
            for j, inner in enumerate(split['inner']):
                path = root / selected['trial']['trial_id'] / f'inner{j}' / f'seed{seed}' / 'inner_predictions.npz'
                with np.load(path) as saved:
                    origins, pred = saved['origins'], saved['prediction']
                    np.testing.assert_array_equal(origins, market.select(inner['validation_dates']))
                    value = scalar_mse(pred, market.y(origins))
                total += value * len(origins)
                count += len(origins)
            totals.append(total / count)
        checks[name + '_selected_validation_mse_recomputed'] = bool(np.isclose(np.mean(totals), selected['mse'], rtol=1e-11, atol=1e-18))
        dvalues = []
        for seed in seeds_for(name):
            path = FORMAL / 'confirmation' / name / f'seed{seed}' / 'predictions.npz'
            receipt = read_json(path.parent / 'receipt.json')
            checks[f'{name}_D_seed{seed}_receipt'] = receipt['prediction_sha256'] == digest(path) and receipt['protocol_sha256'] == digest(ROOT / 'protocol_locked.json')
            with np.load(path) as saved:
                origins, pred = saved['origins'], saved['prediction']
                np.testing.assert_array_equal(origins, market.select(split['confirmation']['evaluation_dates']))
                dvalues.append(scalar_mse(pred, market.y(origins)))
        checks[name + '_D_mse_recomputed'] = bool(np.isclose(np.mean(dvalues), decision['models'][name]['confirmation_mse'], rtol=1e-11, atol=1e-18))
    if summary['holdout_evaluated']:
        with np.load(FORMAL / 'holdout_labels_for_independent_verification.npz') as saved:
            origins, y, dates = saved['origins'], saved['y'], saved['dates']
        zero_mse = scalar_mse(np.zeros_like(y), y)
        checks['holdout_zero_mse_recomputed'] = bool(np.isclose(zero_mse, summary['zero_mse'], rtol=1e-12))
        checks['at_most_two_sealed_model_families'] = len(summary['models']) <= 2
        for name, row in summary['models'].items():
            values = []
            for seed in seeds_for(name):
                path = FORMAL / 'holdout' / name / f'seed{seed}' / 'predictions.npz'
                with np.load(path) as saved:
                    np.testing.assert_array_equal(origins, saved['origins'])
                    values.append(scalar_mse(saved['prediction'], y))
                checks[f'{name}_holdout_seed{seed}_receipt'] = read_json(path.parent / 'receipt.json')['prediction_sha256'] == digest(path)
            checks[name + '_holdout_mse_and_gain_recomputed'] = bool(np.isclose(np.mean(values), row['mse'], rtol=1e-11) and np.isclose(1 - np.mean(values) / zero_mse, row['relative_mse_improvement_vs_zero'], rtol=1e-10, atol=1e-12))
    else:
        accesses = [read_json(p) for p in (ROOT / 'outputs/holdout_access').glob('*.json')]
        checks['negative_development_did_not_unseal'] = not (FORMAL / 'holdout').exists() and all(r['kind'] != 'authorized_read' for r in accesses)
    checks['summary_and_decision_identity'] = summary['protocol_sha256'] == decision['protocol_sha256'] == digest(ROOT / 'protocol_locked.json')
    result = {'passed': all(checks.values()), 'completed_at_utc': utc_now(), 'checks': checks,
              'coverage': coverage, 'primary_metrics_independently_recomputed': True,
              'final_summary_sha256': digest(FORMAL / 'final_summary.json')}
    write_json(FORMAL / 'independent_verification.json', result)
    if not result['passed']:
        raise RuntimeError('Independent research verification failed')
    return result


if __name__ == '__main__':
    print(verify(), flush=True)
