"""Recompute all original monthly split-group path errors without fitting."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from multiscale.historical_coverage import digest, read_json, require, snapshot
from multiscale.historical_design_audit import load_metadata
from multiscale.historical_minute_check import MODELS, calculate, fold_sources, score_fold
from multiscale.historical_results import ADMISSION, checked_coverage

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
COVERAGE = OUT / 'private_coverage_verified/20261108_timing_design_snapshot.json'
PRIOR = OUT / 'private_three_scale_development/20261107T134128870773Z/PRIVATE_MONTHLY_SPLIT_SENSITIVITY.json'
DESTINATION = OUT / 'historical_split_group_audits/20261108_original_monthly_all_metrics'


def main():
    rows = checked_coverage(read_json(COVERAGE), snapshot(), allow_partial=True)
    admission = read_json(ADMISSION / 'admission.json')
    meta, metadata_source = load_metadata(ADMISSION, admission, 'day')
    split_dates = [a['date'] for a in admission['split_actions']]
    require(split_dates == ['2020-08-31'], 'Recorded action list differs')
    prior = read_json(PRIOR)
    require(prior['split_dates'] == split_dates and prior['independent_test'] is False, 'Prior grouping differs')
    groups, sources, results, flat = {}, {}, {}, []
    for target in ('future_close', 'future_log_return'):
        folds = [r for r in rows if (r['profile'], r['horizon'], r['target']) == ('monthly', 21, target)]
        require(len(folds) == 5 and all(r['status'] == 'completed' for r in folds), 'Monthly folds incomplete')
        actuals, anchors, origins, predictions = [], [], [], {m: [] for m in MODELS}
        for row in folds:
            score_fold(row)  # Independently verify all saved metrics and array identities first.
            folder, folders, saved, _, identity = fold_sources(row)
            path = folder / 'targets.npz'
            require(digest(path) == identity, 'Target archive changed')
            sources[str(path)] = identity
            with np.load(path, allow_pickle=False) as z:
                actuals.append(z['actual'].copy())
                anchors.append(z['anchor'].copy())
                origins.append(z['origins'].copy())
            for model in MODELS:
                path = folders[model] / f'prediction_{model}.npz'
                require(digest(path) == saved[model]['prediction_sha256'], 'Prediction archive changed')
                sources[str(path)] = digest(path)
                with np.load(path, allow_pickle=False) as z:
                    require(np.array_equal(z['origins'], origins[-1]), 'Model origins differ')
                    predictions[model].append(z['prediction'].copy())
        actual, anchor, index = np.concatenate(actuals), np.concatenate(anchors), np.concatenate(origins)
        prediction = {m: np.concatenate(parts) for m, parts in predictions.items()}
        crossing = np.zeros(len(index), dtype=bool)
        for date in split_dates:
            crossing |= (meta.sessions[index] < date) & (meta.sessions[index + 21] >= date)
        masks = {'crosses_split': crossing, 'no_split': ~crossing}
        groups[target] = {}
        baseline_loss = np.square(prediction['last_close'][:, -1] - actual[:, -1])
        total_loss = baseline_loss.sum()
        for group, mask in masks.items():
            require(mask.sum() > 0, 'Empty split group')
            metrics = {m: calculate(actual[mask], anchor[mask], prediction[m][mask], target) for m in MODELS}
            base = metrics['last_close']['endpoint_mse']
            fraction = float(baseline_loss[mask].sum() / total_loss)
            old = prior['groups'][target][group]
            require(mask.sum() == old['origins'] and np.isclose(base, old['baseline_endpoint_mse'],
                    rtol=1e-12, atol=1e-16) and np.isclose(fraction,
                    old['fraction_of_total_baseline_squared_error'], rtol=1e-12, atol=1e-16),
                    'Recomputed split groups differ from the previous reported evidence')
            for model in MODELS:
                metrics[model]['relative_improvement'] = 1 - metrics[model]['endpoint_mse'] / base
                require(np.isclose(metrics[model]['relative_improvement'], old['relative_improvement'][model],
                                   rtol=1e-10, atol=1e-12), 'Prior model subgroup comparison differs')
                flat.append({'target': target, 'group': group, 'model': model,
                             'origins': int(mask.sum()), 'baseline_squared_error_fraction': fraction,
                             **{k: metrics[model][k] for k in ('endpoint_mse', 'endpoint_rmse', 'endpoint_mae',
                                 'path_mse', 'path_mae', 'relative_improvement')}})
            groups[target][group] = {'origins': int(mask.sum()), 'baseline_squared_error_fraction': fraction,
                                      'first_origin_session': str(meta.sessions[index[mask][0]]),
                                      'last_origin_session': str(meta.sessions[index[mask][-1]]), 'models': metrics}
        require(crossing.sum() == 21 and (~crossing).sum() == 1949, 'Original group sizes changed')
        for model in MODELS:
            pooled = calculate(actual, anchor, prediction[model], target)
            for metric in ('endpoint_mse', 'endpoint_mae', 'path_mse', 'path_mae'):
                weighted = sum(g['origins'] * g['models'][model][metric] for g in groups[target].values()) / len(index)
                require(np.isclose(weighted, pooled[metric], rtol=1e-12, atol=1e-16), 'Subgroups do not pool to the original errors')
    DESTINATION.mkdir(parents=True, exist_ok=False)
    results = {'status': 'original_monthly_split_groups_all_errors_recomputed',
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'split_dates': split_dates,
               'groups': groups, 'source_sha256': sources, 'bar_metadata_source': metadata_source,
               'prior_group_evidence': str(PRIOR), 'prior_group_evidence_sha256': digest(PRIOR),
               'all_prior_group_values_match': True, 'weighted_group_errors_match_pooled_errors': True,
               'group_model_rows': len(flat), 'new_training_started': False, 'Q3_values_read': False,
               'independent_test': False, 'script_sha256': digest(Path(__file__)),
               'limitations': ['Post-exposure descriptive grouping of one recorded split event.',
                              'Overlapping paths are not independent corporate-action events.',
                              'Only the original nominal targets are analyzed; cash dividends are excluded.']}
    with (DESTINATION / 'SPLIT_GROUP_AUDIT.json').open('x') as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write('\n')
    with (DESTINATION / 'SPLIT_GROUP_METRICS.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({'status': results['status'], 'rows': len(flat), 'prior_group_values_match': True,
                      'group_errors_pool_to_original': True, 'destination': str(DESTINATION)}))


if __name__ == '__main__': main()
