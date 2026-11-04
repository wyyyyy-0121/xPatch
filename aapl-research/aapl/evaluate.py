"""One fixed development decision and one sealed-test family from saved predictions."""
import numpy as np

from .common import ROOT, digest, identity, read_json, utc_now, write_json
from .data import Market
from .statistics import paired_blocks, holm

FORMAL = ROOT / 'outputs/formal'


def seeds_for(name):
    return [2026] if name in ('ridge', 'chronos2_frozen', 'chronos2_matched_frozen') else [2026, 2027, 2028]


def family(stage, name, market, dates, protocol_hash):
    origins = market.select(dates)
    y = market.y(origins)
    predictions, calibration, hashes = [], [], {}
    for seed in seeds_for(name):
        directory = FORMAL / stage / name / f'seed{seed}'
        path = directory / 'predictions.npz'
        receipt = read_json(directory / 'receipt.json')
        if receipt['protocol_sha256'] != protocol_hash or receipt['prediction_sha256'] != digest(path):
            raise ValueError('Prediction protocol/bytes identity differs')
        if receipt['calibration_sha256'] != digest(directory / 'calibration.json'):
            raise ValueError('Calibration changed after prediction')
        with np.load(path) as data:
            np.testing.assert_array_equal(data['origins'], origins)
            np.testing.assert_array_equal(data['bar_start_ns'], market.bars.index[origins].as_unit('ns').asi8)
            pred = data['prediction'].copy()
        if pred.shape != y.shape or not np.isfinite(pred).all():
            raise ValueError('Incomplete/nonfinite saved predictions')
        predictions.append(pred)
        calibration.append(read_json(directory / 'calibration.json')['additive_residual_quantiles'])
        hashes[str(path.relative_to(ROOT))] = digest(path)
    predictions = np.stack(predictions)
    errors = predictions - y
    return {'origins': origins, 'dates': market.dates[origins], 'y': y, 'predictions': predictions,
            'loss': (errors ** 2).mean(axis=0), 'per_seed_mse': (errors ** 2).mean(axis=1),
            'mse': float(np.mean(errors ** 2)), 'calibration': np.asarray(calibration),
            'prediction_files': hashes}


def secondary_metrics(row):
    predictions, y = row['predictions'], row['y']
    ensemble = predictions.mean(axis=0)
    issued = ensemble != 0
    active = issued & (y != 0)
    up, down = y > 0, y < 0
    quantiles = predictions[:, None, :] + row['calibration'][:, :, None]
    lower, upper = quantiles[:, 0], quantiles[:, 2]
    return {'mae_mean_of_seed_mae': float(np.mean(np.abs(predictions - y))),
            'direction_forecast_coverage': float(issued.mean()),
            'direction_accuracy_nonzero_targets': float(np.mean(np.sign(ensemble[active]) == np.sign(y[active]))) if active.any() else None,
            'direction_balanced_accuracy': float((np.mean(ensemble[up] > 0) + np.mean(ensemble[down] < 0)) / 2) if up.any() and down.any() else None,
            'actual_zero_return_fraction': float(np.mean(y == 0)),
            'interval80_coverage_mean_across_seeds': float(np.mean((y >= lower) & (y <= upper))),
            'interval80_mean_width': float(np.mean(upper - lower)),
            'ensemble_mse_secondary': float(np.mean((ensemble - y) ** 2)),
            'primary_aggregation': 'Mean of per-seed losses, without substituting the prediction ensemble'}


def failure_paths():
    return [str(p.relative_to(ROOT)) for p in FORMAL.rglob('*failure.json')]


def development(protocol, split, market):
    destination = ROOT / 'outputs/development_decision.json'
    phash = digest(ROOT / 'protocol_locked.json')
    if destination.exists():
        saved = read_json(destination)
        if saved['protocol_sha256'] != phash:
            raise ValueError('Development decision protocol changed')
        for path, checksum in saved['input_files'].items():
            if digest(ROOT / path) != checksum:
                raise ValueError('Completed development decision inputs changed')
        return saved
    val_origins = np.concatenate([market.select(i['validation_dates']) for i in split['inner']])
    zero_validation_mse = float(np.mean(market.y(val_origins) ** 2))
    confirmation_dates = split['confirmation']['evaluation_dates']
    dorigins = market.select(confirmation_dates)
    zero_d_mse = float(np.mean(market.y(dorigins) ** 2))
    models, inputs, eligible = {}, {}, []
    for name in protocol['models_primary']:
        if name == 'zero':
            continue
        selection_file = FORMAL / 'search' / name / 'selection.json'
        selected = read_json(selection_file)[0]
        inputs[str(selection_file.relative_to(ROOT))] = digest(selection_file)
        row = family('confirmation', name, market, confirmation_dates, phash)
        inputs.update(row['prediction_files'])
        valid_gain = 1 - selected['mse'] / zero_validation_mse
        d_gain = 1 - row['mse'] / zero_d_mse
        models[name] = {'validation_mse': selected['mse'], 'validation_relative_gain_vs_zero': valid_gain,
                        'confirmation_mse': row['mse'], 'confirmation_relative_gain_vs_zero': d_gain,
                        'confirmation_per_seed_mse': row['per_seed_mse'].tolist(),
                        'chosen_trial': selected['trial'], 'fixed_budget': selected['final_budget'],
                        'secondary': secondary_metrics(row)}
        if valid_gain > 0 and d_gain > 0:
            eligible.append(name)
    best = min(models, key=lambda k: (models[k]['validation_mse'], k))
    selected_holdout = list(dict.fromkeys(['xpatch', best]))
    primary = family('confirmation', 'xpatch', market, confirmation_dates, phash)
    alternative = family('confirmation', 'xpatch_loss_ablation', market, confirmation_dates, phash)
    lora = family('confirmation', 'chronos2_lora', market, confirmation_dates, phash)
    frozen = family('confirmation', 'chronos2_matched_frozen', market, confirmation_dates, phash)
    inputs.update(alternative['prediction_files'])
    inputs.update(frozen['prediction_files'])
    mechanisms = {'xpatch_selected_minus_opposite_loss': paired_blocks(primary['loss'] - alternative['loss'], primary['dates']),
                  'chronos_lora_minus_matched_frozen': paired_blocks(lora['loss'] - frozen['loss'], lora['dates']),
                  'scope': 'Exploratory D-block mechanisms; no additional sealed-test models or primary hypotheses'}
    report = {'completed_at_utc': utc_now(), 'protocol_sha256': phash,
              'zero_validation_mse': zero_validation_mse, 'zero_confirmation_mse': zero_d_mse,
              'models': models, 'eligible_by_validation_and_confirmation': eligible,
              'enter_holdout': bool(eligible) and not failure_paths(),
              'selected_for_holdout': selected_holdout,
              'selection_basis': 'Fixed xPatch slot plus lowest pooled validation MSE model; D only governs entry',
              'exploratory_mechanisms': mechanisms, 'technical_failures': failure_paths(), 'input_files': inputs,
              'confirmation_evaluation_policy': 'Single immutable decision; subsequent invocations verify and reuse it',
              'holdout_outcomes_read': False}
    write_json(destination, report)
    return report


def final(protocol, split, market):
    destination = FORMAL / 'final_summary.json'
    phash = digest(ROOT / 'protocol_locked.json')
    decision_path = ROOT / 'outputs/development_decision.json'
    decision = read_json(decision_path)
    if destination.exists():
        saved = read_json(destination)
        if saved['protocol_sha256'] != phash or saved['development_decision_sha256'] != digest(decision_path):
            raise ValueError('Final summary identity changed')
        return saved
    report = {'completed_at_utc': utc_now(), 'protocol_sha256': phash,
              'development_decision_sha256': digest(decision_path), 'development': decision,
              'technical_failures': failure_paths(), 'holdout_evaluated': False,
              'scope': 'AAPL afternoon 12-minute cumulative log returns; prediction metrics without trading-profit inference'}
    if not decision['enter_holdout']:
        report.update(status='completed_development_negative', conclusion='No candidate passed both development conditions; final-test data remain sealed')
        write_json(destination, report)
        return report
    manifest = read_json(ROOT / 'holdout_manifest.json')
    authorization = identity({'protocol_sha256': phash, 'decision_sha256': digest(decision_path),
                              'holdout_csv_sha256': manifest['csv_sha256']})
    holdout = Market.load_holdout(authorization)
    dates = split['holdout_dates_calendar_only']
    origins = holdout.select(dates)
    y = holdout.y(origins)
    zero_loss = y ** 2
    zero_mse = float(np.mean(zero_loss))
    models, pvalues = {}, {}
    for name in decision['selected_for_holdout']:
        row = family('holdout', name, holdout, dates, phash)
        stats = paired_blocks(row['loss'] - zero_loss, row['dates'])
        months = {}
        for month in ('2026-07', '2026-08', '2026-09'):
            mask = np.char.startswith(row['dates'].astype(str), month)
            months[month] = float(1 - np.mean(row['loss'][mask]) / np.mean(zero_loss[mask]))
        models[name] = {'mse': row['mse'], 'relative_mse_improvement_vs_zero': 1 - row['mse'] / zero_mse,
                        'per_seed_mse': row['per_seed_mse'].tolist(),
                        'per_seed_relative_improvement': (1 - row['per_seed_mse'] / zero_mse).tolist(),
                        'monthly_relative_improvement': months, 'primary_test': stats,
                        'sensitivity_block1': paired_blocks(row['loss'] - zero_loss, row['dates'], block=1),
                        'sensitivity_block10': paired_blocks(row['loss'] - zero_loss, row['dates'], block=10),
                        'secondary': secondary_metrics(row), 'prediction_files': row['prediction_files']}
        pvalues[name] = stats['one_sided_p']
    adjusted = holm(pvalues)
    exploratory_only = read_json(ROOT / 'outputs/power_planning.json')['exploratory_only_if_proxy_exceeds_3pct']
    for name, value in models.items():
        value['primary_test']['holm_p'] = adjusted[name]
        checks = {'relative_improvement_at_least1pct': value['relative_mse_improvement_vs_zero'] >= 0.01,
                  'holm_one_sided_p_below0_05': adjusted[name] < 0.05,
                  'at_least2_months_improved': sum(v > 0 for v in value['monthly_relative_improvement'].values()) >= 2,
                  'all_fixed_seeds_improved': all(v > 0 for v in value['per_seed_relative_improvement']),
                  'no_technical_failure': not report['technical_failures']}
        value['qualification_checks'] = checks
        value['stable_better_under_locked_rules'] = all(checks.values())
        value['confirmatory_claim_allowed'] = all(checks.values()) and not exploratory_only
    np.savez_compressed(FORMAL / 'holdout_labels_for_independent_verification.npz', origins=origins,
                        dates=holdout.dates[origins], y=y)
    report.update(status='completed_sealed_test', holdout_evaluated=True, holdout_csv_sha256=manifest['csv_sha256'],
                  holdout_effective_days=manifest['effective_sessions'], holdout_origins=len(origins),
                  zero_mse=zero_mse, models=models, power_planning_exploratory_only=exploratory_only,
                  tested_family=decision['selected_for_holdout'], tested_family_size=len(models))
    write_json(destination, report)
    return report


if __name__ == '__main__':
    from .runner import require_lock
    protocol = require_lock()
    final(protocol, read_json(ROOT / 'outputs/development_splits.json'), Market.load(protocol))
