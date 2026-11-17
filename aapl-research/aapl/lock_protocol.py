"""P4-P5: measured local-plan admission and a lock before development scores exist."""
import json

import numpy as np

from .common import ROOT, digest, read_json, runtime, utc_now, write_json
from .data import Market
from .search_space import generate
from .training import source_identity

MODELS = ('ridge', 'lightgbm', 'xpatch', 'patchtst', 'chronos2_frozen', 'chronos2_lora')


def profiles():
    records = {}
    for name in MODELS:
        records[name] = {}
        for mode in ('small', 'maximum'):
            suffix = '_implementation' if mode == 'maximum' else ''
            path = ROOT / f'outputs/performance/{name}_{mode}{suffix}/result.json'
            row = read_json(path)
            if not row.get('passed') or not row['resources']['passed']:
                raise ValueError(f'Incomplete performance admission: {name}/{mode}')
            if row['resources']['max_affinity_cpus'] > 18 or row['resources']['swap_growth_mib'] > 8:
                raise ValueError('CPU affinity/swap admission failed')
            if mode == 'maximum' and row['training_samples'] != row['max_planned_training_samples']:
                raise ValueError('Maximum planned sample shape was not exercised')
            if mode == 'maximum' and name in ('xpatch', 'patchtst') and row['training_wall_seconds'] < 600:
                raise ValueError('Maximum sustained training shorter than ten minutes')
            if name == 'chronos2_lora' and row['updates'] != 900:
                raise ValueError('Full 900-update LoRA pilot missing')
            records[name][mode] = row
    return records


def project(split, market, measured, shapes, trials):
    """Charge every increment, seed, validation, refit, ablation, load and save."""
    rows = {}
    nmax = len(market.select(split['final_fit_dates']))
    fit_ratio = sum(len(market.select(i['fit_dates'])) / nmax for i in split['inner'])
    val_n = sum(len(market.select(i['validation_dates'])) for i in split['inner'])
    d_ratio = len(market.select(split['confirmation']['fit_dates'])) / nmax
    d_n = len(market.select(split['confirmation']['calibration_dates'])) + len(market.select(split['confirmation']['evaluation_dates']))
    final_n = len(market.select(split['calibration_dates'])) + 8832
    for name in MODELS:
        maximum = measured[name]['maximum']
        components = dict(search=0.0, seed_review=0.0, validation=0.0, confirmation=0.0,
                          mechanism=0.0, final_if_selected=0.0, checkpoint_and_load_io=0.0)
        if name in ('xpatch', 'patchtst'):
            epoch = max(r['epoch_wall_seconds'] for r in maximum['training_runs'])
            anchor = next(r for r in shapes['records'][name] if r['trial']['trial_id'] == 'pressure_reference')
            available = {r['trial']['trial_id']: r for r in shapes['records'][name]}
            unit = sorted([epoch * max(available[t['trial_id']]['warm_epoch_seconds']) / max(anchor['warm_epoch_seconds']) for t in trials[name]], reverse=True)
            n, p1, p2 = len(unit), min(11, len(unit)), min(4, len(unit))
            search_epochs = n * 3 + p1 * 6 + p2 * 18
            components['search'] = (3 * sum(unit) + 6 * sum(unit[:p1]) + 18 * sum(unit[:p2])) * fit_ratio
            components['seed_review'] = 2 * 27 * sum(unit[:2]) * fit_ratio
            inf = max(maximum['validation_inference_wall_seconds']) / maximum['validation_samples']
            components['validation'] = (search_epochs + 108) * val_n * inf
            components['confirmation'] = 81 * unit[0] * d_ratio + 3 * d_n * inf
            components['final_if_selected'] = 81 * unit[0] + 3 * final_n * inf
            if name == 'xpatch':
                components['mechanism'] = components['confirmation']
            save, load = maximum['phases_seconds']['save_checkpoint'], maximum['phases_seconds']['reload_checkpoint']
            # Charge the full optional final even if this family ultimately is not selected.
            epochs = (search_epochs + 108) * len(split['inner']) + (243 if name == 'xpatch' else 162)
            jobs = (n + p1 + p2 + 12) * len(split['inner']) + 9
            components['checkpoint_and_load_io'] = 2 * epochs * save + jobs * load
        elif name == 'chronos2_lora':
            stages = maximum['training_runs']
            step = max(stages[0]['increment_wall_seconds'] / 100, stages[1]['increment_wall_seconds'] / 200,
                       stages[2]['increment_wall_seconds'] / 600)
            n = len(trials[name])
            search_updates = n * 100 + min(n, 4) * 200 + min(n, 2) * 600
            components['search'] = search_updates * len(split['inner']) * step
            components['seed_review'] = 3600 * len(split['inner']) * step
            inf = max(maximum['validation_inference_wall_seconds']) / maximum['validation_samples']
            components['validation'] = (search_updates / 100 + 36) * val_n * inf
            components['confirmation'] = 2700 * step + 3 * d_n * inf
            components['final_if_selected'] = 2700 * step + 3 * final_n * inf
            # A frozen inference at the identical selected configuration is the D-block mechanism.
            components['mechanism'] = d_n * inf + maximum['phases_seconds']['model_load']
            saves = (search_updates / 100 + 36) * len(split['inner']) + 54
            io = maximum['phases_seconds']['save_checkpoint'] + maximum['phases_seconds']['reload_checkpoint']
            loads = (n + min(n, 4) + min(n, 2) + 12) * len(split['inner']) + 6
            components['checkpoint_and_load_io'] = saves * 2 * io + loads * maximum['phases_seconds']['model_load']
        elif name == 'chronos2_frozen':
            inf = max(maximum['validation_inference_wall_seconds']) / maximum['validation_samples']
            components['search'] = len(trials[name]) * val_n * inf
            components['confirmation'] = d_n * inf
            components['final_if_selected'] = final_n * inf
            components['checkpoint_and_load_io'] = (len(trials[name]) * len(split['inner']) + 2) * maximum['phases_seconds']['model_load']
        else:
            fit = max(maximum['fit_wall_seconds']) * (2 if name == 'lightgbm' else 1)
            components['search'] = len(trials[name]) * fit * fit_ratio
            if name == 'lightgbm':
                components['seed_review'] = 4 * fit * fit_ratio
            seeds = 3 if name == 'lightgbm' else 1
            inf = maximum['phases_seconds']['validation_inference'] / maximum['validation_samples']
            components['validation'] = (len(trials[name]) + (4 if seeds == 3 else 0)) * val_n * inf
            components['confirmation'] = seeds * fit * d_ratio + seeds * d_n * inf
            components['final_if_selected'] = seeds * fit + seeds * final_n * inf
            components['checkpoint_and_load_io'] = (len(trials[name]) * len(split['inner']) + 6) * maximum['phases_seconds']['save_reload_verify']
        rows[name] = {'components_seconds': components, 'hours_with50pct_margin': 1.5 * sum(components.values()) / 3600}
    # Final-test selection is fixed to xPatch plus at most one other family. Charge the slowest possible second family.
    gpu_names = ('xpatch', 'patchtst', 'chronos2_frozen', 'chronos2_lora')
    gpu_base = sum(sum(v for k, v in rows[n]['components_seconds'].items() if k != 'final_if_selected') for n in gpu_names)
    gpu_final = rows['xpatch']['components_seconds']['final_if_selected'] + max(rows[n]['components_seconds']['final_if_selected'] for n in gpu_names if n != 'xpatch')
    cpu = sum(sum(r['components_seconds'].values()) for n, r in rows.items() if n not in gpu_names)
    gpu = (gpu_base + gpu_final) * 1.5 / 3600
    return {'models': rows, 'gpu_hours_with_margin': gpu, 'cpu_hours_with_margin': cpu * 1.5 / 3600,
            'wall_hours_with_margin_and_overhead': gpu + cpu * 1.5 / 3600 + 2,
            'safety_factor': 1.5, 'fixed_wall_overhead_hours': 2,
            'admitted_validation_blocks': [i['name'] for i in split['inner']],
            'starting_counts': {k: len(v) for k, v in trials.items()},
            'assumptions': ['All trial geometries timed and calibrated to maximum-N sustained pressure tests',
                            'Promotions, reviewers and refits charged at the slowest eligible geometry',
                            'No early stopping, lower final budget, sampling, reduced precision or concurrency speedup credited',
                            'Distinct-data LightGBM factor 2 plus the common 50 percent margin',
                            'D block, paired mechanisms, calibration and worst possible two-family sealed inference included',
                            'Cold load/checkpoint IO and two wall hours of scheduling/statistical overhead included']}


def main():
    runtime()
    if (ROOT / 'protocol_locked.json').exists():
        raise SystemExit('A frozen protocol already exists; never overwrite it silently')
    required = ['data_admission', 'model_admission', 'optimization_admission', 'executor_smoke', 'parallel_admission', 'parallel_executor_smoke']
    gates = {name: read_json(ROOT / f'outputs/{name}.json') for name in required}
    if not all(g['passed'] for g in gates.values()):
        raise ValueError('Some implementation/admission gate has not passed')
    manifest = read_json(ROOT / 'holdout_manifest.json')
    if not manifest['sealed'] or manifest['predictive_scores_computed']:
        raise ValueError('Final data were not sealed before scoring')
    measured = profiles()
    shapes = read_json(ROOT / 'outputs/shape_profiles.json')
    if not shapes['passed'] or not shapes['resources']['passed']:
        raise ValueError('Shape timing failed')
    compile_result = read_json(ROOT / 'outputs/compile_admission.json')
    if not compile_result['completed']:
        raise ValueError('Bounded compile test has not finished')
    draft = read_json(ROOT / 'protocol_draft.json')
    if digest(ROOT / 'IMPLEMENTATION_PLAN.md') != draft['authoritative_plan']['sha256']:
        raise ValueError('Draft refers to an older user plan')
    market = Market.load()
    split = read_json(ROOT / 'outputs/development_splits.json')
    trials = generate()
    projections = []
    projections.append(project(split, market, measured, shapes, trials))
    if projections[-1]['gpu_hours_with_margin'] > 24:
        trials['xpatch'], trials['chronos2_lora'] = trials['xpatch'][:16], trials['chronos2_lora'][:6]
        projections.append(project(split, market, measured, shapes, trials))
    if projections[-1]['gpu_hours_with_margin'] > 24:
        split = {**split, 'inner': [i for i in split['inner'] if i['name'] in ('V2', 'V3')]}
        projections.append(project(split, market, measured, shapes, trials))
    selected = projections[-1]
    performance = {'passed': selected['gpu_hours_with_margin'] <= 24 and selected['wall_hours_with_margin_and_overhead'] <= 48,
                   'completed_at_utc': utc_now(), 'candidate_projections_in_predeclared_order': projections,
                   'selected_projection': selected,
                   'profiles': {n: {m: {'path': f'outputs/performance/{n}_{m}' + ('_implementation' if m == 'maximum' else '') + '/result.json',
                      'sha256': digest(ROOT / ('outputs/performance/' + n + '_' + m + ('_implementation' if m == 'maximum' else '') + '/result.json'))}
                     for m in ('small', 'maximum')} for n in MODELS},
                   'all_supervised_shapes_sha256': digest(ROOT / 'outputs/shape_profiles.json')}
    # All retained successful tensors, active first rungs, trees, pilots, and generous metadata allowance.
    current_bytes = sum(p.stat().st_size for p in (ROOT / 'outputs').rglob('*') if p.is_file())
    lora_bytes = (ROOT / 'outputs/performance/chronos2_lora_maximum_implementation/latest.pt').stat().st_size
    tree_bytes = (ROOT / 'outputs/performance/lightgbm_maximum_implementation/lightgbm.txt').stat().st_size
    projected_bytes = current_bytes + 2 * (36 + 18 + 6) * lora_bytes + 165 * tree_bytes + 2 * 1024**3
    performance['storage_projection'] = {'gib': projected_bytes * 1.2 / 1024**3, 'cap_gib': 20,
                                       'successful_loser_tensor_pruning_required': True}
    cpu_peaks = sum(measured[n]['maximum']['resources']['peak_rss_gib'] for n in ('ridge', 'lightgbm'))
    worst_group_available = min(r['group']['min_available_ram_gib'] for r in gates['parallel_admission']['records'].values() if r['passed'])
    performance['combined_schedule_memory_bound'] = {
        'available_ram_lower_bound_gib': worst_group_available - cpu_peaks - 0.5,
        'assumption': 'Worst measured three-GPU available RAM minus both isolated CPU peak RSS and0.5GiB manager increment; ongoing actual-memory guard still required'}
    performance['passed'] &= performance['storage_projection']['gib'] <= 20
    write_json(ROOT / 'outputs/performance_admission.json', performance)
    if not performance['passed']:
        raise SystemExit('Local plan not admitted after its declared reductions; stop with a feasibility result')
    write_json(ROOT / 'trial_lists.json', trials)
    locked = {**draft, 'status': 'local_implementation_plan_admitted_locked', 'locked_at_utc': utc_now(),
              'scope': 'three_validation_blocks_D_and_conditional_sealed_test',
              'development': {**draft['development'], 'admitted_validation_blocks': selected['admitted_validation_blocks']},
              'admission': dict(data_passed=True, sealed_data_passed=True, adapters_passed=True,
                                optimization_passed=True, performance_passed=True, formal_training_allowed=True),
              'execution': {'model_options': gates['optimization_admission']['adopted'],
                            'gpu_concurrency': gates['parallel_admission']['adopted_concurrency'],
                            'cpu_ownership': {'gpu_workers': [[0, 1], [2, 3], [4, 5]], 'lightgbm': list(range(6, 14)),
                                              'ridge': [14, 15], 'manager': [16, 17]},
                            'compile': False, 'sampling': 'all_origins', 'precision': 'fp32'},
              'implementation_sha256': source_identity(), 'trial_list_file': 'trial_lists.json',
              'trial_list_sha256': digest(ROOT / 'trial_lists.json'), 'measured_budget': selected,
              'power_planning_sha256': digest(ROOT / 'outputs/power_planning.json'),
              'holdout_manifest_sha256': digest(ROOT / 'holdout_manifest.json'),
              'fine_tuning': {**draft['fine_tuning'], 'chronos2_adapter': 'official_native_forward_quantile_loss_custom_resumable_LoRA_trainer',
                              'schedule': 'linear_over900_updates_same_prefix_at_every_rung'},
              'D_calibration_detail': '20 sessions before D excluded from fitting, with one-session purges before calibration and before D; final calibration is June last20 sessions',
              'artifact_policy': 'Retain all trial identities, metrics, predictions and failures; prune verified successful eliminated tensors with hash tombstones'}
    write_json(ROOT / 'protocol_locked.json', locked)
    files = {str(p.relative_to(ROOT)): digest(p) for folder in ('aapl', 'scripts', 'tests') for p in sorted((ROOT / folder).rglob('*.py'))}
    files['scripts/python.sh'] = digest(ROOT / 'scripts/python.sh')
    write_json(ROOT / 'protocol_lock_receipt.json', {'protocol_sha256': digest(ROOT / 'protocol_locked.json'),
               'environment_sha256': digest(ROOT / 'environment_lock.json'), 'implementation_files': files,
               'admission_files': {str(p.relative_to(ROOT)): digest(p) for p in [ROOT / f'outputs/{n}.json' for n in required] +
                    [ROOT / 'outputs/performance_admission.json', ROOT / 'outputs/power_planning.json', ROOT / 'holdout_manifest.json',
                     ROOT / 'outputs/holdout_exposure_audit.json', ROOT / 'model_availability.json']},
               'locked_at_utc': utc_now()})
    print(json.dumps({'locked': True, 'budget': selected}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
