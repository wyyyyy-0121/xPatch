"""Recompute the fixed pilot and its exploratory expansion decision on CPU.

Reads only the three known development quarters; no Q2/Q3 arrays, training,
prediction adjustment, candidate search or original gate replacement occurs.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fusion.next.common import ROOT, OUT as BASE, digest, identity, read_json, utc_now, write_json
from fusion.next.evaluation import point_scores
from fusion.next.recipe import for_fit
from fusion.next.splits import fold
from scripts.followup_admission import check_frozen, REVIEW_POLICY, NUMERICAL, identified
from scripts.followup_runtime import FOLLOW
from scripts.review_fusion_development import QUARTERS, validate_rows


def time_segments(stamps):
    clock = pd.to_datetime(stamps,utc=True).tz_convert('America/New_York')
    minute = clock.hour.to_numpy()*60+clock.minute.to_numpy()
    return np.select([minute<630,minute<870],REVIEW_POLICY['segments'][:2],default=REVIEW_POLICY['segments'][2])


def robustness(target,prediction,control,dates,segments):
    """Retain every origin; remove the fixed top five days only as sensitivity."""
    target,prediction,control = [np.asarray(a,dtype=np.float64) for a in (target,prediction,control)]
    point_scores(target,prediction);point_scores(target,control)
    if dates.shape != target.shape or segments.shape != target.shape:
        raise ValueError('Paired daily/time labels differ from the common origins')
    gain = (target-control)**2-(target-prediction)**2
    days,inverse = np.unique(dates,return_inverse=True)
    gains = np.bincount(inverse,weights=gain)
    positive = [i for i in range(len(days)) if gains[i]>0]
    ordered = sorted(positive,key=lambda i:(-gains[i],str(days[i])))
    removed = ordered[:REVIEW_POLICY['remove_top_positive_gain_days']]
    kept = ~np.isin(inverse,removed)
    after = float(gain[kept].mean()) if kept.any() else None
    by_time = {name:{'n':int((segments==name).sum()),
                    'mean_loss_reduction':float(gain[segments==name].mean())}
               for name in REVIEW_POLICY['segments'] if np.any(segments==name)}
    return {'paired_mean_loss_reduction':float(gain.mean()),
            'positive_gain_time_segments':sum(r['mean_loss_reduction']>0 for r in by_time.values()),
            'time_of_day':by_time,'days':len(days),'positive_gain_days':int((gains>0).sum()),
            'removed_top_positive_days':[{'date':str(days[i]),'loss_reduction_sum':float(gains[i])} for i in removed],
            'origins_remaining_after_day_sensitivity':int(kept.sum()),
            'mean_loss_reduction_after_top_days_removed':after,
            'positive_after_top_days_removed':bool(after is not None and after>0),
            'day_sensitivity_changes_predictions_or_primary_score':False}


def evaluate(target,quarters,dates,segments,predictions,reference,incumbent):
    """Fixed policy allocation; single-seed pilot is never a three-seed result."""
    target = np.asarray(target,dtype=np.float64)
    reference,incumbent = np.asarray(reference,dtype=np.float64),np.asarray(incumbent,dtype=np.float64)
    controls = {'zero':np.zeros_like(target),'fixed_frozen_seed2026':reference,
                'original_selected_three_seed_mean':incumbent}
    if set(np.unique(quarters)) != set(QUARTERS):
        raise ValueError('All three validation quarters are required')
    results = {}
    for name,p in predictions.items():
        p = np.asarray(p,dtype=np.float64)
        pooled = point_scores(target,p)
        by_quarter = {q:point_scores(target[quarters==q],p[quarters==q]) for q in QUARTERS}
        paired = {key:robustness(target,p,c,dates,segments) for key,c in controls.items()}
        gates = {'pooled_beats_zero':pooled['mse'] < point_scores(target,controls['zero'])['mse'],
                 'pooled_beats_fixed_reference':pooled['mse'] < point_scores(target,reference)['mse'],
                 'pooled_beats_original_selected_ensemble':pooled['mse'] < point_scores(target,incumbent)['mse'],
                 'at_least_two_positive_quarters':sum(s['mse_improvement_vs_zero']>0 for s in by_quarter.values())>=REVIEW_POLICY['minimum_positive_quarters'],
                 'gain_in_at_least_two_time_segments_for_each_control':all(r['positive_gain_time_segments']>=REVIEW_POLICY['minimum_positive_time_segments'] for r in paired.values()),
                 'gain_survives_five_best_days_removed_for_each_control':all(r['positive_after_top_days_removed'] for r in paired.values())}
        results[name] = {'pooled':pooled,'quarters':by_quarter,'paired_robustness':paired,
                         'exploratory_checks':gates,'eligible_for_seed_expansion':all(gates.values())}
    eligible = sorted((name for name,r in results.items() if r['eligible_for_seed_expansion']),
                      key=lambda name:(results[name]['pooled']['mse'],name))
    return {'policy':REVIEW_POLICY,'controls':{name:point_scores(target,p) for name,p in controls.items()},
            'candidates':results,'selected_for_possible_expansion':eligible[0] if eligible else None,
            'expansion_requires_new_budget_and_runtime_admission':True,'training_started':False,
            'independent_research_success_claimed':False,'original_research_gates_replaced':False,
            'seed_count_disclosure':'new recipes and fixed reference: seed 2026; original selected system: three-seed mean'}


def factorial(target,predictions,reference):
    """Matched seed-2026 cells; the old three-seed ensemble is not a matrix cell."""
    cells = {'on_scaled':point_scores(target,reference)['mse'],
             'on_raw':point_scores(target,predictions['A_foundation_raw'])['mse'],
             'off_scaled':point_scores(target,predictions['B_no_foundation_scaled'])['mse'],
             'off_raw':point_scores(target,predictions['C_no_foundation_raw'])['mse']}
    on_raw,off_raw = cells['on_raw'],cells['off_raw']
    on_scaled,off_scaled = cells['on_scaled'],cells['off_scaled']
    return {'raw_MSE_cells':cells,'foundation_on_minus_off_under_scaled':on_scaled-off_scaled,
            'foundation_on_minus_off_under_raw':on_raw-off_raw,
            'raw_minus_scaled_with_foundation':on_raw-on_scaled,
            'raw_minus_scaled_without_foundation':off_raw-off_scaled,
            'interaction_difference_in_foundation_effects':(on_raw-off_raw)-(on_scaled-off_scaled),
            'negative_difference_means_lower_MSE_for_first_named_condition':True,
            'role':'descriptive paired development contrasts, not independent causal or significance claims',
            'parameter_count_and_initialization_differ_when_a_branch_is_removed':True}


def full_prediction(directory,config,quarter,manifest,panel,provenance,seed=2026):
    f = fold(panel.dates,quarter);effective = for_fit(config,f['fit_end'])
    path = directory/quarter/('seed_'+str(seed))/('updates_'+str(effective['total_updates']))
    report = read_json(path/'result.json')
    details = {'config':effective,'fold':f,'seed':seed,'updates':effective['total_updates'],
               'source':manifest['source_identity'],'evaluation_precision':'float32','numerical_policy_identity':NUMERICAL}
    if report['details'] != details or digest(path/'prediction.npz') != report['prediction_sha256']:
        raise ValueError('Pilot/reference prediction identity, precision or full recipe differs')
    with np.load(path/'prediction.npz',allow_pickle=False) as saved:
        rows,p = saved['rows'].copy(),saved['prediction'].copy()
    validate_rows(rows,p,panel.dates,quarter)
    origins = panel.origins['AAPL']
    expected = origins[(panel.dates[origins]>=f['score_start'])&(panel.dates[origins]<=f['score_end'])]
    if not np.array_equal(rows,expected) or len(rows) != manifest['folds'][quarter]['score_origins']:
        raise ValueError('Pilot/reference omits eligible common origins')
    scores = point_scores(panel.targets['AAPL'][rows],p)
    if scores != report['scores']:
        raise ValueError('Pilot/reference score differs from raw predictions and local targets')
    for member in ('result.json','prediction.npz'):
        source = path/member
        provenance[str(source.relative_to(ROOT))] = digest(source)
    return rows,p,report


def original_seeds(path,manifest,panel,rows,quarters,target):
    relative = str(path.relative_to(BASE))
    if (relative not in manifest['original_inputs']
            or digest(path) != manifest['original_inputs'][relative]['sha256']):
        raise ValueError('Original development ensemble was not frozen or changed')
    with np.load(path,allow_pickle=False) as saved:
        for key,value in {'rows':rows,'quarters':quarters,'target':target,
                          'timestamps_ns':panel.stamps[rows].as_unit('ns').asi8}.items():
            if not np.array_equal(saved[key],value):raise ValueError('Original ensemble changed common '+key)
        seeds = saved['predictions'].astype(np.float64)
        if seeds.shape != (3,len(rows)) or not np.isfinite(seeds).all() or not np.array_equal(seeds.mean(0),saved['ensemble']):
            raise ValueError('Original ensemble is not the exact three-seed mean')
        return seeds


def expansion_members(manifest,report):
    """One preselected recipe and only its fixed paired control receive new seeds."""
    selected = report['selected_for_possible_expansion']
    names = {r['name'] for r in manifest['candidates']}
    if (report['manifest_identity'] != manifest['identity'] or report['policy'] != REVIEW_POLICY
            or report['independent_research_success_claimed'] is not False
            or report['original_research_gates_replaced'] is not False):
        raise ValueError('Pilot review does not follow the frozen exploratory policy')
    if selected is None:
        raise RuntimeError('No fixed pilot recipe qualifies for seed expansion')
    if selected not in names or not report['candidates'][selected]['eligible_for_seed_expansion']:
        raise ValueError('Selected recipe is outside the eligible frozen matrix')
    paired = REVIEW_POLICY['paired_control_by_selected'][selected]
    train = [selected] if paired == 'original_frozen' else [selected,paired]
    if not set(train).issubset(names):
        raise ValueError('Fixed paired recipe is missing from the matrix')
    return {'selected_candidate':selected,'paired_control':paired,'train_candidates':train,
            'new_seeds':[2027,2028],'pilot_seed_reused':2026,'new_full_fits':len(train)*6,
            'pilot_decision_identity':report['decision_identity'],
            'original_frozen_control_reused':paired == 'original_frozen',
            'paired_control_can_replace_selected_candidate':False}


def checked_review(manifest_path,manifest=None):
    """Recompute the saved pilot decision before granting any seed extension."""
    manifest = manifest or check_frozen(manifest_path)
    directory = FOLLOW/'runs'/manifest['identity']/'reviews'
    pointer = read_json(directory/'latest.json')
    path = (ROOT/pointer['report']).resolve()
    if (not path.is_relative_to(directory.resolve()) or path.name != 'report.json'
            or pointer['manifest_identity'] != manifest['identity']):
        raise ValueError('Pilot review pointer leaves the exact matrix review directory')
    saved = identified(path)
    stable = {k:v for k,v in saved.items() if k not in ('identity','reviewed_at_utc','decision_identity')}
    if (path.parent.name != saved['identity'] or pointer['identity'] != saved['identity']
            or pointer['selected_for_possible_expansion'] != saved['selected_for_possible_expansion']
            or identity(stable) != saved['decision_identity']):
        raise ValueError('Pilot review pointer or stable decision changed')
    _,fresh = review(manifest_path,publish=False)
    if saved['decision_identity'] != fresh['decision_identity']:
        raise ValueError('Saved pilot decision differs from recomputed predictions')
    plan = expansion_members(manifest,fresh)
    return {'report':str(path.relative_to(ROOT)),'identity':saved['identity'],
            'decision_identity':saved['decision_identity']},plan


def review(manifest_path,publish=True):
    manifest = check_frozen(manifest_path)
    run = FOLLOW/'runs'/manifest['identity']
    pilot = run/'pilot'
    complete = read_json(pilot/'pilot_result.json')
    if (complete['manifest_identity'] != manifest['identity'] or complete['stage'] != 'pilot_complete_review_required'
            or complete['expected_fits'] != 9 or complete['total_coverage_equivalent_epochs'] != 2
            or complete['Q2_Q3_prediction_arrays_loaded'] is not False
            or complete['extension_or_final_fit_started'] is not False):
        raise ValueError('The prospective pilot is not complete or belongs to another matrix')
    expected = {(r['name'],q,2026) for r in manifest['candidates'] for q in QUARTERS}
    recorded = [(r['candidate'],r['quarter'],r['seed']) for r in complete['completed_fits']]
    if len(recorded) != 9 or set(recorded) != expected:
        raise ValueError('Pilot completion omitted or duplicated a fixed fit')
    from scripts.run_fusion_followup import spent_seconds,attempts,PILOT_SECONDS,ENTRY
    valid_guard = False
    for request in attempts(run,'pilot'):
        if not request.exists():continue
        spent_seconds(request,PILOT_SECONDS)
        state = read_json(request.parent/'guard_state.json')
        admission = identified(request.parent/'launch_admission.json')
        if (state['stage']=='completed' and state['worker_returncode']==0
                and admission['manifest_identity']==manifest['identity'] and admission['mode']=='pilot'
                and admission['entry_sha256']==digest(ENTRY)):
            valid_guard = True
    if not valid_guard:
        raise ValueError('No completed exact guarded pilot attempt exists')
    from fusion.next.panel import Panel
    panel = Panel.load()
    provenance = {str((pilot/'pilot_result.json').relative_to(ROOT)):digest(pilot/'pilot_result.json')}
    predictions,common,reference,targets = {},{},[],[]
    for quarter in QUARTERS:
        rows,p,_ = full_prediction(BASE/'trials/fusion'/manifest['reference']['candidate_id'],
                                   manifest['reference'],quarter,manifest,panel,provenance)
        common[quarter] = rows;reference.append(p);targets.append(panel.targets['AAPL'][rows])
    for candidate in manifest['candidates']:
        parts = []
        for quarter in QUARTERS:
            rows,p,result = full_prediction(pilot/'trials/pilot'/candidate['config']['candidate_id'],
                                           candidate['config'],quarter,manifest,panel,provenance)
            if not np.array_equal(rows,common[quarter]):raise ValueError('Pilot common rows differ')
            recorded_result = next(r for r in complete['completed_fits'] if (r['candidate'],r['quarter']) == (candidate['name'],quarter))
            if recorded_result['prediction_sha256'] != result['prediction_sha256'] or recorded_result['scores'] != result['scores']:
                raise ValueError('Pilot complete receipt differs from recomputed fit')
            parts.append(p)
        predictions[candidate['name']] = np.concatenate(parts)
    rows = np.concatenate([common[q] for q in QUARTERS])
    quarters = np.concatenate([np.full(len(common[q]),q) for q in QUARTERS])
    target = np.concatenate(targets)
    path = BASE/manifest['original_selected_predictions']
    incumbent = original_seeds(path,manifest,panel,rows,quarters,target).mean(0)
    provenance[str(path.relative_to(ROOT))] = digest(path)
    result = evaluate(target,quarters,panel.dates[rows],time_segments(panel.stamps[rows]),
                      predictions,np.concatenate(reference),incumbent)
    result['factorial_contrasts'] = {'pooled':factorial(target,predictions,np.concatenate(reference)),
        'quarters':{q:factorial(target[quarters==q],{k:p[quarters==q] for k,p in predictions.items()},
                                np.concatenate(reference)[quarters==q]) for q in QUARTERS}}
    frozen = read_json(BASE/'system_freeze.json')
    result['original_fixed_control_comparison'] = {
        'single':frozen['fixed_strongest_single'],'simple':frozen['fixed_strongest_simple'],
        'pilot_candidates':{name:{'beats_original_strongest_single':r['pooled']['mse']<frozen['fixed_strongest_single']['mse'],
                                  'beats_original_strongest_simple':r['pooled']['mse']<frozen['fixed_strongest_simple']['mse']}
                            for name,r in result['candidates'].items()},
        'role':'original controls were recomputed in the completed original delivery; formal three-seed gates remain required'}
    result.update(manifest_identity=manifest['identity'],
                  prediction_provenance_sha256=provenance,Q2_Q3_prediction_arrays_loaded=False,
                  horizon_minutes=12,full_pilot_fits=9,pilot_seed=2026)
    result['decision_identity'] = identity(result)
    result['reviewed_at_utc'] = utc_now()
    result['identity'] = identity(result)
    if not publish:return None,result
    destination = run/'reviews'/result['identity']
    write_json(destination/'report.json',result)
    lines = ['# 下一轮固定试验开发复核','',
             '仅使用三个已知开发季度。单种子结果用于决定是否值得补种子，不支持独立研究成功或交易收益结论。','',
             '| 配方 | 相对零收益 MSE 改善 | 正季度 | 探索性扩展条件 |','| --- | ---: | ---: | --- |']
    for name,row in result['candidates'].items():
        lines.append(f'| {name} | {100*row["pooled"]["mse_improvement_vs_zero"]:+.6f}% | '
                     f'{sum(q["mse_improvement_vs_zero"]>0 for q in row["quarters"].values())}/3 | {row["eligible_for_seed_expansion"]} |')
    lines += ['',f'待预算与运行准入复核的唯一配方：{result["selected_for_possible_expansion"] or "无"}。',
              '', '去除贡献最高五日和固定分时段仅作预定敏感性检查；不删除主评价中的任何起点、不修改预测或原研究门槛。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'reviews/latest.json',{'report':str((destination/'report.json').relative_to(ROOT)),
               'identity':result['identity'],'manifest_identity':manifest['identity'],
               'selected_for_possible_expansion':result['selected_for_possible_expansion'],'training_started':False})
    return destination,result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    path,result = review(parser.parse_args().manifest)
    print({'review':str(path),'selected_for_possible_expansion':result['selected_for_possible_expansion'],
           'training_started':False,'independent_success_claimed':False})
