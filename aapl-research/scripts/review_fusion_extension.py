"""Recompute the fixed three-seed follow-up and its paired development control.

This review cannot substitute the paired control as winner or open final data.
All qualification remains development evidence after selection of one recipe.
"""
import argparse
from pathlib import Path

import numpy as np

from fusion.next.common import ROOT, OUT as BASE, digest, identity, read_json, utc_now, write_json
from fusion.next.evaluation import point_scores, seed_system
from scripts.followup_admission import check_frozen, identified
from scripts.followup_runtime import FOLLOW
from scripts.review_fusion_development import QUARTERS
from scripts.review_fusion_pilot import checked_review as checked_pilot_review, full_prediction, original_seeds, robustness, time_segments


def evaluate(target,quarters,dates,segments,selected,paired,incumbent,fixed,plan):
    target = np.asarray(target,dtype=np.float64)
    if set(np.unique(quarters)) != set(QUARTERS):
        raise ValueError('Extension review requires all three development quarters')
    scores,mean = seed_system(target,selected)
    control,control_mean = seed_system(target,paired)
    incumbent_scores,incumbent_mean = seed_system(target,incumbent)
    by_quarter = {q:point_scores(target[quarters==q],mean[quarters==q]) for q in QUARTERS}
    gates = {'at_least_one_percent_vs_zero':scores['system']['mse_improvement_vs_zero'] is not None
                 and scores['system']['mse_improvement_vs_zero']>=.01,
             'beats_fixed_strongest_single':scores['system']['mse']<fixed['fixed_strongest_single']['mse'],
             'beats_fixed_strongest_simple':scores['system']['mse']<fixed['fixed_strongest_simple']['mse'],
             'at_least_two_positive_quarters':sum(r['mse_improvement_vs_zero']>0 for r in by_quarter.values())>=2,
             'all_three_seeds_positive':all(r['mse_improvement_vs_zero']>0 for r in scores['individual_seeds'].values())}
    paired_delta = scores['system']['mse']-control['system']['mse']
    on_minus_off = paired_delta if plan['selected_candidate']=='A_foundation_raw' else -paired_delta
    return {'selected_candidate':plan['selected_candidate'],'paired_control':plan['paired_control'],
            'seeds':[2026,2027,2028],'selected':scores,'quarters':by_quarter,'paired_control_scores':control,
            'original_selected_system':incumbent_scores,
            'fixed_strongest_single':fixed['fixed_strongest_single'],
            'fixed_strongest_simple':fixed['fixed_strongest_simple'],
            'development_checks':gates,'development_qualification_passed':all(gates.values()),
            'beats_original_selected_system':scores['system']['mse']<incumbent_scores['system']['mse'],
            'paired_selected_minus_control_MSE':paired_delta,
            'foundation_on_minus_off_MSE':on_minus_off,
            'negative_foundation_contrast_means_lower_MSE_with_foundation':True,
            'paired_seed_MSE_differences':{str(seed):scores['individual_seeds'][str(seed)]['mse']
                 - control['individual_seeds'][str(seed)]['mse'] for seed in (2026,2027,2028)},
            'paired_robustness':robustness(target,mean,control_mean,dates,segments),
            'incumbent_robustness':robustness(target,mean,incumbent_mean,dates,segments),
            'paired_control_can_replace_selected_candidate':False,
            'development_checks_authorize_final_data_access':False,
            'independent_research_success_claimed':False,
            'role':'post-selection development evidence; future independent validation still required',
            'parameter_count_and_initialization_differ_when_a_branch_is_removed':True}


def checked_review(manifest_path,manifest=None):
    manifest = manifest or check_frozen(manifest_path)
    directory = FOLLOW/'runs'/manifest['identity']/'extension_reviews'
    pointer = read_json(directory/'latest.json');path = (ROOT/pointer['report']).resolve()
    if (not path.is_relative_to(directory.resolve()) or path.name!='report.json'
            or pointer['manifest_identity']!=manifest['identity']):
        raise ValueError('Extension review pointer leaves its matrix directory')
    saved = identified(path)
    stable = {k:v for k,v in saved.items() if k not in ('identity','decision_identity','reviewed_at_utc')}
    if (path.parent.name!=saved['identity'] or pointer['identity']!=saved['identity']
            or identity(stable)!=saved['decision_identity']
            or pointer['development_qualification_passed']!=saved['development_qualification_passed']
            or pointer['final_data_access_authorized'] is not False):
        raise ValueError('Extension review pointer or stable decision differs')
    _,fresh = review(manifest_path,publish=False)
    if saved['decision_identity']!=fresh['decision_identity']:
        raise ValueError('Saved extension review differs from recomputed predictions')
    return {'report':str(path.relative_to(ROOT)),'identity':saved['identity'],
            'decision_identity':saved['decision_identity']},fresh


def review(manifest_path,publish=True):
    manifest = check_frozen(manifest_path)
    saved,plan = checked_pilot_review(manifest_path,manifest)
    run = FOLLOW/'runs'/manifest['identity'];extension = run/'extension'
    path = extension/'extension_result.json';complete = read_json(path)
    if (complete['manifest_identity'] != manifest['identity'] or complete['extension_plan'] != plan
            or complete['stage'] != 'extension_complete_review_required'
            or complete['expected_fits'] != plan['new_full_fits']
            or complete['total_coverage_equivalent_epochs'] != 2
            or complete['Q2_Q3_prediction_arrays_loaded'] is not False
            or complete['independent_success_claimed'] is not False or complete['final_fit_started'] is not False):
        raise ValueError('The fixed seed extension is not complete or its decision differs')
    expected = {(name,q,seed) for name in plan['train_candidates'] for q in QUARTERS for seed in plan['new_seeds']}
    recorded = [(r['candidate'],r['quarter'],r['seed']) for r in complete['completed_fits']]
    if len(recorded) != plan['new_full_fits'] or set(recorded) != expected:
        raise ValueError('Extension completion omitted or duplicated a fixed fit')
    from scripts.run_fusion_followup import attempts,spent_seconds,EXTENSION_SECONDS,ENTRY
    provenance = {str(path.relative_to(ROOT)):digest(path),saved['report']:digest(ROOT/saved['report'])}
    valid_guard = False
    for request in attempts(run,'extend'):
        if not request.exists():continue
        spent_seconds(request,EXTENSION_SECONDS)
        state = read_json(request.parent/'guard_state.json')
        admission = identified(request.parent/'launch_admission.json')
        if (state['stage']=='completed' and state['worker_returncode']==0
                and admission['manifest_identity']==manifest['identity'] and admission['mode']=='extend'
                and admission['entry_sha256']==digest(ENTRY) and admission['budget']['extension_plan']==plan
                and admission['budget']['pilot_review']==saved):
            valid_guard = True
            for member in ('guard_request.json','guard_state.json','launch_admission.json'):
                source = request.parent/member;provenance[str(source.relative_to(ROOT))] = digest(source)
    if not valid_guard:raise ValueError('No completed exact guarded extension attempt exists')
    from fusion.next.panel import Panel
    panel = Panel.load();configs = {r['name']:r['config'] for r in manifest['candidates']}
    common,targets,predictions = {},[],{}
    for name in plan['train_candidates']:
        seeds = []
        for seed in (2026,2027,2028):
            parts = []
            for quarter in QUARTERS:
                directory = (run/'pilot/trials/pilot' if seed == 2026 else extension/'trials/extension')/configs[name]['candidate_id']
                rows,p,result = full_prediction(directory,configs[name],quarter,manifest,panel,provenance,seed)
                if quarter not in common:
                    common[quarter] = rows;targets.append(panel.targets['AAPL'][rows])
                elif not np.array_equal(common[quarter],rows):raise ValueError('Extension common rows differ')
                if seed != 2026:
                    recorded_fit = next(r for r in complete['completed_fits'] if (r['candidate'],r['quarter'],r['seed'])==(name,quarter,seed))
                    if recorded_fit['prediction_sha256'] != result['prediction_sha256'] or recorded_fit['scores'] != result['scores']:
                        raise ValueError('Extension complete receipt differs from recomputed fit')
                parts.append(p)
            seeds.append(np.concatenate(parts))
        predictions[name] = np.stack(seeds)
    rows = np.concatenate([common[q] for q in QUARTERS]);target = np.concatenate(targets)
    quarters = np.concatenate([np.full(len(common[q]),q) for q in QUARTERS])
    def original(relative):
        source = BASE/relative;provenance[str(source.relative_to(ROOT))] = digest(source)
        return original_seeds(source,manifest,panel,rows,quarters,target)
    control = (original(manifest['original_frozen_predictions']) if plan['original_frozen_control_reused']
               else predictions[plan['paired_control']])
    fixed_path = BASE/'system_freeze.json';provenance[str(fixed_path.relative_to(ROOT))] = digest(fixed_path)
    result = evaluate(target,quarters,panel.dates[rows],time_segments(panel.stamps[rows]),
                      predictions[plan['selected_candidate']],control,
                      original(manifest['original_selected_predictions']),read_json(fixed_path),plan)
    result.update(manifest_identity=manifest['identity'],extension_plan=plan,pilot_review=saved,
                  prediction_provenance_sha256=provenance,Q2_Q3_prediction_arrays_loaded=False,
                  total_coverage_equivalent_epochs=2,new_full_fits=plan['new_full_fits'])
    result['decision_identity'] = identity(result)
    result['reviewed_at_utc'] = utc_now()
    result['identity'] = identity(result)
    if not publish:return None,result
    destination = run/'extension_reviews'/result['identity'];write_json(destination/'report.json',result)
    lines = ['# 下一轮三种子开发复核','',
             f'固定配方：{plan["selected_candidate"]}；固定配对对照：{plan["paired_control"]}。',
             f'三种子均值相对零收益 MSE 改善：{100*result["selected"]["system"]["mse_improvement_vs_zero"]:+.6f}%。',
             f'原开发数值门槛检查：{result["development_qualification_passed"]}。','',
             '这是选择后的开发证据，不授权最终集访问或独立研究成功结论。配对对照不能替换已选配方。',
             '基础模型对照包含支路移除造成的参数量和初始化差异；结果须结合完整预测与逐种子、季度、分时段及异常日敏感性报告解读。']
    (destination/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(run/'extension_reviews/latest.json',{'report':str((destination/'report.json').relative_to(ROOT)),
               'identity':result['identity'],'manifest_identity':manifest['identity'],
               'development_qualification_passed':result['development_qualification_passed'],
               'final_data_access_authorized':False})
    return destination,result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    path,result = review(parser.parse_args().manifest)
    print({'review':str(path),'development_qualification_passed':result['development_qualification_passed'],
           'independent_success_claimed':False,'final_data_access_authorized':False})
