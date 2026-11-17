"""Derive permitted research statements from authoritative follow-up outputs."""
from pathlib import Path
import os,sys
import numpy as np

BASE=Path(__file__).resolve().parent;PHASE=BASE.parent;V3=PHASE.with_name('20261109_v3')
sys.path.insert(0,str(V3/'code'))
from study import read,sha,atomic,now


def main():
    assert set(os.sched_getaffinity(0)).issubset({24,26}) and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    long=read(PHASE/'longscale_replication/analysis/RESULTS.json')
    means=read(PHASE/'mean_controls/audit/RESULTS.json')
    history=read(PHASE/'history_sensitivity/analysis/RESULTS.json')
    primary=read(V3/'analysis/CANDIDATE_RESULTS.json')
    assert read(BASE/'REVIEW.json')['status']=='passed' and read(BASE/'HISTORY_REVIEW.json')['status']=='passed'
    assert read(PHASE/'mean_controls/audit/REPLAY.json')['status']=='passed'
    tasks=[];stability=[];primary_rows=[]
    for task in sorted({r['task'] for r in long['pooled']}):
        get=lambda method:next(r for r in long['pooled'] if r['task']==task and r['method']==method)
        repair=get('repair_shrink');default=get('default_shrink');mean=get('fit_mean_shrink')
        assert repair['seed_count']==3
        row={'task':task,'origins':repair['origins'],'repair_shrink_gain_pct':repair['improvement_pct'],
             'repair_raw_gain_pct':get('repair_trained')['improvement_pct'],
             'repair_seed_gains_pct':repair['seed_improvements_pct'],'repair_positive_folds':repair['positive_folds'],
             'default_shrink_gain_pct':default['improvement_pct'],'mean_shrink_gain_pct':mean['improvement_pct'],
             'ridge_history_selected_gain_pct':get('ridge_history_selected')['improvement_pct']}
        row['history_counterfactual_gains_pct']={r['method']:r['improvement_over_baseline_pct'] for r in history['pooled'] if r['task']==task}
        tasks.append(row)
        local=[r for r in long['sensitivity'] if r['task']==task]
        phases=[r for r in local if r['rule']=='absolute_bar_phase'];years=[r for r in local if r['rule']=='leave_origin_year_out']
        assert len(phases)==int(task.split('_h')[1].split('_')[0])
        with np.load(PHASE/'longscale_replication/analysis/paired_losses'/f'{task}.npz',allow_pickle=False) as a:
            expected_years=sorted(set(str(d)[:4] for d in a['session_date']))
        assert sorted(r['year'] for r in years)==expected_years
        greedy=next(r for r in local if r['rule']=='original_global_earliest_greedy')
        crossed=next(r for r in local if r['rule']=='crosses_split');clear=next(r for r in local if r['rule']=='no_split')
        assert crossed['origins']+clear['origins']==repair['origins']
        h=int(task.split('_h')[1].split('_')[0]);profile=task.split('_')[0]
        seed_stable=all(v>0 for v in repair['seed_improvements_pct'].values())
        comparisons={ref:{str(r['block']):r['supports_improvement'] for r in long['paired_intervals'] if r['task']==task and r['primary']=='repair_shrink' and r['reference']==ref} for ref in ['fit_mean','fit_mean_shrink']}
        out={'task':task,'seed_directions_all_positive':seed_stable,
             'all_absolute_bar_phases_reported':True,'phase_count':len(phases),
             'phase_gain_range_pct':[min(r['methods']['repair_shrink']['improvement_pct'] for r in phases),max(r['methods']['repair_shrink']['improvement_pct'] for r in phases)],
             'all_origin_years_left_out_once':True,'leave_year_out_count':len(years),
             'leave_year_out_gain_range_pct':[min(r['methods']['repair_shrink']['improvement_pct'] for r in years),max(r['methods']['repair_shrink']['improvement_pct'] for r in years)],
             'earliest_greedy_origins':greedy['origins'],
             'earliest_greedy_repair_gain_pct':greedy['methods']['repair_shrink']['improvement_pct'],
             'earliest_greedy_mean_shrink_gain_pct':greedy['methods']['fit_mean_shrink']['improvement_pct'],
             'split_crossing_origins':crossed['origins'],'no_split_origins':clear['origins'],
             'split_exclusion_contrast_estimable':bool(crossed['origins'] and clear['origins']),
             'mean_control_pair_interval_support':comparisons}
        if profile=='annual':assert greedy['origins']==4 and crossed['origins']==0
        if profile=='quarterly':assert crossed['origins']==0
        if profile=='monthly':assert crossed['origins']==21 and crossed['distinct_events']==1
        stability.append(out)
    for gate in primary['primary_performance_gates']:
        task=gate['task'];row=next(r for r in primary['pooled'] if r['task']==task and r['method']=='candidate')
        m=[r for r in means['pooled'] if r['task']==task and r['method']=='shrink'][0]
        assert not gate['performance_gate_passed']
        primary_rows.append({'task':task,'original_candidate_gain_pct':row['improvement_pct'],
                             'mean_shrink_gain_pct':m['improvement_pct'],'original_performance_gate_passed':False,
                             'same_original_task_role_preserved':True})
    plan=(PHASE/'PLAN_ZH.md').read_text()
    assert all(s in plan for s in ['180','150','二十任务','所有起始相位','逐年去除','2026Q3','1794205909.6228158'])
    bn_complete=next((p for p in (PHASE/'bn_recalibration').rglob('*SUMMARY*.json')),None)
    requirements=[
        {'requirement':'Plan1 six long-scale tasks, default versus initialization repair, five folds and three seeds',
         'status':'verified','evidence':'longscale_replication/analysis/RESULTS.json; longscale_review/REVIEW.json',
         'limit':'180 cells include30exact reusedv3cells and150newfits; original eight-method comparison retained from preceding study, not every prior architecture rerun with three seeds.'},
        {'requirement':'Plan2 all twenty fit-only mean path controls, raw and identically inner-shrunk',
         'status':'verified','evidence':'mean_controls/audit/RESULTS.json; mean_controls/audit/REPLAY.json'},
        {'requirement':'Plan3 frozen BatchNorm intervention and conditional full-stage procedure',
         'status':'not_audited_in_this_review','evidence':'bn_recalibration/BN_RULES.json; GATE_AND_ADMISSION.json',
         'limit':'BN specialist owns ongoing execution and acceptance. This review does not infer completion from a plan, gate or state file.'},
        {'requirement':'Plan4 original greedy, every horizon phase, leave each origin year out and split groups',
         'status':'verified_for_all_six_longscale_tasks','evidence':'longscale_replication/analysis/RESULTS.json:sensitivity; longscale_review/REVIEW.json',
         'limit':'All phasechains and leaveyearout subsets are dependent historical sensitivities, not independent tests.'},
        {'requirement':'Same original absolute deadline and constrained CPU execution',
         'status':'verified_for_longscale_and_mean_control_branches','evidence':'longscale_review/REVIEW.json; mean_controls/audit/INVARIANTS.json',
         'limit':'No newOSwide continuousresource sampling is claimed.'},
        {'requirement':'Original title and five scales/twenty task presentation in new manuscript',
         'status':'pending_new_manuscript_review','evidence':'v3 title retained; v3a/manuscript not yet inspected'},
        {'requirement':'Mechanism-motivated stable predictive superiority on original designated return tasks',
         'status':'not_achieved','evidence':'v3 analysis/CANDIDATE_RESULTS.json and retained primary_rows; BN still requires separate result review'},
        {'requirement':'Independent unseen-period confirmation',
         'status':'not_executed','evidence':'allstudyresultflags;2026Q3sealed'},
        {'requirement':'Shared versus independent horizon real-market comparison',
         'status':'not_executed_separate_scope','evidence':'v3 PLAN_ZH.md and confirmation design; v3a longscale fit is not shared learning'},
    ]
    batch=read(PHASE/'longscale_replication/audit/RESTORATION_BATCH_EXPLANATION.json')
    alpha_zero=read(BASE/'HISTORY_REVIEW.json')['zero_alpha_configs'];assert alpha_zero==27
    report={'status':'claims_reviewed_research_goal_not_achieved','longscale_summary':tasks,'stability_summary':stability,
            'original_primary_tasks':primary_rows,'requirements':requirements,
            'necessary_interpretation_limits':[
                'Allv3aexperimentsfollowinspectionofv3scores; no retrospectiveregistration or independentdiscovery claim.',
                'Annual3seed gains remain positive but no nominalpairedinterval proves superiority over raw or shrunk fitmean controls.',
                'Annual47overlappingorigins and4earliestgreedynonoverlappingpaths remain the same market history; moreseeds do not add marketyears.',
                'Historycounterfactuals keep weights,anchors,scalers and originalalpha fixed; substitutions can be outofdistribution and are not refitteddeployablemethods.',
                '27of90historyconfigs havealpha0; reportedcalibratedsensitivity doesnot establish absenceofrawnetworkhistorydependence.',
                'Originaldefault30weightsets and best/stop epochs reproduce exactly; annualquarterly20forecasts exact, monthly10requireoriginal512batchforbitwisereplay;128batchdifferences disclosed.',
                'Annual/quarterlyfuture split contrast notestimable becausezero crossingorigins; monthly21crossingpaths correspondtooneevent.',
                'Allnominalblockintervalsaredescriptiveexposeddevelopment,withoutmultiplicitycorrection.',
            ],
            'zero_alpha_history_configurations':alpha_zero,'restoration_batch_claim':batch['claim'],
            'manuscript_review_status':'pending','research_goal_achieved':False,'independent_confirmation':False,'Q3_values_read':False,
            'sources':{str(p.relative_to(PHASE)):sha(p) for p in [PHASE/'PLAN_ZH.md',PHASE/'longscale_replication/analysis/RESULTS.json',PHASE/'mean_controls/audit/RESULTS.json',PHASE/'history_sensitivity/analysis/RESULTS.json',BASE/'REVIEW.json',BASE/'HISTORY_REVIEW.json']},
            'at_utc':now()}
    atomic(BASE/'CLAIMS_REVIEW.json',report)
    def f(x):return f'{x:+.4f}%'
    lines=['# v3a 研究主张与要求审查','',
           '本审查根据已保存的结果和独立复放回执生成，不把工程验收、历史开发改善或论文交付改称研究 Goal 已完成。原主要收益终点和门槛保留。','',
           '| 任务 | 修复＋收缩，三种子 | 正向折次 | 默认＋收缩，三种子 | 均值路径＋收缩 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for r in tasks:lines.append(f"| {r['task']} | {f(r['repair_shrink_gain_pct'])} | {r['repair_positive_folds']}/5 | {f(r['default_shrink_gain_pct'])} | {f(r['mean_shrink_gain_pct'])} |")
    lines+=['','年度价格、收益修复在三个种子均为正；但相对原始／收缩无条件均值路径的配对区间，两种块长均包含零。因此年度结果没有证明读取序列所带来的可靠增量。','',
            '| 年度任务 | 原最早贪心四路径修复 | 同四路径均值＋收缩 | 十二相位修复范围 | 逐年去除修复范围 |',
            '| --- | ---: | ---: | ---: | ---: |']
    for r in stability:
        if r['task'].startswith('annual'):
            lines.append(f"| {r['task']} | {f(r['earliest_greedy_repair_gain_pct'])} | {f(r['earliest_greedy_mean_shrink_gain_pct'])} | {f(r['phase_gain_range_pct'][0])} 至 {f(r['phase_gain_range_pct'][1])} | {f(r['leave_year_out_gain_range_pct'][0])} 至 {f(r['leave_year_out_gain_range_pct'][1])} |")
    lines+=['','六个长尺度任务的全部相位和逐年去除均已保存并独立复算。年度每种目标仍只有 47 个重叠起点、4 条最早贪心不重叠路径；相位和去一年子集互相依赖。年度／季度没有未来拆股跨越路径，拆股对照不可估计。月度 21 条跨越路径只对应一个拆股事件。','',
            '| 原主要收益任务 | v3 固定候选 | 新均值＋收缩控制 | 原门槛 |','| --- | ---: | ---: | --- |']
    for r in primary_rows:lines.append(f"| {r['task']} | {f(r['original_candidate_gain_pct'])} | {f(r['mean_shrink_gain_pct'])} | 未通过，未改终点 |")
    lines+=['','历史输入反事实只替换固定模型的输入路径，不重训、不重新校准，也不以外层分数选部署方案。年度实际历史的三种子误差改善为价格 33.7309%、收益 38.1052%；平坦历史为 49.7560%、49.1760%；拟合段平均历史为 2.8137%、14.8262%。两类替代路径可能分布外，且 90 个配置中 27 个的原 alpha 为零；这些结果描述校准后预测函数的敏感性，不能证明网络不读取历史、揭示市场因果机制，或已经开发出更优可部署模型。','',
            '默认复现：30 组权重、最佳轮次和停止轮次精确一致。年度和季度 20 份预测逐位一致；月度 10 份在原 batch512 下逐位复放一致，在新引擎 batch128 下有小量浮点差异。对应断言修复已保留前后源码哈希、原因和已曝光任务，不改变评分、区间、子集或性能门槛。','',
            '| 要求 | 当前审查状态 |','| --- | --- |']
    for r in requirements:lines.append('| '+r['requirement']+' | '+r['status']+' |')
    lines+=['','BatchNorm 分支由专项负责人执行；本审查未据状态文件推断完成。v3a 新稿尚待独立数值与内容保留检查。未见时期确认和共享／独立期限市场实验仍未完成，研究 Goal 保持未达成。','']
    (BASE/'CLAIMS_REVIEW_ZH.md').write_text('\n'.join(lines),encoding='utf-8')
    print({'status':report['status'],'six_task_claims':len(tasks),'all_phase_and_leaveyearout_counts_checked':True},flush=True)


if __name__=='__main__':main()
