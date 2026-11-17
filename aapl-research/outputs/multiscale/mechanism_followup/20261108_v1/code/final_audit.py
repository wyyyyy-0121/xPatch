from common import *
from datetime import datetime,timezone
import subprocess


def main():
    jobs=read(BASE/'protocol/JOBS.json')['jobs']; completion=read(BASE/'audit/TRAINING_COMPLETE.json')
    decision=read(BASE/'protocol/STAGE_C_DECISION.json'); result=read(BASE/'analysis/RESULTS.json')
    assert code_hashes()==read(BASE/'protocol/CODE_FREEZE.json')['hashes']
    auth=read(BASE/'protocol/AUTHORIZATION.json')
    reference=BASE/'code/reference_source'
    assert all(sha(reference/name)==value for name,value in auth['source_files'].items())
    cache=read(BASE/'protocol/CACHE.json')
    assert all(sha(BASE/'cache'/name)==value for name,value in cache['files'].items())
    data=read(BASE/'protocol/DATA.json')
    assert sha(SOURCE/'admission.json')==data['admission_sha256']
    assert all(sha(SOURCE/name)==value for name,value in data['source_files'].items())
    expected=12+len(decision['formal_kinds'])*15; assert len(jobs)==expected
    assert len({j['id'] for j in jobs})==len(jobs)
    actual={p.parent.name for p in (BASE/'runs').glob('*/DONE.json')}
    assert actual=={j['id'] for j in jobs}
    new=0;wall=0.;cpu=0.;last_invocation=0.
    for job in jobs:
        folder=BASE/'runs'/job['id']; d=read(folder/'DONE.json')
        assert d['job']==job and d['status']=='completed'
        assert sha(folder/'best.pt')==d['checkpoint_sha256']
        assert sha(folder/'prediction.npz')==d['prediction_sha256']
        assert d['Q3_values_read'] is False and d['independent_confirmation'] is False
        if not d.get('reused_original',False):
            new+=1; assert d['code_hashes']==code_hashes()
            assert len(d['affinity'])==2
            wall+=d['training_wall_seconds'];cpu+=d['training_cpu_seconds']
            last_invocation+=d['total_last_invocation_seconds']
    assert new==completion['new_fits'] and new<=auth['maximum_new_market_fits']
    assert not list((BASE/'runs').glob('*/ERROR.json'))
    budget=read(BASE/'protocol/BUDGET.json')
    completed_epoch=datetime.fromisoformat(completion['at_utc']).timestamp()
    assert completed_epoch<=budget['deadline_epoch']
    live=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:argv=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        owned=[str(BASE/'code'/name) for name in ('controller.py','worker.py','postprocess.py','verify_recurrent_and_probes.py')]
        if any(x in argv for x in owned):live.append(int(proc.name))
    assert not live,live
    replay=read(BASE/'audit/PREDICTOR_REPLAY.json'); assert replay['status']=='passed'
    recovery=read(BASE/'audit/ENGINE_RECOVERY.json');assert recovery['status']=='passed'
    reference_replay=read(BASE/'audit/REFERENCE_REPLAY.json');assert reference_replay['status']=='passed' and reference_replay['predictors']==27
    release=read(BASE/'audit/RELEASE.json'); assert release['status']=='passed'
    assert sha(Path(release['archive']))==release['archive_sha256']
    visual=read(BASE/'audit/VISUAL_REVIEW.json');assert visual['status']=='passed'
    for item in visual['pdfs']:
        assert sha(BASE/item['pdf'])==item['sha256']
        assert len(item['reviewed_pages'])==item['pages']
    requirements={
        'four_model_mechanism_diagnosis':(BASE/'diagnostics/MECHANISM_REPORT_ZH.md').exists(),
        'twelve_probe_outcomes':len([x for x in decision['evidence'] if 'probe'in x])==6,
        'evidence_conditioned_small_model':replay['predictors']==(len(decision['formal_kinds'])-1)*15,
        'complete_paired_and_ablation_evidence':len(result['fold_means'])==(len(decision['formal_kinds'])+3)*5,
        'all_registered_performance_requirements_reported':len(result['performance_checks'])==6,
        'english_main_and_supplement':all((BASE/'manuscript'/name).exists() for name in ('MAIN_PAPER_EN.pdf','MAIN_PAPER_EN.docx','SUPPLEMENT_EN.pdf','SUPPLEMENT_EN.docx')),
        'full_prediction_restore':replay['status']=='passed',
        'reference_and_probe_full_prediction_restore':reference_replay['status']=='passed',
        'process_exit_and_resume_acceptance':recovery['status']=='passed',
        'portable_archive_extracted_and_executed':release['isolated_standalone_inference_actually_executed'],
        'all_pages_reviewed':visual['status']=='passed',
        'owned_training_and_postprocess_stopped':not live,
        'source_and_budget_invariants_verified':True}
    assert all(requirements.values()),requirements
    p=next(x for x in result['pooled'] if x['kind']==result['primary'])
    audit={'status':'complete_deliverables_verified','at_utc':datetime.now(timezone.utc).isoformat(),
        'requirements':requirements,'declared_jobs':len(jobs),'new_market_fits':new,'reused_fits':len(jobs)-new,
        'performance_goal_passed':result['performance_goal_passed'],'primary':result['primary'],
        'primary_improvement_pct':p['h60_improvement_pct'],'independent_confirmation':False,'Q3_values_read':False,
        'training_wall_window_elapsed_seconds':completed_epoch-budget['start_epoch'],
        'summed_training_invocation_wall_seconds':wall,'summed_training_process_cpu_seconds':cpu,
        'summed_last_invocation_wall_seconds':last_invocation,
        'cost_scope':'training process preparation/fit/inner validation/checkpoint; final prediction and analysis not comprehensively included',
        'four_training_sources_match_freeze':True,'reference_source_files_verified':len(auth['source_files']),
        'input_source_files_verified':len(data['source_files']), 'maximum_training_workers':8,
        'logical_cpus_per_worker':2,'gpu_training':False,'archive_sha256':release['archive_sha256']}
    atomic(BASE/'audit/FINAL_COMPLETION_AUDIT.json',audit)
    lines=['# 本轮研究交付与结论','',
        f'已完成四模型诊断、12 个探针、75 格正式比较（5 格复用、70 次新增）和强对照，共 {new} 次新市场拟合。',
        f'主候选为局部卷积加状态缩放（1,310 个可训练参数）。H60 相对零收益率的三种子平均损失改善：{p["h60_improvement_pct"]:+.6f}%。',
        f'预登记性能目标：{"达到，仅限历史开发" if result["performance_goal_passed"] else "未达到"}。已形成完整结果，论文如实报告失败项。','',
        '| 对照/变体 | H60 MSE 改善（对零） | 正改善折数 |','| --- | ---: | ---: |']
    for x in result['pooled']:lines.append(f'| {x["kind"]} | {x["h60_improvement_pct"]:+.6f}% | {x["positive_folds"]}/5 |')
    lines+=['','## 论文推进方向','',
        '原五尺度比较作为前期证据；新主稿围绕“结构假设 → 重训探针 → 条件选型 → 小模型和强对照 → 配对与消融”组织。',
        '本轮未支持预定趋势/残差互补，因此没有把双路融合写成成果。主模型是否优于零和简单收缩，按全部固定标准判断；失败时写为具体配方和假设的否定证据。',
        '2026Q3 保持封存，全部评分仍是已曝光历史开发资料。模型可复现与预测有效是不同结论。','',
        '## 文件','',
        '- 英文双列主稿：manuscript/MAIN_PAPER_EN.pdf、MAIN_PAPER_EN.docx。',
        '- 完整英文补充：manuscript/SUPPLEMENT_EN.pdf、SUPPLEMENT_EN.docx。',
        '- 中文机制诊断：diagnostics/MECHANISM_REPORT_ZH.md。',
        '- 全部结果与配对区间：analysis/RESULTS.json、CSV 和逐折配对损失。',
        '- 小模型、预处理及推理示例：delivery/。',
        '- 便携审阅与推理包：release/AAPL_MECHANISM_STUDY.zip。',
        '- 逐项验收：audit/FINAL_COMPLETION_AUDIT.json。','',
        '作者 Wang Yang；学校与城市邮编已保留，邮箱由作者填写。未经外部提交。',
        '便携包不包含全部原始训练数据、完整预测和优化器检查点；这些留在工作区并由哈希绑定。归档已解压并实际执行独立推理示例。']
    (BASE/'RESEARCH_SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'completed':True,'performance_goal_passed':result['performance_goal_passed'],
        'primary_improvement_pct':p['h60_improvement_pct']}))


if __name__=='__main__':main()
