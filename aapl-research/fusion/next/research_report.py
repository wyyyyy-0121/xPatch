"""Evidence-based F8 research, cost and cleanup reports; never load market data/model weights."""
import argparse
import json
from pathlib import Path
import stat

import numpy as np
import pandas as pd

from .common import ROOT,OUT,PLAN,digest,read_json,write_json,utc_now,source_identity
from .accounting import totals
from .delivery import score_journal
from .evaluation import compare


def optional(name):
    path=OUT/name
    return read_json(path) if path.exists() else None


def artifact_inventory():
    count=0;logical=0;allocated=0;links=0;inodes=set();by_folder={}
    for path in OUT.rglob('*'):
        try:info=path.lstat()
        except FileNotFoundError:continue
        if stat.S_ISLNK(info.st_mode):links+=1;continue
        if not stat.S_ISREG(info.st_mode):continue
        count+=1;logical+=info.st_size;key=(info.st_dev,info.st_ino)
        if key not in inodes:allocated+=info.st_blocks*512;inodes.add(key)
        folder=path.relative_to(OUT).parts[0];by_folder[folder]=by_folder.get(folder,0)+info.st_size
    deleted=[];failures=[];planned=[]
    for path in (OUT/'training').rglob('*.pruning.json'):
        for name,row in read_json(path)['files'].items():
            item={'file':str((path.parent/name).relative_to(ROOT)),**row}
            if row['status'] in ('deleted','deleted_after_interruption'):deleted.append(item)
            elif row['status']=='failed':failures.append(item)
            else:planned.append(item)
    return {'regular_files':count,'symlinks':links,'logical_regular_file_bytes':logical,
            'allocated_bytes_unique_inode':allocated,'logical_bytes_by_output_folder':by_folder,
            'cleanup':{'deleted_files':len(deleted),'deleted_logical_bytes':sum(r['bytes'] for r in deleted),
                       'failed_files':failures,'pending_files':planned,'deletion_receipts':deleted},
            'retained_for_reproduction':['raw acquisition receipts/pages','prepared development shards','source/plan/model locks',
                'trial recipes and raw predictions','OOF predictions','current and previous training checkpoints',
                'fixed fitted system and controls','statistics and all failures'],
            'inventory_reads_file_metadata_only':True}


def cost_report(acceptance=None):
    jobs=[read_json(p) for p in (OUT/'usage_events').glob('*.json')];usage=totals()
    setup=optional('environment_setup.json');profile=optional('execution_profile.json') or {}
    setups=[read_json(p) for p in (OUT/'environment_setup_events').glob('*.json')]
    quote=profile.get('price_cny_per_hour_quoted');gpu_hours=usage['gpu_card_hours']
    return {'usage':usage,'GPU_budget_card_hours':24.,
            'quoted_price_cny_per_hour':quote,'compute_only_cost_at_quote_cny':None if quote is None else quote*gpu_hours,
            'actual_provider_billed_cost_cny':None,'billing_note':'Only measured accounted GPU computation is estimated; provider idle/install/network billing is not inferred',
            'environment_setup':setup,'environment_setup_measured':setup is not None,
            'environment_setup_attempts':setups,'all_recorded_setup_wall_seconds':sum(r['wall_seconds'] for r in setups),
            'local_replay_wall_seconds':None if acceptance is None else acceptance.get('total_replay_wall_seconds'),
            'local_replay_process_CPU_seconds':None if acceptance is None else acceptance.get('total_replay_process_cpu_seconds'),
            'job_wall_seconds':sum(j['wall_seconds'] for j in jobs),
            'job_wall_sum_is_not_elapsed_makespan':True,'jobs':jobs,
            'sampling_note':'Per-job RSS/thread samples and F2/F8 resource peaks are retained; CPU preparation/network/user wait outside accounted jobs is not invented',
            'storage_and_cleanup':artifact_inventory()}


def paired_mechanisms(f5):
    if f5 is None:return None
    full={r['config']['paired_base_id']:r for r in f5['mechanisms']['full']}
    result={}
    for variant,rows in f5['mechanisms'].items():
        if variant=='full':continue
        result[variant]=[]
        for row in rows:
            base=full[row['config']['paired_base_id']]
            result[variant].append({'paired_base_id':row['config']['paired_base_id'],
                'full_config':base['config'],'variant_config':row['config'],
                'full_three_seed_mean_MSE':base['system']['mse'],'variant_three_seed_mean_MSE':row['system']['mse'],
                'variant_minus_full_MSE':row['system']['mse']-base['system']['mse'],
                'full_seed_scores':base['individual_seeds'],'variant_seed_scores':row['individual_seeds'],
                'full_quarters':base['quarter_scores'],'variant_quarters':row['quarter_scores'],
                'interpretation':'paired development evidence; no economic causality or held-out significance implied'})
    return result


def recompute_final():
    """Recompute the locked statistics from saved predictions, without new model/data access."""
    path=OUT/'final_test/predictions.npz';original=read_json(OUT/'final_test/report.json')
    if digest(path)!=original['raw_prediction_sha256']:raise ValueError('Final raw prediction identity changed')
    with np.load(path,allow_pickle=False) as saved:
        controls={'zero':np.zeros(len(saved['target'])),'single':saved['single'],'simple':saved['simple']}
        result=compare(saved['target'],saved['prediction'],controls,saved['dates'],pd.to_datetime(saved['decision_time_ns'],utc=True))
    differences=[key for key,value in result.items() if original.get(key)!=value]
    if differences:raise ValueError('Independent final-statistics recomputation differs: '+','.join(differences))
    value={'raw_prediction_sha256':digest(path),'original_report_sha256':digest(OUT/'final_test/report.json'),
           'identical_prespecified_statistics':True,'no_new_model_inference':True,'recomputed_at_utc':utc_now(),'result':result}
    write_json(OUT/'final_statistics_recomputed.json',value);return value


def build():
    inventory=optional('inventory.json');data=optional('data_audit.json');admission=optional('data_admission.json')
    performance=optional('performance_admission.json');f5=optional('F5_complete.json');frozen=optional('system_freeze.json')
    confirmation=optional('confirmation.json');final=optional('final_test/report.json');acceptance=None;deployment=None
    if frozen:
        root=OUT/frozen['final_fit_policy']['artifact']
        if (root/'deployment.json').exists():
            deployment=read_json(root/'deployment.json');acceptance=optional('deployment_acceptance/'+deployment['identity']+'/report.json')
    stages={'F0':bool(inventory and inventory.get('F0_passed')),'F1':bool(data and data.get('passed')),
            'F2':bool(performance and performance.get('passed')),
            'F3':(OUT/'F3_complete.json').exists(),'F4':(OUT/'F4_complete.json').exists(),
            'F5':f5 is not None,'F6':confirmation is not None,'F7_evaluated':final is not None,
            'F8_engineering_passed':bool(acceptance and acceptance.get('engineering_acceptance_passed'))}
    stopped=bool(confirmation and not confirmation.get('proceed_to_final_test',False))
    research_status=('final_gate_passed' if final and final['research_success'] else
                     ('final_gate_failed' if final else ('stopped_before_final_by_prespecified_rule' if stopped else 'in_progress')))
    violations=[]
    if stopped and (OUT/'final_test/entry.json').exists():violations.append('Final entry exists despite a failed pre-final gate')
    if final and (not confirmation or not confirmation.get('proceed_to_final_test')):violations.append('Final result lacks the required Q2 entry condition')
    sources={}
    for name in ('inventory.json','data_audit.json','data_admission.json','search_manifest.json','performance_admission.json',
                 'F3_complete.json','F4_complete.json','F5_complete.json','system_freeze.json','confirmation.json','final_test/report.json'):
        path=OUT/name
        if path.exists():sources[name]=digest(path)
    value={'generated_at_utc':utc_now(),'plan_sha256':digest(PLAN),'source_identity':source_identity(),
           'stages':stages,'research_status':research_status,'violations':violations,'source_artifacts_sha256':sources,
           'data':admission,'data_audit':None if data is None else {'passed':data['passed'],'pending':data['pending'],'checks':data['checks']},
           'selected_development_system':None if frozen is None else frozen['selected_system'],
           'fixed_controls':None if frozen is None else {'single':frozen['fixed_strongest_single'],'simple':frozen['fixed_strongest_simple']},
           'paired_mechanisms':paired_mechanisms(f5),'Q2_confirmation':confirmation,'final_test':final,
           'final_test_skipped_by_stop_rule':stopped and final is None,
           'Q3_preserved_on_negative_path':bool(stopped and not (OUT/'final_test/entry.json').exists()),
           'deployment_identity':None if deployment is None else deployment['identity'],'engineering_acceptance':acceptance,
           'cost_and_cleanup':cost_report(acceptance),
           'paper_claim_boundary':{'journal_acceptance_demonstrated':False,'novelty_literature_review_completed':False,
               'prediction_research_gate_passed':bool(final and final['research_success']),
               'trading_profitability_tested':False,'scope':'fixed convenience sample; next12-minute AAPL log-return MSE; known Q2 process confirmation',
               'mechanism_claims_require_paired_retraining':'attention/gate visualizations are descriptive'},
           'completion_not_inferred_from_file_existence':True}
    write_json(OUT/'research_report.json',value)
    lines=['# 全天基础模型融合研究记录','',f'生成时间：{value["generated_at_utc"]}。研究状态：`{research_status}`。',
           '', '本报告汇总已有产物；代码或单元测试存在不代表正式训练、预测改善或部署验收已经完成。','',
           '| 项目 | 当前证据 |','|---|---|']
    lines.extend(f'| {name} | {"已取得阶段通过证据" if passed else "未取得通过证据"} |' for name,passed in stages.items())
    lines+=['','主任务为 AAPL 未来 12 分钟累计对数收益，主指标为原始收益 MSE。Q2 仅作已知流程确认；Q3 只有通过开发与确认门槛才读取。',
            '',f'累计核算 GPU 计算：{value["cost_and_cleanup"]["usage"]["gpu_card_hours"]:.6f} 卡小时。实际供应商账单尚未取得，不能据此推断总租赁费用。',
            '',f'已记录删除 {value["cost_and_cleanup"]["storage_and_cleanup"]["cleanup"]["deleted_files"]} 个过期检查点文件；逐项字节、哈希与失败记录见 JSON。',
            '',f'[完整研究、成本与清理记录]({OUT / "research_report.json"})',
            '', '论文中的正向主张须有最终保留检验支持；投稿新颖性仍需相关工作核查，预测验收不代表可交易性。','']
    (OUT/'research_report.md').write_text('\n'.join(lines),encoding='utf-8')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recompute-final',action='store_true');args=parser.parse_args()
    if args.recompute_final:print({'identical_prespecified_statistics':recompute_final()['identical_prespecified_statistics']})
    else:
        report=build();print({'research_status':report['research_status'],'stages':report['stages'],'violations':report['violations']})
