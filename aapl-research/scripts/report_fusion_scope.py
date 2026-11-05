"""Report the amended study scope without modifying the frozen research engine."""
import argparse
from pathlib import Path

from fusion.next.common import OUT,digest,identity,read_json,source_identity,utc_now,write_json
from fusion.next.restore import allowed


def inspect(evidence_root=OUT):
    root=Path(evidence_root)
    path=root/'performance/scope_100h.json';scope=read_json(path)
    if scope['identity']!=identity({k:v for k,v in scope.items() if k!='identity'}):
        raise ValueError('Scope identity changed')
    parent=root/'search_manifest.json'
    if digest(parent)!=scope['parent_search_sha256'] or read_json(parent)['identity']!=scope['parent_search_identity']:
        raise ValueError('Scope belongs to another parent search')
    if scope['source_identity']!=source_identity():raise ValueError('Frozen model code changed')
    runner=root/'performance/scope_100h_runner.txt'
    if digest(runner)!=scope['execution_script_sha256']:raise ValueError('External runner differs from frozen scope')
    time_limit=root/'performance/scope_100h_time_limit.json'
    if digest(time_limit)!=scope['hard_limit_receipt_sha256']:raise ValueError('Time limit receipt changed')
    provenance=[path,parent,runner,time_limit]
    for artifact in provenance:
        if not allowed('outputs/fusion_next/'+str(artifact.relative_to(root))):
            raise ValueError('Scope provenance cannot pass the formal return allowlist')
    stages={}
    for name in ('F3','F4','F5'):
        artifact=root/(name+'_complete.json')
        stages[name]={'completed':False}
        if not artifact.exists():continue
        row=read_json(artifact)
        if (row.get('scope_identity')!=scope['identity'] or
            row.get('scope_amendment_sha256')!=digest(path) or
            row.get('source_identity')!=scope['source_identity'] or
            row.get('original_complete_scope_claimed') is not False):
            raise ValueError('Stage completion does not bind the amended scope: '+name)
        stages[name]={'completed':True,'receipt_sha256':digest(artifact)};provenance.append(artifact)
    return {'generated_at_utc':utc_now(),'scope_identity':scope['identity'],
        'source_identity':scope['source_identity'],'authorization':scope['authorization'],
        'parent_search_identity':scope['parent_search_identity'],
        'counts':scope['counts'],'simple_count':scope['simple_count'],
        'validation':scope['validation'],'seeds':scope['seeds'],
        'epochs':scope['training_budget_policy']['total_coverage_equivalent_epochs'],
        'mechanisms_retained':scope['mechanisms_retained'],'mechanisms_deferred':scope['mechanisms_deferred'],
        'projection':scope['projection'],'stages':stages,
        'original_complete_scope_claimed':False,'Q3_values_read_by_this_report':False,
        'claim_limits':['Omitted mechanisms do not provide evidence of foundation, market, feature or transfer contributions',
            'Search and seed reviews use the reduced scope, not all original configurations',
            'Runtime projection does not prove completion or predictive improvement'],
        'provenance_sha256':{str(p.relative_to(root)):digest(p) for p in provenance},
        'scope_provenance_passes_existing_return_allowlist':True}


def run(evidence_root=OUT):
    value=inspect(evidence_root)
    write_json(OUT/'bounded_scope_report.json',value)
    lines=['# 100 小时精简研究范围记录','',
        '本报告与原完整研究报告一并阅读。它只核验范围身份和阶段收据，不加载行情、模型、标签或预测数值。',
        '',f'范围身份：`{value["scope_identity"]}`。',
        '',f'每次完整拟合保留 {value["epochs"]} 个等价轮；验证季度：'+', '.join(value['validation'])+'；种子：'+', '.join(map(str,value['seeds']))+'。',
        '', '| 阶段 | 精简范围完成收据 |','| --- | --- |']
    lines.extend(f'| {name} | {"已核验" if row["completed"] else "尚无完成证据"} |' for name,row in value['stages'].items())
    lines+=['','保留机制：'+', '.join(value['mechanisms_retained'])+'。',
        '', '未执行机制：'+', '.join(value['mechanisms_deferred'])+'。',
        '', '不得把这些阶段的完成记录解释成原完整候选或七组机制全部完成。缺少的机制不能支持基础表示、市场变量、字段或迁移贡献的独立归因。',
        '', '预测改善仍须由固定开发门槛、一次 Q2 确认及条件式最终检验证明；运行包验收另行记录。']
    (OUT/'bounded_scope_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--evidence-root',type=Path,default=OUT)
    value=run(parser.parse_args().evidence_root)
    print({'scope_identity':value['scope_identity'],'stages':value['stages']})
