"""Verify and summarize three post-hoc seeds on the same monthly folds."""
from datetime import datetime,timezone
import json

import numpy as np

from multiscale.bars import file_digest
from multiscale.spec import MODELS,OUT


def main():
    base=OUT/'private_split_neutral_monthly'
    sources={2026:base/'20261107T145708694676Z',2027:base/'20261107T150504678124Z',
             2028:base/'20261107T150527182499Z'}
    target=OUT/'private_split_neutral_seed_replication/20261107T150345851174Z'
    protocol=target/'PRIVATE_REPLICATION_PROTOCOL.json'
    prereg=json.loads(protocol.read_text());rows={};plans={};states={}
    if prereg['additional_seeds']!=[2027,2028] or prereg['baseline_seed']!=2026:
        raise ValueError('Unexpected seed replication plan')
    for seed,root in sources.items():
        plan=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
        state=json.loads((root/'PRIVATE_QUEUE_STATE.json').read_text())
        result_path=root/'PRIVATE_RESULTS.json';result=json.loads(result_path.read_text())
        check=json.loads((root/'PRIVATE_LABEL_AND_AGGREGATE_CHECK.json').read_text())
        if (plan['recipe']['seed']!=seed or state['status']!='completed_posthoc_development_only' or
                state['completed']!=5 or state['results_sha256']!=file_digest(result_path) or
                result['validation_origins']!=1970 or
                check['status']!='independent_label_and_saved_prediction_check'):
            raise ValueError(f'Incomplete or changed seed {seed}')
        if seed!=2026 and plan['seed_replication_protocol_sha256']!=file_digest(protocol):
            raise ValueError('A seed is not bound to the replication plan')
        plans[seed]=plan;states[seed]=state;rows[seed]=result
    for seed in (2027,2028):
        for key in ('admission_sha256','original_bar_sha256','adjusted_bar_sha256','target','actions',
                    'origins_sha256','profile','horizon','lookback','seasonal_period','jobs'):
            if plans[seed][key]!=plans[2026][key]:raise ValueError('Seed data or splits differ: '+key)
        expected=dict(plans[2026]['recipe']);expected['seed']=seed
        if plans[seed]['recipe']!=expected:raise ValueError('Seed recipe differs beyond its seed')
    for number in range(1,6):
        reference=None
        for seed,root in sources.items():
            key=f'fold_{number:02d}';folder=root/key;entry=states[seed]['jobs'][key]
            if (entry['status']!='completed' or entry['receipt_sha256']!=file_digest(folder/'fold_receipt.json') or
                    entry['verification_sha256']!=file_digest(root/(key+'_verification.json'))):
                raise ValueError('A fold verification changed')
            with np.load(folder/'targets.npz',allow_pickle=False) as data:
                actual=data['actual'].copy();origins=data['origins'].copy();decision=data['decision_ns'].copy()
            if reference is None:reference=(actual,origins,decision)
            elif any(not np.array_equal(a,b) for a,b in zip(reference,(actual,origins,decision),strict=True)):
                raise ValueError('Seeds do not share identical target arrays')
    metrics={name:{str(seed):rows[seed]['models'][name]['relative_improvement'] for seed in sources} for name in MODELS}
    gru=metrics['gru']
    if not (gru['2026']>0 and gru['2027']<0 and gru['2028']>0):
        raise ValueError('The stated sign pattern is unsupported')
    result={'status':'private_posthoc_three_seed_comparison','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'task':'monthly H21 split-neutral price log return','seeds':list(sources),'folds_per_seed':5,
            'validation_origins_per_seed':1970,'relative_mse_improvement_vs_zero_return':metrics,
            'sources':{str(seed):str(root) for seed,root in sources.items()},
            'source_result_sha256':{str(seed):file_digest(root/'PRIVATE_RESULTS.json') for seed,root in sources.items()},
            'same_targets_and_splits':True,'independent_test':False,'publication_rights':'not confirmed',
            'limits':['The two new seeds were selected after the initial result was observed.',
                      'All seeds reuse the same exposed periods; they do not supply independent market samples.',
                      'A single 20-epoch training budget and three seeds do not establish convergence.',
                      'Returns exclude cash dividends.']}
    out=target/'PRIVATE_SEED_COMPARISON.json'
    with out.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    names=[('mlp','MLP'),('gru','GRU'),('lstm','LSTM'),('xpatch','xPatch')]
    lines=['# 月度拆股中性收益：三个随机种子复跑','',
           f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}。复跑使用相同的五个开发折和相同的 1,970 个重叠验证起点；两个新种子是在看到首个结果后追加的。','',
           '表中为相对零收益基线的终点均方误差改善，正数表示误差较小，负数表示误差较大；不是投资收益率。','',
           '| 方法 | 种子 2026 | 种子 2027 | 种子 2028 |','| --- | ---: | ---: | ---: |']
    for name,label in names:
        lines.append('| '+label+' | '+' | '.join(f'{100*metrics[name][str(seed)]:.2f}%' for seed in sources)+' |')
    lines += ['',f"GRU 的三个结果分别为 {100*gru['2026']:.2f}%、{100*gru['2027']:.2f}%、{100*gru['2028']:.2f}%；其中一个种子明显差于基线，因此目前不能称为稳定优势。",
              '三个种子使用相同的历史时期，不能视为独立市场样本；该对照也是事后探索。此前首个种子的 21/42 交易日描述性区间均包含零。收益不含现金分红，数据发表权尚待确认。','',
              '已逐折核对三组目标和时间起点完全一致，全部模型预测及汇总误差均有复算回执。','',
              '[机器可读结果]('+str(out)+')  ',
              '[种子复跑计划]('+str(protocol)+')  ',
              '[原种子与区间报告]('+str(sources[2026]/'PRIVATE_REPORT_ZH.md')+')','']
    report=target/'PRIVATE_SEED_COMPARISON_ZH.md'
    with report.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':result['status'],'report':str(report),'result':str(out),
                      'sha256':file_digest(out)},ensure_ascii=False))


if __name__=='__main__':main()
