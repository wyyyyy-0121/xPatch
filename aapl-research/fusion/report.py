"""Independent persisted-prediction checks and a concrete delivery report."""
import math
import numpy as np

from aapl.common import ROOT,digest,read_json,write_json,utc_now
from .config import OUT,EXPERTS


def report(protocol,selection,result):
    from .pipeline import load_oof,assert_protocol
    assert_protocol();data=load_oof([protocol['confirmation_quarter']]);checks={};values=[]
    for seed in (2026,2027,2028):
        path=OUT/'confirmation'/f'seed{seed}'/'predictions.npz'
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved['origins'],data['origins'])
            values.append(math.fsum((float(p)-float(y))**2 for p,y in zip(saved['prediction'],data['y']))/len(data['y']))
    checks['primary_mse_scalar_recomputed']=bool(np.isclose(np.mean(values),result['mse'],rtol=1e-11,atol=1e-18))
    checks['zero_mse_scalar_recomputed']=bool(np.isclose(math.fsum(float(y)**2 for y in data['y'])/len(data['y']),result['zero_mse'],rtol=1e-11))
    checks['confirmation_decision_recomputed']=result['qualified']==all(result['qualification_checks'].values())
    checks['serving_verified']=read_json(OUT/'serving_verification.json')['passed']
    for quarter in protocol['oof_quarters']+[protocol['confirmation_quarter']]:
        m=read_json(OUT/'oof'/quarter/'manifest.json')
        checks[quarter+'_strict_past_expert_fit']=max(m['fit_dates'])<min(m['purge_dates'])<min(m['evaluation_dates'])
        checks[quarter+'_foundation_resource_limits']=m['resources']['passed']
    for row in selection['ranking']:
        for block in row['blocks']:
            checks[row['config']['candidate_id']+block['quarter']+'_past_only_meta_fit']=block['train_max_date']<block['validation_min_date']
    checks['no_holdout_access']=not result['holdout_accessed'] and not (OUT/'holdout').exists()
    audit={'passed':all(checks.values()),'checks':checks,'completed_at_utc':utc_now(),
           'protocol_sha256':digest(OUT/'protocol.json'),'confirmation_sha256':digest(OUT/'confirmation.json')}
    write_json(OUT/'independent_verification.json',audit)
    if not audit['passed']:raise RuntimeError('Fusion final artifact verification failed')
    serving=read_json(OUT/'serving_verification.json')
    lines=['# Chronos-2 驱动的 AAPL 融合模型：第一版结果','',
           f"研究身份 `{protocol['research_id']}`，生成于 {utc_now()}。",'',
           '实际结构为 Chronos-2 编码器量价表示、五个已有模型专家、因果行情统计和直接优化 12 分钟累计收益 MSE 的融合层。',
           f"最终融合配置：`{selection['main']['config']}`。",'',
           f"开发期合并验证相对零收益改善：{1-selection['main']['mse']/selection['validation_zero_mse']:.3%}。",
           f"2026Q2 确认相对零收益改善：{result['relative_gain_vs_zero']:.3%}；相对最强对照 `{result['strongest_control']}` 改善：{result['gain_vs_strongest_control']:.3%}。",'',
           '| 方法 | 2026Q2 MSE | 相对零收益改善 |','|---|---:|---:|',
           f"| 基础模型融合（主指标，逐种子损失平均） | {result['mse']:.9g} | {result['relative_gain_vs_zero']:.3%} |"]
    for name,mse in result['controls_mse'].items():
        lines.append(f"| {name} | {mse:.9g} | {1-mse/result['zero_mse']:.3%} |")
    lines+=['',f"预设开发验收结果：{'通过，仍需独立最终检验' if result['qualified'] else '未通过，不声称优于现有方案'}。",'']
    for k,v in result['qualification_checks'].items():lines.append(f"- {k}：{'通过' if v else '未通过'}")
    lines+=['',f"本地完整分钟柱输入到预测的热启动延迟：中位数 {serving['latency_p50_seconds']:.3f} 秒，P95 {serving['latency_p95_seconds']:.3f} 秒。",
            '完整推理已与保存的确认预测逐点核对。输出包括累计收益、派生价格、逐专家预测和训练截止日；不输出未经校准的置信区间。', '',
            '开发加载器只访问截至 2026-06-30 的数据，2026Q3 保持封存。旧轮次的实施协议、代码和结果保留。', '',
            '运行：`AAPL_CPU_THREADS=4 AAPL_CPU_IDS=0-7 bash scripts/python.sh -m fusion.inference --csv /absolute/minute_bars.csv --as-of 2026-06-30T19:00:00Z`。',
            'CSV 使用既有 Alpaca 格式，需包含目标时点前至少 241 根同日连续且已完成的分钟柱。当前入口为本地推理与历史回放，没有接入实时行情或订单执行。', '',
            '研发对照中的旧模型超参选自此前研究；确认在新选择固定后进行。单股票、下午时段、预训练边界和行情版本时点限制仍然适用。', '',
            '证据：`outputs/fusion/protocol.json`、`selection.json`、`confirmation.json`、`independent_verification.json`、`serving_verification.json`。','']
    (ROOT/'FUSION_RESULTS.md').write_text('\n'.join(lines))
