"""Complete the four-model mechanism report with all probe outcomes and structure limits."""
from common import *
import torch
from multiscale.models import make_model


def main():
    d=read(BASE/'diagnostics/DIAGNOSTICS.json'); decision=read(BASE/'protocol/STAGE_C_DECISION.json')
    data=read(BASE/'protocol/DATA.json'); cache=cache_arrays(); bars,_,folds,_=bars_and_folds()
    labels={'mlp':'MLP','gru':'GRU','lstm':'LSTM','xpatch':'xPatch'}
    structures={
        'mlp':'Flattened 480-point normalized history plus anchor → 64-unit ReLU → dropout 0.1 → 60 outputs',
        'gru':'One 32-unit GRU layer → final state plus anchor → 60-output linear head',
        'lstm':'One 32-unit LSTM layer → final state plus anchor → 60-output linear head',
        'xpatch':'Causal EMA (coefficient 0.1) → trend/residual dual-stream xPatch adapter plus learned anchor readout'}
    counts={m:sum(p.numel() for p in make_model(m,480,60).parameters()) for m in labels}
    lines=['# 四模型结构、机制探针与解释边界','',
        '范围：AAPL 分钟 H60 累计对数收益，原五折、376,890 个共同验证起点；全部是已曝光历史开发资料。',
        '本报告把直接算出的误差事实、针对性干预结果、仍未证实的机制解释分别写明。2026Q3 未读取或评分。','',
        '## 结构与训练结果','',
        '| 模型 | 具体实现 | 参数量 | 原五折 H60 改善（对零） |','| --- | --- | ---: | ---: |']
    for m in labels: lines.append(f'| {labels[m]} | {structures[m]} | {counts[m]:,} | {d["pooled"][m]["improvement_pct"]:+.5f}% |')
    lines+=['','输入窗口、价格锚点和每步标签都仅在内层拟合段估计均值/尺度；路径标准化训练损失与原尺度 H60 停止误差不是同一指标。',
        '原 20 次拟合都在轮数上限之前停止；这排除了“本任务因为撞上 20 轮上限才停下”的直接解释，未证明优化充分收敛。','',
        '| 模型 | 折 | 完成轮数 | 最佳轮数 |','| --- | ---: | ---: | ---: |']
    for f in folds:
        for m in labels:
            summary=read(data['folds'][str(f['fold'])]['models'][m]['summary'])
            ck=torch.load(data['folds'][str(f['fold'])]['models'][m]['checkpoint'],map_location='cpu',weights_only=True)
            lines.append(f'| {labels[m]} | {f["fold"]} | {len(summary["history"])} | {ck["epoch"]} |')
    lines+=['','## 幅度与对应','',
        '`MSE = E[y²] + E[p²] − 2E[py]` 精确成立。下面两项都除以零基准 MSE。它不证明经济机制，也不把事后校准变为可实现收益。','',
        '| 模型 | 预测能量 | 对应抵消项 | 与真实收益相关 |','| --- | ---: | ---: | ---: |']
    for m in labels:
        x=d['pooled'][m]; lines.append(f'| {labels[m]} | {x["energy_ratio"]:.8f} | {x["alignment_ratio"]:.8f} | {x["correlation"]:.6f} |')
    lines+=['',f'四预测等权平均的 H60 改善为 {d["equal_mean_improvement_pct"]:+.5f}%；架构不同没有自动形成有效组合。','',
        '## 全部单因素探针','',
        '| 探针 | 对应原模型误差改善：折 1 / 折 2 | 两折筛查支持 |','| --- | ---: | --- |']
    for x in decision['evidence']:
        if 'probe' in x:
            a,b=x['improvement_vs_original_pct']
            lines.append(f'| {x["probe"]} | {a:+.5f}% / {b:+.5f}% | {"是" if x["screen_supported"] else "否"} |')
    lines+=['','锚点常数化保持参数布局；xPatch 输入置零保持两路计算网络、偏置和锚点读出。输入减少会改变有效容量和训练轨迹，不能归因于单独的经济成分。',
        'GRU/LSTM 的均值读出在两折都更差，因此这项具体改变不支持“最后状态读出是主要可修复原因”。',
        '去锚点并无两折一致增益，因此不能把所有失败统一归于价格水平输入。',
        'xPatch 两种输入置零都改善误差，说明这组完整配方的新增变化并未自动提供有效预测；仍不能由此推出所有分解架构都无效。','',
        '## 互补筛查与后续结构','',
        '高低波动阈值由每折内层拟合段过去 60 根收益标准差中位数定义。趋势输入保留的探针在两折、两种状态均优于残差输入保留的探针，未发生要求的排名反转。',
        '本轮双路互补筛查失败。按新增拟合前已经登记的条件路径，正式研究局部卷积、常数缩放、状态缩放及强对照；不把双路融合写成实证成果。',
        '这一筛查只能决定本轮是否有足够证据进入原双路候选，不能排除其他状态、其他专家或其他市场的互补。','',
        '## 预测相关性','',
        '| 模型 | MLP | GRU | LSTM | xPatch |','| --- | ---: | ---: | ---: | ---: |']
    for i,m in enumerate(labels): lines.append('| '+labels[m]+' | '+' | '.join(f'{x:.5f}' for x in d['forecast_correlations'][i])+' |')
    lines+=['','低预测相关性只说明输出不同；是否有可用互补还需配对损失和固定控制。状态、偏差与相关性明细见 FOUR_MODEL_METRICS.csv 和 DIAGNOSTICS.json。','',
        '## 尚不能作出的结论','',
        '不能把误差恒等式称为因果证明；不能把两折探针当成跨市场定律；不能把更少参数或通过恢复测试称为预测优势。',
        '状态缩放与可训练头存在重参数化关系，不提供硬性幅度上限或已校准可靠性。最终性能和模块结论由五折三种子、零基准、同信息 Ridge 与内层标定收缩共同决定。']
    (BASE/'diagnostics/MECHANISM_REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    atomic(BASE/'diagnostics/STRUCTURE_COUNTS.json',{'parameters':counts,'structures':structures,
        'probe_results_bound_to':sha(BASE/'protocol/STAGE_C_DECISION.json'),'Q3_values_read':False})
    print(json.dumps({'four_model_report':'completed','parameters':counts}))


if __name__=='__main__':main()
