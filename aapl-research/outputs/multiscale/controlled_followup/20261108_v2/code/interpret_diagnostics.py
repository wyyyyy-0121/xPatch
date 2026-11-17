"""Readable interpretation of completed original-model diagnostics, not new fits."""
from common import *

def main():
    data=read(BASE/'diagnostics/DIAGNOSTICS_20_TASKS.json')
    table=[];mechanism=[];training=[]
    for profile in PROFILES:
        for target in ('future_close','future_log_return'):
            rr=[r for r in data['rows'] if r['profile']==profile and r['target']==target]
            training.append({'profile':profile,'target':target,'fits':len(rr),
                'at_epoch_cap':sum(r['epoch_cap_reached'] for r in rr),
                'median_epochs':float(np.median([r['epochs'] for r in rr]))})
    for task in read(BASE/'protocol/DATA.json')['task_files']:
        for model in ('mlp','gru','lstm','xpatch'):
            rr=[r for r in data['rows'] if r['task']==task and r['model']==model]
            N=sum(r['origins'] for r in rr)
            baseline=sum(r['baseline_mse']*r['origins'] for r in rr)/N
            mse=sum(r['endpoint_mse']*r['origins'] for r in rr)/N
            mean_error=sum(r['mean_error']*r['origins'] for r in rr)/N
            energy=sum(r['prediction_energy_over_baseline']*r['baseline_mse']*r['origins'] for r in rr)/N/baseline
            alignment=sum(r['alignment_offset_over_baseline']*r['baseline_mse']*r['origins'] for r in rr)/N/baseline
            assert np.isclose(mse/baseline,1+energy-alignment,rtol=1e-9)
            record={'task':task,'model':model,'origins':N,'endpoint_mse':mse,
                'improvement_pct':100*(1-mse/baseline),'prediction_energy_over_baseline':energy,
                'alignment_offset_over_baseline':alignment,'mean_error':mean_error,
                'pooled_bias_squared_fraction_of_mse':mean_error**2/mse,
                'epoch_cap_fits':sum(r['epoch_cap_reached'] for r in rr)}
            table.append(record)
            if task in [key(*t) for t in MECHANISM]:mechanism.append(record)
    text=['# 二十任务诊断的实际发现\n',
        '本报告只解释已完成的原种子 2026 比较，使用五折的原始预测与训练历史。新小模型正在按固定矩阵运行，这些诊断不用于修改矩阵或筛选新模型。\n',
        '## 1. 训练轮数的结论必须限定任务\n',
        f"四百个原神经拟合中，{sum(r['epoch_cap_fits'] for r in table)} 个达到二十轮上限。分钟 H60 收益的二十个拟合均未达到上限；这一结论不能扩展到其他尺度。小时价格有三十六个拟合达到上限，年度、季度也有上限拟合。因此，‘原模型都不是因为预算而停’不是全研究事实。是否增加轮次会改善外层误差仍未检验，本轮保留原停止规则。\n",
        '| 尺度 / 目标 | 原拟合 | 达二十轮上限 | 完成轮数中位数 |\n| --- | ---: | ---: | ---: |']
    for r in training:text.append(f"| {r['profile']} / {r['target']} | {r['fits']} | {r['at_epoch_cap']} | {r['median_epochs']:.1f} |")
    text+=['\n## 2. 两个机制任务的误差分解\n',
        'E[(y-p)²]/E[y²] = 1 + 预测能量比 − 对应抵消比。价格先减去起点价格，收益直接使用累计对数收益。该恒等式分解误差构成，不解释结构或经济因果。\n',
        '| 任务 / 模型 | 相对零预测改善 | 预测能量比 | 对应抵消比 |\n| --- | ---: | ---: | ---: |']
    for r in mechanism:text.append(f"| {r['task']} / {r['model']} | {r['improvement_pct']:+.4f}% | {r['prediction_energy_over_baseline']:.6f} | {r['alignment_offset_over_baseline']:.6f} |")
    text+=['\n两个任务的四种原配方均未超过零预测。分钟 H60 的四种输出都存在能量代价大于对应抵消的情况。小时 H8 中，GRU 的对应抵消项甚至为负；这是该组预测与目标乘积均值的描述，不能推广为 GRU 总是反向预测。xPatch 在两任务中有较高能量比，但单凭它不能定位到某个分支或保证缩放有效。\n',
        '## 3. 内部检查能支持到哪一步\n',
        '固定检查点在第一折最多一百二十八个均匀起点上接受历史片段、价格锚点及部分 xPatch 输入扰动。输出变化衡量配方对这些输入的敏感性。置零可能产生训练分布外输入，均值读出没有在本检查中重训，不能用这张敏感性表替代干预重训或作因果归因。全部数值保存在 INTERNAL_SENSITIVITY.csv。\n',
        '上一轮实际重训的均值读出与锚点探针另行保留；本轮新增的原结构损失探针只检验训练目标变化。新小模型的四组配对才能分开检验感受野、损失及交互。即使完整候选改善，也不能将所有组件都解释成有效。\n',
        '## 4. 可以直接写入论文与仍待验证的内容\n',
        '可以写入：二十任务的原配方误差、偏差、能量与对应分解、训练停止事实和有限输入敏感性。可以据此说明具体配方与评价的差异。\n',
        '仍待新结果验证：扩大局部感受野能否改善；给终点更高权重能否改善；两者是否有超过相加的增量；固定专家状态权重是否有超过零与常数融合的优势。任何单一任务的增量均不能自动外推到其他尺度。\n',
        '年度样本少、标签重叠及月度拆股影响保留原分析。不同目标的平方损失量纲不混合；较多起点不能被当作同样多的独立市场事件。\n']
    (BASE/'diagnostics/INTERPRETATION_ZH.md').write_text('\n'.join(text),encoding='utf-8')
    atomic(BASE/'diagnostics/POOLED_DIAGNOSTIC_FACTS.json',{'model_task_rows':table,'training_stop_summary':training,
        'Q3_values_read':False,'scope':'original seed-2026 exposed historical comparison only'})
    print(json.dumps({'original_task_model_pairs':len(table),'epoch_cap_fits':sum(r['epoch_cap_fits'] for r in table)}))

if __name__=='__main__':main()
