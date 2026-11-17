"""Summarize verified post-hoc monthly split-neutral development locally."""
from datetime import datetime,timezone
import json

from multiscale.bars import file_digest
from multiscale.spec import MODELS,OUT


def main():
    root=OUT/'private_split_neutral_monthly/20261107T145708694676Z'
    intervals=OUT/'private_split_neutral_uncertainty/20261107T145946279609Z/PRIVATE_DESCRIPTIVE_INTERVALS.json'
    state=json.loads((root/'PRIVATE_QUEUE_STATE.json').read_text())
    result_path=root/'PRIVATE_RESULTS.json';result=json.loads(result_path.read_text())
    verified=json.loads((root/'PRIVATE_LABEL_AND_AGGREGATE_CHECK.json').read_text())
    uncertainty=json.loads(intervals.read_text())
    if (state['status']!='completed_posthoc_development_only' or state['completed']!=5 or
            state['results_sha256']!=file_digest(result_path) or
            verified['status']!='independent_label_and_saved_prediction_check' or
            uncertainty['source_results_sha256']!=file_digest(result_path)):
        raise ValueError('Result or uncertainty verification changed')
    rows=[]
    for number in range(1,6):
        p=root/f'fold_{number:02d}'/'fold_receipt.json';x=json.loads(p.read_text())
        rows.append(x['models']['gru']['against_last_close_or_zero']['relative_improvement'])
    ci=uncertainty['models']['gru']
    if any(v['confidence_interval'] is None or not v['confidence_interval'][0]<=0<=v['confidence_interval'][1]
           for v in ci):raise ValueError('The stated GRU interval interpretation is not supported')
    labels={'last_close':'零收益','drift':'漂移','seasonal':'季节','ridge':'Ridge',
            'mlp':'MLP','gru':'GRU','lstm':'LSTM','xpatch':'xPatch'}
    lines=['# 月度 H21 拆股中性收益：事后开发对照','',
           f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}。这是观察原结果后增加的本机探索，不是独立测试或投稿结论。','',
           '使用与原月度任务相同的五个时间折和 1,970 个验证起点，重新训练八类方法。每根价格按该日期之前已生效的拆股比例调整；未来拆股仅影响实现后的标签。价格收益不含现金分红，不是总回报。','',
           '过去对数相对价格窗口与原方案核对一致；模型的价格锚点数值和训练目标发生变化，因此原方案与此处的差异不能归因于单一因素。','',
           '| 方法 | 相对零收益基线的终点 MSE 改善 |','| --- | ---: |']
    for name in MODELS:
        lines.append(f"| {labels[name]} | {100*result['models'][name]['relative_improvement']:.2f}% |")
    lines += ['',f"GRU 的五折汇总 MSE 比基线低 {100*result['models']['gru']['relative_improvement']:.2f}%，"
              f"但逐折分别为 {', '.join(f'{100*x:.2f}%' for x in rows)}。",
              '按 21 和 42 个交易日分块计算的描述性损失差区间均包含零；验证窗口重叠，且这是事后选择的一种子开发实验，不能据此宣称稳定或独立显著优势。','',
              '五折标签已按另一公式逐步复核，预测和汇总误差也已复算；新的零收益基线与先前仅改标签的诊断一致。这些校验不能证明数据发表授权或模型在独立数据上的效果。','',
              '[全部模型结果]('+str(root/'PRIVATE_RESULTS.csv')+')  ',
              '[标签和汇总核验]('+str(root/'PRIVATE_LABEL_AND_AGGREGATE_CHECK.json')+')  ',
              '[描述性区间]('+str(intervals)+')','']
    path=root/'PRIVATE_REPORT_ZH.md'
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':'private_split_neutral_report_created','path':str(path),'sha256':file_digest(path)},ensure_ascii=False))


if __name__=='__main__':main()
