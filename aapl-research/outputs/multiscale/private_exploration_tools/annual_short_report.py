"""Render the verified short-history annual development result for local use."""
from datetime import datetime,timezone
import json

from multiscale.bars import file_digest
from multiscale.spec import MODELS,OUT


def main():
    root=OUT/'private_annual_short_development/20261107T143734746737Z'
    plan=json.loads((root/'PRIVATE_PROTOCOL.json').read_text())
    state=json.loads((root/'PRIVATE_QUEUE_STATE.json').read_text())
    result_path=root/'PRIVATE_ANNUAL_RESULTS.json'
    results=json.loads(result_path.read_text())
    check=json.loads((root/'PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json').read_text())
    if (state['status']!='completed_short_history_development_only' or state['completed']!=10 or
            state['results_sha256']!=file_digest(result_path) or check['status']!='saved_arrays_recalculated'):
        raise ValueError('Annual result is incomplete or has changed')
    for job in plan['jobs']:
        entry=state['jobs'][job['key']]
        verification=root/(job['key']+'.verification.json')
        if entry['status']!='completed' or entry['verification_sha256']!=file_digest(verification):
            raise ValueError('A fold has not been verified')
    fold_gru={j['fold']:json.loads((root/j['key']/'fold_receipt.json').read_text())['models']['gru']
              ['against_last_close_or_zero']['relative_improvement'] for j in plan['jobs']
              if j['target']=='future_log_return'}
    labels=[('last_close','持续性／零收益'),('drift','漂移'),('seasonal','季节'),('ridge','Ridge'),
            ('mlp','MLP'),('gru','GRU'),('lstm','LSTM'),('xpatch','xPatch')]
    lines=['# AAPL 年度 H12：10 年短历史本机探索','',
           f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}。价格与收益各五折，八类方法；仅供个人本机开发。','',
           '## 数据与方案','',
           '- Alpaca 日线候选约覆盖 2016 年至 2026 年 6 月；用户选择先按约 10 年做实验。年度任务采用 12 个月回看和 12 个月预测。',
           '- 这是对原定至少 15 年历史的方案调整。按时间先后划分五折，并在训练、早停和验证之间按实际标签完成时间清除重叠。',
           '- 五折的拟合起点分别为 24、33、41、49、56，早停起点为 7、8、10、11、13；各目标共 47 个验证起点。年度预测窗口相互重叠，不能当作 47 个独立年度样本。',
           '- 八种方法都使用相同的各折验证起点；神经模型一种随机种子、最多 20 轮，结论尚不能视为模型收敛或稳健优势。账号具体协议及论文发表权仍待确认。','',
           '## 五折汇总','',
           '下表为相对基线的终点均方误差改善：`100 × (1 − 模型 MSE / 基线 MSE)`。正数表示误差较小，负数表示误差较大；不是投资收益率。价格基线是当前价格，收益基线是零收益。','',
           '| 方法 | 未来 12 个月累计对数收益 | 未来 12 个月名义价格 |','| --- | ---: | ---: |']
    for name,label in labels:
        values=[results[t]['metrics'][name]['relative_improvement'] for t in ('future_log_return','future_close')]
        lines.append('| '+label+' | '+' | '.join(f'{100*v:.2f}%' if v is not None else '不可计算' for v in values)+' |')
    lines += ['',
              '收益任务中，GRU 的汇总误差比零收益基线低约 16.62%；价格任务中，神经模型均未超过持续性基线。这只是已曝光的小样本开发结果，不能认定为独立显著或稳定优势。',
              f"例如收益 GRU 在第一折比基线差 {100*abs(fold_gru[1]):.2f}%，第二折则好 {100*fold_gru[2]:.2f}%；不同时间段的结果波动较大。",
              '季节基线在第 12 个月恰好重复当前月份的价格，因此其终点误差与持续性基线相同。',
              '名义价格和收益标签可能受到拆股影响，收益不含现金分红；数据口径仍需进一步核查。2026 年第三季度保持封存。','',
              '## 核验与文件','',
              '10 折均已独立复算保存的预测与标签；另从保存数组按验证起点数重新汇总所有模型。算术复核不等于独立市场样本或发表授权。','',
              '[结果表 CSV]('+str(root/'PRIVATE_ANNUAL_RESULTS.csv')+')  ',
              '[方案与逐折起点]('+str(root/'PRIVATE_PROTOCOL.json')+')  ',
              '[汇总复算回执]('+str(root/'PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json')+')','']
    path=root/'PRIVATE_ANNUAL_REPORT_ZH_v2.md'
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':'private_annual_report_created','path':str(path),'sha256':file_digest(path)},ensure_ascii=False))


if __name__=='__main__':main()
