"""Create a timestamped Chinese status snapshot from verified local artifacts."""
import csv
from datetime import datetime,timezone
import json

from multiscale.bars import file_digest
from multiscale.spec import OUT


def main():
    coverage_paths=sorted((OUT/'private_coverage').glob('*.csv'))
    aggregate_paths=sorted((OUT/'private_minute_h60_aggregate').glob('*/PRIVATE_PARTIAL_AGGREGATE.json'))
    if not coverage_paths or not aggregate_paths:raise ValueError('Coverage or minute aggregate is unavailable')
    coverage_path=coverage_paths[-1]
    with coverage_path.open(encoding='utf8',newline='') as f:
        coverage={(r['profile'],int(r['horizon']),r['target']):r for r in csv.DictReader(f)}
    aggregate_path=aggregate_paths[-1];aggregate=json.loads(aggregate_path.read_text())
    annual_root=OUT/'private_annual_short_development/20261107T143734746737Z'
    annual=json.loads((annual_root/'PRIVATE_ANNUAL_RESULTS.json').read_text())
    seed_root=OUT/'private_split_neutral_seed_replication/20261107T150345851174Z'
    seed=json.loads((seed_root/'PRIVATE_SEED_COMPARISON.json').read_text())
    if (aggregate['status']!='partial_h60_full_origin_development_only' or
            seed['status']!='private_posthoc_three_seed_comparison'):
        raise ValueError('Expected verified development snapshots')
    def row(h,target):return coverage[('minute',h,target)]
    def md_link(label,path):return f'[{label}]({path})'
    lines=['# AAPL 多尺度实验：当前进度','',
           f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}。结果为个人本机开发快照，尚不是独立确认或投稿结论。','',
           '## 已完成和正在运行','',
           '- 年度：按用户同意的约 10 年数据、12 个月回看，价格和收益各五折均已完成；共 47 个相互重叠的验证起点，低于原 15 年方案的历史长度。',
           '- 小时、月度、季度：10 个价格／收益任务各五折已完成并复算。',
           f"- 分钟 H60 收益：已汇总 {len(aggregate['completed_folds'])}/5 折、{aggregate['validation_origins']:,} 个完整验证起点；其余折以覆盖表为准。"]
    for h in (5,15,30,60):
        price=row(h,'future_close');ret=row(h,'future_log_return')
        if h==60:
            lines.append(f"- 分钟 H60 价格：{price['status']}，完整折 {price['completed_full_folds']}/5。")
        else:
            lines.append(f"- 分钟 H{h}：收益 {ret['status']}（{ret['completed_full_folds']}/5），价格 {price['status']}（{price['completed_full_folds']}/5）。")
    gru=seed['relative_mse_improvement_vs_zero_return']['gru']
    h60=aggregate['models']['gru']['relative_improvement']
    lines += ['', '## 当前结果怎么理解','',
              f"- 年度收益 GRU 五折汇总误差相对零收益基线改善 {100*annual['future_log_return']['metrics']['gru']['relative_improvement']:.2f}%，但样本小且各折波动很大。",
              f"- 月度拆股中性收益 GRU 三个种子的相对改善分别为 {100*gru['2026']:.2f}%、{100*gru['2027']:.2f}%、{100*gru['2028']:.2f}%；结果对种子敏感。首个种子的描述性分块区间包含零。",
              f"- 分钟 H60 收益当前汇总中，GRU 相对零收益基线的误差改善为 {100*h60:.2f}%；数值接近零，尚不能认定有稳定优势。",
              '- 这些百分比是预测均方误差相对基线的变化，不是投资收益率；月度收益不包含现金分红。','',
              '## 仍需处理','',
              '继续完成分钟各跨度的剩余折并复算；确认实际账号的行情协议和论文发表衍生结果的许可。用户表示会发送询问信，目前不能假设已发出或收到回复。','',
              md_link('完整覆盖快照',coverage_path.with_suffix('.md'))+'  ',
              md_link('年度详细报告',annual_root/'PRIVATE_ANNUAL_REPORT_ZH_v2.md')+'  ',
              md_link('月度三个种子报告',seed_root/'PRIVATE_SEED_COMPARISON_ZH.md')+'  ',
              md_link('分钟前几折汇总',aggregate_path),'']
    parent=OUT/'private_current_reports';parent.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');path=parent/(stamp+'_PRIVATE_CURRENT_ZH.md')
    with path.open('x',encoding='utf8') as f:f.write('\n'.join(lines))
    print(json.dumps({'status':'private_current_report_created','path':str(path),'sha256':file_digest(path)},ensure_ascii=False))


if __name__=='__main__':main()
