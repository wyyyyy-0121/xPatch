"""Explain completed evidence and its limits without selecting another recipe."""
from common import *
from datetime import datetime


def percentage(value):
    return f'{value:+.4f}%'


def table(headers, rows):
    return '| ' + ' | '.join(headers) + ' |\n| ' + ' | '.join(['---'] * len(headers)) + ' |\n' + ''.join(
        '| ' + ' | '.join(map(str, row)) + ' |\n' for row in rows)


def main():
    result = read(BASE / 'analysis/RESULTS.json')
    complete = read(BASE / 'audit/TRAINING_COMPLETE.json')
    jobs = read(BASE / 'protocol/SELECTED_JOBS.json')['jobs']
    assert result['completed_new_fits'] == complete['jobs'] == len(jobs)
    primary = [row for row in result['pooled'] if row['variant'] == 'long_aligned']
    assert len(primary) == 20
    met = result['both_mechanism_tasks_pass']
    sections = [
        '# 本轮推进的实际研究结论\n',
        '论文仍为 Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales。'
        '保留五尺度、价格和收益两目标、二十任务；没有更换题目。\n',
        f'新增 {len(jobs)} 次神经拟合已完成。两个指定机制任务同时满足预设性能目标：'
        f'{"是，但仅限已暴露的历史开发数据" if met else "否"}。'
        '实验执行完成不等于提出了优于基线的预测方法；论文排版和复现另行验收。\n',
        '## 1. 旧研究提供了什么依据\n',
        '原四百次神经拟合给出了固定比较起点、保存的专家输出和可追溯的失败模式。'
        '误差分解定位到预测能量与目标对齐的差额，不能仅凭恒等式判断某个分支导致错误。'
        '固定检查点的输入扰动只能说明敏感性；本轮用真实重训的损失探针与同参数四格消融进一步检验。\n',
        '原拟合中 81/400 达二十轮上限，包含 36 个小时价格拟合；分钟 H60 收益的二十个拟合均未达上限。'
        '因此不能对全部模型作“与训练上限无关”的结论。详见 diagnostics/INTERPRETATION_ZH.md。\n',
        '## 2. 固定主候选在二十任务的表现\n',
        '以下均为三个种子的每起点平均损失再汇总，不是平均预测的集成成绩。'
        '正数表示优于各自的价格持续性或零收益参考；不能跨量纲平均 MSE。\n',
        table(['任务', '主候选改善', '正向折数 / 5', '种子 2026 / 2027 / 2028'], [
            [row['task'], percentage(row['improvement_pct']), row['positive_fold_means'],
             ' / '.join(percentage(row['seed_improvements_pct'][str(seed)]) for seed in (2026, 2027, 2028))]
            for row in primary]),
        '## 3. 两个机制任务与简单控制\n'
    ]
    factor_pairs = [
        ('long_path', 'short_path', '原路径损失下扩大局部感受野'),
        ('short_aligned', 'short_path', '短感受野下增加终点权重'),
        ('long_aligned', 'short_aligned', '终点加权下扩大局部感受野'),
        ('long_aligned', 'long_path', '长感受野下增加终点权重')]
    for profile, horizon, target in MECHANISM:
        task = key(profile, horizon, target)
        criterion = next(row for row in result['mechanism_performance'] if row['task'] == task)
        sections.append(f'### {task}\n')
        labels = {
            'minimum_one_percent': '相对零预测至少改善 1%',
            'all_three_seeds_positive': '三个种子均为正',
            'four_positive_folds': '至少四折平均损失为正向改善',
            'all_control_intervals_both_blocks': '两个块长度下全部四个控制的区间下界为正'}
        sections.append(table(['预设要求', '实际满足'], [
            [labels[name], '是' if value else '否'] for name, value in criterion['checks'].items()]))
        controls = [row for row in result['intervals'] if row['task'] == task and
                    row['primary'] == 'long_aligned' and row['reference'] in
                    ('zero', 'original_lstm', 'ridge', 'shrink_lstm')]
        sections.append(table(['主候选相对控制', '块长度（交易日）', '改善', 'MSE 差值 95% 区间'], [
            [row['reference'], row['block'], percentage(row['relative_improvement_pct']), row['difference_ci95']]
            for row in controls]))
        for name, reference, description in factor_pairs:
            intervals = [row for row in result['intervals'] if row['task'] == task and
                         row['primary'] == name and row['reference'] == reference]
            assert len(intervals) == 2
            support = all(row['supports_improvement'] for row in intervals)
            sections.append(f'- {description}：相对误差改善 {percentage(intervals[0]["relative_improvement_pct"])}；'
                            f'两个预设块长度的区间均支持改善：{"是" if support else "否"}。\n')
        interaction = [row for row in result['factorial_interactions'] if row['task'] == task]
        sections.append(table(['交互块长度', '损失交互', '95% 区间', '支持超过相加的收益'], [
            [row['block'], f'{row["mean_loss_interaction"]:.9g}', row['difference_ci95'],
             '是' if row.get('joint_gain_beyond_additivity_supported') else '否'] for row in interaction]))
    sections += [
        '感受野因素只改变膨胀率，四格参数量相同；短模型仍通过池化和输入处理利用窗口，'
        '不能把这个因素解释成所有长期记忆。新旧模型比较还改变输入、规模和初始化，'
        '只有小模型内部的配对能支持对应实现因素。区间是已暴露开发期的描述性区间，没有多重比较校正。\n',
        '## 4. 融合是否确实带来增量\n'
    ]
    for fusion in result['fusion']:
        sections.append('### ' + fusion['task'] + '\n')
        sections.append(table(['方法', '相对零预测改善'], [
            [row['method'], percentage(row['improvement_pct'])] for row in fusion['pooled']]))
        sections.append('全部预设门控配对要求通过：' + ('是' if fusion['gate_gain_supported'] else '否') + '。\n')
    sections += [
        '这些融合数字只使用相同的第三至第五折，不能与五折主模型表直接作成绩比较。'
        '专家固定为原种子 2026，三个门控种子不能证明整个专家训练流程的稳定性。'
        '二十五个门控参数以外仍需运行四个完整专家。\n',
        '## 5. 本轮实际计算与复现范围\n'
    ]
    receipts = [read(BASE / 'runs' / job['id'] / 'DONE.json') for job in jobs]
    budget = read(BASE / 'protocol/BUDGET.json')
    end = datetime.fromisoformat(complete['at_utc']).timestamp()
    cost = {
        'new_neural_fits': len(jobs),
        'original_core_neural_fits_reused': 400,
        'maximum_training_workers': budget['maximum_workers'],
        'logical_cpus_per_worker': budget['cpus_per_worker'],
        'gpu_used': False,
        'training_queue_wall_seconds_including_benchmark_and_recovery': end - budget['start_epoch'],
        'absolute_deadline_epoch': budget['deadline_epoch'],
        'completed_by_deadline': end <= budget['deadline_epoch'],
        'sum_recorded_fit_wall_seconds': sum(row['training_wall_seconds'] for row in receipts),
        'sum_recorded_fit_cpu_seconds': sum(row['training_cpu_seconds'] for row in receipts),
        'fit_time_boundary': 'recorded worker initialization/training/stopping/checkpoint section; final prediction and separate analysis excluded',
        'new_fits_reaching_twenty_epoch_cap': sum(row['epochs'] == 20 for row in receipts),
        'tiny_fits_selecting_epoch_zero': sum(row['job']['family'] == 'tiny' and row['best_epoch'] == 0 for row in receipts),
        'Q3_values_read': False,
        'independent_confirmation': False}
    atomic(BASE / 'analysis/ACTUAL_COMPUTE.json', cost)
    sections.append(f'新增拟合记录的 CPU 时间合计 {cost["sum_recorded_fit_cpu_seconds"]/3600:.3f} 核时；'
                    f'队列从计费起点到完成 {cost["training_queue_wall_seconds_including_benchmark_and_recovery"]/3600:.3f} 小时，'
                    f'早于绝对截止：{"是" if cost["completed_by_deadline"] else "否"}。'
                    '这两个量不同；前者不包含末端预测和另行分析。详细边界保存在 ACTUAL_COMPUTE.json。\n')
    sections.append(f'新拟合中 {cost["new_fits_reaching_twenty_epoch_cap"]} 格达到二十轮上限；'
                    f'{cost["tiny_fits_selecting_epoch_zero"]} 格小模型由内层停止选择初始持续性输出。'
                    '这项统计说明实际训练和选择行为，不表示该任务不存在可预测性。\n')
    sections += [
        '## 6. 论文能主张什么，尚未完成什么\n',
        ('两个机制任务达到了事先规定的历史开发性能要求；主张仍限于这些配方、信息和开发期。'
         if met else '本轮没有完成“两个机制任务稳定优于强控制”的方法目标。论文可报告受控比较、'
         '诊断证据和失败边界，不能把模型组装或训练完成写成预测方法突破。') + '\n',
        '年度价格和收益点估计分别改善约 6.93% 和 6.82%，三个种子都为正；预设 12/24 个月块的区间下界也为正。'
        '但四十七个重叠起点只能提供很少的长块，区间对重采样构造敏感，又属于已暴露开发期，不能当作独立验证。'
        '这是保留的局部跨尺度发现，不把年度任务事后换成预设的两个机制主任务。\n',
        '年度只有四十七个重叠起点；月度拆股路径集中在一个事件。跨尺度变化不等于二十个独立复现。'
        '2026Q3 没有读取或评分，没有独立验证或交易收益结论。\n',
        '原共享多期限与独立多期限的真实市场比较仍未运行。它是明确未解决的科学问题；'
        '本轮路径头与融合不能替代该控制。本轮不增加训练格、改主候选或启动新的搜索。\n',
        '主稿、补充材料、逐页审阅和脱离工作区的复现检查分别记录在 audit/。'
        '只有实际验收通过后，才可将论文交付目标标为完成；邮箱仍由作者填写。\n'
    ]
    (BASE / 'analysis/RESEARCH_FINDINGS_ZH.md').write_text('\n\n'.join(sections), encoding='utf-8')


if __name__ == '__main__':
    main()
