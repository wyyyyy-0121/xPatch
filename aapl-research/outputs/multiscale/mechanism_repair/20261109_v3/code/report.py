"""A candid goal ledger, driven by the final development evidence."""
from study import *
import csv


def main():
    screen = read(BASE/'analysis/SCREENING_RESULTS.json')
    rep = read(BASE/'analysis/REPLICATION_RESULTS.json'); cap = read(BASE/'analysis/CAP_RESULTS.json')
    result = read(BASE/'analysis/CANDIDATE_RESULTS.json')
    cand = read(BASE/'protocol/CANDIDATE_JOBS.json'); rows = []
    fresh = {j['id']: j for j in read(BASE/'protocol/JOBS.json')['jobs']}
    fresh.update({j['id']: j for j in read(BASE/'protocol/REPLICATION_JOBS.json')['jobs']})
    fresh.update({j['id']: j for j in cand['jobs']})
    assert len(fresh) == 345
    caps = read(BASE/'protocol/CAP_JOBS.json')['jobs']; assert len(caps) == 30
    done = [read(BASE/'runs'/j['id']/'DONE.json') for j in fresh.values()]
    assert all(r['status'] == 'completed' for r in done)
    cap_cpu = sum(read(BASE/'runs'/j['cfg']['id']/'DONE.json')['training_cpu_seconds']-
                  read(BASE/'runs'/j['source_job']/'DONE.json')['training_cpu_seconds'] for j in caps)
    trained_cpu = sum(r['training_cpu_seconds'] for r in done)+cap_cpu
    entries = [
        ('原题和五尺度二十任务', '已保留', '主稿题目、CANDIDATE_RESULTS.json 的 20 个任务；原比较完整保留'),
        ('比较混杂审计与精确复现', '已完成', '12 个筛查、20 个完整原配方及 4 个小模型前缀均匹配'),
        ('实现机制的配对证据', '部分取得支持', '完整四格、五折三种子的干预与两种块长；不等于市场机制'),
        ('训练上限影响', '已完成指定诊断', 'H1 30 个精确前缀对照，17 个增加训练；不代表全部架构充分训练'),
        ('有依据的固定候选检验', '已完成本轮检验', '同一初始化和内层收缩规则覆盖全部二十任务'),
        ('稳定超过强基准的预测方法', '未达到', '两个预定收益任务的性能门槛未通过，不能以修复弱默认替代'),
        ('未见时期的独立确认', '未执行', '2026Q3 继续封存；已有历史曝光身份不变'),
        ('共享／独立期限市场比较', '未执行', '单独列账，当前路径输出不代替此实验'),
        ('论文增补与计算复现', '进行中', '以文档、逐项预测复放和独立统计回执为完成依据')]
    for requirement, status, evidence in entries:
        rows.append({'requirement': requirement, 'status': status, 'evidence': evidence})
    atomic(BASE/'analysis/GOAL_LEDGER.json', {'requirements': rows, 'research_goal_achieved': False,
           'fresh_configurations': len(fresh), 'cap_comparisons': len(caps),
           'recorded_training_process_cpu_seconds_excluding_copied_prefix': trained_cpu,
           'Q3_values_read': False, 'independent_confirmation': False, 'at_utc': now()})
    lines = ['# 本轮推进与尚未完成的研究目标', '',
      '原题保持 **Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales**。Goal 继续开启。', '',
      '## 已核实的结果', '',
      '- 52 格筛查完整执行；四组完整配对复核共 240 格；固定候选覆盖二十任务共 130 格。各阶段包含复用，不能简单相加。',
      '- 本轮共 345 个新训练配置及 30 个复制前缀的训练上限对照；后者只有 17 个增加训练，其余 13 个按原早停保留。',
      '- 原配方权重、最佳轮次和停止轮次的精确复现通过；退出／恢复保留 Adam、随机状态和批次进度。', '']
    for task in ('minute_h60_future_log_return', 'hourly_h8_future_log_return'):
        contrast = next(r for r in rep['contrasts'] if r['task'] == task and r['model'] == 'xpatch' and
                        r['primary'] == 'anchor_baseline' and r['reference'] == 'anchor_random' and r['mode'] == 'trained')
        point = next(r for r in result['pooled'] if r['task'] == task and r['method'] == 'candidate')
        name = '分钟 H60' if task.startswith('minute') else '小时 H8'
        lines.append(f"- {name}：基准函数初始化相比默认 xPatch 误差降低 {contrast['pooled_improvement_over_reference_pct']:.2f}%，但最终收缩候选相对零基准改善为 {point['improvement_pct']:+.4f}%；预测性能目标未达到。")
    lines += ['', '## Goal 验收台账', '', '| 要求 | 状态 | 依据 |', '| --- | --- | --- |']
    for row in rows:
        lines.append('| '+' | '.join(row.values())+' |')
    lines += ['', '## 接下来与边界', '',
      '本轮补齐新稿、数值检查、逐页查看和保存模型／独立统计复算。成功交付这些材料只完成论文与工程部分，不会自动关闭预测研究 Goal。',
      '本轮结果支持修复一些原默认实现的输出与优化问题；它没有支持“收益信号已被可靠提取”。后续若要追求新的正向方法，应先提出可区别于本轮的可检验机制及有限实验，保留已有失败证据与原预算截止，不从已曝光任务随意挑一项改作成功终点。',
      '所有训练使用 CPU，最多 8 个并行进程、每进程 2 个逻辑核；分析共享 2 核。GPU 关闭，无租赁或付款。2026Q3 未读取；另行终止的基础模型项目未启动。', '']
    (BASE/'PROGRESS_ZH.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'fresh_fits': 345, 'cap_checks': 30, 'goal_achieved': False,
                      'training_process_cpu_hours': trained_cpu/3600}), flush=True)


if __name__ == '__main__':
    main()
