"""Publish a current requirements ledger without rewriting previous releases."""
from pathlib import Path
import csv
import hashlib
import json
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]
TITLE = 'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def export(name, rows):
    with (BASE / 'analysis' / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def main():
    results = read(BASE / 'analysis/STATISTICS.json')
    outer = read(BASE / 'analysis/OUTER_RESULTS.json')
    old = read(BASE.with_name('20261109_v3a') / 'analysis/GOAL_LEDGER.json')
    ledger = dict(old)
    ledger.update({'bounded_phase_status': 'context_experiment_complete_delivery_acceptance_pending',
        'current_phase': 'v3b_origin_context_and_compressed_history',
        'goal_status': 'active', 'research_goal_achieved': False,
        'current_matrix_cells': 40, 'current_candidate': results['candidate'],
        'deterministic_training_seed_count': None,
        'current_results': {task: {'candidate': entry['summary'][results['candidate']],
            'checks': entry['checks'], 'inner_gate': entry['inner_direction_gate_passed'],
            'performance_gate': entry['performance_gate_passed'],
            'confirmation_eligible': entry['independent_confirmation_eligible']}
            for task, entry in results['tasks'].items()},
        'current_protocol_sha256': hashlib.sha256((BASE / 'protocol/RULES.json').read_bytes()).hexdigest(),
        'at_utc': datetime.now(timezone.utc).isoformat()})
    ledger['requirements'] = [dict(x) for x in old['requirements']]
    for requirement in ledger['requirements']:
        name = requirement['requirement']
        if name == '有依据的小模型或修复／组合检验':
            requirement.update(status='固定初始化／收缩、BN诊断及本次40格上下文小模型检验均已执行',
                evidence='唯一D_joint_shrink候选未通过内层方向及全部开发门槛；不改选A/B/C、raw或超参数')
        elif name == '稳定超过强基准的预测方法':
            requirement.update(status='仍未达到',
                evidence='新候选分钟−0.2216048125%、小时−0.6836574982%，仅1/5与2/5折正，31项控制区间门槛失败；原失败记录保留')
        elif name == '历史输入的条件价值':
            requirement.update(status='新增一个受控低维条件均值检验，未取得可推广的稳定信号证据',
                evidence='A均值／B时刻／C历史／D加性组合四格配对；不推论所有时间上下文或波动模型均无效')
        elif name == '英文论文与实际独立解压复现':
            requirement.update(status='v3/v3a已交付；v3b数值复核与独立包验收进行中',
                evidence='本目录manuscript、evaluation_review、replay_review及后续audit/RELEASE.json；不覆盖先前交付')
    write(BASE / 'analysis/GOAL_LEDGER.json', ledger)
    export('CONTEXT_METRICS.csv', outer['metrics'])
    export('CONTEXT_SUMMARY.csv', [{'task': task, 'method': name, **row}
        for task, entry in results['tasks'].items() for name, row in entry['summary'].items()])
    export('CONTEXT_INTERVALS.csv', [row for entry in results['tasks'].values()
        for rows in list(entry['comparisons'].values()) + [entry['diagnostic_contrasts']] for row in rows])
    plan = f'''# 继续推进到论文验收的当前计划

题目保持 **{TITLE}**。原五尺度、二十项价格／收益任务及 Wang Yang 的署名保留；邮箱由作者填写。本计划承接 v3 的机制研究与 v3a 补充，Goal 为 active，不能以交付论文代替取得预测方法成果。

## 希望达到的研究成果

1. 用匹配输入、目标、初始化、停止与预算的配对干预，明确哪些实现因素造成原结果，以及哪些解释仍无法区分。
2. 根据该证据冻结一个具体的小模型或组合，在事先指定的任务上稳定优于零收益／持续价格、均值、Ridge 和原模型。原神经候选保留三种子标准；确定性解不伪造种子。本轮两项主要任务的性能门槛都必须满足，不能从二十项里事后挑一项代替。
3. 候选通过开发条件后，才在合格且标签完整的未见时期确认。主张范围须与确认范围一致；一个季度无法确认年度目标。
4. 将结果、失败、样本局限和复现材料完整写进原题英文论文；若未获得方法增益，论文应明确报告，不承诺一定能找到正结果。

## 本轮已执行的具体工作与并行分工

| 工作 | 当前证据 | 验收或下一步 |
| --- | --- | --- |
| 原公平性与机制研究 | v3 已完成锚点、初始化、输出及训练上限配对；v3a 完成长周期复验与BN干预 | 保留局部因果范围，不推成市场经济机制 |
| 单一新假设 | 决策时刻与星期能否在压缩历史之上增加条件均值信息 | 只检验固定加性表示；不声称新型Ridge |
| 匹配消融 | 两任务×五折×A均值/B时刻/C历史/D组合，全部40格完成 | sole candidate为D的内层收缩形式，不改选其他格 |
| 恢复 | 实际退出于4096起点并从同一游标恢复，与连续运行比特一致 | 保留检查点、冻结身份和全过程回执 |
| 数值复核（并行） | 独立人员重建特征、40模型、inner alpha、完整路径、31控制与136区间 | 必须与主实现核对；失败则定位并保留差异 |
| 论文（并行） | 独立v3b主稿与完整补充材料 | 主文双列，保留姓名、学校、城市、邮编、国家、邮箱占位；逐页检查 |
| 便携复现（并行） | 输入样例、模型、逐起点端点/路径损失及完整日格 | 实际解压并屏蔽原工作区/网络后运行；不冒充原始数据重训 |

本轮固定候选分钟H60为 **−0.221605%**、小时H8为 **−0.683657%**，正向折次分别为1/5与2/5；两任务的内层方向与完整开发条件均失败。当前配方无独立确认资格。全部原始输出与对照照实保存；禁止按这些外层分数修改滞后、谐波、正则化强度或候选形式再筛选。

## 从当前证据到最终论文的剩余步骤

**步骤一：完成本轮验收和解释收口。** 将独立重建、统计复算、恢复、文稿结构和逐页渲染的证据写入同一验收记录；用D/C、D/B、B/A说明被检验的信息增量，保留未支持和退化区间。交付主稿、完整补充、结果CSV、源码和可执行包。

**步骤二：保留未达成研究项，明确下一假设准入。** 现有初始化／归一化和本次加性时刻表示都不能保证强基准以上的条件信号。下一轮须先说明哪条现有证据支持一个与已失败配方可区分的机制，并写出单因素干预、同信息控制、唯一候选、成本与失败条件，再在剩余授权内执行。不能无限增加架构或利用当前外层分数反复调参。对既有机制解释与预测效果分别验收；无法取得增益时，保留负面结论和未完成Goal。

**步骤三：共享／独立期限单独处理。** 原市场对照仍未执行，当前多输出线性解不算完成。若最终保留共享学习优势主张，先冻结相同起点、输入、可用标签、更新／参数预算和停止规则，再做真实共享与分开训练。当前论文不声称该优势，不能通过H60前缀补账。

**步骤四：通过开发条件后才独立确认。** 冻结方法、适用任务、强控制和确认指标，核验确认期未参与方法选择且标签完整。当前2026Q3继续封存；所有已曝光历史维持开发身份。短期确认不能替代长周期证据，也不能补救候选开发失败。

**步骤五：最终论文与投稿前验收。** 将所有研究问题逐项对应方法、表格和适用边界；检查作者邮箱、官网学校英文名、图表、公式、参考文献和数据许可记录。提交前交付可编辑文稿及PDF、完整配置/代码/结果清单和独立复现回执。以实际证据决定摘要与贡献，不改原题；本轮没有对外投稿或公开发布。

## 时间和计算边界

原绝对截止为 **北京时间2026-11-09 14:31:49.623**（epoch1794205909.6228158），不重新起算四小时。当前解析拟合已全部结束；剩余主要是独立核验、文字与归档。训练上限仍为8进程×2逻辑核，全部分析合计限CPU24/26，GPU关闭，不租赁、付款或联网取数。到截止停止新增计算并保存恢复状态。

Goal保持active。只有对应证据真正满足条件，才能登记机制支持、方法成功或独立确认；若预算先结束，保留工作与未完成项，不能自动追加预算或将目标标完成。文稿和可复现交付是独立的工程验收项。
'''
    (BASE / 'PLAN_TO_COMPLETION_ZH.md').write_text(plan)
    report = f'''# 本轮上下文假设结果

原题：{TITLE}

本轮在原研究证据基础上，固定检验“决策时刻和星期是否给压缩历史增加条件均值信息”。两任务×五折×四格共40模型全部完成；确定性求解无训练种子。D_joint_shrink是评分前冻结的唯一候选，A/B/C和原始D不按外层成绩晋升。

| 固定候选 | 相对零收益基准 | 正向折次 | 内层方向 | 全部开发条件 |
| --- | --- | --- | --- | --- |
| 分钟H60 | −0.221605% | 1/5 | 未通过 | 未通过 |
| 小时H8 | −0.683657% | 2/5 | 未通过 | 未通过 |

正百分比才表示误差下降。配对统计保留全部31控制及两个块长，每任务68项区间，合计136项。只支持固定表示和当前历史的判断，不能据此说所有时刻信息或所有模型都无效。市场预测增益目标仍未实现，2026Q3未打开，独立确认未执行，原共享／独立期限市场对照仍单独列账。

恢复测试已完成实际进程退出与重启，并与不中断累积逐位核对。独立40模型重建、完整路径与统计核验，以及解压屏蔽原环境复现的最新状态见evaluation_review、replay_review和audit。原v3/v3a归档不覆盖。

完整计划：PLAN_TO_COMPLETION_ZH.md。论文：manuscript/MAIN_PAPER_EN.md、SUPPLEMENT_EN.md及对应Word/PDF。所有80个输出度量、66项方法汇总和136区间分别见analysis/CONTEXT_METRICS.csv、CONTEXT_SUMMARY.csv、CONTEXT_INTERVALS.csv。
'''
    (BASE / 'RESULTS_ZH.md').write_text(report)
    print('Current plan, Goal ledger and complete result CSV files written.')


if __name__ == '__main__':
    main()
