"""Keep engineering delivery and the active, unachieved research Goal distinct."""
from pathlib import Path
from datetime import datetime, timezone
import json

BASE = Path(__file__).resolve().parents[1]
TITLE = 'Multi-Horizon Forecasting of Apple Stock Prices and Returns Across Temporal Scales'


def read(p):
    return json.loads(Path(p).read_text())


def main():
    analysis = BASE / 'analysis'; analysis.mkdir(exist_ok=True)
    long = read(BASE / 'longscale_replication/analysis/RESULTS.json')
    bn = read(BASE / 'bn_recalibration/CONTROL_AUDIT.json')
    jobs = read(BASE / 'longscale_replication/protocol/JOBS.json')
    gate = read(BASE / 'bn_recalibration/GATE_AND_ADMISSION.json')
    assert jobs['deadline_epoch'] == gate['deadline_epoch'] == 1794205909.6228158
    release = read(BASE / 'audit/RELEASE.json') if (BASE / 'audit/RELEASE.json').exists() else None
    delivered = release is not None and release['status'] == 'passed'
    requirements = [
        {'requirement': '原题、作者与五尺度二十任务', 'status': '已保留',
         'evidence': '英文主稿及全部二十项结果；仅六项长周期更新种子深度'},
        {'requirement': '原模型比较混杂及实现机制', 'status': '部分得到配对支持',
         'evidence': 'v3完整四格初始化／锚点干预与训练上限对照；BN缓冲区干预补充局部推断误差证据，不解释经济机制'},
        {'requirement': '长周期复现、强简单控制及稳健性', 'status': '本补充阶段已完成并独立复核',
         'evidence': '180配置、三种子；100折均值控制；全部相位、去一年、拆股跨越；年度均值对照区间跨零且非重叠仅4路径'},
        {'requirement': '历史输入的条件价值', 'status': '已取得限制性诊断，未证明稳定条件信号',
         'evidence': '90模型×3未重训输入干预；平坦历史并不更差，27配置alpha0会掩盖网络敏感性'},
        {'requirement': '有依据的小模型或修复／组合检验', 'status': '已执行原固定修复及本次BN诊断',
         'evidence': '原二十任务初始化与收缩；分钟15配置BN扩展，小时仅2格失败筛查，无新增架构成功主张'},
        {'requirement': '稳定超过强基准的预测方法', 'status': '未达到',
         'evidence': '原主要收益门槛仍失败；BN分钟收缩−0.04375884%，三个种子均负，仅1折正，全部性能条件失败'},
        {'requirement': '合格未见时期的独立确认', 'status': '未准入、未执行',
         'evidence': '没有通过开发门槛的固定候选；2026Q3继续封存，旧历史曝光身份不变，单季度无法确认年度标签'},
        {'requirement': '共享／独立期限市场对照', 'status': '未执行，单独列账',
         'evidence': '不以H60路径前缀替代，不在当前论文中声称共享学习已优于独立训练'},
        {'requirement': '英文论文与实际独立解压复现', 'status': '已验证交付' if delivered else '文稿已渲染、解压验收待发布阶段记录',
         'evidence': '主文5页双列、补充4页；原v3不覆盖；预测／保存损失复算入口、包清单和解压验收见audit/RELEASE.json'},
    ]
    ledger = {'title': TITLE, 'goal_status': 'active', 'research_goal_achieved': False,
        'bounded_phase_status': 'delivered_research_goal_not_achieved' if delivered else 'final_acceptance_in_progress',
        'requirements': requirements, 'fresh_training_configurations_this_addendum': 150,
        'longscale_cells': 180, 'exact_reused_cells': 30, 'BN_fixed_weight_cells': 17,
        'mean_control_folds': 100, 'history_fixed_weight_cells': 90,
        'longscale_primary_tasks_replaced': False, 'primary_endpoints_changed': False,
        'independent_confirmation': False, 'Q3_values_read': False,
        'deadline_epoch': jobs['deadline_epoch'], 'max_training_processes': 8,
        'logical_CPUs_per_training_process': 2, 'all_analysis_affinity': [24, 26], 'GPU_used': False,
        'at_utc': datetime.now(timezone.utc).isoformat()}
    (analysis / 'GOAL_LEDGER.json').write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
    state = '已通过实际解压复现' if delivered else '正在执行最终解压复现验收'
    progress = f'''# 当前进度与研究状态

原题：**{TITLE}**。

Goal 继续 **active**。核心预测改进目标 **未达到**，独立确认未执行；本轮实验完成、文件排版或复现通过不替代这些标准。

本补充阶段已完成：六项长周期两配方、五折三种子的180格复验，其中精确复用30格、新增150格；全二十任务100折拟合均值控制；90个固定模型的三种历史输入条件；17个固定权重的BatchNorm统计重估配置。长周期、均值和历史敏感性已通过独立复核。BN和新稿的专项审阅见longscale_review各验收记录。

年度价格／收益三种子汇总改善为 **33.730917%／38.105180%**。与均值收缩控制的两种块长区间跨零；4条原非重叠路径上仅为17.584957%／7.469751%，均值控制为48.348760%／51.316552%附近，以RESULTS.json精确值为准。相位与去一年子集依赖，不增加独立市场历史。年度及季度未来拆股对照不可估计。

历史替换是未重训敏感性，不是新候选；27/90配置alpha为零。BN分钟原始误差相对原修复下降0.943571%，但仍低于零基准；内层收缩后的改善为 **−0.04375884%**，原修复为−0.03807042%。三个适配种子均负，仅一折正。小时BN未通过两折内层筛查，没有扩展为完整结果。该分支也未满足预测目标。

主稿5页、正文双列，补充4页；作者邮箱保持待作者填写。原v3全部文件与4.63GiB压缩包保持原样。新包仅引用原包校验值。当前交付状态：**{state}**，实际验收记录为audit/RELEASE.json；最终归档与验收摘要在audit/FINAL_AUDIT.json。

当前有效总计划见PLAN_TO_COMPLETION_ZH.md。余下缺口：可重复且胜过强控制的方法、合格独立确认，以及单列的共享／独立期限市场比较。候选未通过开发门槛时，不能启封2026Q3，也不能用年度正向结果替代原主要任务。后续新假设须先写明依据、对照、预算及失败判据，不能从当前外层结果连续试到获胜。

计算继续使用原绝对截止 **2026-11-09 14:31:49.623 北京时间**；不重新起算四小时。GPU关闭，训练上限8进程×2逻辑核，分析合计仅CPU24、26。训练与固定权重实验已完成并保存，当前只做文档和复现验收。
'''
    # Pull the exact nonoverlap return-mean number rather than trusting a
    # rounded prose carryover when the authoritative result is available.
    greedy = next(r for r in long['sensitivity'] if r['task'] == 'annual_h12_future_log_return' and
                  r['rule'] == 'original_global_earliest_greedy')
    progress = progress.replace('51.316552%附近，以RESULTS.json精确值为准',
                                f"{greedy['methods']['fit_mean_shrink']['improvement_pct']:.6f}%")
    (BASE / 'PROGRESS_ZH.md').write_text(progress)
    print(json.dumps({'goal_status': 'active', 'research_goal_achieved': False, 'delivered': delivered}), flush=True)


if __name__ == '__main__':
    main()
