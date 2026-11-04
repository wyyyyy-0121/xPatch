"""Human-readable research conclusion after independent numerical verification."""
from .common import ROOT, read_json, utc_now


def write_results():
    result = read_json(ROOT / 'outputs/formal/final_summary.json')
    verification = read_json(ROOT / 'outputs/formal/independent_verification.json')
    if not verification['passed']:
        raise RuntimeError('Do not write a conclusion before independent verification passes')
    protocol = read_json(ROOT / 'protocol_locked.json')
    usage = read_json(ROOT / 'outputs/formal/usage.json')
    decision = result['development']
    lines = ['# AAPL 12 分钟收益预测：本轮研究结果', '', f'生成时间：{utc_now()}。按用户的 IMPLEMENTATION_PLAN.md 和锁定协议执行。', '',
             '主任务为 AAPL 下午 13:30–15:47 ET 的未来 12 分钟累计 log 收益，主指标为固定种子分别计算损失后的平均 MSE。', '',
             '| 方法 | 开发验证 MSE | 验证相对零收益改善 | D 块 MSE | D 块相对改善 | 固定重训预算 |',
             '|---|---:|---:|---:|---:|---:|']
    for name, row in decision['models'].items():
        lines.append(f"| {name} | {row['validation_mse']:.8g} | {row['validation_relative_gain_vs_zero']:.3%} | {row['confirmation_mse']:.8g} | {row['confirmation_relative_gain_vs_zero']:.3%} | {row['fixed_budget']} |")
    lines += ['', f"零收益 MSE：开发验证 {decision['zero_validation_mse']:.8g}；D 块 {decision['zero_confirmation_mse']:.8g}。",
              '开发验证的配置选择本身会使这些分数偏乐观；D 块和条件封存检验承担后续验证。', '']
    if result['holdout_evaluated']:
        lines += ['封存检验使用 2026 年 7—9 月共 64 个交易日、8,832 个共同起点，模型名单在开发决策中固定。', '',
                  '| 方法 | 封存 MSE | 相对零收益改善 | Holm 单侧 p | 改善月份数 | 所有种子改善 | 满足稳定改善规则 |',
                  '|---|---:|---:|---:|---:|---|---|']
        for name, row in result['models'].items():
            count = sum(x > 0 for x in row['monthly_relative_improvement'].values())
            lines.append(f"| {name} | {row['mse']:.8g} | {row['relative_mse_improvement_vs_zero']:.3%} | {row['primary_test']['holm_p']:.5g} | {count}/3 | {'是' if row['qualification_checks']['all_fixed_seeds_improved'] else '否'} | {'是' if row['stable_better_under_locked_rules'] else '否'} |")
        lines += ['', '1 日与 10 日块敏感性、每月与每种子结果、方向、MAE、区间覆盖等完整指标见机器汇总。']
    else:
        lines += ['没有候选同时在开发验证和 D 块低于零收益，因此按协议停止进入最终检验；7—9 月数据继续封存。',
                  '这是当前模型、信息范围和训练预算下的开发期阴性结果，不把未运行的封存检验写成模型检验失败。']
    lines += ['', '配对机制分析只用于开发阶段解释：xPatch 固定超参、种子和预算改换损失；Chronos LoRA 对照匹配配置的冻结模型。它们不增加封存检验的模型数。', '',
              f"按实测成本锁定的 GPU 预算含 50% 余量为 {protocol['measured_budget']['gpu_hours_with_margin']:.2f} 小时。",
              f"当前记录的实际物理 GPU 忙碌时间为 {usage['physical_gpu_busy_hours']:.2f} 小时，总墙钟为 {usage['wall_hours']:.2f} 小时；并发作业的重叠时间只计一次。", '',
              '初始搜索、晋级、种子复核、失败和保存预测已保留。成功淘汰配置的张量检查点按锁定规则删除，保留哈希、指标与删除清单。', '',
              '数值完整性和研究搜索覆盖由独立核查通过，主 MSE 从保存的预测重新求和计算。']
    positive = result['holdout_evaluated'] and any(row['confirmatory_claim_allowed'] for row in result['models'].values())
    if positive:
        lines += ['', '论文判断：本轮提供了可以进一步组织研究文章的改善与稳定性证据。正式选题仍需核实相对已有文献的新意，并检验机制及适用范围；单只股票的结果不足以支持普遍优越性，也不能保证期刊接收。']
    else:
        lines += ['', '论文判断：目前证据不足以支持“xPatch 已获得稳定有效改进”的论文主张。可以保留为充分调参与严格流程下的实证材料；是否值得投稿还取决于阴性结论的新颖性、机制分析和其他标的的独立证据。下一步不依据封存结果追加同一检验。']
    lines += ['', '局限：本轮只涵盖下午时段和单一 AAPL 标的；基础模型预训练样本边界未独立证明。分钟最终历史柱不证明实际接收时点。预测改善不等于扣除成本后的可交易收益。', '',
              '云端、Kronos、TimesFM、全天面板未启用，不计为已完成模型比较。', '',
              '可核对文件：`protocol_locked.json`、`trial_lists.json`、`outputs/development_decision.json`、`outputs/formal/final_summary.json`、`outputs/formal/independent_verification.json`。', '']
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines))
    status = '# 当前执行状态\n\n'
    status += f"更新：{utc_now()}。本地实施流程已完成，独立主指标复核通过。\n\n"
    status += ('已按固定名单完成封存检验。' if result['holdout_evaluated'] else '开发期进入条件未满足，封存数据保持未解封。')
    status += '\n\n结果和论文方向判断见 [RESULTS.md](RESULTS.md)。工程准入见 [ADMISSION_REPORT.md](ADMISSION_REPORT.md)。\n'
    (ROOT / 'EXECUTION_STATUS.md').write_text(status)
