"""Generate concise human and machine delivery reports from the frozen run."""
from pathlib import Path

from .io import ROOT, read_json, write_json


def report(config, run_directory):
    directory = Path(run_directory)
    evaluation = read_json(directory / "evaluation.json")
    passed = evaluation["passed_candidates"]
    status = "waiting_conditional_confirmation" if passed else "no_candidate_passed_development_gates"
    confirmation_path = directory / "confirmation/evaluation.json"
    if passed and confirmation_path.exists():
        confirmation = read_json(confirmation_path)
        status = confirmation["status"]
    lines = ["# AAPL 未来 30 分钟 RV 研究结果", "",
        f"运行签名：`{evaluation['signature']}`。当前结论：`{status}`。", "",
        "这是分钟起点的波动率研究，输出累计对数收益平方之和；不预测 AAPL 涨跌，不证明交易盈利。历史数据已有项目曝光，结果不是干净盲测。", "",
        "| 信息集 | 模型 | 日等权 QLIKE | log-RV MSE | 80% 覆盖 | 90% 覆盖 | 合格会话 |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for key, values in evaluation["descriptions"].items():
        panel, name = key.split("/")
        lines.append(f"| {panel} | {name} | {values['qlike']:.6f} | {values['log_rv_mse']:.6f} | {values.get('coverage_80', 0):.3f} | {values.get('coverage_90', 0):.3f} | {values['sessions']} |")
    lines += ["", "比较对象是对应信息集在各外折内部选出的 HAR，差值为候选减 HAR，负数表示改善。Holm 家族固定为十项。", "",
        "| 信息集/候选 | QLIKE 差值 | 保守 p | Holm p | 改善季度 | 全部门槛通过 |",
        "|---|---:|---:|---:|---:|---|"]
    for key, values in evaluation["comparisons"].items():
        if "mean_difference" not in values:
            lines.append(f"| {key} | 未完成 | 1 | 1 | 未完成 | 否 |")
        else:
            improving = sum(value < 0 for value in values["quarter_differences"].values())
            lines.append(f"| {key} | {values['mean_difference']:.6f} | {values['p_conservative']:.6g} | {values['holm_p']:.6g} | {improving}/10 | {'是' if values['passed'] else '否'} |")
    lines += ["", "`evaluation.json` 保存两个块置信区间、DM/HAC、年度/季度/时段分段、各随机种子、经验区间覆盖与 MZ 回归。MZ 使用每日平均的日内重叠 RV 与预测，HAC 为五日滞后；它是辅助校准诊断。", "",
        "`selection.json`、各任务 `request.json`/`result.json`、模型检查点及 `predictions.npz` 保留完整选参和回放证据。失败与剪枝任务保留，不能从模型表或多重比较家族中消失。", "",
        "xPatch 的市场面板使用共享参数进行多资产辅助训练，原网络保持通道独立；其他表格与 Chronos 面板使用起点内过去的市场协变量。Chronos 固定权重，其 API 的 point 输出实际为 log 尺度中位数，再经独立校准回变得到 RV 均值。", "",
        "全部历史柱经过日历与缺口审计，但最终历史柱不能证明当时的发布与接收时间；成交量及 SPY/QQQ 价格尚未取得独立供应商复核。SPY 的九根缺柱保留，公共输入资格造成部分日期或时段不可用。", "",
        "近期 SIP 权限未准入，前瞻数据必须实际按时收到并在目标开始前记录预测。模型未完成前瞻验收时不能标为有效实时信号。最佳报价探针不等于多档盘口，订单流方向本轮未准入。", ""]
    if not passed:
        lines += ["没有候选通过全部开发门槛时，历史确认、交易/执行用途与前瞻观察按协议跳过。完整搜索及两轮复核通过后，交付明确标识的 HAR 参考基线和阴性结果；不追加模型或放宽门槛。", ""]
    else:
        lines += [f"开发门槛通过者：{', '.join(passed)}。条件阶段状态：`{status}`。历史确认使用已冻结配置；确认通过后仍须至少六十个合格交易日的事前预测验收。", ""]
    destination = ROOT / "delivery/REPORT.md"
    destination.write_text("\n".join(lines))
    write_json(ROOT / "delivery/results.json", {"signature": evaluation["signature"], "development_status": status,
        "passed_candidates": passed, "engineering_final_acceptance": False,
        "evaluation": str((directory / "evaluation.json").relative_to(ROOT)), "historical_exposure": True,
        "direction_or_profit_claim": False})
    return destination
