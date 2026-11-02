"""Aggregate fixed-seed tests without selecting a winner on held-out data."""
from pathlib import Path

import numpy as np

from .metrics import forecast_metrics, paired_day_bootstrap
from .models import NEURAL_MODELS
from .run import read_json, write_json


def generate_report(output):
    output = Path(output)
    config = read_json(output / "manifest.json")["config"]
    protocol = read_json(output / "frozen.json")
    audit = read_json(output / "audit.json")
    results, predictions = {}, {}
    common_origins = common_y = common_dates = None
    for name in config["models"]:
        group, forecasts = [], []
        for seed in (config["seeds"] if name in NEURAL_MODELS else [0]):
            directory = output / "test" / f"{name}_seed{seed}"
            result = read_json(directory / "result.json")
            with np.load(directory / "predictions.npz") as saved:
                origins, y, dates, pred = (saved[k] for k in ("origins", "y", "dates", "pred"))
            if common_origins is None:
                common_origins, common_y, common_dates = origins, y, dates
            else:
                for actual, expected in ((origins, common_origins), (y, common_y), (dates, common_dates)):
                    if not np.array_equal(actual, expected):
                        raise AssertionError("Models evaluated on different samples or labels")
            recalculated = forecast_metrics(y, pred)
            if result["metrics"] != recalculated:
                raise AssertionError("Stored metrics disagree with saved predictions")
            group.append(result)
            forecasts.append(pred)
        predictions[name] = np.stack(forecasts)
        metrics = {}
        for h in ("1", "5", "12"):
            metrics[h] = {}
            for key in ("mse", "mae", "rmse", "skill_vs_zero", "direction_accuracy"):
                values = [r["metrics"][h][key] for r in group]
                metrics[h][key] = {"mean": float(np.mean(values)), "std": float(np.std(values)),
                                   "per_seed": values}
        results[name] = {"metrics": metrics, "parameters": group[0]["parameters"],
                         "peak_cuda_allocated_mib": max((r["peak_cuda_allocated_mib"] or 0) for r in group),
                         "inference_ms_per_sample": float(np.mean([r["inference_ms_per_sample"] for r in group])),
                         "training_seconds": sum(r["training_seconds"] for r in group),
                         "checkpoint_reproduction": all(r["checkpoint_exact_reproduction"] for r in group)}
    # Bootstrap the seed-averaged loss, NOT the loss of an ensemble forecast.
    # Stacking seeds into each date block preserves pairing and gives exactly that estimand.
    comparisons = {}
    if "ls" in predictions:
        candidate = predictions["ls"]
        seed_count, sample_count, horizon = candidate.shape
        tiled_y = np.tile(common_y, (seed_count, 1))
        tiled_dates = np.tile(common_dates, seed_count)
        for name, baseline in predictions.items():
            if name == "ls":
                continue
            if baseline.shape[0] == 1:
                baseline = np.repeat(baseline, seed_count, axis=0)
            if baseline.shape != candidate.shape:
                raise AssertionError("Unpaired seeds")
            ci = paired_day_bootstrap(tiled_y, candidate.reshape(-1, horizon), baseline.reshape(-1, horizon),
                                      tiled_dates, repeats=config["bootstrap_repeats"])
            deltas = []
            for i in range(seed_count):
                a = forecast_metrics(common_y, candidate[i])["12"]["mse"]
                b = forecast_metrics(common_y, baseline[i])["12"]["mse"]
                deltas.append(a - b)
            ci["per_seed_mse_differences"] = deltas
            ci["all_seeds_improve"] = all(d < 0 for d in deltas)
            comparisons[name] = ci
    required = {"zero", "ridge", protocol["selected_original"]}
    adequate = bool(comparisons) and None not in required and required.issubset(comparisons)
    supported = adequate and all(comparisons[n]["ci95"][1] < 0 and comparisons[n]["all_seeds_improve"]
                                 for n in required)
    summary = {"models": results, "ls_minus_baseline": comparisons,
               "selected_original_from_validation": protocol["selected_original"],
               "long_short_effectiveness_supported": bool(supported),
               "conclusion": "Supports LS under this single-asset protocol" if supported else
                             "Does not establish stable LS improvement under this protocol",
               "test_samples": len(common_y), "test_dates": len(np.unique(common_dates)),
               "data_provenance_verified": False,
               "return_quality": audit["return_quality"],
               "uncertainty": "95% day-block paired bootstrap; seed-averaged losses, no ensemble; exploratory ablations uncorrected"}
    # Show concentration of held-out squared error without changing any targets.
    actual_cumulative = common_y.sum(axis=1)
    summary["test_zero_mse_concentration"] = []
    total_loss = float(np.sum(actual_cumulative ** 2))
    for date in np.unique(common_dates):
        mask = common_dates == date
        loss = float(np.sum(actual_cumulative[mask] ** 2))
        summary["test_zero_mse_concentration"].append({"date": str(date), "samples": int(mask.sum()),
                                                       "zero_squared_error_sum": loss,
                                                       "share": loss / total_loss if total_loss > 0 else 0.0})
    summary["test_zero_mse_concentration"].sort(key=lambda r: r["share"], reverse=True)
    write_json(output / "summary.json", summary)
    lines = ["# xPatch-LS 实验报告", "", f"结论：{summary['conclusion']}。", "",
             f"测试集：{len(common_y):,} 个相同预测时刻，{summary['test_dates']} 个日期；"
             f"验证集预选原版：{protocol['selected_original']}。", "",
             "表中为各随机种子指标的均值；预测目标均为累计对数收益率。", "",
             "| 模型 | 12分钟MSE | 相对零预测改善 | 1分钟MAE/RMSE | 5分钟MAE/RMSE | 12分钟MAE/RMSE | 参数 | 峰值CUDA MiB | ms/样本 |",
             "|---|---:|---:|---|---|---|---:|---:|---:|"]
    for name, r in results.items():
        m = r["metrics"]
        pairs = [f"{m[h]['mae']['mean']:.6g} / {m[h]['rmse']['mean']:.6g}" for h in ("1", "5", "12")]
        lines.append(f"| {name} | {m['12']['mse']['mean']:.8g} | {m['12']['skill_vs_zero']['mean']:.2%} | " +
                     " | ".join(pairs) + f" | {r['parameters']:,} | {r['peak_cuda_allocated_mib']:.1f} | {r['inference_ms_per_sample']:.5f} |")
    lines += ["", "## 长短双流相对各对照的差异", "",
              "差异为 LS MSE 减去对照 MSE，负值有利于 LS。置信区间按完整日期分块抽样，保留日内重叠窗口。", "",
              "| 对照 | MSE差异 | 95%区间 | 所有种子改善 |", "|---|---:|---|---|"]
    for name, ci in comparisons.items():
        lines.append(f"| {name} | {ci['mse_difference']:.8g} | [{ci['ci95'][0]:.8g}, {ci['ci95'][1]:.8g}] | {ci['all_seeds_improve']} |")
    contrasts = {"xpatch48": "较长趋势历史（参数量匹配）", "ls_unpooled": "长流压缩",
                 "ls_no_decomp": "EMA分解", "long_only": "加入短流", "short_only": "加入长流"}
    lines += ["", "## 因素判断", ""]
    for name, factor in contrasts.items():
        if name not in comparisons:
            continue
        ci = comparisons[name]
        direction = "有利" if ci["mse_difference"] < 0 else "不利"
        certainty = "区间跨零，证据不足" if ci["ci95"][0] <= 0 <= ci["ci95"][1] else "区间不跨零"
        lines.append(f"- {factor}：点估计{direction}于主方案；{certainty}。")
    lines += ["", "## 数据与限制", "",
              f"- 数据 SHA256：`{audit['sha256']}`；{audit['rows']:,} 行价格、{audit['eligible_origins']:,} 个有效窗口。",
              f"- 时区：{audit['timezone']}；数据来源和复权信息尚未由CSV证实。保留提供的日期边界，不猜测交易所时区或筛选常规交易时段。",
              f"- 因缺失分钟额外排除 {sum(d['excluded_for_gaps'] for d in audit['coverage_by_date']):,} 个按行数原本可能组成的窗口；不回填、不跨夜。",
              "- 只使用AAPL及单一收益率特征，不能外推至其他资产或宣称交易盈利；方向准确率不等于可交易性。",
              "- 日分块区间依赖日期块可交换假设，未建模跨日相关；消融比较属于探索性分析。",
              "- GPU显存为PyTorch峰值已分配内存，包含驻留数据和模型；CPU基线不占CUDA显存。",
              "- 延迟为预热后的批量推理，排除数据准备及主机传输；CPU与GPU延迟不能解释为同硬件算法速度。",
              "- 原版EMA已替换为稳定等价计算；网络默认路径保持原结构，长短窗口扩展及单流消融显式启用。",
              "", "最大的连续分钟价格跳变如下。部分跳变随后迅速回归，来源未确认；它们可能主导MSE，不能将拟合这份CSV的结果直接视为真实市场预测能力。原始目标未删改。", "",
              "| 时间戳 | 前收盘 | 收盘 | 分钟对数收益率 |", "|---|---:|---:|---:|"]
    for row in audit["return_quality"]["largest_absolute_returns"]:
        lines.append(f"| {row['timestamp']} | {row['previous_close']:.4f} | {row['close']:.4f} | {row['log_return']:.6f} |")
    lines += ["", "测试集零预测平方误差贡献最大的五个日期：", ""]
    for row in summary["test_zero_mse_concentration"][:5]:
        lines.append(f"- {row['date']}：{row['share']:.2%}。")
    lines += ["", "完整指标、种子差异、置信区间与工件索引见 `summary.json`、`manifest.json`、`frozen.json` 及各运行目录。", ""]
    (output / "REPORT.md").write_text("\n".join(lines))
    print(f"Report: {output / 'REPORT.md'}", flush=True)
    return summary
