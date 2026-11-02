"""Recompute aligned seed-mean losses, calibration scores and predeclared stability gates."""
from pathlib import Path

import numpy as np

from .data import confirmation_split, quarterly_splits
from .io import read_json, write_json
from .metrics import forecast_metrics, holm, paired_block_bootstrap


def mean_tree(values):
    if isinstance(values[0], dict):
        return {key: mean_tree([v[key] for v in values]) for key in values[0]}
    if values[0] is None:
        return None
    return float(np.mean(values))


def load_panel(output, config, splits, feature, names):
    from .run import run_path, seeds_for
    model_data = {}
    reference = None
    for name in names:
        predictions, quantiles, results = [], [], []
        for seed in seeds_for(name, config):
            chunks = []
            for split in splits:
                directory = run_path(output, split, feature, name, seed)
                with np.load(directory / "predictions.npz") as saved:
                    chunks.append({k: saved[k].copy() for k in saved.files})
                results.append(read_json(directory / "result.json"))
            joined = {k: np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
            if reference is None:
                reference = {k: joined[k] for k in ("origins", "dates", "timestamps", "y")}
            for key, expected in reference.items():
                if not np.array_equal(joined[key], expected):
                    raise AssertionError(f"Unpaired forecasts: {feature}/{name}/seed{seed}/{key}")
            predictions.append(joined["pred"])
            quantiles.append(joined["quantiles"])
        model_data[name] = {"pred": np.stack(predictions), "quantiles": np.stack(quantiles),
                            "results": results}
    return reference, model_data


def compare(reference, candidate, baseline, config):
    horizon_index = config["horizons_minutes"].index(config["statistics"]["primary_horizon_minutes"])
    actual = reference["y"][:, horizon_index]
    a = np.mean((candidate["pred"][:, :, horizon_index] - actual) ** 2, axis=0)
    b = np.mean((baseline["pred"][:, :, horizon_index] - actual) ** 2, axis=0)
    difference = a - b
    s = config["statistics"]
    record = paired_block_bootstrap(difference, reference["dates"], s["bootstrap_repeats"], s["bootstrap_seed"])
    record["five_day_block_sensitivity"] = paired_block_bootstrap(
        difference, reference["dates"], s["bootstrap_repeats"], s["bootstrap_seed"], 5)
    dates = reference["dates"].astype(str)
    quarters = np.array([f"{d[:4]}Q{(int(d[5:7])-1)//3+1}" for d in dates])
    record["by_quarter_mse_difference"] = {q: float(difference[quarters == q].mean()) for q in np.unique(quarters)}
    years = np.array([d[:4] for d in dates])
    record["by_year_mse_difference"] = {y: float(difference[years == y].mean()) for y in np.unique(years)}
    # Compare each seed against the baseline seed-mean loss. A deterministic
    # baseline is not needlessly replicated, and no forecast ensemble is used.
    record["per_seed_mse_difference"] = [float(np.mean((p[:, horizon_index] - actual) ** 2 - b))
                                         for p in candidate["pred"]]
    record["all_seeds_improve"] = all(v < 0 for v in record["per_seed_mse_difference"])
    return record


def generate_report(output, stage="development"):
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    config = manifest["identity"]["config"]
    if stage == "development":
        splits = quarterly_splits(config)
        panel_names = {f: config["models"] for f in config["feature_sets"]}
    else:
        frozen = read_json(output / "frozen.json")
        splits = [confirmation_split(config)]
        panel_names = {f: list(dict.fromkeys(["zero", "ridge", frozen["selected_original"][f], selected]))
                       for f, selected in frozen["selected"].items() if selected is not None}
    summary = {"signature": manifest["signature"], "stage": stage, "panels": {},
               "independent_test": False,
               "uncertainty": "conditional seed-mean losses; circular day or five-session block bootstrap; no minute iid or ensemble",
               "pretraining_exposure": "Kronos paper cutoff 2024-06 is not an independent checkpoint audit; Chronos financial exposure unknown",
               "point_in_time_available": False, "trading_profitability_evaluated": False}
    p_values = {}
    for feature, names in panel_names.items():
        reference, forecasts = load_panel(output, config, splits, feature, names)
        panel = {"samples": len(reference["y"]), "sessions": len(np.unique(reference["dates"])),
                 "models": {}, "vs_zero": {}, "vs_ridge": {}}
        for name, saved in forecasts.items():
            metrics = [forecast_metrics(reference["y"], pred, config["horizons_minutes"], q,
                                       config["calibration"]["quantile_levels"])
                       for pred, q in zip(saved["pred"], saved["quantiles"])]
            losses = (saved["pred"] - reference["y"]) ** 2
            dates = reference["dates"].astype(str)
            years = np.array([d[:4] for d in dates])
            quarters = np.array([f"{d[:4]}Q{(int(d[5:7])-1)//3+1}" for d in dates])
            panel["models"][name] = {"metrics": mean_tree(metrics), "per_seed_metrics": metrics,
                "by_year_12m_mse": {y: float(losses[:, years == y, -1].mean()) for y in np.unique(years)},
                "by_quarter_12m_mse": {q: float(losses[:, quarters == q, -1].mean()) for q in np.unique(quarters)},
                "total_seconds": sum(r["total_seconds"] for r in saved["results"]),
                "peak_cuda_allocated_mib": max((r["peak_cuda_allocated_mib"] or 0) for r in saved["results"])}
            if name != "zero":
                record = compare(reference, saved, forecasts["zero"], config)
                panel["vs_zero"][name] = record
                p_values[f"{feature}/{name}"] = record["p_value_one_sided_candidate_better"]
            if name not in {"ridge", "zero"}:
                panel["vs_ridge"][name] = compare(reference, saved, forecasts["ridge"], config)
        summary["panels"][feature] = panel
    adjusted = holm(p_values)
    for feature, panel in summary["panels"].items():
        originals = [n for n in ("xpatch48", "xpatch240") if n in panel["models"]]
        selected_original = (min(originals, key=lambda n: panel["models"][n]["metrics"]["12"]["mse"])
                             if stage == "development" else frozen["selected_original"][feature])
        panel["selected_original_from_development"] = selected_original
        for name, comparison in panel["vs_zero"].items():
            comparison["holm_adjusted_p"] = adjusted[f"{feature}/{name}"]
            comparison["positive_quarters"] = sum(v < 0 for v in comparison["by_quarter_mse_difference"].values())
            if stage == "development":
                comparison["development_gate_passed"] = bool(
                    comparison["mse_difference"] < 0 and comparison["holm_adjusted_p"] < 0.05 and
                    comparison["ci95"][1] < 0 and comparison["five_day_block_sensitivity"]["ci95"][1] < 0 and
                    comparison["positive_quarters"] >= config["statistics"]["positive_quarters_required"] and
                    set(comparison["by_year_mse_difference"]) == {"2023", "2024", "2025"} and
                    all(v < 0 for v in comparison["by_year_mse_difference"].values()) and comparison["all_seeds_improve"])
        panel["vs_selected_original"] = {}
        reference, forecasts = load_panel(output, config, splits, feature, panel_names[feature])
        for name in panel_names[feature]:
            if name not in {"zero", selected_original}:
                panel["vs_selected_original"][name] = compare(reference, forecasts[name], forecasts[selected_original], config)
    summary["any_development_gate_passed"] = any(r.get("development_gate_passed", False)
        for panel in summary["panels"].values() for r in panel["vs_zero"].values())
    write_json(output / f"{stage}_summary.json", summary)
    lines = [f"# AAPL v2：{stage}", "", "本报告为历史回顾性比较；评价期已有项目曝光，预训练金融数据边界也未独立确认。", "",
             "各模型先计算每个种子的损失再平均，不将预测平均成集成。所有适用模型使用相同 origin。2025 开发结果仅覆盖上半年。", ""]
    for feature, panel in summary["panels"].items():
        lines += [f"## {feature}", "", f"{panel['samples']:,} 个起点，{panel['sessions']} 个交易日。", "",
                  "| 模型 | 1分钟 RMSE | 5分钟 RMSE | 12分钟 MSE | 对零改善 | 80%区间覆盖 | Holm p | 改善季度 | 开发门槛 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name, model in panel["models"].items():
            m = model["metrics"]
            comparison = panel["vs_zero"].get(name)
            p = f"{comparison['holm_adjusted_p']:.5g}" if comparison else "—"
            quarters = str(comparison["positive_quarters"]) if comparison else "—"
            gate = str(comparison.get("development_gate_passed", "历史确认")) if comparison else "基线"
            lines.append(f"| {name} | {m['1']['rmse']:.6g} | {m['5']['rmse']:.6g} | {m['12']['mse']:.8g} | "
                         f"{m['12']['skill_vs_zero']:.3%} | {m['12']['central_interval_coverage']:.2%} | {p} | {quarters} | {gate} |")
        lines += ["", "| 模型对零 | MSE差 | 日块95%区间 | 五会话块95%区间 |", "|---|---:|---|---|"]
        for name, c in panel["vs_zero"].items():
            lo, hi = c["ci95"]
            flo, fhi = c["five_day_block_sensitivity"]["ci95"]
            lines.append(f"| {name} | {c['mse_difference']:.6g} | [{lo:.6g}, {hi:.6g}] | [{flo:.6g}, {fhi:.6g}] |")
        lines += ["", "区间为未校正的配对 bootstrap 区间；Holm p 用于预声明的双面板候选对零族。年度、季度、种子、pinball、校准覆盖、对 Ridge 和原版 xPatch 的差异均保存在 JSON。", ""]
    lines += ["任何模型未通过门槛时，停止其依赖的历史确认及交易成本研究。历史预测误差改善也不构成可交易性证明。", ""]
    (output / f"{stage.upper()}_REPORT.md").write_text("\n".join(lines))
    return summary
