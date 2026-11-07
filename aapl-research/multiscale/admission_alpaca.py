"""Prepare a limited-history development fallback from the existing Alpaca account.

It does not use EODHD/Twelve Data prices. Annual remains insufficient under the
original 15-year minimum; publication rights and the final protocol remain open.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .admission import checked_source, comparison, daily_bars, fold_summary, read_json, save_bars, stamp, verify_page_rows
from .bars import aggregate_intraday, aggregate_months, eligible_origins, file_digest, load_admitted_minutes
from .exposure import legacy_ledger, overlapping_origin_count
from .spec import DEV_END, MODELS, NY, OUT, PROFILES, TARGETS
from .splits import warmup_folds


def official_split():
    """Use Apple's public 2020 announcement, not a vendor-adjusted close field."""
    folder = OUT / "source_docs"
    path = folder / "apple_2020_split_announcement.html"
    receipt_path = folder / "apple_2020_split_announcement.receipt.json"
    receipt = read_json(receipt_path)
    if file_digest(path) != receipt["sha256"]:
        raise ValueError("Apple announcement bytes differ from the saved receipt")
    html = path.read_text(encoding="utf-8", errors="replace").lower()
    if "four-for-one stock split" not in html or "august 31, 2020" not in html:
        raise ValueError("Apple announcement does not contain the expected split facts")
    return [{"date": "2020-08-31", "ratio": 4.0}], {"url": receipt["url"], "sha256": receipt["sha256"]}


def run(alpaca_path: Path, nasdaq_path: Path):
    alpaca_dir, alpaca_receipt, alpaca = checked_source(alpaca_path, "alpaca_daily_raw.csv")
    nasdaq_dir, nasdaq_receipt, nasdaq = checked_source(nasdaq_path, "nasdaq_daily.csv")
    verify_page_rows(alpaca_dir, alpaca_receipt, alpaca, "alpaca")
    verify_page_rows(nasdaq_dir, nasdaq_receipt, nasdaq, "nasdaq")
    actions, action_source = official_split()
    ledger = legacy_ledger(Path("/home/ayaka/xPatch/appl/Min-by-Min/results/test_use_ledger.json"))
    minute, minute_evidence = load_admitted_minutes()
    hour, minute_days, aggregation = aggregate_intraday(minute)
    day, calendar_audit = daily_bars(alpaca)
    month = aggregate_months(day)
    series = {"minute": minute, "hour": hour, "day": day, "month": month}
    minute_close = pd.DataFrame({"date": minute_days.session_date, "close": minute_days.close})
    comparisons = {"alpaca_vs_nasdaq": comparison(alpaca, nasdaq, actions, right_is_split_adjusted=True),
                   "alpaca_vs_last_regular_minute": comparison(alpaca, minute_close, actions, right_is_split_adjusted=False)}
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUT / "admission" / ("alpaca_" + run_id)
    directory.mkdir(parents=True, exist_ok=False)
    for frequency, bars in series.items():
        save_bars(directory / f"bars_{frequency}.npz", bars)
    result = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "limited-history Alpaca development data and provisional task inventory; no model scores",
              "development_cutoff_new_york": DEV_END, "Q3_values_read": False, "experiment_scored": False,
              "provider": "Alpaca SIP/raw daily and previously admitted SIP/raw minutes",
              "source_receipt_hashes": {"alpaca_daily": file_digest(alpaca_dir / "receipt.json"),
                                        "nasdaq_comparison": file_digest(nasdaq_dir / "receipt.json")},
              "source": {"first": alpaca_receipt["first"], "last": alpaca_receipt["last"],
                         "rows": alpaca_receipt["rows"]},
              "minute_source": minute_evidence, "intraday_aggregation": aggregation,
              "daily_calendar": calendar_audit, "cross_provider": comparisons,
              "legacy_test_exposure": ledger,
              "split_actions": actions,
              "split_source": action_source,
              "limitations": ["2016-2026 daily history is shorter than the original 15-year annual minimum.",
                              "Apple's announcement establishes the 2020 split; completeness of other corporate actions needs review.",
                              "Daily bars use conservative next-New-York-midnight completion; final revisions may arrive later and point-in-time receipt is not verified.",
                              "Academic publication and redistribution rights for the user's account have not been verified.",
                              "Raw close labels crossing a split include the nominal price change."],
              "tasks": [], "code_sha256": {}, "long_source_academic_license_verified": False}
    for name in ("spec.py", "bars.py", "splits.py", "admission.py", "admission_alpaca.py", "exposure.py"):
        result["code_sha256"][name] = file_digest(Path(__file__).parent / name)
    for profile in PROFILES:
        bars = series[profile.frequency]
        years = ((datetime.fromisoformat(str(bars.session_date[-1])) -
                  datetime.fromisoformat(str(bars.session_date[0]))).days / 365.2425)
        for horizon in profile.horizons:
            origins = eligible_origins(bars, max(profile.lookbacks), horizon,
                                       same_session=(profile.frequency == "minute"))
            np.save(directory / f"origins_{profile.name}_h{horizon}.npy", origins, allow_pickle=False)
            folds = fold_summary(bars, origins, horizon)
            alternative = None
            if profile.name == "quarterly" and len(folds) == 5 and min(f["inner_fit_origins"] for f in folds) < 8:
                proposed = warmup_folds(bars, origins, horizon, min_inner_fit=24,
                                        min_inner_stop=4, min_validation_groups=6)
                alternative = {"status": "proposed_not_frozen" if len(proposed) == 5 else "infeasible",
                               "rule": {"minimum_inner_fit_origins": 24, "minimum_inner_stop_origins": 4,
                                        "minimum_validation_months_per_fold": 6,
                                        "selection": "earliest feasible warm-up by sample counts only"},
                               "folds": [{"fold": f["fold"], "warmup_groups": f["warmup_groups"],
                                          "inner_fit_origins": len(f["inner"]["fit"]),
                                          "inner_stop_origins": len(f["inner"]["stop"]),
                                          "validation_origins": len(f["validation"]),
                                          "validation_first_decision": stamp(bars.end_ns[f["validation"][0]]),
                                          "validation_last_decision": stamp(bars.end_ns[f["validation"][-1]])}
                                         for f in proposed]}
            last = bars.end_ns[origins[-1] + horizon] if len(origins) else None
            if last is not None and pd.Timestamp(int(last), tz="UTC").tz_convert(NY).strftime("%Y-%m-%d") > DEV_END:
                raise ValueError("Target crosses the development cutoff")
            if years < profile.minimum_history_years:
                status = "blocked_insufficient_original_minimum_history"
            elif len(folds) != 5 or not all(f["status"] == "ok" for f in folds):
                status = "blocked_insufficient_purged_folds"
            elif min(f["inner_fit_origins"] for f in folds) < 8:
                status = "candidate_early_fold_has_fewer_than_eight_fit_origins"
            else:
                status = "candidate_pending_protocol_and_publication_rights"
            for target in TARGETS:
                result["tasks"].append({"profile": profile.name, "frequency": profile.frequency,
                                        "horizon": horizon, "target": target,
                                        "lookback_candidates": list(profile.lookbacks),
                                        "common_max_lookback": max(profile.lookbacks),
                                        "primary_endpoint": horizon == profile.primary_horizon,
                                        "history_span_years": years, "minimum_history_years": profile.minimum_history_years,
                                        "minimum_history_met": years >= profile.minimum_history_years,
                                        "minimum_inner_fit_origins": min((f["inner_fit_origins"] for f in folds), default=0),
                                        "known_legacy_overlap_origins": overlapping_origin_count(bars.session_date, origins, horizon, ledger),
                                        "origins": len(origins), "folds": folds, "proposed_warmup_folds": alternative,
                                        "first_decision": stamp(bars.end_ns[origins[0]]) if len(origins) else None,
                                        "last_decision": stamp(bars.end_ns[origins[-1]]) if len(origins) else None,
                                        "last_target_completion": stamp(last) if last is not None else None,
                                        "status": status})
    if len(result["tasks"]) != 20:
        raise AssertionError("Expected twenty tasks")
    result["files"] = {p.name: file_digest(p) for p in sorted(directory.iterdir()) if p.is_file()}
    (directory / "admission.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    lines = ["# Alpaca 日线的多尺度候选任务", "", f"生成时间：{result['created_at_utc']}。无模型评分。", "",
             f"日线 {result['source']['first']} 至 {result['source']['last']}，共 {result['source']['rows']} 行；"
             "分钟来源沿用已有 SIP/raw 准入。仅做开发工程准备，数据发表权限仍待核查。", "",
             "| 尺度 | 跨度 | 目标 | 起点 | 历史年数 | 状态 |",
             "| --- | ---: | --- | ---: | ---: | --- |"]
    for item in result["tasks"]:
        lines.append(f"| {item['profile']} | {item['horizon']} | {item['target']} | {item['origins']:,} | "
                     f"{item['history_span_years']:.2f} | {item['status']} |")
    lines.extend(["", "年度任务未达到原定 15 年最低历史长度，不得作为正式年度比较。"
                  "少于八个内层拟合起点只表示明显的工程样本不足，并非八个以上即足以支持统计结论。"
                  "其他任务也只是候选；须先冻结实验矩阵并处理数据使用与评价边界。", "",
                  "日线以纽约自然日结束作为保守的最早完成时间，2026-06-30 的最终日柱在该边界之后完成，"
                  "故从可用标签中排除；后续修订和实际接收延迟仍无法证明。", "",
                  "季度另外生成按样本数选择初始训练期的五折候选，未冻结也未评分；详见 admission.json。"
                  "即使达到这一工程样本底线，也不能证明神经模型充分训练或长期结论具有足够统计效力。", "",
                  "旧项目测试使用台账及每项任务与其已曝光日期的重叠计数保存在 admission.json；"
                  "换数据源或跨度不能让同一行情期间恢复为独立测试。", "",
                  "本报告没有把原开发季度作为独立确认，也没有读取 2026Q3 的市场价格。", ""])
    (directory / "ADMISSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"directory": str(directory), "task_count": len(result["tasks"]),
                      "statuses": {task["profile"]: task["status"] for task in result["tasks"]},
                      "daily_calendar": calendar_audit, "cross_provider": comparisons,
                      "experiment_scored": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpaca", type=Path, required=True)
    parser.add_argument("--nasdaq", type=Path, required=True)
    args = parser.parse_args()
    run(args.alpaca, args.nasdaq)


if __name__ == "__main__":
    main()
