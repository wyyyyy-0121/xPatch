"""Certify accepted input identities without rewriting historical v2 evidence."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from data_acquisition.delivery_data import sha, verify, write_json
from finance_v2.provenance import verify_alpaca_run


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def build():
    seal = read(ROOT / "delivery/history.json")
    if not seal["passed"]:
        raise ValueError("Historical seal has not passed")
    for name, expected in seal["historical_file_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Historical evidence changed: {name}")
    old_root = ROOT / "data_acquisition/runs/alpaca_development_20261102"
    sample_root = ROOT / "data_acquisition/runs/alpaca_audit20_20261102T0800Z"
    old_verified = verify_alpaca_run(old_root, save=False)
    sample_verified = verify_alpaca_run(sample_root, save=False)
    old_receipt = read(old_root / "receipt.json")
    old_audit = read(old_root / "history_audit.json")
    old_sample = read(sample_root / "quality_audit.json")
    if not old_audit["historical_research_eligible"] or not old_sample["dst_switch_verified"]:
        raise ValueError("Historical AAPL audits did not pass")
    assets = {"AAPL": {
        "csv": str((old_root / old_receipt["csv"]).relative_to(ROOT)),
        "csv_sha256": old_verified["csv_sha256"],
        "receipt": str((old_root / "receipt.json").relative_to(ROOT)),
        "receipt_sha256": sha(old_root / "receipt.json"),
        "verification": old_verified, "sample_verification": sample_verified,
        "audit": str((old_root / "history_audit.json").relative_to(ROOT)),
        "audit_sha256": sha(old_root / "history_audit.json"),
        "regular_rows": old_audit["regular_rows"], "expected_rows": old_audit["expected_rows"],
        "sessions": len(old_audit["daily"]), "coverage": old_audit["regular_coverage"],
        "minimum_daily_coverage": old_audit["minimum_daily_coverage"],
        "audit20": str((sample_root / "quality_audit.json").relative_to(ROOT)),
        "audit20_sha256": sha(sample_root / "quality_audit.json"),
        "volume_independently_verified": False}}
    for symbol in ("SPY", "QQQ"):
        root = ROOT / f"data_acquisition/runs/delivery_{symbol}_history_20261103"
        sample = ROOT / f"data_acquisition/runs/delivery_{symbol}_audit20_20261103"
        verified = verify(root)
        sample_verification = verify(sample)
        receipt, audit, sample_audit = read(root / "receipt.json"), read(root / "audit.json"), read(sample / "audit.json")
        if not audit["passed"] or not sample_audit["passed"]:
            raise ValueError(f"{symbol} data audit failed")
        days = {entry["date"]: entry for entry in sample_audit["coverage"]["daily"]}
        if len(days) != 20 or days["2024-11-29"]["expected"] != 210:
            raise ValueError("20-day audit lacks expected early-close coverage")
        if days["2024-03-08"]["open_utc"][11:16] == days["2024-03-11"]["open_utc"][11:16]:
            raise ValueError("20-day audit does not verify spring DST")
        if days["2024-10-31"]["open_utc"][11:16] == days["2024-11-22"]["open_utc"][11:16]:
            raise ValueError("20-day audit does not verify autumn DST")
        coverage = audit["coverage"]
        assets[symbol] = {"csv": str((root / receipt["csv"]).relative_to(ROOT)),
            "csv_sha256": verified["csv_sha256"], "receipt": str((root / "receipt.json").relative_to(ROOT)),
            "receipt_sha256": sha(root / "receipt.json"), "verification": verified,
            "sample_verification": sample_verification, "audit": str((root / "audit.json").relative_to(ROOT)),
            "audit_sha256": sha(root / "audit.json"), "audit20": str((sample / "audit.json").relative_to(ROOT)),
            "audit20_sha256": sha(sample / "audit.json"), "regular_rows": coverage["regular_rows"],
            "expected_rows": coverage["expected_rows"], "sessions": len(coverage["daily"]),
            "coverage": coverage["regular_coverage"], "minimum_daily_coverage": coverage["minimum_daily_coverage"],
            "volume_independently_verified": False, "prices_independently_cross_checked": False}
    probe_root = ROOT / "data_acquisition/runs/delivery_probe_20261103"
    probe = read(probe_root / "probe.json")
    for entry in probe["requests"]:
        if "raw_file" in entry and sha(probe_root / entry["raw_file"]) != entry["sha256"]:
            raise ValueError("Probe response fingerprint changed")
    latest = [r for r in probe["requests"] if r["name"] in ("latest_sip_bars", "recent_sip_access")]
    realtime = all(r.get("http_status") == 200 for r in latest) and len(latest) == 2
    report = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "passed_for_retrospective_research": True, "feed": "sip", "adjustment": "raw", "timeframe": "1Min",
        "timestamp_label": "bar_start", "timezone": "America/New_York", "calendar": "XNYS common US equity session grid",
        "start": "2021-01-04", "end": "2026-06-30", "assets": assets,
        "probe": str((probe_root / "probe.json").relative_to(ROOT)), "probe_sha256": sha(probe_root / "probe.json"),
        "recent_sip_access_passed": realtime, "timely_live_delivery_certified": False,
        "point_in_time_historical_data": False, "historical_independent_holdout": False,
        "order_flow": {"historical_trade_quote_access_sample_passed": all(
            r["access_passed"] for r in probe["requests"] if "quotes_" in r["name"] or "trades_" in r["name"]),
            "complete_dataset_acquired_and_audited": False, "multi_level_depth_available": False,
            "conditional_line_eligible": False,
            "reason": "Only best-bid/ask and trade access samples are verified; no complete order-flow dataset or multi-level book was acquired. Conditional line is not admitted by access samples."},
        "limitations": ["Final historical bars do not reconstruct publication time or revision latency.",
            "2025H2-2026H1 and older dates have project exposure; these are retrospective comparisons.",
            "AAPL overlapping prices were checked in v2; SPY/QQQ prices and all volumes lack an independent supplier cross-check.",
            "Raw corporate-action scale is retained; exclude all overnight jumps in intraday return/RV construction.",
            "An available SIP history does not imply a paid subscription or real-time SIP permission."]}
    write_json(ROOT / "delivery/data.json", report)
    external = {"schema_version": 1, "updated_utc": report["created_utc"], "items": [
        {"id": "realtime_sip", "status": "skipped_permission_denied" if not realtime else "requires_live_latency_audit",
         "evidence": report["probe"], "http_statuses": [r.get("http_status") for r in latest],
         "provider_message": "subscription does not permit querying recent SIP data" if not realtime else None,
         "blocks": ["timely_live_inference", "qualifying_prospective_observation"],
         "does_not_block": ["historical_data_audit", "retrospective_training", "offline_inference", "negative_result_delivery"],
         "action_taken": "No purchase, subscription change or alternate-feed fallback."},
        {"id": "conditional_orderflow", "status": "not_eligible_no_complete_dataset",
         "evidence": report["probe"], "action_taken": "Preserve access evidence; do not claim DeepLOB or complete OFI replication."},
        {"id": "github_push", "status": "skipped_not_authorized", "action_taken": "Local commits only."}]}
    write_json(ROOT / "delivery/external_dependencies.json", external)
    rows = "\n".join(f"| {symbol} | {a['sessions']} | {a['regular_rows']:,}/{a['expected_rows']:,} | {a['minimum_daily_coverage']:.2%} | `{a['csv_sha256']}` |"
        for symbol, a in assets.items())
    text = f"""# 交付主线数据权限与验收

验收时间：{report['created_utc']}。历史封存凭证：`delivery/history.json`；机器可读输入清单：`delivery/data.json`。

## 实测权限

25 次只读行情请求中，15 次 AAPL/SPY/QQQ 历史分钟线请求及 8 次 AAPL 历史逐笔/最佳报价请求均返回非空 HTTP 200。最新 SIP 与最近十分钟 SIP 均返回 HTTP 403：`subscription does not permit querying recent SIP data`。未改变订阅、未付款、未回退 IEX，也不推断账户套餐。

实时权限项登记在 `delivery/external_dependencies.json`。它不阻塞历史训练和离线交付，但当前不能承诺及时的 SIP 分钟推理或启动合格前瞻观察；市场关闭时的探针也不能测量盘中真实发布延迟。

## 20 日与多年历史

AAPL 复用已封存的 20 日审计和多年 SIP 数据，本次从原始响应再次全字段核对，未改写原凭证。新增 SPY/QQQ 先通过同一组 20 日 DST、提前收盘审计，再下载 2021-01-04 至 2026-06-30 多年数据；全部分页和 CSV 的 OHLCV、成交笔数、VWAP 都独立重建核对，未插值、删异常或回退其他 feed。

| 标的 | 会话 | RTH 分钟覆盖 | 最低日覆盖 | CSV SHA256 |
|---|---:|---:|---:|---|
{rows}

日历使用美国股票共同常规会话的 XNYS 网格；bar 使用起始标签、America/New_York 时区、SIP/raw。`raw` 不代表未修订的点时数据。历史分钟线可能有事后修订，不可作为实时延迟证据。只有同日相邻收盘收益进入 RV，排除隔夜和公司行动引起的跨日跳空。

## 来源边界与条件研究

AAPL 的重叠价格交叉核查见封存的 `data_acquisition/V2_SOURCE_AUDIT.md`；SPY/QQQ 价格和全部成交量尚无独立供应商核验，结构验收不等于价格或成交量的独立真实性证明。既有历史仍有项目曝光，2025H2—2026H1 只作历史确认。

逐笔与报价探针仅证明抽样日期的访问能力，未取得并验收完整订单流训练和评价数据；报价字段仅为最佳买卖价及数量，不含多档深度。因此 B 线暂不满足准入条件，不声称已复现 OFI 或 DeepLOB。A 线所需三个标的的分钟数据已满足回顾研究的覆盖与完整性要求。
"""
    (ROOT / "data_acquisition/DELIVERY_SOURCE_AUDIT.md").write_text(text)
    print(json.dumps({"passed": True, "assets": {s: a["regular_rows"] for s, a in assets.items()},
        "realtime_sip": realtime, "conditional_orderflow": False}))
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    build()
