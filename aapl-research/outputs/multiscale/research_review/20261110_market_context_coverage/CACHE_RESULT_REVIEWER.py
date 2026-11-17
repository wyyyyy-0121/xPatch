"""Independent offline SPY cache review; standard library, CPU 26, no adapter imports."""
from __future__ import annotations
import argparse
from datetime import date, datetime, time as midnight, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from zoneinfo import ZoneInfo

SOURCE_SHA = "9cc0d1e2af800d4dfe33ac10db009d483f66f5e0912f2a4ca3f3953863c9b32a"
FIXED = {"symbols": "SPY", "timeframe": "1Min", "feed": "sip", "adjustment": "raw", "currency": "USD", "asof": "-", "sort": "asc", "limit": 10000}
BUDGET = {"requests": 300, "bytes": 268435456, "window_seconds": 600, "workers": 2, "CPU": 24, "socket_seconds": 15, "page_bytes": 4194304}
NY = ZoneInfo("America/New_York")
UTC = timezone.utc
LIMIT = datetime(2026, 7, 1, tzinfo=NY).astimezone(UTC)

class Invalid(Exception):
    pass

def require(ok, code):
    if not ok:
        raise Invalid(code)

def sha(body):
    return hashlib.sha256(body).hexdigest()

def unique(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate_json_key")
        value[key] = item
    return value

def reject_constant(value):
    raise Invalid("nonfinite_json_constant")

def decode(body):
    return json.loads(body, object_pairs_hook=unique, parse_constant=reject_constant)

def number(value):
    return type(value) in (int, float) and math.isfinite(value)

def date_ranges():
    intervals = []
    day = date(2021, 1, 4)
    final = date(2026, 6, 30)
    while day <= final:
        end = min(day + timedelta(days=29), final)
        intervals.append([day.isoformat(), end.isoformat()])
        day = end + timedelta(days=1)
    require(len(intervals) == 67, "independent_range_count")
    return intervals

RANGES = date_ranges()

def boundary(index):
    start, end = (date.fromisoformat(x) for x in RANGES[index])
    return (datetime.combine(start, midnight(), NY).astimezone(UTC),
            datetime.combine(end + timedelta(days=1), midnight(), NY).astimezone(UTC))

def request(index, token):
    lower, upper = boundary(index)
    result = dict(FIXED)
    result.update(start=lower.isoformat(), end=(upper - timedelta(minutes=1)).isoformat())
    if token is not None:
        result["page_token"] = token
    return result

def minute_stamp(text):
    require(type(text) is str, "bar_timestamp_type")
    if text.endswith("Z"):
        core = text[:-1]
    elif text.endswith("+00:00"):
        core = text[:-6]
    else:
        raise Invalid("bar_timestamp_not_UTC")
    parts = core.split(".")
    require(len(parts) <= 2 and len(parts[0]) == 19, "bar_timestamp_shape")
    if len(parts) == 2:
        require(1 <= len(parts[1]) <= 9 and set(parts[1]) == {"0"}, "bar_timestamp_subminute")
    stamp = datetime.fromisoformat(parts[0]).replace(tzinfo=UTC)
    require(stamp.strftime("%Y-%m-%dT%H:%M:%S") == parts[0] and stamp.second == 0, "bar_timestamp_minute_grid")
    return stamp

def page_check(body, index, previous, guard):
    page = decode(body)
    require(type(page) is dict and type(page.get("bars")) is dict and set(page["bars"]) == {"SPY"}, "page_container_symbol")
    require("next_page_token" in page, "page_token_absent")
    next_token = page["next_page_token"]
    require(next_token is None or type(next_token) is str and bool(next_token), "page_token_invalid")
    require("currency" not in page or page["currency"] == "USD", "page_currency")
    rows = page["bars"]["SPY"]
    require(type(rows) is list and 1 <= len(rows) <= 10000, "page_row_limit")
    low, high = boundary(index)
    first = None
    last = previous
    for ordinal, bar in enumerate(rows):
        if ordinal % 256 == 0:
            guard()
        require(type(bar) is dict, "bar_container")
        stamp = minute_stamp(bar.get("t"))
        require(low <= stamp < high and stamp < LIMIT, "bar_range_or_July_cutoff")
        require(last is None or stamp > last, "bar_duplicate_or_unordered")
        for key in ("o", "h", "l", "c", "v", "n", "vw"):
            value = bar.get(key)
            require(number(value), "bar_numeric_type_or_finiteness")
            require(value >= 0 and (key in ("v", "n") or value > 0), "bar_numeric_sign")
        require(bar["l"] <= bar["o"] <= bar["h"] and bar["l"] <= bar["c"] <= bar["h"], "bar_OHLC_schema")
        for key in ("v", "n"):
            require(bar[key] == int(bar[key]) and bar[key] <= 2**63 - 1, "bar_count_schema")
        if first is None:
            first = stamp
        last = stamp
    return {"rows": len(rows), "first_UTC": first.isoformat(), "last_UTC": last.isoformat(), "next_token": next_token}, last

def review(protocol_path, acquisition, source_path, gate_path, output):
    started_review = time.time()
    report = {"kind": "independent_SPY_cache_result_review_v1", "CPU_affinity": sorted(os.sched_getaffinity(0)),
              "accepted_integrity": False, "all67source_qualified": False,
              "original_RTH_context_coverage": "not_tested", "unacquired_ranges_are_not_negative_source_evidence": True,
              "research_goal_achieved": False, "returns_models_scores_or_causal_claims": False,
              "uses_main_validator": False, "errors": [], "review_started_epoch": started_review}
    deadline = None
    checks = 0
    def guard():
        if deadline is not None:
            require(time.time() < deadline, "original_600_second_deadline_reached")
    try:
        require(report["CPU_affinity"] == [26], "reviewer_CPU_affinity")
        pbytes = protocol_path.read_bytes()
        protocol = decode(pbytes)
        deadline = protocol.get("deadline_epoch")
        start = protocol.get("started_epoch")
        require(number(start) and number(deadline) and deadline - start == 600, "protocol_absolute_window")
        report.update(protocol_sha256=sha(pbytes), original_started_epoch=start, original_deadline_epoch=deadline)
        guard()
        require(start <= time.time(), "protocol_window_not_started")
        require(protocol.get("kind") == "fixed_SPY_source_coverage_v1" and protocol.get("execution_admitted") is True, "protocol_admission")
        require(protocol.get("source_sha256") == SOURCE_SHA == sha(source_path.read_bytes()), "source_identity")
        require(protocol.get("cost_gate_sha256") == sha(gate_path.read_bytes()), "cost_gate_identity")
        require(protocol.get("endpoint") == "https://data.alpaca.markets/v2/stocks/bars", "protocol_endpoint")
        require(protocol.get("fixed") == FIXED and protocol.get("ranges") == RANGES and protocol.get("budget") == BUDGET, "protocol_scope_or_budget")
        require(all(protocol.get(key) is False for key in ("GPU", "Q3_allowed", "fits_or_scores_allowed")), "protocol_forbidden_scope")
        guard()
        jbytes = (acquisition / "JOURNAL.json").read_bytes()
        journal = decode(jbytes)
        report["journal_sha256"] = sha(jbytes)
        require(journal.get("protocol_sha256") == sha(pbytes) and journal.get("started_epoch") == start and journal.get("deadline_epoch") == deadline, "journal_protocol_binding")
        attempts, saved = journal.get("attempts"), journal.get("ranges")
        require(type(attempts) is list and len(attempts) <= 300 and type(saved) is list and len(saved) == 67, "journal_structure")
        require(journal.get("sealed_snapshot") is True, "journal_not_terminal_snapshot")
        report["journal_status"] = journal.get("status")
        expected_names = set()
        charged = raw_bytes = verified_pages = 0
        rebuilt = [{"pages": [], "complete": False} for _ in RANGES]
        prior = [None] * 67
        tokens = [set() for _ in RANGES]
        pending = []
        all_received_in_window = True
        for ordinal, attempt in enumerate(attempts):
            guard()
            require(type(attempt) is dict and attempt.get("id") == ordinal, "attempt_id")
            index, page_no, charge = attempt.get("range"), attempt.get("page"), attempt.get("charged_bytes")
            require(type(index) is int and 0 <= index < 67 and type(page_no) is int and page_no >= 0, "attempt_range_page")
            require(type(charge) is int and 0 <= charge <= 4194305, "attempt_charge")
            state = rebuilt[index]
            require(not state["complete"] and page_no == len(state["pages"]), "page_gap_repeat_or_after_terminal")
            expected_token = state["pages"][-1]["next_token"] if state["pages"] else None
            require(attempt.get("params") == request(index, expected_token), "request_params_or_token_chain")
            name = f"range_{index:03d}_request_{ordinal:03d}.json"
            require(attempt.get("raw") == name, "raw_filename")
            expected_names.add(name)
            expected_names.add(Path(name).with_suffix(".partial").name)
            status = attempt.get("status")
            require(status in ("reserved", "body_journaled", "raw_saved", "qualified"), "attempt_status")
            charged += charge
            require(charged <= 268435456, "cumulative_response_budget")
            raw = acquisition / name
            partial = raw.with_suffix(".partial")
            require(not (raw.exists() and partial.exists()), "duplicate_raw_and_partial")
            if status == "reserved":
                require(charge == 4194305 and not raw.exists(), "reserved_attempt_identity")
                if partial.exists():
                    size = partial.stat().st_size
                    require(0 <= size <= charge, "partial_exceeds_reservation")
                    raw_bytes += size
                pending.append(ordinal)
                continue
            size, body_hash = attempt.get("body_bytes"), attempt.get("body_sha256")
            require(type(size) is int and 0 <= size <= 4194304 and charge == size, "committed_size_charge")
            require(type(body_hash) is str and len(body_hash) == 64 and set(body_hash) <= set("0123456789abcdef"), "committed_sha_shape")
            path = raw if raw.exists() else partial
            require(path.is_file() and (raw.exists() or status == "body_journaled"), "committed_body_missing")
            require(path.stat().st_size == size, "raw_stat_size")
            body = path.read_bytes()
            require(len(body) == size and sha(body) == body_hash, "raw_bytes_or_sha_mismatch")
            raw_bytes += size
            received = datetime.fromisoformat(attempt["received_utc"])
            require(received.utcoffset() == timedelta(0), "received_timestamp_not_UTC")
            all_received_in_window &= start <= received.timestamp() < deadline
            meta, prior[index] = page_check(body, index, prior[index], guard)
            require(meta["next_token"] is None or meta["next_token"] not in tokens[index], "pagination_token_cycle")
            if meta["next_token"] is not None:
                tokens[index].add(meta["next_token"])
            meta["attempt_id"] = ordinal
            state["pages"].append(meta)
            state["complete"] = meta["next_token"] is None
            verified_pages += 1
            if status != "qualified":
                pending.append(ordinal)
        require(journal.get("requests_charged") == len(attempts) and journal.get("bytes_charged") == charged, "journal_budget_summary")
        require(raw_bytes <= charged and all_received_in_window, "actual_bytes_or_received_outside_window")
        for child in acquisition.iterdir():
            if child.name.startswith("range_"):
                require(child.name in expected_names and child.is_file(), "unaccounted_response_file")
        for index, state in enumerate(rebuilt):
            require(type(saved[index]) is dict, "range_journal_type")
            original_pages = saved[index].get("pages")
            qualified_pages = [p for p in state["pages"] if attempts[p["attempt_id"]]["status"] == "qualified"]
            require(original_pages == qualified_pages, "range_page_metadata_mismatch")
            require(saved[index].get("complete") is bool(qualified_pages and qualified_pages[-1]["next_token"] is None), "range_completion_flag")
        first = [a for a in attempts if a["range"] == 0]
        if journal.get("first_range_cost") is not None:
            cost = journal["first_range_cost"]
            elapsed = journal["first_range_completed_epoch"] - journal["first_range_started_epoch"]
            io = sum(a["io_wall_seconds"] for a in first)
            require(number(io) and io >= 0 and 0 < elapsed <= 600, "first_range_cost_time")
            require(cost.get("p0") == len(first) and cost.get("b0") == sum(a["charged_bytes"] for a in first) and cost.get("t0_seconds") == elapsed and cost.get("io_wall_seconds") == io, "first_range_cost_identity")
            gate = journal.get("continuation_gate_evaluation")
            if gate is not None:
                predicted_requests = math.ceil(66 * len(first) * 1.5)
                predicted_bytes = math.ceil(66 * sum(a["charged_bytes"] for a in first) * 1.5)
                predicted_wall = 1.5 * 66 * (io / 2 + max(0.0, elapsed - io))
                require(gate.get("cost_gate_sha256") == protocol["cost_gate_sha256"], "evaluated_gate_identity")
                require(gate.get("remaining_requests_predicted") == predicted_requests and gate.get("remaining_bytes_predicted") == predicted_bytes and gate.get("remaining_wall_seconds_predicted") == predicted_wall, "evaluated_gate_projection")
                first_count, first_bytes = len(first), sum(a["charged_bytes"] for a in first)
                accepted_at_gate = first_count + predicted_requests <= 300 and first_bytes + predicted_bytes + 2 * 4194304 <= 268435456 and gate["evaluated_epoch"] + predicted_wall + 60 <= deadline
                require(gate.get("accepted") is accepted_at_gate, "evaluated_gate_decision")
                require(accepted_at_gate or not any(a["range"] != 0 for a in attempts), "remaining_requested_after_gate_rejection")
                report["continuation_gate_accepted"] = accepted_at_gate
        guard()
        require((acquisition / "JOURNAL.json").read_bytes() == jbytes and protocol_path.read_bytes() == pbytes, "snapshot_changed_during_review")
        completed = [index for index, state in enumerate(rebuilt) if state["complete"]]
        all_qualified = len(completed) == 67 and not pending
        report.update(accepted_integrity=True, all67source_qualified=all_qualified,
                      status="all67_source_responses_integrity_accepted" if all_qualified else "partial_cache_integrity_accepted",
                      ranges_total=67, source_response_ranges_complete=len(completed), not_acquired_ranges=sum(not r["pages"] for r in rebuilt),
                      requests_recomputed=len(attempts), response_bytes_charged_recomputed=charged,
                      response_file_bytes_recomputed=raw_bytes, raw_pages_sha_and_schema_recomputed=verified_pages,
                      source_rows_recomputed=sum(p["rows"] for r in rebuilt for p in r["pages"]),
                      pending_attempts=pending, received_timestamps_within_original_window=all_received_in_window,
                      timestamp_cutoff_UTC=LIMIT.isoformat(),
                      per_range=[{"range": i, "dates_NY": RANGES[i], "pages": len(r["pages"]), "rows": sum(p["rows"] for p in r["pages"]), "source_response_complete": r["complete"]} for i, r in enumerate(rebuilt)],
                      limitations=["Request send times and runtime thread count/affinity are not independently observable from this cache journal.", "Source-response completeness does not establish original RTH or predictor-context coverage.", "Unacquired ranges have no source-availability inference."])
    except Exception as exc:
        report["status"] = "independent_review_not_accepted"
        report["errors"].append(str(exc) if isinstance(exc, Invalid) else type(exc).__name__)
    report["review_finished_epoch"] = time.time()
    report["review_within_original_deadline"] = deadline is not None and report["review_finished_epoch"] < deadline
    if not report["review_within_original_deadline"]:
        report["accepted_integrity"] = False
        report["all67source_qualified"] = False
    output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    return report

def self_test():
    passed = []
    stamp = "2021-01-04T14:30:00Z"
    bar = {"t": stamp, "o": 1, "h": 1, "l": 1, "c": 1, "v": 0, "n": 0, "vw": 1}
    def body(rows, token=None):
        return json.dumps({"bars": {"SPY": rows}, "next_page_token": token}).encode()
    meta, last = page_check(body([bar], "next"), 0, None, lambda: None)
    assert meta["rows"] == 1 and meta["next_token"] == "next"
    passed.append("independent_fixture_valid_page")
    cases = [(body([bar, bar]), "bar_duplicate_or_unordered"),
             (body([dict(bar, t="2026-07-01T04:00:00Z")]), "bar_range_or_July_cutoff"),
             (body([dict(bar, t="2021-01-04T14:30:00.000000001Z")]), "bar_timestamp_subminute"),
             (body([dict(bar, v=True)]), "bar_numeric_type_or_finiteness"),
             (body([dict(bar, l=2)]), "bar_OHLC_schema")]
    for data, reason in cases:
        try:
            page_check(data, 0, None, lambda: None)
        except Invalid as error:
            assert str(error) == reason
            passed.append(reason)
        else:
            raise AssertionError(reason)
    try:
        page_check(body([bar]), 0, last, lambda: None)
    except Invalid as error:
        assert str(error) == "bar_duplicate_or_unordered"
        passed.append("cross_page_duplicate")
    else:
        raise AssertionError("cross_page_duplicate")
    assert request(2, None)["start"] == "2021-03-05T05:00:00+00:00"
    assert request(2, None)["end"] == "2021-04-04T03:59:00+00:00"
    assert RANGES[-1] == ["2026-06-07", "2026-06-30"]
    passed.extend(["NY_DST_bounds", "67_fixed_ranges"])
    print(json.dumps({"manual_fixture_checks": len(passed), "passed": passed, "CPU_affinity": sorted(os.sched_getaffinity(0)), "real_cache_read": False}))

if __name__ == "__main__":
    os.sched_setaffinity(0, {26})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root-notified", action="store_true")
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.error("Need explicit --root-notified and --protocol") if not args.root_notified or not args.protocol else None
        base = Path(__file__).resolve().parent
        result = review(args.protocol, base / "acquisition", base / "download_spy_coverage.py", base / "COST_CONTINUATION_GATE.json", base / "INDEPENDENT_CACHE_RESULT_REVIEW.json")
        print(json.dumps({key: result.get(key) for key in ("status", "accepted_integrity", "all67source_qualified", "errors", "review_within_original_deadline")}))
        raise SystemExit(0 if result["accepted_integrity"] else 2)
