"""Independent compressed SPY result verifier; stdlib only; no adapter imports or HTTP."""
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
import re
import signal
import zlib
from zoneinfo import ZoneInfo

SOURCE_SHA = "51baf69a08427a1ca0f9ba7de3bf1e5444f439e81df89f1f93fb06dc9c713c94"
FIXED = {"symbols": "SPY", "timeframe": "1Min", "feed": "sip", "adjustment": "raw", "currency": "USD", "asof": "-", "sort": "asc", "limit": 10000}
BUDGET = {"window_seconds":3600,"global_GET":300,"prior_actual_GET":4,"payload_storage_bytes":268435456,"prior_payload_bytes":2127635,"wire_page_bytes":4194304,"canonical_page_bytes":4194304,"side_guard_bytes":1,"attempt_reserve_bytes":8388610,"workers":2,"CPU":24,"socket_seconds":15,"closing_reserve_seconds":2}
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


BASE = Path(__file__).resolve().parent
CAP = 4194304
SIDE = CAP + 1
MANIFEST_SHA = "3addf97adf1a53a0d7d4c25a03f6a8321158eac429d05021a0de21d82f5961b7"
GATE_SHA = "7cef35c94f41d534e4fe3d8bf303982fe44da6bc3b24ec441665725f3d7e01c6"
SUPERVISOR_SHA = "921c7ccd7c122c3b26a3670b44214cd94e5d1f4078d71712f26bf7437b1e0909"


def file_hash(path, expected_bytes, expected_sha, guard, cap=None):
    guard()
    require(path.is_file() and path.stat().st_size == expected_bytes, "file_size_or_absence")
    require(cap is None or expected_bytes <= cap, "file_cap")
    total = 0
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            guard()
            block = stream.read(65536)
            if not block:
                break
            total += len(block)
            require(cap is None or total <= cap, "stream_file_cap")
            hashed.update(block)
    require(total == expected_bytes and hashed.hexdigest() == expected_sha, "file_sha_or_length")


def gzip_check(wire, wire_meta, canonical, canonical_meta, guard):
    """Independent bounded decoder; canonical comparison is byte-for-byte, not only SHA."""
    require(wire.stat().st_size == wire_meta["bytes"] <= CAP, "gzip_wire_stat")
    decoder = zlib.decompressobj(31)
    encoded = decoded = 0
    wire_hash = hashlib.sha256()
    plain_hash = hashlib.sha256()
    plain = canonical.open("rb") if canonical is not None else None
    try:
        with wire.open("rb") as stream:
            while True:
                guard()
                block = stream.read(65536)
                if not block:
                    break
                encoded += len(block)
                require(encoded <= CAP, "gzip_wire_cap")
                wire_hash.update(block)
                require(not decoder.eof, "gzip_extra_member_or_trailing")
                pending = block
                while pending:
                    guard()
                    before = len(pending)
                    piece = decoder.decompress(pending, min(65536, CAP - decoded + 1))
                    decoded += len(piece)
                    require(decoded <= CAP, "gzip_decoded_cap")
                    plain_hash.update(piece)
                    if plain is not None:
                        require(plain.read(len(piece)) == piece, "canonical_not_exact_gzip_decode")
                    pending = decoder.unconsumed_tail
                    require(not decoder.unused_data, "gzip_extra_member_or_trailing")
                    if decoder.eof:
                        require(not pending, "gzip_trailing_unconsumed")
                        break
                    require(bool(piece) or len(pending) != before, "gzip_no_progress")
        require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail, "gzip_not_complete_single_member")
        require(encoded == wire_meta["bytes"] and wire_hash.hexdigest() == wire_meta["sha256"], "gzip_wire_identity")
        if canonical_meta is not None:
            require(decoded == canonical_meta["bytes"] and plain_hash.hexdigest() == canonical_meta["sha256"], "gzip_canonical_identity")
            require(plain is not None and plain.read(1) == b"", "canonical_extra_bytes")
        return {"encoded_bytes": encoded, "decoded_bytes": decoded, "decoded_sha256": plain_hash.hexdigest(), "CRC_EOF_single_member": True}
    finally:
        if plain is not None:
            plain.close()


def index_page(index, page, path, size, checksum, meta, token):
    return {"range": index, "page": page,
            "canonical_ref": {"path": str(path), "bytes": size, "sha256": checksum},
            "request_page_token": token, "next_page_token": meta["next_token"],
            "rows": meta["rows"], "first_UTC": meta["first_UTC"], "last_UTC": meta["last_UTC"]}


def supervisor_check(base, scope, protocol_hash, start, deadline, guard):
    guard()
    body = (base / (scope.upper() + "_SUPERVISOR.json")).read_bytes()
    receipt = decode(body)
    require(receipt.get("protocol_sha256") == protocol_hash and receipt.get("supervisor_sha256") == SUPERVISOR_SHA,
            "supervisor_identity")
    require(receipt.get("scope") == scope and receipt.get("status") == "terminal"
            and receipt.get("child_terminal") is True and receipt.get("deadline_reset") is False
            and receipt.get("deadline_epoch") == deadline and receipt.get("CPU_affinity") == [24], "supervisor_scope_or_terminal")
    lower, upper = receipt.get("started_epoch"), receipt.get("ended_epoch")
    require(number(lower) and number(upper) and start <= lower <= upper < deadline, "supervisor_original_window")
    require(type(receipt.get("PID")) is int and receipt["PID"] > 0 and type(receipt.get("exit_code")) is int, "supervisor_PID_or_exit")
    require(number(receipt.get("seconds")) and abs(receipt["seconds"] - (upper - lower)) < 0.001, "supervisor_duration_record")
    return receipt, sha(body)


def review(protocol_path, acquisition, output, scope, calibration_snapshot=None):
    require(not output.exists() and not output.with_suffix(output.suffix + ".tmp").exists(),
            "independent_output_exists_do_not_reread_cache")
    began = time.time()
    report = {"kind": "independent_compressed_SPY_source_result_review_v1", "scope": scope,
              "CPU_affinity": sorted(os.sched_getaffinity(0)), "review_started_epoch": began,
              "accepted_integrity": False, "all67source_qualified": False, "errors": [],
              "original_RTH_context_coverage": "not_tested", "uses_author_validator_or_decoder": False,
              "HTTP_calls": 0, "AAPL_labels_returns_models_Q3_read": False, "research_goal_achieved": False}
    deadline = None
    timer = False
    def guard():
        require(deadline is None or time.time() < deadline - 0.5, "same_original3600s_review_deadline")
    def stop(_sig, _frame):
        raise Invalid("same_original3600s_review_deadline")
    try:
        require(report["CPU_affinity"] == [26], "reviewer_CPU26")
        protocol_bytes = protocol_path.read_bytes()
        protocol = decode(protocol_bytes)
        start, deadline = protocol.get("started_epoch"), protocol.get("deadline_epoch")
        require(number(start) and number(deadline) and deadline - start == 3600 and start <= began, "protocol_original3600s")
        guard()
        signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, deadline - time.time() - 0.5)
        timer = True
        protocol_hash = sha(protocol_bytes)
        report.update(protocol_sha256=protocol_hash, original_started_epoch=start, original_deadline_epoch=deadline)
        require(protocol.get("kind") == "SPY_gzip_continuation_v1" and protocol.get("execution_admitted") is True, "protocol_admission")
        require(protocol.get("source_sha256") == SOURCE_SHA == sha((BASE / "download_compressed_spy.py").read_bytes()), "adapter_source_identity")
        require(protocol.get("inheritance_manifest_sha256") == MANIFEST_SHA == sha((BASE / "INHERITANCE_MANIFEST.json").read_bytes()), "manifest_identity")
        require(protocol.get("cost_gate_sha256") == GATE_SHA == sha((BASE / "COST_GATE.json").read_bytes()), "gate_identity")
        require(sha((BASE / "deadline_supervisor.py").read_bytes()) == SUPERVISOR_SHA, "supervisor_source_identity")
        dependencies = protocol.get("dependencies")
        require(type(dependencies) is dict and set(dependencies) == {"source", "transport"}, "dependency_set")
        expected_dep = {"source": "9cc0d1e2af800d4dfe33ac10db009d483f66f5e0912f2a4ca3f3953863c9b32a",
                        "transport": "f7eaf73d8e5a0849761ec7839e91fad099d5cdaa007a055ceb9d5f9c677d6dcc"}
        for name, checksum in expected_dep.items():
            require(dependencies[name].get("sha256") == checksum == sha(Path(dependencies[name]["path"]).read_bytes()), "locked_dependency_identity")
        require(protocol.get("endpoint") == "https://data.alpaca.markets/v2/stocks/bars"
                and protocol.get("fixed") == FIXED and protocol.get("ranges") == RANGES
                and protocol.get("budget") == BUDGET and protocol.get("Accept-Encoding") == "gzip", "protocol_scope_or_budget")
        require(all(protocol.get(k) is False for k in ("GPU", "Q3_allowed", "features_fits_scores_allowed"))
                and protocol.get("prior_window_closed_not_reset") is True, "protocol_forbidden_scope")
        supervisor, supervisor_hash = supervisor_check(BASE, scope, protocol_hash, start, deadline, guard)
        report.update(supervisor_sha256=supervisor_hash, primary_PID=supervisor["PID"], primary_exit_code=supervisor["exit_code"],
                      primary_terminal_before_original_deadline=True)
        journal_bytes = (acquisition / "JOURNAL.json").read_bytes()
        journal = decode(journal_bytes)
        require(journal.get("protocol_sha256") == protocol_hash and journal.get("started_epoch") == start
                and journal.get("deadline_epoch") == deadline and journal.get("sealed_snapshot") is True, "journal_binding_or_terminal")
        report.update(journal_sha256=sha(journal_bytes), journal_status=journal.get("status"))
        manifest = decode((BASE / "INHERITANCE_MANIFEST.json").read_bytes())
        require(manifest.get("prior_actual_GET") == 4 and manifest.get("prior_payload_bytes") == 2127635, "prior_budget_contract")
        entities = manifest.get("entities")
        require(type(entities) is list and len(entities) == len({e["path"] for e in entities}) == 4
                and sum(e["bytes"] for e in entities) == 2127635, "prior_entity_ledger")
        old_entity = {e["path"]: e for e in entities}
        for e in entities:
            file_hash(Path(e["path"]), e["bytes"], e["sha256"], guard)
        for e in manifest["evidence"]:
            guard()
            require(sha(Path(e["path"]).read_bytes()) == e["sha256"], "prior_acceptance_evidence_changed")
        rebuilt = [{"pages": [], "complete": False} for _ in RANGES]
        previous = [None] * 67
        seen_tokens = [set() for _ in RANGES]
        inherited = sorted(manifest["prefix_pages"], key=lambda x: x["page"])
        require(len(inherited) == 2, "inherited_two_pages")
        for ordinal, p in enumerate(inherited):
            require(p["page"] == ordinal and p["path"] in old_entity
                    and p["bytes"] == old_entity[p["path"]]["bytes"]
                    and p["sha256"] == old_entity[p["path"]]["sha256"], "inherited_page_identity")
            state = rebuilt[0]
            require(not state["complete"], "inherited_after_terminal")
            token = state["pages"][-1]["next_page_token"] if state["pages"] else None
            require(p["params"] == request(0, token), "inherited_request_token")
            meta, previous[0] = page_check(Path(p["path"]).read_bytes(), 0, previous[0], guard)
            state["pages"].append(index_page(0, ordinal, Path(p["path"]), p["bytes"], p["sha256"], meta, token))
            state["complete"] = meta["next_token"] is None
        require(rebuilt[0]["complete"], "inherited_range_terminal")
        attempts, saved = journal.get("attempts"), journal.get("ranges")
        require(type(attempts) is list and len(attempts) <= 296 and type(saved) is list and len(saved) == 67, "journal_structure")
        charged, actual, known_wire, known_canonical = 2127635, 2127635, 0, 0
        allowed = set()
        located = {}
        pending = []
        verified_gzip = 0
        for ordinal, a in enumerate(attempts):
            guard()
            require(type(a) is dict and a.get("id") == ordinal, "attempt_ID")
            index, page_no, status = a.get("range"), a.get("page"), a.get("status")
            require(type(index) is int and 1 <= index < 67 and type(page_no) is int and page_no >= 0, "attempt_range_page")
            require(status in ("reserved", "wire_committed", "canonical_committed", "qualified"), "attempt_status")
            require((status == "reserved") is (a.get("wire_meta") is None)
                    and (status in ("canonical_committed", "qualified")) is (a.get("canonical_meta") is not None), "attempt_transaction_state")
            for part, suffix in (("wire", "wire"), ("canonical", "canonical.json")):
                name = f"range_{index:03d}_attempt_{ordinal:03d}.{suffix}"
                require(a.get(part + "_name") == name, "attempt_filename")
                final = acquisition / name
                partial = final.with_suffix(final.suffix + ".partial")
                allowed.update((final.name, partial.name))
                require(not (final.exists() and partial.exists()), "duplicate_entity_copy")
                path = final if final.exists() else partial if partial.exists() else None
                meta, charge = a.get(part + "_meta"), a.get(part + "_charged")
                if meta is None:
                    require(type(charge) is int and charge == SIDE and not final.exists(), "unknown_side_reservation")
                    if path is not None:
                        require(path.is_file() and path.stat().st_size <= SIDE, "unknown_partial_guard_cap")
                    require(part != "canonical" or status != "reserved" or path is None, "canonical_before_wire_transaction")
                else:
                    require(type(meta) is dict and type(meta.get("bytes")) is int and 0 <= meta["bytes"] <= CAP
                            and type(charge) is int and charge == meta["bytes"]
                            and type(meta.get("sha256")) is str and re.fullmatch("[0-9a-f]{64}", meta["sha256"]), "known_side_meta_or_charge")
                    require(path is not None and (status != "qualified" or final.exists()), "committed_entity_missing")
                    if part == "wire":
                        headers = meta.get("headers")
                        require(type(headers) is dict and str(headers.get("content-encoding", "")).strip().lower() == "gzip", "committed_wire_encoding")
                        known_wire += charge
                    else:
                        known_canonical += charge
                charged += charge
                if path is not None:
                    actual += path.stat().st_size
                    require(path.stat().st_size <= charge, "actual_side_exceeds_charge")
                located[(ordinal, part)] = path
            require(charged <= 268435456, "cumulative_storage_cap")
            if a.get("wire_meta") is not None:
                canonical = located[(ordinal, "canonical")] if a.get("canonical_meta") is not None else None
                gzip_check(located[(ordinal, "wire")], a["wire_meta"], canonical, a.get("canonical_meta"), guard)
                verified_gzip += 1
            if status != "qualified":
                pending.append(ordinal)
        require(actual <= charged and 4 + len(attempts) <= 300
                and journal.get("global_GET_charged") == 4 + len(attempts)
                and journal.get("payload_storage_bytes_charged") == charged, "cumulative_budget_summary")
        for child in acquisition.iterdir():
            if ".wire" in child.name or ".canonical.json" in child.name or child.name.startswith("range_"):
                require(child.name in allowed and child.is_file(), "orphan_payload")
        for index in range(1, 67):
            state = rebuilt[index]
            entries = sorted((a for a in attempts if a["range"] == index), key=lambda x: x["page"])
            for a in entries:
                require(not state["complete"] and a["page"] == len(state["pages"]), "page_gap_repeat_or_after_terminal")
                token = state["pages"][-1]["next_page_token"] if state["pages"] else None
                require(a.get("params") == request(index, token), "fixed_request_or_page_token_chain")
                if a["status"] != "qualified":
                    require(a is entries[-1], "pending_attempt_followed_by_page")
                    continue
                path = located[(a["id"], "canonical")]
                meta, previous[index] = page_check(path.read_bytes(), index, previous[index], guard)
                require(meta["next_token"] is None or meta["next_token"] not in seen_tokens[index], "page_token_cycle")
                if meta["next_token"] is not None:
                    seen_tokens[index].add(meta["next_token"])
                cm = a["canonical_meta"]
                state["pages"].append(index_page(index, a["page"], path, cm["bytes"], cm["sha256"], meta, token))
                state["complete"] = meta["next_token"] is None
        for i, state in enumerate(rebuilt):
            require(type(saved[i]) is dict and saved[i].get("pages") == state["pages"]
                    and saved[i].get("complete") is state["complete"], "saved_range_metadata_or_completion")
        calibration = [a for a in attempts if a["range"] == 1]
        cost = journal.get("calibration_cost")
        if cost is not None:
            require(rebuilt[1]["complete"] and all(a["status"] == "qualified" for a in calibration), "calibration_qualification")
            ios = [a.get("io_wall_seconds") for a in calibration]
            require(all(number(x) and x >= 0 for x in ios), "calibration_individual_IO_time")
            total = journal.get("calibration_completed_epoch")
            require(number(total) and start <= total < deadline, "calibration_completion_time")
            total -= start
            io = sum(ios)
            b1 = sum(a["wire_charged"] + a["canonical_charged"] for a in calibration)
            require(cost == {"p1": len(calibration), "b1": b1, "t1_seconds": total, "io_wall_seconds": io}, "frozen_calibration_cost")
            report["calibration_cost_verified"] = cost
            gate = journal.get("continuation_gate")
            if gate is not None:
                req, size = math.ceil(65 * len(calibration) * 1.5), math.ceil(65 * b1 * 1.5)
                serial = max(0.0, total - io)
                wall = 65 * 1.5 * (io / 2 + serial)
                when = gate.get("evaluated_epoch")
                require(number(when) and start <= when < deadline and gate.get("cost_gate_sha256") == GATE_SHA
                        and gate.get("remaining_GET_predicted") == req and gate.get("remaining_storage_bytes_predicted") == size
                        and gate.get("remaining_wall_seconds_predicted") == wall and gate.get("serial_seconds") == serial
                        and gate.get("calibration_cost") == cost and gate.get("not_completion_guarantee") is True, "cost_gate_projection")
                require(calibration_snapshot is not None, "gate_requires_frozen_calibration_snapshot")
                snapshot_body = calibration_snapshot.read_bytes()
                snapshot = decode(snapshot_body)
                require(snapshot.get("protocol_sha256") == protocol_hash and snapshot.get("sealed_snapshot") is True
                        and snapshot.get("calibration_cost") == cost and all(a["range"] == 1 for a in snapshot["attempts"])
                        and snapshot["attempts"] == calibration, "calibration_snapshot_identity")
                base_req = 4 + len(snapshot["attempts"])
                base_storage = 2127635 + sum(a["wire_charged"] + a["canonical_charged"] for a in snapshot["attempts"])
                expected_accept = base_req + req <= 300 and base_storage + size + 16777220 <= 268435456 and when + wall + 60 <= deadline
                require(gate.get("accepted") is expected_accept, "gate_first_remaining_decision")
                require(expected_accept or not any(a["range"] >= 2 for a in attempts), "requests_after_rejected_gate")
                report.update(continuation_gate_accepted=expected_accept, calibration_snapshot_sha256=sha(snapshot_body),
                              gate_cumulative_base_GET=base_req, gate_cumulative_base_storage=base_storage)
        if scope == "calibration":
            require(all(a["range"] == 1 for a in attempts), "remaining_started_during_calibration_review")
        else:
            require(cost is not None and journal.get("continuation_gate") is not None, "remaining_requires_calibration_and_gate")
            first_supervisor, first_supervisor_hash = supervisor_check(BASE, "calibration", protocol_hash, start, deadline, guard)
            require(first_supervisor["PID"] != supervisor["PID"] and first_supervisor["ended_epoch"] <= supervisor["started_epoch"], "distinct_sequential_real_PIDs")
            report.update(real_distinct_calibration_remaining_PIDs=True, calibration_supervisor_sha256=first_supervisor_hash)
        all_complete = all(s["complete"] for s in rebuilt) and not pending
        index_path = acquisition / "CANONICAL_INDEX.json"
        if all_complete:
            expected_index = {"kind": "SPY_qualified_canonical_index_v1", "protocol_sha256": protocol_hash,
                              "inheritance_manifest_sha256": MANIFEST_SHA, "all_67_ranges_complete": True,
                              "ranges": RANGES, "pages": [p for r in rebuilt for p in r["pages"]],
                              "source_only_not_RTH_or_context_admission": True, "Q3_allowed": False}
            index_body = index_path.read_bytes()
            require(decode(index_body) == expected_index, "canonical_index_metadata")
            report["canonical_index_sha256"] = sha(index_body)
        else:
            require(not index_path.exists(), "incomplete_stage_has_full_index")
        guard()
        require((acquisition / "JOURNAL.json").read_bytes() == journal_bytes
                and protocol_path.read_bytes() == protocol_bytes, "snapshot_changed_during_review")
        report.update(accepted_integrity=True, all67source_qualified=all_complete,
                      status="all67_source_responses_integrity_accepted" if all_complete else "partial_source_cache_integrity_accepted",
                      prior_actual_GET=4, cumulative_GET_recomputed=4 + len(attempts), new_attempts=len(attempts),
                      cumulative_storage_charged=charged, cumulative_entity_file_bytes=actual,
                      new_committed_wire_bytes=known_wire, new_committed_canonical_bytes=known_canonical,
                      independently_verified_gzip_pages=verified_gzip, pending_attempts=pending,
                      source_response_ranges_complete=sum(r["complete"] for r in rebuilt),
                      not_acquired_ranges=sum(not r["pages"] for r in rebuilt),
                      source_rows_recomputed=sum(p["rows"] for r in rebuilt for p in r["pages"]),
                      per_range=[{"range": i, "dates_NY": RANGES[i], "pages": len(r["pages"]),
                                  "rows": sum(p["rows"] for p in r["pages"]), "complete": r["complete"]} for i, r in enumerate(rebuilt)],
                      limitations=["No per-attempt send/receipt timestamps exist in this journal; primary process-window evidence and static guards do not independently prove each GET time.",
                                   "First remaining gate base comes from preserved calibration snapshot and explicit sequential process receipts; no later resumed-gate base is inferred.",
                                   "Recorded CPU affinity and worker cap are not an independent runtime trace.",
                                   "Source-response qualification is separate from original RTH/origin coverage, historical PIT receipt proof, added information or prediction success."])
    except Exception as exc:
        report["status"] = "independent_review_not_accepted"
        report["errors"].append(str(exc) if isinstance(exc, Invalid) else type(exc).__name__)
    finally:
        if timer:
            signal.setitimer(signal.ITIMER_REAL, 0)
    report["review_finished_epoch"] = time.time()
    report["review_within_original_deadline"] = number(deadline) and report["review_finished_epoch"] < deadline
    if not report["review_within_original_deadline"]:
        report["accepted_integrity"] = report["all67source_qualified"] = False
    require(not output.exists(), "independent_output_exists_do_not_overwrite")
    temp = output.with_suffix(output.suffix + ".tmp")
    with temp.open("x") as stream:
        json.dump(report, stream, separators=(",", ":"), allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temp.replace(output)
    return report


def self_test():
    import gzip
    checks = []
    def good(name, condition):
        require(condition, "selftest_" + name)
        checks.append(name)
    def bad(name, call):
        try:
            call()
        except (Invalid, zlib.error):
            checks.append(name)
        else:
            raise Invalid("selftest_missing_rejection_" + name)
    plain = json.dumps({"artificial": "x" * 10000}).encode()
    packed = gzip.compress(plain, mtime=0)
    with tempfile.TemporaryDirectory(prefix="independent_compressed_verifier_") as root:
        root = Path(root)
        wire, canon = root / "fake.wire", root / "fake.canonical"
        def check(data, expected=plain):
            wire.write_bytes(data); canon.write_bytes(expected)
            return gzip_check(wire, {"bytes": len(data), "sha256": sha(data)}, canon,
                              {"bytes": len(expected), "sha256": sha(expected)}, lambda: None)
        def wire_only(data):
            wire.write_bytes(data)
            return gzip_check(wire, {"bytes": len(data), "sha256": sha(data)}, None, None, lambda: None)
        good("valid_single_member_exact_canonical", check(packed)["decoded_bytes"] == len(plain))
        bad("canonical_byte_difference", lambda: check(packed, plain[:-1] + b"z"))
        bad("second_gzip_member", lambda: check(packed + packed))
        bad("trailing_zero", lambda: check(packed + b"\0"))
        bad("truncated_footer", lambda: check(packed[:-1]))
        broken = bytearray(packed); broken[-8] ^= 1
        bad("CRC_mismatch", lambda: check(bytes(broken)))
        broken = bytearray(packed); broken[-4] ^= 1
        bad("ISIZE_mismatch", lambda: check(bytes(broken)))
        exact = b"x" * CAP
        good("exact_decoded_cap", check(gzip.compress(exact, mtime=0), exact)["decoded_bytes"] == CAP)
        bad("decoded_cap_guard", lambda: check(gzip.compress(exact + b"x", mtime=0), exact + b"x"))
        bad("zipbomb_guard", lambda: wire_only(gzip.compress(exact * 4, mtime=0)))
        bad("encoded_cap_guard", lambda: wire_only(b"x" * (CAP + 1)))
        bar = {"t": "2021-02-03T05:00:00Z", "o": 1, "h": 1, "l": 1, "c": 1, "v": 0, "n": 0, "vw": 1}
        def body(rows):
            return json.dumps({"bars": {"SPY": rows}, "next_page_token": None}).encode()
        meta, last = page_check(body([bar]), 1, None, lambda: None)
        good("independent_schema_valid", meta["rows"] == 1)
        bad("duplicate_time", lambda: page_check(body([bar, bar]), 1, None, lambda: None))
        bad("cross_page_time", lambda: page_check(body([bar]), 1, last, lambda: None))
        bad("count_bool", lambda: page_check(body([dict(bar, v=True)]), 1, None, lambda: None))
        bad("OHLC", lambda: page_check(body([dict(bar, l=2)]), 1, None, lambda: None))
        bad("July_cutoff", lambda: page_check(body([dict(bar, t="2026-07-01T04:00:00Z")]), 66, None, lambda: None))
        bad("nonzero_subminute", lambda: page_check(body([dict(bar, t="2021-02-03T05:00:00.000000001Z")]), 1, None, lambda: None))
        bad("duplicate_JSON_key", lambda: decode(b'{"x":1,"x":2}'))
        bad("nonfinite_JSON", lambda: decode(b'{"x":NaN}'))
        good("fixed67_and_DST", len(RANGES) == 67 and request(2, None)["end"] == "2021-04-04T03:59:00+00:00")
        good("prior_and_dual_budget", BUDGET["prior_actual_GET"] == 4 and BUDGET["prior_payload_bytes"] == 2127635 and 2 * SIDE == 8388610)
    receipt = {"kind": "independent_verifier_artificial_selftest_v1", "checks_passed": len(checks), "checks": checks,
               "CPU_affinity": sorted(os.sched_getaffinity(0)), "actual_raw_read": False, "HTTP_calls": 0,
               "author_fixtures_run": False, "verifier_sha256": sha(Path(__file__).read_bytes())}
    output = BASE / "INDEPENDENT_VERIFIER_SELFTEST.json"
    require(not output.exists(), "selftest_receipt_exists")
    output.write_text(json.dumps(receipt, separators=(",", ":")) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    os.sched_setaffinity(0, {26})
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--root-notified", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--scope", choices=("calibration", "remaining"))
    parser.add_argument("--calibration-snapshot", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if not args.root_notified or not args.protocol or not args.scope or not args.output:
            parser.error("Explicit --root-notified --protocol --scope --output required; no cache read")
        result = review(args.protocol, BASE / "acquisition", args.output, args.scope, args.calibration_snapshot)
        print(json.dumps({k: result.get(k) for k in ("status", "accepted_integrity", "all67source_qualified", "errors", "review_within_original_deadline")}))
        raise SystemExit(0 if result["accepted_integrity"] else 2)

