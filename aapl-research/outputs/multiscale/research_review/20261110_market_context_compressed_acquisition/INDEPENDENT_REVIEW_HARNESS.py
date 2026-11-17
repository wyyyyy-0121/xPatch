"""Artificial end-to-end aggregation checks for independent verifier; no actual raw/HTTP."""
from pathlib import Path
import copy
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib.util
import json
import os
import tempfile
import time

BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("independent_verifier_under_test", BASE / "INDEPENDENT_SOURCE_VERIFIER.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
assert os.sched_getaffinity(0) == {26}
checks = []


def put(path, data):
    path.write_bytes(data)
    return {"path": str(path), "bytes": len(data), "sha256": m.sha(data)}


def dump(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")))


def constant_page(index, minute=0, token=None):
    stamp = m.boundary(index)[0] + timedelta(minutes=minute)
    bar = {"t": stamp.isoformat(), "o": 1.0, "h": 1.0, "l": 1.0, "c": 1.0, "v": 0, "n": 0, "vw": 1.0}
    body = json.dumps({"bars": {"SPY": [bar]}, "next_page_token": token}, separators=(",", ":")).encode()
    meta = {"rows": 1, "first_UTC": stamp.isoformat(), "last_UTC": stamp.isoformat(), "next_token": token}
    return body, meta


with tempfile.TemporaryDirectory(prefix="independent_end_to_end_") as temporary:
    root = Path(temporary)
    acquisition = root / "acquisition"
    inherited = root / "inherited"
    acquisition.mkdir(); inherited.mkdir()
    # Copy code bytes only for identity checks, without importing author code.
    for name in ("download_compressed_spy.py", "deadline_supervisor.py"):
        (root / name).write_bytes((BASE / name).read_bytes())
    m.BASE = root
    entities, prefix = [], []
    inherited_index = []
    for page in range(2):
        token = "artificial-next" if page == 0 else None
        body, meta = constant_page(0, page, token)
        ref = put(inherited / (str(page) + ".json"), body)
        entities.append(ref)
        request_token = None if page == 0 else "artificial-next"
        prefix.append(dict(ref, page=page, params=m.request(0, request_token)))
        inherited_index.append(m.index_page(0, page, Path(ref["path"]), ref["bytes"], ref["sha256"], meta, request_token))
    entities.append(put(inherited / "old_fake_probe.wire", gzip.compress(constant_page(0)[0], mtime=0)))
    filler = 2127635 - sum(e["bytes"] for e in entities)
    entities.append(put(inherited / "artificial_filler.bin", b"0" * filler))
    evidence_path = root / "old_acceptance.json"
    dump(evidence_path, {"artificial": True})
    manifest = {"prior_actual_GET": 4, "prior_payload_bytes": 2127635, "entities": entities,
                "prefix_pages": prefix, "evidence": [{"path": str(evidence_path), "sha256": m.sha(evidence_path.read_bytes())}]}
    manifest_path = root / "INHERITANCE_MANIFEST.json"
    dump(manifest_path, manifest)
    m.MANIFEST_SHA = m.sha(manifest_path.read_bytes())
    gate_path = root / "COST_GATE.json"
    dump(gate_path, {"artificial": True})
    m.GATE_SHA = m.sha(gate_path.read_bytes())
    start = time.time() - 30
    protocol = {"kind": "SPY_gzip_continuation_v1", "execution_admitted": True, "source_sha256": m.SOURCE_SHA,
                "inheritance_manifest_sha256": m.MANIFEST_SHA, "cost_gate_sha256": m.GATE_SHA,
                "dependencies": {
                    "source": {"path": str(BASE.parent / "20261110_market_context_coverage/download_spy_coverage.py"), "sha256": "9cc0d1e2af800d4dfe33ac10db009d483f66f5e0912f2a4ca3f3953863c9b32a"},
                    "transport": {"path": str(BASE.parent / "20261110_market_context_coverage/transport_compression/probe_transport_gzip.py"), "sha256": "f7eaf73d8e5a0849761ec7839e91fad099d5cdaa007a055ceb9d5f9c677d6dcc"}},
                "endpoint": "https://data.alpaca.markets/v2/stocks/bars", "fixed": m.FIXED, "ranges": m.RANGES,
                "budget": m.BUDGET, "Accept-Encoding": "gzip", "started_epoch": start, "deadline_epoch": start + 3600,
                "GPU": False, "Q3_allowed": False, "features_fits_scores_allowed": False, "prior_window_closed_not_reset": True}
    protocol_path = root / "protocol.json"
    dump(protocol_path, protocol)
    protocol_hash = m.sha(protocol_path.read_bytes())

    def supervisor(scope, pid, lower, upper):
        dump(root / (scope.upper() + "_SUPERVISOR.json"), {"protocol_sha256": protocol_hash,
             "supervisor_sha256": m.SUPERVISOR_SHA, "scope": scope, "status": "terminal", "child_terminal": True,
             "deadline_reset": False, "deadline_epoch": start + 3600, "CPU_affinity": [24], "started_epoch": lower,
             "ended_epoch": upper, "seconds": upper - lower, "PID": pid, "exit_code": 0})
    supervisor("calibration", 700001, start + 1, start + 10)
    journal = {"protocol_sha256": protocol_hash, "started_epoch": start, "deadline_epoch": start + 3600,
               "sealed_snapshot": True, "status": "calibration_complete_awaiting_root_gate", "attempts": [],
               "ranges": [{"pages": [], "complete": False} for _ in range(67)]}
    journal["ranges"][0] = {"pages": inherited_index, "complete": True, "inherited": True}

    def add_range(index):
        ordinal = len(journal["attempts"])
        body, meta = constant_page(index)
        encoded = gzip.compress(body, mtime=0)
        wire_name = f"range_{index:03d}_attempt_{ordinal:03d}.wire"
        canonical_name = f"range_{index:03d}_attempt_{ordinal:03d}.canonical.json"
        wire_ref = put(acquisition / wire_name, encoded)
        canonical_ref = put(acquisition / canonical_name, body)
        attempt = {"id": ordinal, "range": index, "page": 0, "status": "qualified", "params": m.request(index, None),
                   "wire_name": wire_name, "canonical_name": canonical_name,
                   "wire_meta": {"bytes": len(encoded), "sha256": m.sha(encoded), "headers": {"content-encoding": "gzip"}},
                   "canonical_meta": {"bytes": len(body), "sha256": m.sha(body)},
                   "wire_charged": len(encoded), "canonical_charged": len(body), "io_wall_seconds": 0.05}
        journal["attempts"].append(attempt)
        journal["ranges"][index] = {"pages": [m.index_page(index, 0, acquisition / canonical_name, len(body), m.sha(body), meta, None)], "complete": True}

    def save():
        journal["global_GET_charged"] = 4 + len(journal["attempts"])
        journal["payload_storage_bytes_charged"] = 2127635 + sum(a["wire_charged"] + a["canonical_charged"] for a in journal["attempts"])
        dump(acquisition / "JOURNAL.json", journal)

    add_range(1)
    first = journal["attempts"][0]
    b1 = first["wire_charged"] + first["canonical_charged"]
    cost = {"p1": 1, "b1": b1, "t1_seconds": 10.0, "io_wall_seconds": 0.05}
    journal.update(calibration_completed_epoch=start + 10, calibration_cost=cost)
    save()
    calibration_snapshot = root / "calibration_snapshot.json"
    calibration_snapshot.write_bytes((acquisition / "JOURNAL.json").read_bytes())
    result = m.review(protocol_path, acquisition, root / "calibration_result.json", "calibration")
    assert result["accepted_integrity"] and not result["all67source_qualified"] and result["source_response_ranges_complete"] == 2, result
    checks.append("synthetic_calibration_end_to_end")
    for index in range(2, 67):
        add_range(index)
    journal["status"] = "all_ranges_downloaded_field_validated"
    journal["continuation_gate"] = {"accepted": True, "cost_gate_sha256": m.GATE_SHA,
        "remaining_GET_predicted": 98, "remaining_storage_bytes_predicted": m.math.ceil(65 * b1 * 1.5),
        "remaining_wall_seconds_predicted": 65 * 1.5 * (0.05 / 2 + 9.95), "serial_seconds": 9.95,
        "calibration_cost": copy.deepcopy(cost), "evaluated_epoch": start + 11, "not_completion_guarantee": True}
    index = {"kind": "SPY_qualified_canonical_index_v1", "protocol_sha256": protocol_hash,
             "inheritance_manifest_sha256": m.MANIFEST_SHA, "all_67_ranges_complete": True, "ranges": m.RANGES,
             "pages": [p for r in journal["ranges"] for p in r["pages"]], "source_only_not_RTH_or_context_admission": True, "Q3_allowed": False}
    dump(acquisition / "CANONICAL_INDEX.json", index)
    supervisor("remaining", 700002, start + 11, start + 20)
    save()
    result = m.review(protocol_path, acquisition, root / "full_result.json", "remaining", calibration_snapshot)
    assert result["accepted_integrity"] and result["all67source_qualified"] and result["cumulative_GET_recomputed"] == 70, result
    checks.append("synthetic_all67_and_cumulative_ledgers")
    good_journal = copy.deepcopy(journal)
    def reject(name, mutate, reason):
        global journal
        journal = copy.deepcopy(good_journal)
        mutate()
        save()
        result = m.review(protocol_path, acquisition, root / (name + ".json"), "remaining", calibration_snapshot)
        assert result["accepted_integrity"] is False and reason in result["errors"], result
        checks.append(name)
    reject("wire_charge_low", lambda: journal["attempts"][1].update(wire_charged=0), "known_side_meta_or_charge")
    reject("bad_next_request", lambda: journal["attempts"][1]["params"].update(page_token="wrong"), "fixed_request_or_page_token_chain")
    reject("saved_range_flag", lambda: journal["ranges"][2].update(complete=False), "saved_range_metadata_or_completion")
    reject("gate_projection_wrong", lambda: journal["continuation_gate"].update(remaining_GET_predicted=97), "cost_gate_projection")
    reject("expired_primary_receipt", lambda: supervisor("remaining", 700002, start + 11, start + 3601), "supervisor_original_window")
    supervisor("remaining", 700002, start + 11, start + 20)
    journal = copy.deepcopy(good_journal); save()
    wire = acquisition / journal["attempts"][1]["wire_name"]
    original = wire.read_bytes(); damaged = bytearray(original); damaged[-8] ^= 1; wire.write_bytes(damaged)
    result = m.review(protocol_path, acquisition, root / "bad_wire_crc.json", "remaining", calibration_snapshot)
    assert not result["accepted_integrity"] and "error" in result["errors"], result
    checks.append("whole_review_CRC_rejection")
    wire.write_bytes(original)
    canonical = acquisition / journal["attempts"][1]["canonical_name"]
    original = canonical.read_bytes(); canonical.write_bytes(original[:-1] + b" ")
    result = m.review(protocol_path, acquisition, root / "bad_canonical.json", "remaining", calibration_snapshot)
    assert not result["accepted_integrity"] and "canonical_not_exact_gzip_decode" in result["errors"], result
    checks.append("whole_review_canonical_byte_rejection")
    canonical.write_bytes(original)
    try:
        m.review(protocol_path, acquisition, root / "full_result.json", "remaining", calibration_snapshot)
    except m.Invalid as exc:
        assert str(exc) == "independent_output_exists_do_not_reread_cache"
    else:
        raise AssertionError("duplicate output should stop before cache read")
    checks.append("existing_output_stops_before_cache_reread")

receipt = {"kind": "independent_verifier_artificial_end_to_end_v1", "checks_passed": len(checks), "checks": checks,
           "CPU_affinity": sorted(os.sched_getaffinity(0)), "actual_raw_read": False, "HTTP_calls": 0,
           "author_fixtures_run": False, "actual_processes_created": False,
           "verifier_sha256": m.sha((BASE / "INDEPENDENT_SOURCE_VERIFIER.py").read_bytes()),
           "harness_sha256": m.sha(Path(__file__).read_bytes())}
output = BASE / "INDEPENDENT_END_TO_END_FIXTURE.json"
assert not output.exists()
output.write_text(json.dumps(receipt, separators=(",", ":")) + "\n")
print(json.dumps(receipt))
