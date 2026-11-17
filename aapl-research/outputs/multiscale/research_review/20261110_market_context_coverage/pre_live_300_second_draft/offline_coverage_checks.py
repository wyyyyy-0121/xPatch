"""Source-only artificial fixtures; never reads credentials or opens a network."""
from datetime import timedelta
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import sys
import tempfile
import threading
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "download_spy_coverage.py"
spec = importlib.util.spec_from_file_location("coverage_offline", SOURCE)
m = importlib.util.module_from_spec(spec)
sys.dont_write_bytecode = True
spec.loader.exec_module(m)
checks, forbidden = [], []


def check(name, passed):
    checks.append({"name": name, "passed": bool(passed)})
    if not passed:
        raise AssertionError(name)


def blocked(*args, **kwargs):
    forbidden.append(True)
    raise AssertionError("Real network or credential reader forbidden in offline checks")


def body(index=0, offset=0, token=None):
    stamp = m.bounds(index)[0] + timedelta(hours=9, minutes=30 + offset)
    bars = [{"t": (stamp + timedelta(minutes=i)).isoformat(), "o": 100.,
             "h": 101., "l": 99., "c": 100., "v": 10, "n": 2, "vw": 100.}
            for i in range(2)]
    return json.dumps({"bars": {"SPY": bars}, "next_page_token": token,
                       "currency": "USD"}, allow_nan=False).encode()


class Response:
    def __init__(self, data, url):
        self.data, self.url, self.cursor = data, url, 0
        self.status, self.headers = 200, {"Content-Type": "application/json"}
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, length):
        result = self.data[self.cursor:self.cursor + length]
        self.cursor += len(result)
        return result


class Opener:
    def __init__(self, data): self.data = data
    def open(self, request, timeout):
        check("fake_request_socket_bound", timeout == 15)
        return Response(self.data, request.full_url)


def main():
    source_hash = m.digest(SOURCE.read_bytes())
    gate = json.loads((HERE / "COST_CONTINUATION_GATE.json").read_bytes())
    with tempfile.TemporaryDirectory(prefix="offline_coverage_", dir=HERE) as temp:
        base = Path(temp)
        (base / "COST_CONTINUATION_GATE.json").write_text(json.dumps(gate))
        counter = [0]
        def acquire():
            counter[0] += 1
            root = base / str(counter[0]); root.mkdir()
            protocol = m.template()
            protocol.update(execution_admitted=True, started_epoch=1000., deadline_epoch=1300.)
            path = root / "PROTOCOL.json"; path.write_text(json.dumps(protocol))
            return m.Acquisition(path, root / "data", clock=lambda: 1001.)
        def error(name, action):
            try: action()
            except Exception: check(name, True)
            else: check(name, False)
        def fake_fetch(acq, attempt, data):
            with patch.object(m, "build_opener", lambda *args: Opener(data)):
                acq.fetch(attempt, ("synthetic-only", "synthetic-only"))
        with patch.object(m, "BASE", base), patch.object(m, "credentials", blocked), \
             patch.object(m, "build_opener", blocked), \
             patch.object(m.os, "sched_setaffinity", lambda *args: None):
            check("67_fixed_ranges_and_cutoff", len(m.RANGES) == 67 and m.RANGES[0] == ["2021-01-04", "2021-02-02"]
                  and m.RANGES[-1] == ["2026-06-07", "2026-06-30"]
                  and m.parameters(66)["end"] == "2026-07-01T03:59:00+00:00")
            check("synthetic_page_valid", m.validate_page(body(), 0)["rows"] == 2)
            for name, edit in [
                ("nonzero_nanos_rejected", lambda x: x["bars"]["SPY"][0].__setitem__("t", "2021-01-04T14:30:00.000000001Z")),
                ("missing_token_rejected", lambda x: x.pop("next_page_token")),
                ("empty_token_rejected", lambda x: x.__setitem__("next_page_token", "")),
                ("AAPL_member_rejected", lambda x: x["bars"].__setitem__("AAPL", [])),
                ("wrong_currency_rejected", lambda x: x.__setitem__("currency", "EUR")),
                ("duplicate_timestamp_rejected", lambda x: x["bars"]["SPY"][1].__setitem__("t", x["bars"]["SPY"][0]["t"])),
                ("range_overrun_rejected", lambda x: x["bars"]["SPY"][1].__setitem__("t", "2021-02-03T05:00:00Z")),
                ("fractional_volume_rejected", lambda x: x["bars"]["SPY"][0].__setitem__("v", 1.5)),
                ("invalid_OHLC_rejected", lambda x: x["bars"]["SPY"][0].__setitem__("h", 99.5)),
            ]:
                x = json.loads(body()); edit(x)
                error(name, lambda x=x: m.validate_page(json.dumps(x).encode(), 0))
            x = json.loads(body(66)); x["bars"]["SPY"][1]["t"] = "2026-07-01T04:00:00Z"
            error("Q3_boundary_rejected", lambda: m.validate_page(json.dumps(x).encode(), 66))

            acq = acquire(); first = acq.reserve(0, None)
            fake_fetch(acq, first, body(token="page2")); acq.qualify(first)
            check("successful_page_settles_reservation_to_actual_bytes", first["charged_bytes"] == len(body(token="page2")) < m.RESERVATION)
            second = acq.reserve(0, "page2")
            fake_fetch(acq, second, body(offset=2)); acq.qualify(second)
            check("two_page_terminal_chain", acq.state["ranges"][0]["complete"] and len(acq.state["ranges"][0]["pages"]) == 2)
            restored = m.Acquisition(acq.protocol_path, acq.root, clock=lambda: 1001.)
            restored.recover()
            check("completed_pages_recover_without_GET", restored.state["requests_charged"] == 2 and restored.state["ranges"][0]["complete"])
            error("terminal_range_rejects_another_GET_reservation", lambda: restored.reserve(0, None))

            acq = acquire(); attempt = acq.reserve(0, None)
            with patch.object(m.time, "monotonic", side_effect=[100., 101., 200., 202., 300., 303.]):
                fake_fetch(acq, attempt, body())
            check("IO_timing_sums_only_open_and_reads", attempt["io_wall_seconds"] == 6.)

            acq = acquire(); attempt = acq.reserve(0, None)
            fake_fetch(acq, attempt, body())
            restored = m.Acquisition(acq.protocol_path, acq.root, clock=lambda: 1001.)
            restored.recover()
            check("raw_saved_before_qualification_recovers", restored.state["attempts"][0]["status"] == "qualified")

            acq = acquire(); attempt = acq.reserve(0, None); data = body()
            partial = (acq.root / attempt["raw"]).with_suffix(".partial"); partial.write_bytes(data)
            attempt.update(status="body_journaled", body_bytes=len(data), body_sha256=m.digest(data),
                           charged_bytes=len(data), io_wall_seconds=.01); acq._save()
            restored = m.Acquisition(acq.protocol_path, acq.root, clock=lambda: 1001.)
            restored.recover()
            check("precommitted_complete_temp_recovers", not partial.exists() and restored.state["ranges"][0]["complete"])

            acq = acquire(); attempt = acq.reserve(0, None)
            fake_fetch(acq, attempt, body())
            (acq.root / attempt["raw"]).write_bytes(body() + b" ")
            restored = m.Acquisition(acq.protocol_path, acq.root, clock=lambda: 1001.)
            error("corrupt_completed_page_rejected", restored.recover)

            acq = acquire(); attempt = acq.reserve(0, None)
            partial = (acq.root / attempt["raw"]).with_suffix(".partial"); partial.write_bytes(b'{"bars":')
            restored = m.Acquisition(acq.protocol_path, acq.root, clock=lambda: 1001.)
            error("uncertain_partial_not_retried", restored.recover)
            check("uncertain_attempt_keeps_worst_case_charge", restored.state["bytes_charged"] == m.RESERVATION and partial.exists())

            acq = acquire(); attempt = acq.reserve(0, None)
            error("oversize_response_rejected", lambda: fake_fetch(acq, attempt, b"x" * m.RESERVATION))
            check("oversize_failed_request_charge_preserved", acq.state["requests_charged"] == 1 and acq.state["bytes_charged"] == m.RESERVATION)

            acq = acquire(); failures = []
            def reserve_many(index):
                for _ in range(50):
                    try: acq.reserve(index, None)
                    except m.Halt: failures.append(True); return
            ts = [threading.Thread(target=reserve_many, args=(i,)) for i in (0, 1)]
            for t in ts: t.start()
            for t in ts: t.join()
            attempts = acq.state["attempts"]
            check("two_threads_budget_atomic_unique_ids", failures and len(attempts) == len({a["id"] for a in attempts})
                  and len(attempts) <= 300 and sum(a["charged_bytes"] for a in attempts) <= m.BUDGET["bytes"])
            check("byte_cap_stops_before_overallocation", (len(attempts) + 1) * m.RESERVATION > m.BUDGET["bytes"])
            acq = acquire()
            # Isolate the request-count boundary from the independently tested byte cap.
            acq.state["attempts"] = [{"charged_bytes": 0} for _ in range(300)]
            error("300_request_counter_boundary_rejected", lambda: acq.reserve(0, None))

            acq = acquire(); acq.clock = lambda: 1301.
            error("expired_deadline_no_request", lambda: acq.reserve(0, None))
            check("expired_deadline_not_reset", acq.state["deadline_epoch"] == 1300. and not acq.state["attempts"])
            acq = acquire(); changed = json.loads(acq.protocol_path.read_text()); changed["started_epoch"] += 1; changed["deadline_epoch"] += 1
            acq.protocol_path.write_text(json.dumps(changed))
            error("changed_protocol_deadline_rejected", lambda: m.Acquisition(acq.protocol_path, acq.root, clock=lambda: 1001.))

            acq = acquire(); started, release = threading.Event(), threading.Event()
            tasks = queue.Queue(); tasks.put(0)
            def late_fetch(attempt, auth):
                started.set(); release.wait(2)
                partial = (acq.root / attempt["raw"]).with_suffix(".partial"); partial.write_bytes(body())
                acq.commit_body(attempt, partial, {})
            with patch.object(acq, "fetch", late_fetch):
                thread = threading.Thread(target=acq.worker, args=(tasks, ("offline", "offline")))
                thread.start(); check("late_worker_started", started.wait(2))
                acq.seal("stopped_preserve_progress"); terminal = acq.path.read_bytes()
                release.set(); thread.join(2)
            check("late_worker_cannot_overwrite_sealed_terminal", not thread.is_alive() and acq.path.read_bytes() == terminal)
            check("late_complete_partial_preserved", len(list(acq.root.glob("*.partial"))) == 1)

            acq = acquire(); attempt = acq.reserve(0, None); fake_fetch(acq, attempt, body()); acq.qualify(attempt)
            acq.state["first_range_started_epoch"] = 1000.; acq.clock = lambda: 1002.; acq.record_first_cost()
            original = copy.deepcopy(acq.state["first_range_cost"])
            acq.continuation_gate()
            prediction = acq.state["continuation_gate_evaluation"]
            expected_wall = 99 * (original["io_wall_seconds"] / 2 + max(0., 2. - original["io_wall_seconds"]))
            check("CPU_serial_cost_not_divided_by_two", prediction["remaining_wall_seconds_predicted"] == expected_wall)
            acq.clock = lambda: 1299.
            error("late_remaining_cost_gate_fails", acq.continuation_gate)
            check("first_range_cost_not_recalculated_on_resume", acq.state["first_range_cost"] == original)

        check("no_real_HTTP_or_credentials", not forbidden)
        check("review_CPU26", sorted(os.sched_getaffinity(0)) == [26])
        check("source_unchanged_during_fixtures", m.digest(SOURCE.read_bytes()) == source_hash)
    report = {"accepted": True, "status": "offline_engineering_checks_only_no_live_admission",
              "source_sha256": source_hash, "fixture_source_sha256": m.digest(Path(__file__).read_bytes()),
              "cost_gate_sha256": m.digest((HERE / "COST_CONTINUATION_GATE.json").read_bytes()),
              "checks_total": len(checks), "checks_passed": len(checks), "checks": checks,
              "real_HTTP_requests": 0, "credential_files_read": False,
              "market_rows_arrays_weights_AAPL_or_Q3_read": False,
              "synthetic_constant_bars_only": True, "fits_or_returns_or_scores": False,
              "runtime_thread_affinity_calls_mocked_to_keep_review_CPU26": True,
              "research_goal_achieved": False}
    target = HERE / "OFFLINE_CHECKS.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": True, "checks": len(checks), "source_sha256": source_hash,
                      "receipt_sha256": m.digest(target.read_bytes())}))


if __name__ == "__main__": main()
