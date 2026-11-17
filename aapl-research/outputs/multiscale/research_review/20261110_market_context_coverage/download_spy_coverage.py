"""Fixed SPY-only source acquisition. No returns, AAPL reads, models or scoring."""
from __future__ import annotations

import argparse
from datetime import date, datetime, time as day_time, timedelta, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import signal
import threading
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
NY = ZoneInfo("America/New_York")
FIRST, LAST = date(2021, 1, 4), date(2026, 6, 30)
CUTOFF = datetime(2026, 7, 1, tzinfo=NY).astimezone(timezone.utc)
MAX_PAGE = 4 * 1024 * 1024
RESERVATION = MAX_PAGE + 1
BUDGET = {"requests": 300, "bytes": 256 * 1024 * 1024,
          "window_seconds": 600, "workers": 2, "CPU": 24,
          "socket_seconds": 15, "page_bytes": MAX_PAGE}
FIXED = {"symbols": "SPY", "timeframe": "1Min", "feed": "sip",
         "adjustment": "raw", "currency": "USD", "asof": "-",
         "sort": "asc", "limit": 10000}
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)")


class Halt(Exception):
    """A constant reason code; never provider/error/credential text."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def ranges():
    result, first = [], FIRST
    while first <= LAST:
        last = min(LAST, first + timedelta(days=29))
        result.append([first.isoformat(), last.isoformat()])
        first = last + timedelta(days=1)
    return result


RANGES = ranges()


def bounds(index):
    first, last = map(date.fromisoformat, RANGES[index])
    low = datetime.combine(first, day_time(), NY).astimezone(timezone.utc)
    high = datetime.combine(last + timedelta(days=1), day_time(), NY).astimezone(timezone.utc)
    return low, high


def parameters(index, token=None):
    low, high = bounds(index)
    result = dict(FIXED, start=low.isoformat(),
                  end=(high - timedelta(minutes=1)).isoformat())
    if token is not None:
        result["page_token"] = token
    return result


def template():
    gate = BASE / "COST_CONTINUATION_GATE.json"
    return {"kind": "fixed_SPY_source_coverage_v1", "execution_admitted": False,
            "source_sha256": digest(Path(__file__).read_bytes()),
            "cost_gate_sha256": digest(gate.read_bytes()) if gate.is_file() else None,
            "endpoint": ENDPOINT, "fixed": FIXED, "ranges": RANGES,
            "budget": BUDGET, "started_epoch": None, "deadline_epoch": None,
            "GPU": False, "Q3_allowed": False, "fits_or_scores_allowed": False}


def credentials():
    names = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    values = {name: os.environ.get(name, "") for name in names}
    path = Path.home() / ".config/alpaca/credentials.env"
    if not all(values.values()) and path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            name, sep, value = line.partition("=")
            name = name.strip()
            if sep and name in names and not values[name]:
                parsed = shlex.split(value, comments=True)
                if len(parsed) != 1:
                    raise Halt("credential_literal_invalid")
                values[name] = parsed[0]
    if not all(values.values()):
        raise Halt("credentials_unavailable_no_request")
    return tuple(values[name] for name in names)


def validate_page(body, index, previous=None, guard=None):
    payload = json.loads(body)
    if (not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict)
            or set(payload["bars"]) != {"SPY"} or "next_page_token" not in payload):
        raise Halt("container_or_symbol_invalid")
    token = payload["next_page_token"]
    if token is not None and (not isinstance(token, str) or not token):
        raise Halt("token_invalid")
    if "currency" in payload and payload["currency"] != "USD":
        raise Halt("currency_invalid")
    bars = payload["bars"]["SPY"]
    if not isinstance(bars, list) or not 1 <= len(bars) <= 10000:
        raise Halt("page_row_count_invalid")
    low, high = bounds(index)
    last = datetime.fromisoformat(previous) if previous else None
    first_stamp = None
    for number, bar in enumerate(bars):
        if guard is not None and number % 256 == 0:
            guard()
        if not isinstance(bar, dict) or not isinstance(bar.get("t"), str) or not STAMP.fullmatch(bar["t"]):
            raise Halt("timestamp_format_invalid")
        stamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
        if not low <= stamp < high or stamp >= CUTOFF:
            raise Halt("range_or_Q3_boundary_invalid")
        if last is not None and stamp <= last:
            raise Halt("timestamp_duplicate_or_unordered")
        for field in ("o", "h", "l", "c", "v", "n", "vw"):
            value = bar[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise Halt("source_numeric_invalid")
            if value < 0 or (field in ("o", "h", "l", "c", "vw") and value <= 0):
                raise Halt("source_sign_invalid")
        if not (bar["l"] <= bar["o"] <= bar["h"] and bar["l"] <= bar["c"] <= bar["h"]):
            raise Halt("OHLC_invalid")
        if any(bar[k] != int(bar[k]) or bar[k] > 2**63 - 1 for k in ("v", "n")):
            raise Halt("source_count_invalid")
        first_stamp = first_stamp or stamp.isoformat()
        last = stamp
    return {"rows": len(bars), "first_UTC": first_stamp,
            "last_UTC": last.isoformat(), "next_token": token}


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Acquisition:
    def __init__(self, protocol_path, root=None, clock=time.time):
        self.root = Path(root) if root is not None else BASE / "acquisition"
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "JOURNAL.json"
        self.protocol_path = Path(protocol_path)
        protocol_bytes = self.protocol_path.read_bytes()
        self.protocol = json.loads(protocol_bytes)
        expected = template()
        for key in ("kind", "source_sha256", "endpoint", "fixed", "ranges", "budget",
                    "GPU", "Q3_allowed", "fits_or_scores_allowed"):
            if self.protocol.get(key) != expected[key]:
                raise Halt("protocol_identity_invalid")
        if self.protocol.get("execution_admitted") is not True:
            raise Halt("protocol_not_admitted")
        gate_path = BASE / "COST_CONTINUATION_GATE.json"
        if (not gate_path.is_file() or self.protocol.get("cost_gate_sha256") != digest(gate_path.read_bytes())):
            raise Halt("cost_gate_identity_invalid")
        self.cost_gate = json.loads(gate_path.read_bytes())
        for key, value in {"kind": "SPY_first_range_cost_gate_v1", "safety_factor": 1.5,
                           "remaining_ranges": 66, "closure_reserve_seconds": 60,
                           "inflight_page_reserve_count": 2,
                           "maximum_requests": BUDGET["requests"],
                           "maximum_total_response_bytes": BUDGET["bytes"],
                           "total_original_window_seconds": BUDGET["window_seconds"],
                           "maximum_IO_threads_after_gate": BUDGET["workers"]}.items():
            if self.cost_gate.get(key) != value:
                raise Halt("cost_gate_specification_invalid")
        start, end = self.protocol.get("started_epoch"), self.protocol.get("deadline_epoch")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not math.isfinite(start) or not math.isfinite(end) or end - start != BUDGET["window_seconds"]):
            raise Halt("absolute_window_invalid")
        self.clock, self.deadline = clock, end
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.sealed = False
        self.protocol_hash = digest(protocol_bytes)
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "protocol_sha256": self.protocol_hash, "started_epoch": start,
            "deadline_epoch": end, "status": "prepared", "attempts": [],
            "ranges": [{"pages": [], "complete": False} for _ in RANGES]}
        if (self.state.get("protocol_sha256") != self.protocol_hash
                or self.state.get("started_epoch") != start or self.state.get("deadline_epoch") != end):
            raise Halt("journal_protocol_or_deadline_changed")
        attempts = self.state.get("attempts")
        saved_ranges = self.state.get("ranges")
        if (not isinstance(attempts, list) or len(attempts) > BUDGET["requests"]
                or not isinstance(saved_ranges, list) or len(saved_ranges) != len(RANGES)):
            raise Halt("journal_structure_invalid")
        charged = 0
        for number, attempt in enumerate(attempts):
            if not isinstance(attempt, dict):
                raise Halt("journal_attempt_invalid")
            index, page = attempt.get("range"), attempt.get("page")
            charge = attempt.get("charged_bytes")
            if (attempt.get("id") != number or type(index) is not int or not 0 <= index < len(RANGES)
                    or type(page) is not int or page < 0 or type(charge) is not int
                    or not 0 <= charge <= RESERVATION
                    or attempt.get("raw") != f"range_{index:03d}_request_{number:03d}.json"
                    or attempt.get("status") not in ("reserved", "body_journaled", "raw_saved", "qualified")):
                raise Halt("journal_attempt_identity_invalid")
            params = attempt.get("params")
            if not isinstance(params, dict):
                raise Halt("journal_request_invalid")
            token = params.get("page_token")
            if (token is not None and (not isinstance(token, str) or not token)
                    or params != parameters(index, token)):
                raise Halt("journal_request_invalid")
            if attempt["status"] != "reserved":
                size, value = attempt.get("body_bytes"), attempt.get("body_sha256")
                if (type(size) is not int or not 0 <= size <= MAX_PAGE or charge != size
                        or not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None):
                    raise Halt("journal_body_identity_invalid")
            elif charge != RESERVATION:
                raise Halt("journal_unfinished_charge_invalid")
            charged += charge
        if charged > BUDGET["bytes"]:
            raise Halt("journal_byte_budget_invalid")
        self.state["sealed_snapshot"] = False
        self._save()

    def _save(self):
        with self.lock:
            if self.sealed:
                raise Halt("journal_sealed")
            self.state["requests_charged"] = len(self.state["attempts"])
            self.state["bytes_charged"] = sum(a["charged_bytes"] for a in self.state["attempts"])
            atomic(self.path, self.state)

    def guard(self):
        now = self.clock()
        if now < self.protocol["started_epoch"]:
            raise Halt("original_window_not_started")
        if self.sealed or self.stop.is_set() or now >= self.deadline:
            raise Halt("stopped_or_absolute_deadline")

    def seal(self, status):
        with self.lock:
            if self.sealed:
                return
            self.state["status"] = status
            self.state["sealed_snapshot"] = True
            self._save()
            self.sealed = True

    def continuation_gate(self):
        first = [a for a in self.state["attempts"] if a["range"] == 0]
        start = self.state.get("first_range_started_epoch")
        end = self.state.get("first_range_completed_epoch")
        measured = self.state.get("first_range_cost", {})
        if (not first or not self.state["ranges"][0]["complete"]
                or any(a["status"] != "qualified" for a in first)
                or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
                or not 0 < end - start <= BUDGET["window_seconds"]):
            raise Halt("first_range_cost_evidence_invalid")
        p0, b0, t0 = measured.get("p0"), measured.get("b0"), measured.get("t0_seconds")
        t_io = measured.get("io_wall_seconds")
        if (p0 != len(first) or b0 != sum(a["charged_bytes"] for a in first) or t0 != end - start
                or not isinstance(t_io, (int, float)) or not math.isfinite(t_io) or t_io < 0
                or t_io != sum(a["io_wall_seconds"] for a in first)):
            raise Halt("frozen_first_range_cost_changed")
        t_cpu_serial = max(0.0, t0 - t_io)
        predicted_requests = math.ceil(66 * p0 * 1.5)
        predicted_bytes = math.ceil(66 * b0 * 1.5)
        predicted_wall = 1.5 * 66 * (t_io / 2 + t_cpu_serial)
        accepted = (len(self.state["attempts"]) + predicted_requests <= BUDGET["requests"]
                    and sum(a["charged_bytes"] for a in self.state["attempts"])
                    + predicted_bytes + 2 * MAX_PAGE <= BUDGET["bytes"]
                    and self.clock() + predicted_wall + 60 <= self.deadline)
        self.state["continuation_gate_evaluation"] = {
            "cost_gate_sha256": self.protocol["cost_gate_sha256"],
            "p0": p0, "b0": b0, "t0_seconds": t0,
            "io_wall_seconds": t_io, "cpu_serial_seconds": t_cpu_serial,
            "remaining_requests_predicted": predicted_requests,
            "remaining_bytes_predicted": predicted_bytes,
            "remaining_wall_seconds_predicted": predicted_wall,
            "evaluated_epoch": self.clock(), "accepted": accepted,
            "prediction_is_not_completion_guarantee": True}
        self._save()
        if not accepted:
            raise Halt("fixed_first_range_cost_gate_failed")

    def reserve(self, index, token):
        with self.lock:
            self.guard()
            # Leave a full socket timeout and a short commit allowance before cutoff.
            if self.deadline - self.clock() <= BUDGET["socket_seconds"] + 2:
                raise Halt("insufficient_time_for_another_request")
            attempts = self.state["attempts"]
            if self.state["ranges"][index]["complete"]:
                raise Halt("range_already_complete_no_more_requests")
            if (len(attempts) >= BUDGET["requests"]
                    or sum(a["charged_bytes"] for a in attempts) + RESERVATION > BUDGET["bytes"]):
                raise Halt("request_or_byte_budget_exhausted")
            number = len(attempts)
            attempt = {"id": number, "range": index,
                       "page": len(self.state["ranges"][index]["pages"]),
                       "params": parameters(index, token), "status": "reserved",
                       "charged_bytes": RESERVATION,
                       "raw": f"range_{index:03d}_request_{number:03d}.json"}
            attempts.append(attempt)
            self._save()
            return attempt

    def commit_body(self, attempt, temporary, headers):
        count = temporary.stat().st_size
        if count > MAX_PAGE:
            raise Halt("response_size_exceeded")
        body_hash = digest(temporary.read_bytes())
        with self.lock:
            if self.sealed:
                raise Halt("journal_sealed_complete_temp_retained")
            attempt.update(status="body_journaled", body_bytes=count,
                           body_sha256=body_hash, charged_bytes=count, headers=headers,
                           received_utc=datetime.now(timezone.utc).isoformat())
            self._save()
            temporary.replace(self.root / attempt["raw"])
            attempt["status"] = "raw_saved"
            self._save()

    def qualify(self, attempt):
        with self.lock:
            current = self.state["ranges"][attempt["range"]]
            if current["complete"] or attempt["page"] != len(current["pages"]):
                raise Halt("page_gap_or_page_after_terminal")
            prior = current["pages"][-1] if current["pages"] else None
        raw = self.root / attempt["raw"]
        if raw.stat().st_size > MAX_PAGE:
            raise Halt("cached_size_invalid")
        body = raw.read_bytes()
        if len(body) != attempt["body_bytes"] or digest(body) != attempt["body_sha256"]:
            raise Halt("cached_identity_invalid")
        expected_token = prior["next_token"] if prior else None
        if attempt["params"] != parameters(attempt["range"], expected_token):
            raise Halt("cached_pagination_request_invalid")
        meta = validate_page(body, attempt["range"], prior["last_UTC"] if prior else None, self.guard)
        if meta["next_token"] is not None and meta["next_token"] in [p["next_token"] for p in current["pages"]]:
            raise Halt("pagination_cycle")
        with self.lock:
            if self.sealed:
                raise Halt("journal_sealed")
            current["pages"].append(dict(meta, attempt_id=attempt["id"]))
            current["complete"] = meta["next_token"] is None
            attempt["status"] = "qualified"
            self._save()

    def recover(self):
        # Reconstruct pagination from immutable complete pages, not trusted cursors.
        for item in self.state["ranges"]:
            item.update(pages=[], complete=False)
        uncertain = False
        for attempt in sorted(self.state["attempts"], key=lambda a: (a["range"], a["page"])):
            self.guard()
            raw = self.root / attempt["raw"]
            temp = raw.with_suffix(".partial")
            if attempt["status"] in ("body_journaled", "raw_saved", "qualified"):
                if not raw.is_file() and temp.is_file() and temp.stat().st_size <= MAX_PAGE:
                    data = temp.read_bytes()
                    if len(data) == attempt["body_bytes"] and digest(data) == attempt["body_sha256"]:
                        temp.replace(raw)
                if not raw.is_file():
                    raise Halt("committed_page_missing")
                self.qualify(attempt)
            else:
                uncertain = True
        if uncertain:
            raise Halt("incomplete_old_attempt_no_automatic_retry")
        self._save()

    def fetch(self, attempt, auth):
        request = Request(ENDPOINT + "?" + urlencode(attempt["params"]), headers={
            "APCA-API-KEY-ID": auth[0], "APCA-API-SECRET-KEY": auth[1],
            "User-Agent": "AAPL-SPY-source-coverage/1.0"})
        path = (self.root / attempt["raw"]).with_suffix(".partial")
        self.guard()
        io_wall = 0.0
        opener = build_opener(RejectRedirect)
        try:
            io_started = time.monotonic()
            try:
                response = opener.open(request, timeout=BUDGET["socket_seconds"])
            finally:
                io_wall += time.monotonic() - io_started
            with response:
                if response.url.split("?")[0] != ENDPOINT or response.status != 200:
                    raise Halt("response_endpoint_or_status_invalid")
                total = 0
                with path.open("xb") as stream:
                    while True:
                        self.guard()
                        io_started = time.monotonic()
                        try:
                            block = response.read(min(65536, RESERVATION - total))
                        finally:
                            io_wall += time.monotonic() - io_started
                        if not block:
                            stream.flush(); os.fsync(stream.fileno())
                            break
                        stream.write(block); total += len(block)
                        if total > MAX_PAGE:
                            raise Halt("response_size_exceeded")
                headers = {k.lower(): v for k, v in response.headers.items()
                           if k.lower() in ("date", "content-type", "x-request-id")}
        finally:
            with self.lock:
                if not self.sealed:
                    attempt["io_wall_seconds"] = io_wall
                    self._save()
        self.commit_body(attempt, path, headers)

    def worker(self, tasks, auth):
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {BUDGET["CPU"]})
        try:
            while not self.stop.is_set():
                try:
                    index = tasks.get_nowait()
                except queue.Empty:
                    return
                while not self.state["ranges"][index]["complete"]:
                    pages = self.state["ranges"][index]["pages"]
                    token = pages[-1]["next_token"] if pages else None
                    attempt = self.reserve(index, token)
                    self.fetch(attempt, auth)
                    self.qualify(attempt)
        except Exception as exc:
            with self.lock:
                if self.sealed:
                    self.stop.set()
                    return
                self.state["status"] = "stopped_preserve_progress"
                self.state["error_type"] = type(exc).__name__
                if isinstance(exc, Halt):
                    self.state["reason_code"] = str(exc)
                if isinstance(exc, HTTPError):
                    self.state["http_status"] = exc.code
                self.stop.set(); self._save()

    def run(self, scope):
        self.guard(); self.recover()
        if scope == "remaining" and not self.state["ranges"][0]["complete"]:
            raise Halt("first_range_gate_not_complete")
        if scope == "remaining":
            self.continuation_gate()
        elif "first_range_started_epoch" not in self.state:
            self.state["first_range_started_epoch"] = self.protocol["started_epoch"]
            self._save()
        indices = [0] if scope == "first-range" else list(range(1, len(RANGES)))
        tasks = queue.Queue()
        for index in indices:
            if not self.state["ranges"][index]["complete"]:
                tasks.put(index)
        if tasks.empty():
            if scope == "first-range" and "first_range_cost" not in self.state:
                self.record_first_cost()
            self.seal("requested_scope_already_complete"); return
        auth = credentials()
        self.state["status"] = "running"; self.state["scope"] = scope; self._save()
        threads = [threading.Thread(target=self.worker, args=(tasks, auth), daemon=True)
                   for _ in range(min(BUDGET["workers"], tasks.qsize()))]
        for thread in threads:
            thread.start()
        while any(thread.is_alive() for thread in threads) and not self.stop.wait(0.05):
            if self.clock() >= self.deadline:
                self.stop.set()
        with self.lock:
            if not self.stop.is_set():
                status = "first_range_complete_awaiting_root_gate" if scope == "first-range" else "all_ranges_downloaded_field_validated"
                if scope == "first-range":
                    self.record_first_cost()
            else:
                status = "stopped_preserve_progress"
            self.state["full_RTH_coverage_independently_proven"] = False
            self.state["returns_features_fits_scores"] = False
            self.state["research_goal_achieved"] = False
            self.seal(status)

    def record_first_cost(self):
        if "first_range_cost" in self.state:
            return
        first = [a for a in self.state["attempts"] if a["range"] == 0]
        if not first or not self.state["ranges"][0]["complete"] or any(a["status"] != "qualified" for a in first):
            raise Halt("first_range_cost_not_complete")
        self.state["first_range_completed_epoch"] = self.clock()
        t0 = self.state["first_range_completed_epoch"] - self.state["first_range_started_epoch"]
        t_io = sum(a["io_wall_seconds"] for a in first)
        self.state["first_range_cost"] = {"p0": len(first),
            "b0": sum(a["charged_bytes"] for a in first), "t0_seconds": t0,
            "io_wall_seconds": t_io, "cpu_serial_seconds": max(0.0, t0 - t_io)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", action="store_true")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--scope", choices=("first-range", "remaining"))
    args = parser.parse_args()
    if args.template:
        print(json.dumps(template(), indent=2)); return
    if not args.protocol or not args.scope:
        parser.error("Need --protocol and --scope; no request sent")
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {BUDGET["CPU"]})
    output = BASE / "acquisition"
    output.mkdir(exist_ok=True)
    with (output / "PROCESS.lock").open("a") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquisition = None
        try:
            acquisition = Acquisition(args.protocol)
            def stop(_signal, _frame):
                acquisition.stop.set()
            for event in (signal.SIGUSR1, signal.SIGTERM, signal.SIGALRM):
                signal.signal(event, stop)
            remaining = acquisition.deadline - time.time()
            if remaining <= 0:
                raise Halt("original_deadline_expired_no_reset")
            signal.setitimer(signal.ITIMER_REAL, remaining)
            acquisition.run(args.scope)
        except Exception as exc:
            if acquisition is not None:
                with acquisition.lock:
                    if not acquisition.sealed:
                        acquisition.state["error_type"] = type(exc).__name__
                        if isinstance(exc, Halt):
                            acquisition.state["reason_code"] = str(exc)
                        acquisition.seal("stopped_preserve_progress")
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        result = {"status": "not_started"} if acquisition is None else {
            k: acquisition.state.get(k) for k in ("status", "requests_charged", "bytes_charged", "reason_code")}
        print(json.dumps(result))
        if acquisition is None:
            raise SystemExit(2)
        if result["status"] not in ("first_range_complete_awaiting_root_gate",
                                    "all_ranges_downloaded_field_validated", "requested_scope_already_complete"):
            raise SystemExit(75)


if __name__ == "__main__":
    main()
