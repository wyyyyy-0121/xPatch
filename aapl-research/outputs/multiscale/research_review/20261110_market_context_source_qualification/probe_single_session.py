"""Fixed SPY source qualification; no AAPL prices, labels, scores, or model fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

ROOT = Path(__file__).resolve().parent
ENDPOINT = "https://data.alpaca.markets/v2/stocks/bars"
PARAMS = {
    "symbols": "SPY", "timeframe": "1Min", "feed": "sip", "adjustment": "raw",
    "asof": "-", "currency": "USD", "sort": "asc", "limit": 10000,
    "start": "2021-01-04T14:30:00Z", "end": "2021-01-04T20:59:00Z",
}
MAX_BYTES = 1048576


def atomic_json(path, data):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def literal_credentials():
    """Read two literal assignments only; never echo, source, log, or persist them."""
    names = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    values = {name: os.environ.get(name, "") for name in names}
    path = Path.home() / ".config/alpaca/credentials.env"
    if not all(values.values()) and path.is_file():
        for raw in path.read_text().splitlines():
            line = raw.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            name, sep, value = line.partition("=")
            if not sep or name.strip() not in names or values[name.strip()]:
                continue
            fields = shlex.split(value, comments=True)
            if len(fields) != 1:
                raise ValueError("Invalid literal credential assignment; values suppressed")
            values[name.strip()] = fields[0]
    return tuple(values[name] for name in names)


def validate(payload):
    """Require the exact 390-minute source grid, without filling or deleting rows."""
    if not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict):
        raise ValueError("Invalid bars container")
    if (set(payload["bars"]) != {"SPY"} or "next_page_token" not in payload
            or payload["next_page_token"] is not None):
        raise ValueError("Unexpected symbol or incomplete single-session page")
    if "currency" in payload and payload["currency"] != "USD":
        raise ValueError("Unexpected response currency")
    bars = payload["bars"]["SPY"]
    if not isinstance(bars, list) or len(bars) != 390:
        raise ValueError("Expected exactly 390 original minute bars")
    first = datetime(2021, 1, 4, 14, 30, tzinfo=timezone.utc)
    last_end = first + timedelta(minutes=390)
    stamps = []
    for i, bar in enumerate(bars):
        if not isinstance(bar, dict):
            raise ValueError("Invalid bar object")
        text = bar["t"]
        # fromisoformat truncates beyond microseconds; reject nonzero nanoseconds
        # before parsing so an off-grid source cannot silently become on-grid.
        if (not isinstance(text, str) or re.fullmatch(
                r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:00(?:\.0{1,9})?(?:Z|\+00:00)", text) is None):
            raise ValueError("Timestamp must be an exact UTC minute")
        stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("Timestamp must declare UTC")
        if stamp != first + timedelta(minutes=i) or stamp + timedelta(minutes=1) > last_end:
            raise ValueError("Missing, duplicate, unordered, or off-grid timestamp")
        stamps.append(stamp)
        for field in ("o", "h", "l", "c", "v", "n", "vw"):
            v = bar[field]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError("Non-finite/non-numeric source field")
            if v < 0 or (field in ("o", "h", "l", "c", "vw") and v <= 0):
                raise ValueError("Invalid source field sign")
        if not (bar["l"] <= bar["o"] <= bar["h"] and bar["l"] <= bar["c"] <= bar["h"]):
            raise ValueError("OHLC consistency failure")
        if any(bar[field] != int(bar[field]) or bar[field] > 2**63 - 1
               for field in ("v", "n")):
            raise ValueError("Volume or trade count is not a nonnegative int64")
    hours = []
    for start in range(0, 360, 60):
        group = bars[start:start + 60]
        hours.append({"start_utc": group[0]["t"],
                      "end_utc": (first + timedelta(minutes=start + 60)).isoformat(),
                      "source_minutes": len(group)})
    # A deliberate grid lag withholds the current completed SPY bar. Validate
    # this clock convention against parsed source timestamps, not constant dates.
    # It does not establish when historical corrected bars were actually received.
    lag_checks = 0
    for i in range(1, len(stamps)):
        decision = stamps[i] + timedelta(minutes=1)
        selected_end = stamps[i - 1] + timedelta(minutes=1)
        if selected_end + timedelta(minutes=1) != decision or selected_end >= decision:
            raise ValueError("One-grid-bar delayed input clock mismatch")
        lag_checks += 1
    return {"rows": len(bars), "source_fields": ["t", "o", "h", "l", "c", "v", "n", "vw"],
            "first_start_utc": bars[0]["t"], "last_start_utc": bars[-1]["t"],
            "full_hours": hours, "residual_minutes_excluded": 30,
            "one_grid_bar_delay_clock_checks": lag_checks,
            "origin_received_or_original_revision_proven": False,
            "assumption": "Final historical completed bars, as with original AAPL admission; one-grid-bar lag is not proof of original receipt/revision availability.",
            "full_development_coverage_proven": False,
            "forecast_features_or_labels_built": False, "model_fits": 0}


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {24})
    out = ROOT / "single_session_probe"
    out.mkdir(exist_ok=True)
    receipt_path = out / "RECEIPT.json"
    raw_path = out / "raw_response.json"
    raw_temp = raw_path.with_suffix(".tmp")
    old = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
    if old and (old.get("params") != PARAMS or old.get("endpoint") != ENDPOINT):
        raise SystemExit("Cached request differs from frozen source probe")
    if old and (raw_path.is_file() or raw_temp.is_file()) and old.get("requests_attempted") != 1:
        raise SystemExit("Cached response lacks its single-request receipt")
    # Recover only an already committed complete response, without another GET.
    # The response hash is journaled before its bytes can acquire the final name.
    if (old and not raw_path.is_file() and raw_temp.is_file()
            and old.get("body_sha256") and raw_temp.stat().st_size <= MAX_BYTES):
        saved = raw_temp.read_bytes()
        if (len(saved) == old.get("body_bytes")
                and hashlib.sha256(saved).hexdigest() == old["body_sha256"]):
            raw_temp.replace(raw_path)
    if old and old.get("requests_attempted", 0) and not raw_path.is_file():
        raise SystemExit("Prior request did not preserve a complete body; no automatic retry admitted")
    receipt = old or {"endpoint": ENDPOINT, "params": PARAMS,
                      "started_utc": datetime.now(timezone.utc).isoformat(),
                      "requests_attempted": 0, "status": "prepared",
                      "GPU": False, "Q3_read": False, "AAPL_member_decode": False,
                      "model_fits": 0, "credential_values_stored": False}
    if receipt["params"] != PARAMS or receipt["endpoint"] != ENDPOINT:
        raise SystemExit("Cached request differs from frozen source probe")
    started = time.monotonic()

    def stop(_sig, _frame):
        raise TimeoutError("Fixed 50-second source-probe deadline or external stop")

    signal.signal(signal.SIGALRM, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGUSR1, stop)
    signal.setitimer(signal.ITIMER_REAL, 50)
    try:
        if raw_path.is_file():
            if raw_path.stat().st_size > MAX_BYTES:
                raise ValueError("Cached body exceeds fixed 1-MiB bound")
            body = raw_path.read_bytes()
            if (len(body) != receipt.get("body_bytes")
                    or hashlib.sha256(body).hexdigest() != receipt.get("body_sha256")):
                raise ValueError("Cached body identity mismatch")
        else:
            if not args.fetch:
                raise ValueError("No cached body; --fetch is required for the frozen single request")
            key, secret = literal_credentials()
            receipt["credentials_available"] = bool(key and secret)
            if not key or not secret:
                receipt["status"] = "no_credentials_no_request"
                atomic_json(receipt_path, receipt)
                print(json.dumps({"status": receipt["status"], "requests_attempted": 0}))
                return
            request = Request(ENDPOINT + "?" + urlencode(PARAMS), headers={
                "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
                "User-Agent": "AAPL-multiscale-source-qualification/1.0"})
            receipt.update(requests_attempted=1, status="request_started")
            atomic_json(receipt_path, receipt)
            with build_opener(RejectRedirect).open(request, timeout=15) as response:
                if response.url.split("?")[0] != ENDPOINT:
                    raise ValueError("Unexpected response endpoint")
                body = response.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise ValueError("Response exceeds fixed 1-MiB bound")
                receipt.update(http_status=response.status,
                               content_type=response.headers.get("Content-Type"),
                               received_utc=datetime.now(timezone.utc).isoformat())
            receipt.update(body_sha256=hashlib.sha256(body).hexdigest(), body_bytes=len(body),
                           status="complete_body_received_pending_persistence")
            atomic_json(receipt_path, receipt)
            raw_temp.write_bytes(body)
            raw_temp.replace(raw_path)
            receipt.update(body_sha256=hashlib.sha256(body).hexdigest(), body_bytes=len(body),
                           status="complete_body_saved")
            atomic_json(receipt_path, receipt)
        qualified = validate(json.loads(body))
        atomic_json(out / "SOURCE_FIELD_CLOCK_REPORT.json", qualified)
        receipt.update(status="single_session_source_qualified", elapsed_seconds=time.monotonic()-started)
        atomic_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "body_bytes": len(body),
                          "rows": qualified["rows"], "requests_attempted": receipt["requests_attempted"]}))
    except HTTPError as exc:
        receipt.update(status="http_rejected_no_retry", http_status=exc.code,
                       elapsed_seconds=time.monotonic()-started)
        atomic_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "http_status": exc.code}))
        raise SystemExit(2) from None
    except Exception as exc:
        # Suppress exception text: URL/transport libraries must never echo headers or keys.
        receipt.update(status="stopped_or_unqualified", error_type=type(exc).__name__,
                       elapsed_seconds=time.monotonic()-started)
        atomic_json(receipt_path, receipt)
        print(json.dumps({"status": receipt["status"], "error_type": type(exc).__name__}))
        raise SystemExit(2) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == "__main__":
    main()
