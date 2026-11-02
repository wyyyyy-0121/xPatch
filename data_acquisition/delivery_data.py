"""Read-only market-data probes and resumable SIP acquisition for final delivery.

Only the Alpaca market-data host is allowed. Authentication is taken from the
process environment, with literal local assignments loaded without a shell.
Raw responses stay in the Git-ignored runs directory; no trading API is used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import threading
import time as clock
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener, HTTPRedirectHandler
from zoneinfo import ZoneInfo

from data_acquisition.download_alpaca_sip import SAMPLE_DATES


HOST = "https://data.alpaca.markets"
NY = ZoneInfo("America/New_York")
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
BAR_FIELDS = ["o", "h", "l", "c", "v", "n", "vw"]


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def load_environment():
    names = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    if all(os.environ.get(name) for name in names):
        return
    path = Path.home() / ".config/alpaca/credentials.env"
    if path.is_file():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:].strip()
            name, separator, value = line.partition("=")
            name = name.strip()
            if not separator or name not in names or os.environ.get(name):
                continue
            try:
                parts = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError("Invalid credential assignment; contents suppressed") from None
            if len(parts) != 1:
                raise ValueError("Credential assignment must be literal; contents suppressed")
            os.environ[name] = parts[0]
    if not all(os.environ.get(name) for name in names):
        raise RuntimeError("Required Alpaca environment variables unavailable")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class Client:
    def __init__(self):
        load_environment()
        self.opener = build_opener(NoRedirect)

    def get(self, path, params):
        if not path.startswith("/v2/stocks/") or "?" in path or ".." in path:
            raise ValueError("Market-data path outside allowlist")
        request = Request(HOST + path + "?" + urlencode(params), headers={
            "APCA-API-KEY-ID": os.environ["APCA_API_KEY_ID"],
            "APCA-API-SECRET-KEY": os.environ["APCA_API_SECRET_KEY"],
            "User-Agent": "AAPL-volatility-research"})
        for attempt in range(3):
            try:
                with self.opener.open(request, timeout=35) as response:
                    body = response.read(30_000_001)
                    if len(body) > 30_000_000:
                        raise RuntimeError("Market-data response exceeds size guard")
                    return response.status, body, {k.lower(): v for k, v in response.headers.items()
                        if k.lower() in ("date", "x-request-id", "content-type")}
            except HTTPError as error:
                if (error.code == 429 or error.code >= 500) and attempt < 2:
                    clock.sleep(2 ** attempt)
                    continue
                return error.code, error.read(8192), {}
            except (URLError, TimeoutError):
                if attempt < 2:
                    clock.sleep(2 ** attempt)
                    continue
                raise RuntimeError("Market-data network request failed; details suppressed") from None


def bounds(a, b):
    start = datetime.combine(date.fromisoformat(a), time(), NY).astimezone(timezone.utc)
    stop = datetime.combine(date.fromisoformat(b) + timedelta(days=1), time(), NY).astimezone(timezone.utc)
    return start.isoformat(), stop.isoformat()


def probe(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    client = Client()
    jobs = []
    for symbol in ("AAPL", "SPY", "QQQ"):
        for day in ("2021-01-04", "2022-12-30", "2023-01-03", "2025-06-30", "2026-06-30"):
            start, stop = bounds(day, day)
            jobs.append((f"{symbol}_bars_{day}", "/v2/stocks/bars",
                {"symbols": symbol, "timeframe": "1Min", "feed": "sip", "adjustment": "raw",
                 "start": start, "end": stop, "limit": 3, "sort": "asc"}, "bars", symbol))
    for kind in ("quotes", "trades"):
        for day in ("2021-01-04", "2022-12-30", "2023-01-03", "2025-06-30"):
            start, stop = bounds(day, day)
            jobs.append((f"AAPL_{kind}_{day}", f"/v2/stocks/AAPL/{kind}",
                {"feed": "sip", "start": start, "end": stop, "limit": 3, "sort": "asc"}, kind, "AAPL"))
    jobs.append(("latest_sip_bars", "/v2/stocks/bars/latest",
        {"symbols": "AAPL,SPY,QQQ", "feed": "sip"}, "bars", None))
    now = datetime.now(timezone.utc)
    jobs.append(("recent_sip_access", "/v2/stocks/bars",
        {"symbols": "AAPL", "timeframe": "1Min", "feed": "sip", "adjustment": "raw",
         "start": (now - timedelta(minutes=10)).isoformat(), "end": now.isoformat(),
         "limit": 3, "sort": "desc"}, "bars", "AAPL"))
    def fetch(job):
        name, path, params, group, symbol = job
        started = datetime.now(timezone.utc).isoformat()
        try:
            status, body, headers = client.get(path, params)
            target = output / (name + ".json")
            target.write_bytes(body)
            try:
                payload = json.loads(body)
            except (ValueError, UnicodeDecodeError):
                payload = {}
            records = payload.get(group, {})
            if symbol and isinstance(records, dict):
                records = records.get(symbol, []) or []
            if isinstance(records, list):
                count = len(records)
                stamps = [r.get("t") for r in records]
                schema = sorted(set().union(*(set(r) for r in records))) if records else []
            elif isinstance(records, dict):
                count = len(records)
                stamps = {k: v.get("t") for k, v in records.items()}
                schema = []
            else:
                count, stamps, schema = 0, [], []
            return {"name": name, "path": path, "params": params, "http_status": status,
                "request_started_utc": started, "received_utc": datetime.now(timezone.utc).isoformat(),
                "headers": headers, "raw_file": target.name, "sha256": sha(target),
                "rows": count, "timestamps": stamps, "record_fields": schema,
                "access_passed": status == 200 and count > 0,
                "scope": "Access sample only; not a historical coverage or latency certification"}
        except RuntimeError as error:
            return {"name": name, "path": path, "params": params, "access_passed": False,
                "error": str(error), "request_started_utc": started}
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, jobs))
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
        "endpoint": HOST, "requests": results, "requests_count": len(results),
        "account_or_subscription_inferred": False, "market_data_only": True,
        "limitations": ["No account, subscription, purchase or trading endpoint was called.",
            "Historical quotes are best bid/ask, not multi-level book depth.",
            "Access samples do not establish complete trade/quote history or timely live publication."]}
    write_json(output / "probe.json", report)
    print(json.dumps({"output": str(output), "requests": len(results),
        "nonpassing": [{"name": r["name"], "status": r.get("http_status"), "rows": r.get("rows")}
                       for r in results if not r["access_passed"]]}), flush=True)
    return report


def verify(root):
    """Reconstruct CSV from all raw response fields, without modifying evidence."""
    import numpy as np
    import pandas as pd
    root = Path(root)
    receipt = json.loads((root / "receipt.json").read_text())
    if not receipt.get("download_complete"):
        raise ValueError("Incomplete acquisition")
    if receipt["request"] != {"symbol": receipt["symbol"], "feed": "sip", "adjustment": "raw", "timeframe": "1Min"}:
        raise ValueError("Unexpected acquisition identity")
    all_rows, paths = [], set()
    for ri, (a, b) in enumerate(receipt["ranges"]):
        pages = sorted((p for p in receipt["pages"] if p["range_index"] == ri), key=lambda p: p["page"])
        if not pages or [p["page"] for p in pages] != list(range(len(pages))):
            raise ValueError("Missing or non-contiguous pages")
        lower, upper = map(pd.Timestamp, bounds(a, b))
        tokens = set()
        for pi, entry in enumerate(pages):
            path = (root / entry["file"]).resolve()
            if not path.is_relative_to((root / "raw").resolve()) or path in paths:
                raise ValueError("Invalid or duplicate raw path")
            paths.add(path)
            if sha(path) != entry["sha256"] or path.stat().st_size != entry["bytes"]:
                raise ValueError("Raw response hash/size mismatch")
            payload = json.loads(path.read_bytes())
            bars = payload["bars"].get(receipt["symbol"], []) or []
            token = payload.get("next_page_token")
            if len(bars) != entry["rows"] or bool(token) != entry["has_next_page"]:
                raise ValueError("Pagination metadata mismatch")
            if bool(token) != (pi < len(pages) - 1) or token and token in tokens:
                raise ValueError("Incomplete/cyclic pagination")
            if token:
                tokens.add(token)
            for bar in bars:
                stamp = pd.Timestamp(bar["t"])
                if stamp.tzinfo is None:
                    raise ValueError("Naive raw timestamp")
                if lower <= stamp < upper:
                    all_rows.append([stamp, *[bar[k] for k in BAR_FIELDS]])
    if len(paths) != len(receipt["pages"]):
        raise ValueError("Unexpected saved pages")
    rebuilt = pd.DataFrame(all_rows, columns=COLUMNS).sort_values("timestamp").reset_index(drop=True)
    csv_path = (root / receipt["csv"]).resolve()
    if not csv_path.is_relative_to(root.resolve()) or sha(csv_path) != receipt["csv_sha256"]:
        raise ValueError("CSV hash/path mismatch")
    actual = pd.read_csv(csv_path)
    if list(actual.columns) != COLUMNS or len(actual) != len(rebuilt) or len(actual) != receipt["rows"]:
        raise ValueError("CSV schema/length mismatch")
    actual["timestamp"] = pd.to_datetime(actual.timestamp, utc=True)
    if actual.timestamp.duplicated().any() or not actual.timestamp.is_monotonic_increasing:
        raise ValueError("Duplicate/unsorted timestamps")
    if not np.array_equal(actual.timestamp.to_numpy(), rebuilt.timestamp.to_numpy()):
        raise ValueError("Raw and CSV timestamps differ")
    if not np.allclose(actual[COLUMNS[1:]], rebuilt[COLUMNS[1:]], rtol=1e-14, atol=1e-12):
        raise ValueError("Raw and CSV values differ")
    return {"passed": True, "symbol": receipt["symbol"], "pages": len(paths),
        "ranges": len(receipt["ranges"]), "rows": len(actual), "csv_sha256": receipt["csv_sha256"],
        "all_fields_verified": COLUMNS, "point_in_time_available": False}


def download(symbol, ranges, output):
    output = Path(output)
    identity = {"symbol": symbol, "feed": "sip", "adjustment": "raw", "timeframe": "1Min"}
    if symbol not in ("AAPL", "SPY", "QQQ"):
        raise ValueError("Symbol outside preregistered data scope")
    ranges = [list(r) for r in ranges]
    if datetime.fromisoformat(bounds(*ranges[-1])[1]) > datetime.now(timezone.utc) - timedelta(minutes=20):
        raise ValueError("Only completed historical dates are supported")
    output.mkdir(parents=True, exist_ok=True)
    (output / "raw").mkdir(exist_ok=True)
    receipt_path = output / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["request"] != identity or receipt["ranges"] != ranges:
            raise ValueError("Resume identity/ranges differ")
        if receipt.get("download_complete"):
            return verify(output)
    else:
        receipt = {"symbol": symbol, "request": identity, "ranges": ranges, "pages": [],
            "endpoint": HOST + "/v2/stocks/bars", "created_utc": datetime.now(timezone.utc).isoformat(),
            "script_sha256": sha(__file__), "download_complete": False,
            "timezone": "America/New_York", "timestamp_label": "bar_start", "point_in_time_available": False}
    lock = threading.Lock()
    def save():
        write_json(receipt_path, receipt)
    save()
    client = Client()
    def fetch(ri):
        a, b = ranges[ri]
        start, stop = bounds(a, b)
        params = {"symbols": symbol, "feed": "sip", "adjustment": "raw", "timeframe": "1Min",
            "start": start, "end": stop, "limit": 10000, "sort": "asc"}
        pages = sorted((p for p in receipt["pages"] if p["range_index"] == ri), key=lambda p: p["page"])
        seen = set()
        for pi, entry in enumerate(pages):
            path = output / entry["file"]
            if entry["page"] != pi or sha(path) != entry["sha256"]:
                raise ValueError("Corrupt cached response")
            token = json.loads(path.read_bytes()).get("next_page_token")
            if not token:
                if pi != len(pages) - 1:
                    raise ValueError("Cached response after final page")
                return
            if token in seen:
                raise ValueError("Repeated cached token")
            seen.add(token)
            params["page_token"] = token
        pi = len(pages)
        while pi < 500:
            status, body, headers = client.get("/v2/stocks/bars", params)
            if status != 200:
                raise RuntimeError(f"Historical SIP request rejected: HTTP {status}")
            payload = json.loads(body)
            bars = payload["bars"].get(symbol, []) or []
            token = payload.get("next_page_token")
            path = output / "raw" / f"range_{ri:03d}_page_{pi:04d}.json"
            # Recover an unindexed response by comparing its exact bytes first.
            if path.exists() and path.read_bytes() != body:
                path.rename(path.with_suffix(f".orphan_{datetime.now(timezone.utc).strftime('%H%M%S%f')}"))
            path.write_bytes(body)
            entry = {"range_index": ri, "page": pi, "file": str(path.relative_to(output)),
                "sha256": sha(path), "bytes": len(body), "rows": len(bars), "has_next_page": bool(token),
                "received_utc": datetime.now(timezone.utc).isoformat(), "headers": headers}
            with lock:
                receipt["pages"].append(entry)
                save()
            if not token:
                return
            if token in seen:
                raise ValueError("Repeated pagination token")
            seen.add(token)
            params["page_token"] = token
            pi += 1
        raise RuntimeError("Pagination size guard exceeded")
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(fetch, range(len(ranges))))
        rows = []
        for entry in receipt["pages"]:
            lower, upper = bounds(*ranges[entry["range_index"]])
            lower, upper = datetime.fromisoformat(lower), datetime.fromisoformat(upper)
            for bar in json.loads((output / entry["file"]).read_bytes())["bars"].get(symbol, []) or []:
                stamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
                if lower <= stamp < upper:
                    rows.append([stamp.astimezone(NY).isoformat(), *[bar[k] for k in BAR_FIELDS]])
        rows.sort(key=lambda row: datetime.fromisoformat(row[0]))
        path = output / f"{symbol}_sip_1min_raw.csv"
        with path.with_suffix(".partial").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(COLUMNS)
            writer.writerows(rows)
        path.with_suffix(".partial").replace(path)
        receipt.update(csv=path.name, csv_sha256=sha(path), rows=len(rows), download_complete=True)
        receipt.pop("failure", None)
        save()
        verified = verify(output)
        write_json(output / "verification.json", verified)
        print(json.dumps({"output": str(output), **verified}), flush=True)
        return verified
    except (RuntimeError, ValueError, KeyError, TypeError) as error:
        receipt["failure"] = str(error)
        receipt["download_complete"] = False
        save()
        raise


def audit(root):
    from finance_v2.data import audit_bars, read_bars
    root = Path(root)
    verified = verify(root)
    receipt = json.loads((root / "receipt.json").read_text())
    selected = [a for a, b in receipt["ranges"] if a == b]
    bars = read_bars(root / receipt["csv"])
    regular, coverage = audit_bars(bars, start=receipt["ranges"][0][0], end=receipt["ranges"][-1][1],
        requested_dates=selected if len(selected) == len(receipt["ranges"]) else None)
    result = {"passed": bool(coverage["coverage_passed"]), "verification": verified,
        "coverage": coverage, "volume_independently_verified": False,
        "historical_research_only": True, "point_in_time_available": False}
    write_json(root / "audit.json", result)
    print(json.dumps({"passed": result["passed"], "symbol": receipt["symbol"],
        "regular_rows": len(regular), "sessions": len(coverage["daily"]),
        "coverage": coverage["regular_coverage"], "minimum_daily_coverage": coverage["minimum_daily_coverage"]}), flush=True)
    if not result["passed"]:
        raise ValueError("Coverage gate failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("probe", "audit20", "history", "verify", "audit"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--symbol", choices=("AAPL", "SPY", "QQQ"), default="AAPL")
    args = parser.parse_args()
    if args.stage == "probe":
        probe(args.output)
    elif args.stage in ("verify", "audit"):
        result = verify(args.output) if args.stage == "verify" else audit(args.output)
        if args.stage == "verify":
            print(json.dumps(result))
    else:
        if args.stage == "audit20":
            ranges = [(day, day) for day in SAMPLE_DATES]
        else:
            ranges, current, end = [], date(2021, 1, 4), date(2026, 6, 30)
            while current <= end:
                stop = min(end, current + timedelta(days=29))
                ranges.append((current.isoformat(), stop.isoformat()))
                current = stop + timedelta(days=1)
        download(args.symbol, ranges, args.output)
        audit(args.output)


if __name__ == "__main__":
    main()
