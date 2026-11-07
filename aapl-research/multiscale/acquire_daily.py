"""Acquire bounded AAPL daily development history with attributable raw pages.

This uses existing Alpaca credentials locally for read-only market data and
Nasdaq's public historical endpoint for a bounded comparison. No private key is
accepted via command-line arguments, written to receipts, or printed. Every
returned date must fall within the requested development period.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import shlex
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

from .spec import DEV_END, DEV_START, NY, OUT

ALPACA_URL = "https://data.alpaca.markets/v2/stocks/bars"
NASDAQ_URL = "https://api.nasdaq.com/api/quote/AAPL/historical"
MAX_BYTES = 15_000_000


class RejectRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


OPENER = build_opener(RejectRedirect)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def credentials():
    """Read only literal assignments; never evaluate a credentials file."""
    names = ("APCA_API_KEY_ID", "APCA_API_SECRET_KEY")
    values = {name: os.environ.get(name, "") for name in names}
    path = Path.home() / ".config" / "alpaca" / "credentials.env"
    if not all(values.values()) and path.is_file():
        for original in path.read_text(encoding="utf-8").splitlines():
            line = original.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].strip()
            name, sep, value = line.partition("=")
            name = name.strip()
            if not sep or name not in names or values[name]:
                continue
            try:
                parts = shlex.split(value, comments=True)
            except ValueError:
                raise ValueError("Invalid credential assignment; values suppressed") from None
            if len(parts) != 1:
                raise ValueError("Expected a single literal credential; values suppressed")
            values[name] = parts[0]
    if not all(values.values()):
        raise RuntimeError("Existing Alpaca market-data credentials are unavailable")
    return tuple(values[name] for name in names)


def fetch(url: str, params: dict, headers: dict) -> tuple[bytes, dict]:
    req = Request(url + "?" + urlencode(params), headers=headers)
    try:
        with OPENER.open(req, timeout=30) as response:
            if response.url.split("?", 1)[0] != url:
                raise ValueError("Unexpected endpoint")
            body = response.read(MAX_BYTES + 1)
            if len(body) > MAX_BYTES:
                raise ValueError("Response exceeds size guard")
            safe_headers = {k.lower(): v for k, v in response.headers.items()
                            if k.lower() in {"date", "content-type", "x-request-id"}}
            return body, safe_headers
    except HTTPError as exc:
        raise RuntimeError(f"Data provider returned HTTP {exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Data provider request failed or timed out") from None


def date_in_scope(value: date) -> bool:
    return date.fromisoformat(DEV_START) <= value <= date.fromisoformat(DEV_END)


def checked_decimal(value, *, positive=False, nonnegative=False):
    try:
        number = Decimal(str(value).replace("$", "").replace(",", ""))
    except (InvalidOperation, ValueError):
        raise ValueError("Invalid numeric field in response") from None
    if not number.is_finite() or (positive and number <= 0) or (nonnegative and number < 0):
        raise ValueError("Non-finite or invalid numeric field in response")
    return number


def validate_ohlcv(open_, high, low, close, volume):
    o, h, l, c = [checked_decimal(x, positive=True) for x in (open_, high, low, close)]
    checked_decimal(volume, nonnegative=True)
    if h < max(o, l, c) or l > min(o, h, c):
        raise ValueError("Invalid OHLC relationships")


def write_csv(path: Path, rows: list[dict], columns=("date", "open", "high", "low", "close", "volume")):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def fetch_alpaca(directory: Path):
    key, secret = credentials()
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
               "User-Agent": "AAPL-multiscale-academic-data/1.0"}
    params = {"symbols": "AAPL", "timeframe": "1Day", "feed": "sip",
              "adjustment": "raw", "start": "1980-12-12T00:00:00Z",
              "end": "2026-07-01T03:59:59Z", "limit": 10000, "sort": "asc"}
    pages, rows, seen, token = [], [], set(), None
    for page in range(100):
        request_params = dict(params)
        if token:
            request_params["page_token"] = token
        body, headers_received = fetch(ALPACA_URL, request_params, headers)
        payload = json.loads(body)
        if not isinstance(payload.get("bars"), dict) or "AAPL" not in payload["bars"]:
            raise ValueError("Missing AAPL bars in response")
        bars = payload["bars"]["AAPL"] or []
        for bar in bars:
            stamp = datetime.fromisoformat(bar["t"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("Provider timestamp lacks timezone")
            local_date = stamp.astimezone(ZoneInfo(NY)).date()
            if not date_in_scope(local_date):
                raise ValueError("Provider returned a date outside the development range")
            validate_ohlcv(bar["o"], bar["h"], bar["l"], bar["c"], bar["v"])
            rows.append({"date": local_date.isoformat(), "open": str(bar["o"]),
                         "high": str(bar["h"]), "low": str(bar["l"]),
                         "close": str(bar["c"]), "volume": str(bar["v"])})
        filename = f"alpaca_page_{page:04d}.json"
        (directory / filename).write_bytes(body)
        pages.append({"file": filename, "sha256": sha256(body), "bytes": len(body),
                      "rows": len(bars), "headers": headers_received})
        token = payload.get("next_page_token")
        if not token:
            break
        if token in seen:
            raise ValueError("Repeated Alpaca pagination token")
        seen.add(token)
    else:
        raise ValueError("Alpaca pagination exceeds guard")
    dates = [r["date"] for r in rows]
    if not rows or dates != sorted(set(dates)):
        raise ValueError("Empty, duplicate or unordered Alpaca daily dates")
    write_csv(directory / "alpaca_daily_raw.csv", rows)
    return {"provider": "Alpaca", "endpoint": ALPACA_URL, "parameters": params,
            "price_semantics": "unadjusted raw daily bars; not point-in-time certified",
            "first": dates[0], "last": dates[-1], "rows": len(rows), "pages": pages,
            "csv_sha256": sha256((directory / "alpaca_daily_raw.csv").read_bytes())}


def fetch_nasdaq(directory: Path):
    params = {"assetclass": "stocks", "fromdate": DEV_START, "todate": DEV_END,
              "limit": 10000}
    body, headers = fetch(NASDAQ_URL, params,
                          {"User-Agent": "Mozilla/5.0", "Accept": "application/json",
                           "Referer": "https://www.nasdaq.com/"})
    payload = json.loads(body)
    data = payload.get("data")
    if not isinstance(data, dict) or payload.get("status", {}).get("rCode") != 200:
        raise ValueError("Nasdaq did not return a successful data response")
    source_rows = data.get("tradesTable", {}).get("rows", [])
    total = int(data.get("totalRecords", -1))
    if total != len(source_rows) or not source_rows:
        raise ValueError("Nasdaq response appears incomplete")
    rows = []
    for item in source_rows:
        local_date = datetime.strptime(item["date"], "%m/%d/%Y").date()
        if not date_in_scope(local_date):
            raise ValueError("Nasdaq returned a date outside the development range")
        validate_ohlcv(item["open"], item["high"], item["low"], item["close"], item["volume"])
        rows.append({"date": local_date.isoformat(),
                     **{col: str(checked_decimal(item[col]))
                        for col in ("open", "high", "low", "close", "volume")}})
    rows.sort(key=lambda item: item["date"])
    dates = [r["date"] for r in rows]
    if dates != sorted(set(dates)):
        raise ValueError("Duplicate Nasdaq daily dates")
    (directory / "nasdaq_page_0000.json").write_bytes(body)
    write_csv(directory / "nasdaq_daily.csv", rows)
    return {"provider": "Nasdaq", "endpoint": NASDAQ_URL, "parameters": params,
            "price_semantics": "adjustment and historical revision policy not yet independently established",
            "first": dates[0], "last": dates[-1], "rows": len(rows),
            "pages": [{"file": "nasdaq_page_0000.json", "sha256": sha256(body),
                       "bytes": len(body), "rows": len(rows), "headers": headers}],
            "csv_sha256": sha256((directory / "nasdaq_daily.csv").read_bytes())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("alpaca", "nasdaq"), required=True)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUT / "daily_sources" / f"{args.provider}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    result = {"status": "incomplete", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "bounded historical development only", "independent_test": False,
              "upper_date_inclusive_new_york": DEV_END, "provider": args.provider}
    try:
        handler = {"alpaca": fetch_alpaca, "nasdaq": fetch_nasdaq}[args.provider]
        details = handler(directory)
        result.update(details, status="acquired_not_admitted")
    except Exception as exc:
        result["failure"] = f"{type(exc).__name__}: {exc}"
        (directory / "receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"directory": str(directory), "status": result["status"],
                          "failure": result["failure"]}))
        raise SystemExit(1) from None
    (directory / "receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"directory": str(directory), "status": result["status"],
                      "first": result["first"], "last": result["last"], "rows": result["rows"]}))


if __name__ == "__main__":
    main()
