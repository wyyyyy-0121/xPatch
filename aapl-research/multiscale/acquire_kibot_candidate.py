"""Acquire bounded Kibot guest daily bars for internal source diagnostics only.

The public guest account and the historical date bounds are documented by
Kibot. Publication rights for derived results are not established. This tool
does not admit data, train a model, publish prices, or request post-cutoff bars.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
import csv
import hashlib
from http.cookiejar import CookieJar
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPCookieProcessor, Request, build_opener

from .acquire_daily import RejectRedirect, validate_ohlcv, write_csv
from .spec import DEV_END, DEV_START, OUT


ENDPOINT = "https://api.kibot.com/"
MAX_RESPONSE_BYTES = 5_000_000
MAX_LINE_BYTES = 2048


class OutOfScopeDate(ValueError):
    """A provider response contained a date outside the requested interval."""


def parse_response(stream, first: date, last: date):
    if last < first or first < date.fromisoformat(DEV_START) or last > date.fromisoformat(DEV_END):
        raise ValueError("Requested interval is outside the development scope")
    rows, byte_count = [], 0
    while True:
        line = stream.readline(MAX_LINE_BYTES + 1)
        if not line:
            break
        byte_count += len(line)
        if len(line) > MAX_LINE_BYTES or byte_count > MAX_RESPONSE_BYTES:
            raise ValueError("Provider response exceeds size guard")
        # Check the date before parsing, saving, or printing the price fields.
        date_field, sep, rest = line.partition(b",")
        if not sep:
            raise ValueError("Provider response is not the expected CSV format")
        try:
            day = datetime.strptime(date_field.decode("ascii"), "%m/%d/%Y").date()
        except (UnicodeDecodeError, ValueError):
            raise ValueError("Provider response contains an invalid date") from None
        if day < first or day > last:
            raise OutOfScopeDate(f"Provider returned an out-of-scope date: {day.isoformat()}")
        try:
            fields = next(csv.reader([rest.decode("ascii").strip()]))
        except (UnicodeDecodeError, csv.Error):
            raise ValueError("Provider response has invalid CSV fields") from None
        if len(fields) != 5:
            raise ValueError("Provider response has an unexpected number of OHLCV fields")
        validate_ohlcv(*fields)
        rows.append({"date": day.isoformat(), **dict(zip(("open", "high", "low", "close", "volume"), fields, strict=True))})
    if not rows:
        raise ValueError("Provider returned no daily bars")
    rows.sort(key=lambda row: row["date"])
    if len({row["date"] for row in rows}) != len(rows):
        raise ValueError("Provider response has duplicate dates")
    return rows, byte_count


def fetch(directory: Path):
    first, last = date.fromisoformat(DEV_START), date.fromisoformat(DEV_END)
    cookies = CookieJar()
    opener = build_opener(RejectRedirect, HTTPCookieProcessor(cookies))
    headers = {"User-Agent": "AAPL-internal-source-diagnostic/1.0", "Accept": "text/plain"}
    login = {"action": "login", "user": "guest", "password": "guest"}
    history = {"action": "history", "symbol": "AAPL", "interval": "daily",
               "startdate": first.strftime("%m/%d/%Y"), "enddate": last.strftime("%m/%d/%Y"),
               "unadjusted": "1"}
    try:
        with opener.open(Request(ENDPOINT + "?" + urlencode(login), headers=headers), timeout=25) as response:
            if response.geturl().split("?", 1)[0] != ENDPOINT or response.status != 200:
                raise ValueError("Unexpected guest-login endpoint or status")
            # The response can contain session information, so do not log it.
            response.read(2048)
        with opener.open(Request(ENDPOINT + "?" + urlencode(history), headers=headers), timeout=25) as response:
            if response.geturl().split("?", 1)[0] != ENDPOINT or response.status != 200:
                raise ValueError("Unexpected historical endpoint or status")
            rows, byte_count = parse_response(response, first, last)
    except HTTPError as exc:
        raise RuntimeError(f"Kibot returned HTTP {exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Kibot request failed or timed out") from None
    path = directory / "kibot_daily_raw.csv"
    write_csv(path, rows)
    receipt = {"status": "candidate_internal_diagnostic_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "provider": "Kibot guest account", "endpoint": ENDPOINT,
               "requested_first": first.isoformat(), "requested_last": last.isoformat(),
               "first": rows[0]["date"], "last": rows[-1]["date"], "rows": len(rows),
               "response_bytes": byte_count, "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "documented_price_mode": "unadjusted=1; nominal as printed per provider documentation",
               "publication_rights_verified": False, "source_admitted": False, "experiment_scored": False,
               "Q3_values_read": False, "training_authorized": False,
               "limitations": ["Guest access is intended for integration and internal use; publication terms for derived results are not confirmed.",
                               "Provider attribution and documented unadjusted mode do not verify every historical close or point-in-time revision.",
                               "Guest corporate-action access and a complete independent split chain are not established.",
                               "No source or derived price data may be redistributed without applicable rights."]}
    (directory / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf8")
    return receipt


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUT / "daily_sources" / ("kibot_guest_" + stamp)
    directory.mkdir(parents=True, exist_ok=False)
    try:
        receipt = fetch(directory)
    except Exception as exc:
        failure = {"status": "candidate_acquisition_failed", "error_type": type(exc).__name__,
                   "error": str(exc), "source_admitted": False, "experiment_scored": False,
                   "out_of_scope_date_received": isinstance(exc, OutOfScopeDate)}
        (directory / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf8")
        raise
    print(json.dumps({"directory": str(directory), "status": receipt["status"],
                      "first": receipt["first"], "last": receipt["last"], "rows": receipt["rows"],
                      "publication_rights_verified": False, "experiment_scored": False}))


if __name__ == "__main__":
    main()
