"""Reconstruct each saved SIP request and compare every emitted CSV field."""
from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .io import digest, read_json, write_json


def verify_alpaca_run(root, save=True):
    root = Path(root).resolve()
    receipt = read_json(root / "receipt.json")
    if not receipt.get("download_complete"):
        raise ValueError("Download incomplete; no dataset can be accepted")
    if (receipt.get("symbol"), receipt.get("feed"), receipt.get("adjustment"),
            receipt.get("timeframe"), receipt.get("endpoint")) != (
            "AAPL", "sip", "raw", "1Min", "https://data.alpaca.markets/v2/stocks/bars"):
        raise ValueError("Unexpected data request identity")
    ranges = [tuple(pair) for pair in receipt["ranges"]]
    if not ranges or len(set(ranges)) != len(ranges):
        raise ValueError("Missing/duplicated request ranges")
    by_range = {pair: [] for pair in ranges}
    seen_paths = set()
    for entry in receipt["pages"]:
        pair = (entry["range_start"], entry["range_end"])
        if pair not in by_range:
            raise ValueError("Saved page not in requested ranges")
        raw = (root / entry["file"]).resolve()
        if not raw.is_relative_to(root / "raw") or raw in seen_paths:
            raise ValueError("Unexpected/duplicate raw file path")
        seen_paths.add(raw)
        if digest(raw) != entry["sha256"] or raw.stat().st_size != entry["bytes"]:
            raise ValueError("Raw page SHA256 or byte count differs from receipt")
        by_range[pair].append((entry, read_json(raw)))
    rows = []
    for (a, b), pages in by_range.items():
        pages.sort(key=lambda pair: pair[0]["page"])
        if not pages or [p[0]["page"] for p in pages] != list(range(len(pages))):
            raise ValueError("Request range missing pages or has non-contiguous page numbers")
        start = pd.Timestamp(datetime.combine(pd.Timestamp(a).date(), time(), ZoneInfo("America/New_York")))
        stop = pd.Timestamp(datetime.combine(pd.Timestamp(b).date() + timedelta(days=1), time(),
                                             ZoneInfo("America/New_York")))
        tokens = set()
        for i, (entry, payload) in enumerate(pages):
            bars = payload.get("bars", {}).get("AAPL", []) or []
            token = payload.get("next_page_token")
            if len(bars) != entry["rows_returned"] or bool(token) != entry["has_next_page"]:
                raise ValueError("Raw response row count/pagination differs from receipt")
            if bool(token) != (i < len(pages) - 1) or (token and token in tokens):
                raise ValueError("Incomplete or cyclic pagination")
            if token:
                tokens.add(token)
            for bar in bars:
                stamp = pd.Timestamp(bar["t"])
                if stamp.tzinfo is None:
                    raise ValueError("Raw timestamp lacks timezone")
                if start <= stamp < stop:
                    rows.append((stamp.tz_convert("UTC"), bar["o"], bar["h"], bar["l"],
                                 bar["c"], bar["v"], bar.get("n"), bar.get("vw")))
    columns = ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]
    rebuilt = pd.DataFrame(rows, columns=columns).sort_values("timestamp").reset_index(drop=True)
    csv_path = (root / receipt["csv"]).resolve()
    if not csv_path.is_relative_to(root) or digest(csv_path) != receipt["csv_sha256"]:
        raise ValueError("CSV SHA256 differs from receipt")
    actual = pd.read_csv(csv_path)
    if list(actual.columns) != columns or len(actual) != len(rebuilt) or len(actual) != receipt["rows"]:
        raise ValueError("Reconstructed row count/schema differs from CSV/receipt")
    actual["timestamp"] = pd.to_datetime(actual.timestamp, utc=True, errors="raise")
    if actual.timestamp.duplicated().any() or not actual.timestamp.is_monotonic_increasing:
        raise ValueError("CSV timestamps duplicated or unsorted")
    if not np.array_equal(actual.timestamp.to_numpy(), rebuilt.timestamp.to_numpy()):
        raise ValueError("CSV timestamp sequence differs from raw pages")
    expected_values = rebuilt[columns[1:]].to_numpy(np.float64)
    values = actual[columns[1:]].to_numpy(np.float64)
    if not np.allclose(values, expected_values, rtol=1e-14, atol=1e-12, equal_nan=True):
        raise ValueError("CSV values differ from reconstructed raw pages")
    result = {"passed": True, "receipt_sha256": digest(root / "receipt.json"),
              "csv_sha256": digest(csv_path), "pages_verified": len(seen_paths),
              "ranges_verified": len(ranges), "rows_verified": len(actual),
              "fields_verified": columns, "raw_to_csv_all_fields_verified": True,
              "formal_eligible": False, "point_in_time_available": "not established"}
    if save:
        write_json(root / "receipt_verification.json", result)
    return result
