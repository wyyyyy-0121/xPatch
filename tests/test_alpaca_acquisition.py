import csv
import json
from pathlib import Path
import sys

import pytest

from data_acquisition import download_alpaca_sip as downloader
from finance_v2.io import digest, write_json
from finance_v2.provenance import verify_alpaca_run


def fixture_run(root):
    root.mkdir()
    (root / "raw").mkdir()
    rows = [{"t": "2024-03-08T14:30:00Z", "o": 100., "h": 101., "l": 99.,
             "c": 100.5, "v": 123, "n": 4, "vw": 100.25},
            {"t": "2024-03-08T14:31:00Z", "o": 100.5, "h": 102., "l": 99.,
             "c": 101., "v": 456, "n": 5, "vw": 100.75}]
    pages = []
    for i, row in enumerate(rows):
        p = root / "raw" / f"range_000_page_{i:04}.json"
        token = "next-page" if i == 0 else None
        write_json(p, {"bars": {"AAPL": [row]}, "next_page_token": token})
        pages.append({"file": str(p.relative_to(root)), "sha256": digest(p), "bytes": p.stat().st_size,
                      "page": i, "range_start": "2024-03-08", "range_end": "2024-03-08",
                      "has_next_page": bool(token), "rows_returned": 1})
    csv_path = root / "AAPL_sip_1min_raw.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"])
        for i, row in enumerate(rows):
            writer.writerow([f"2024-03-08T09:{30+i}:00-05:00", row["o"], row["h"], row["l"],
                             row["c"], row["v"], row["n"], row["vw"]])
    manifest = {"symbol": "AAPL", "feed": "sip", "adjustment": "raw", "timeframe": "1Min",
                "endpoint": downloader.ENDPOINT, "purpose": "development_data_acquisition_only",
                "download_complete": True, "ranges": [["2024-03-08", "2024-03-08"]],
                "pages": pages, "csv": csv_path.name, "rows": 2, "csv_sha256": digest(csv_path)}
    write_json(root / "receipt.json", manifest)
    return manifest


def test_receipt_compares_values_and_pagination_not_only_csv_hash(tmp_path):
    root = tmp_path / "run"
    manifest = fixture_run(root)
    assert verify_alpaca_run(root)["rows_verified"] == 2
    csv_path = root / manifest["csv"]
    csv_path.write_text(csv_path.read_text().replace("100.5,123", "100.7,123"))
    manifest["csv_sha256"] = digest(csv_path)
    write_json(root / "receipt.json", manifest)
    with pytest.raises(ValueError, match="values differ"):
        verify_alpaca_run(root)


def test_missing_raw_page_and_incomplete_pagination_rejected(tmp_path):
    root = tmp_path / "run"
    manifest = fixture_run(root)
    manifest["pages"] = manifest["pages"][:1]
    write_json(root / "receipt.json", manifest)
    with pytest.raises(ValueError, match="pagination"):
        verify_alpaca_run(root)


def test_resume_verified_cached_pages_never_refetches(tmp_path, monkeypatch):
    root = tmp_path / "run"
    fixture_run(root)
    original = digest(root / "AAPL_sip_1min_raw.csv")
    monkeypatch.setattr(downloader, "read_credentials", lambda path: ("example", "example"))
    def forbidden_request(*args):
        raise AssertionError("Completed raw pages must not be fetched again")
    monkeypatch.setattr(downloader, "request_body", forbidden_request)
    monkeypatch.setattr(sys, "argv", ["download", "--start", "2024-03-08", "--end", "2024-03-08",
                                     "--output", str(root), "--resume"])
    downloader.main()
    assert digest(root / "AAPL_sip_1min_raw.csv") == original
    assert verify_alpaca_run(root)["passed"]
