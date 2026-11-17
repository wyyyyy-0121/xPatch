"""Read historical test-use metadata without reopening historical scores."""
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from .bars import file_digest


def legacy_ledger(path: Path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Historical test-use ledger is empty or malformed")
    intervals = []
    for record in records:
        if record.get("symbol") != "AAPL" or not record.get("status"):
            raise ValueError("Historical ledger contains an unexpected entry")
        start = date.fromisoformat(record["start"][:10])
        end = date.fromisoformat(record["end"][:10])
        if end < start:
            raise ValueError("Historical exposure interval is reversed")
        intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return {"sha256": file_digest(path), "entries": len(records),
            "status_labels": sorted(set(row["status"] for row in records)),
            "merged_intervals": [{"start": str(a), "end": str(b)} for a, b in merged],
            "scope": "known legacy use only; absence of a ledger overlap does not prove independence"}


def overlapping_origin_count(session_dates, origins, horizon, ledger):
    origins = np.asarray(origins, dtype=np.int64)
    labels = np.asarray(session_dates)
    if horizon < 1 or (origins < 0).any() or (origins + horizon >= len(labels)).any():
        raise ValueError("Invalid origins or horizon")
    exposed = np.zeros(len(origins), dtype=bool)
    for interval in ledger["merged_intervals"]:
        exposed |= ((labels[origins] <= interval["end"]) &
                    (labels[origins + horizon] >= interval["start"]))
    return int(exposed.sum())
