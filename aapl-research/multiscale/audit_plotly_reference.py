"""Compare a pinned 2017 public sample with the disputed 2015 source date.

The sample's upstream market-data rights are not established. This downloads no
data after the development cutoff and does not admit a source or score a model.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .acquire_kaggle_candidate import csv_metadata
from .audit_alpha_candidates import compare_raw_to_adjusted, read_candidate
from .audit_kaggle_candidates import candidate, compare, kpatel_splits
from .bars import file_digest
from .spec import DEV_END, OUT


COMMIT = "6a88de1c9a95f0019566d3b568eb26444feda686"
URL = f"https://api.github.com/repos/plotly/datasets/contents/finance-charts-apple.csv?ref={COMMIT}"
COMMIT_URL = f"https://api.github.com/repos/plotly/datasets/commits/{COMMIT}"
MAX_BYTES = 2_000_000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pratham", type=Path, required=True)
    parser.add_argument("--kpatel", type=Path, required=True)
    args = parser.parse_args()
    headers = {"User-Agent": "AAPL-academic-source-audit/1.0", "Accept": "application/vnd.github+json"}
    with urlopen(Request(COMMIT_URL, headers=headers), timeout=20) as response:
        commit_bytes = response.read(MAX_BYTES + 1)
    if len(commit_bytes) > MAX_BYTES:
        raise ValueError("Commit metadata exceeds the download size limit")
    commit = json.loads(commit_bytes)
    committed_at = datetime.fromisoformat(commit["commit"]["committer"]["date"].replace("Z", "+00:00"))
    if commit.get("sha") != COMMIT or committed_at.date().isoformat() > DEV_END:
        raise ValueError("Reference commit identity or date does not meet the cutoff")
    with urlopen(Request(URL, headers=headers), timeout=20) as response:
        metadata_bytes = response.read(2 * MAX_BYTES + 1)
    if len(metadata_bytes) > 2 * MAX_BYTES:
        raise ValueError("Reference response exceeds the download size limit")
    metadata = json.loads(metadata_bytes)
    if metadata.get("encoding") != "base64" or metadata.get("size", MAX_BYTES + 1) > MAX_BYTES:
        raise ValueError("Unexpected reference encoding or size")
    raw = base64.b64decode(metadata["content"])
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if len(raw) != metadata["size"] or blob != metadata["sha"]:
        raise ValueError("Reference bytes do not match the GitHub blob")
    info = csv_metadata(raw)
    if info["last"] >= "2020-01-01" or "AAPL.Close" not in info["columns"]:
        raise ValueError("Unexpected reference date range or price column")
    frame = pd.read_csv(io.BytesIO(raw), usecols=["Date", "AAPL.Close"])
    frame = frame.rename(columns={"Date": "date", "AAPL.Close": "close"}).sort_values("date", ignore_index=True)
    frame["close"] = pd.to_numeric(frame["close"], errors="raise")
    if not np.isfinite(frame.close).all() or (frame.close <= 0).any() or frame.date.duplicated().any():
        raise ValueError("Invalid reference dates or closes")
    alpha_dir, _, alpha = read_candidate(args.pratham, "pratham")
    k_dir, k_receipt, adjusted = candidate(args.kpatel, "kpatel")
    actions = kpatel_splits(k_dir, k_receipt)
    disputed = "2015-09-30"
    p, a, k = (series.loc[series.date == disputed, "close"] for series in (frame, alpha, adjusted))
    if not (len(p) == len(a) == len(k) == 1):
        raise ValueError("The disputed date is absent from a source")
    factor = float(np.prod([event["ratio"] for event in actions if event["date"] > disputed]))
    differences = {"plotly_vs_alpha_relative": float(abs(p.iloc[0] - a.iloc[0]) / a.iloc[0]),
                   "plotly_vs_kpatel_after_split_conversion_relative":
                       float(abs(p.iloc[0] - k.iloc[0] * factor) / (k.iloc[0] * factor))}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUT / "source_references" / f"plotly_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "reference.csv").write_bytes(raw)
    (directory / "github_metadata.json").write_bytes(metadata_bytes)
    (directory / "commit_metadata.json").write_bytes(commit_bytes)
    result = {"status": "diagnostic_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "reference_url": URL, "pinned_commit": COMMIT, "commit_date_utc": committed_at.isoformat(),
              "reference_sha256": hashlib.sha256(raw).hexdigest(), "github_blob_sha1": blob,
              "github_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
              "commit_metadata_sha256": hashlib.sha256(commit_bytes).hexdigest(), "reference_metadata": info,
              "source_receipt_sha256": {"alpha": file_digest(alpha_dir / "receipt.json"),
                                         "kpatel": file_digest(k_dir / "receipt.json")},
              "Q3_values_read": False, "experiment_scored": False,
              "comparisons": {"reference_vs_alpha_nominal": compare(frame, alpha),
                              "reference_vs_kpatel_split_adjusted": compare_raw_to_adjusted(frame, adjusted, actions)},
              "disputed_month_end": {"date": disputed, **differences},
              "limitations": ["The reference's original provider and data licence are not independently verified.",
                              "Several public sources can share upstream data; agreement does not prove independence.",
                              "Price-unit conversion uses the candidate-reported split table.",
                              "This is not permission to train, publish, or redistribute the reference data."]}
    (directory / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"directory": str(directory), "status": result["status"],
                      "reference_metadata": info, "disputed_month_end": result["disputed_month_end"]}))


if __name__ == "__main__":
    main()
