"""Download public, license-labeled historical candidates for source audit only.

The uploader's label is recorded as a claim, not proof of upstream rights or data
quality. Archives are never executed or blindly extracted, and any CSV containing
dates after the development boundary is rejected before writing its bytes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import stat
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile, is_zipfile

from .spec import DEV_END, DEV_START, OUT

DATASETS = {
    "rishabh": "rishabhkmr/appleaapl-historical-share-price-since-listing",
    "shivam": "shiivvvaam/apple-stock-market-historical-data-1980-2024",
    "kpatel": "kpatel00/apple-stock-prices-updated-15-august-2024",
    "pratham": "prathamjyotsingh/apple-stocks",
    "ayesha": "ayeshaaqib/apple-stock-data-1999-2024",
}
LICENSE_LABELS = {"rishabh": "CC0: Public Domain", "shivam": "CC0: Public Domain",
                  "kpatel": "CC0: Public Domain", "pratham": "Apache 2.0",
                  "ayesha": "Attribution 4.0 International (CC BY 4.0)"}
MAX_ARCHIVE = 15_000_000
MAX_UNCOMPRESSED = 30_000_000


def get(url, accept):
    try:
        with urlopen(Request(url, headers={"User-Agent": "AAPL-academic-source-audit/1.0", "Accept": accept}), timeout=30) as response:
            body = response.read(MAX_ARCHIVE + 1)
            if len(body) > MAX_ARCHIVE:
                raise ValueError("Provider response exceeds size bound")
            return body
    except HTTPError as exc:
        raise RuntimeError(f"Public provider request returned HTTP {exc.code}") from None
    except (URLError, TimeoutError):
        raise RuntimeError("Public provider request failed or timed out") from None


def parse_day(value):
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(value[:10] if fmt == "%Y-%m-%d" else value, fmt).date()
        except ValueError:
            continue
    raise ValueError("Unrecognized or missing CSV date")


def csv_metadata(raw):
    try:
        stream = io.StringIO(raw.decode("utf-8-sig"), newline="")
    except UnicodeDecodeError:
        raise ValueError("CSV encoding is not UTF-8") from None
    reader = csv.DictReader(stream)
    if not reader.fieldnames:
        raise ValueError("CSV lacks a header")
    dates = [name for name in reader.fieldnames if name.strip().lower() in {"date", "datetime", "timestamp"}]
    if len(dates) != 1:
        raise ValueError("CSV must contain one explicit date column")
    found = []
    for row in reader:
        if not any(str(v or "").strip() for v in row.values()):
            continue
        day = parse_day(row[dates[0]])
        if not date.fromisoformat(DEV_START) <= day <= date.fromisoformat(DEV_END):
            raise ValueError("CSV contains a date outside the development range")
        found.append(day.isoformat())
    if not found or len(found) != len(set(found)):
        raise ValueError("CSV has no dates or duplicate dates")
    return {"columns": reader.fieldnames, "rows": len(found), "first": min(found), "last": max(found)}


def audit_archive(body):
    stream = io.BytesIO(body)
    if not is_zipfile(stream):
        raise ValueError("Provider did not return a ZIP archive")
    csvs = []
    with ZipFile(stream) as archive:
        infos = archive.infolist()
        if len(infos) > 50 or sum(info.file_size for info in infos) > MAX_UNCOMPRESSED:
            raise ValueError("Archive exceeds safety bounds")
        for item in infos:
            name = PurePosixPath(item.filename)
            mode = item.external_attr >> 16
            if name.is_absolute() or ".." in name.parts or stat.S_ISLNK(mode) or item.flag_bits & 1:
                raise ValueError("Archive has an unsafe or encrypted member")
            if item.filename.lower().endswith(".csv"):
                raw = archive.read(item)
                csvs.append((item.filename, raw, csv_metadata(raw)))
    if not csvs:
        raise ValueError("Archive has no auditable CSV file")
    return csvs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=tuple(DATASETS), required=True)
    args = parser.parse_args()
    ref = DATASETS[args.candidate]
    metadata_url = "https://www.kaggle.com/api/v1/datasets/view/" + ref
    download_url = "https://www.kaggle.com/api/v1/datasets/download/" + ref
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = OUT / "kaggle_candidates" / f"{args.candidate}_{stamp}"
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {"status": "incomplete", "ref": ref, "metadata_url": metadata_url,
               "download_url": download_url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    try:
        metadata_bytes = get(metadata_url, "application/json")
        metadata = json.loads(metadata_bytes)
        updated = datetime.fromisoformat(metadata["lastUpdated"].replace("Z", "+00:00"))
        if metadata.get("licenseName") != LICENSE_LABELS[args.candidate] or updated.date() > date.fromisoformat(DEV_END):
            raise ValueError("Dataset license label or historical snapshot date does not meet the candidate policy")
        raw = get(download_url, "application/zip")
        csvs = audit_archive(raw)
        (directory / "dataset_metadata.json").write_bytes(metadata_bytes)
        (directory / "download.zip").write_bytes(raw)
        entries = []
        for i, (original_name, content, summary) in enumerate(csvs):
            name = f"candidate_{i:02d}.csv"
            (directory / name).write_bytes(content)
            entries.append({"file": name, "original_name": original_name,
                            "sha256": hashlib.sha256(content).hexdigest(), **summary})
        receipt.update(status="candidate_not_admitted", license_label=metadata["licenseName"],
                       last_updated=metadata["lastUpdated"], description=metadata.get("description", ""),
                       upstream_rights_independently_verified=False,
                       metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                       archive_sha256=hashlib.sha256(raw).hexdigest(), files=entries, Q3_values_read=False)
    except Exception as exc:
        receipt["failure"] = f"{type(exc).__name__}: {exc}"
        (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf8")
        print(json.dumps({"directory": str(directory), "status": "incomplete", "failure": receipt["failure"]}))
        raise SystemExit(1) from None
    (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({"directory": str(directory), "status": receipt["status"], "ref": ref,
                      "license_label": receipt["license_label"], "files": receipt["files"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
