"""Measure read-only candidate bar I/O without training or exposing prices.

The output is local metadata for capacity planning, not source admission, a
publication-rights decision, or a forecast experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import socket
import time

import numpy as np
import pandas as pd

from .bars import file_digest
from .spec import DEV_END, NY, OUT, ROOT


SOURCE_FILES = ("multiscale/source_io_benchmark.py", "multiscale/bars.py", "multiscale/spec.py")


def measure(directory: Path, frequencies, repetitions=3):
    directory = directory.resolve()
    if not directory.is_relative_to((OUT / "admission").resolve()):
        raise ValueError("Candidate admission is outside the isolated workspace")
    if not 1 <= repetitions <= 5 or not frequencies or len(set(frequencies)) != len(frequencies):
        raise ValueError("Invalid I/O measurement scope")
    receipt_path = (directory / "admission.json").resolve()
    if not receipt_path.is_relative_to(directory) or not receipt_path.is_file():
        raise ValueError("Invalid admission receipt path")
    admission_hash = file_digest(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if (receipt.get("Q3_values_read") is not False or receipt.get("experiment_scored") is not False or
            receipt.get("development_cutoff_new_york") != DEV_END):
        raise ValueError("Candidate receipt does not establish the required cutoff")
    source_hashes = {name: file_digest(ROOT / name) for name in SOURCE_FILES}
    cutoff = pd.Timestamp("2026-07-01", tz=NY).tz_convert("UTC").as_unit("ns").value
    records = []
    for frequency in frequencies:
        if frequency not in ("minute", "hour", "day", "month"):
            raise ValueError("Unknown bar frequency")
        name = f"bars_{frequency}.npz"
        path = (directory / name).resolve()
        expected = receipt.get("files", {}).get(name)
        if not path.is_relative_to(directory) or not path.is_file() or not expected:
            raise ValueError("Candidate bar path or hash is missing")
        for run in range(1, repetitions + 1):
            start = time.monotonic()
            if file_digest(path) != expected:
                raise ValueError("Candidate bar bytes differ from the admission receipt")
            verified = time.monotonic()
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"start_ns", "end_ns", "close", "observed", "session_date"}:
                    raise ValueError("Unexpected bar fields")
                dates, end, observed = (data[key] for key in ("session_date", "end_ns", "observed"))
                if (dates.dtype.kind != "U" or dates.ndim != 1 or not len(dates) or
                        (dates > DEV_END).any() or observed.dtype != np.bool_ or len(end) != len(observed) or
                        (end[observed] >= cutoff).any()):
                    raise ValueError("Candidate metadata exceeds the protected cutoff")
                metadata_done = time.monotonic()
                close = data["close"]  # Values are neither printed nor used to fit a model.
                if close.ndim != 1 or len(close) != len(dates):
                    raise ValueError("Bar close-array shape differs from metadata")
                close_done = time.monotonic()
                records.append({"frequency": frequency, "run": run, "bar_count": len(dates),
                                "compressed_file_bytes": path.stat().st_size,
                                "decoded_close_bytes": close.nbytes,
                                "identity_hash_seconds": verified - start,
                                "metadata_decode_seconds": metadata_done - verified,
                                "close_decode_seconds": close_done - metadata_done,
                                "total_seconds": close_done - start})
    end_hashes = {name: file_digest(ROOT / name) for name in SOURCE_FILES}
    if end_hashes != source_hashes:
        raise ValueError("I/O measurement source changed during the run")
    return {"status": "candidate_read_only_io_not_admitted", "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "admission_sha256": admission_hash, "host": socket.gethostname(), "frequencies": list(frequencies),
            "repetitions": repetitions, "records": records, "source_sha256_at_start": source_hashes,
            "source_sha256_at_end": end_hashes, "cpu_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "Q3_values_read": False, "model_trained": False, "experiment_scored": False,
            "limitations": ["The operating-system cache was not cleared; even the first run may be warm.",
                            "Measures identity hashing and array decoding only, not feature generation, training, validation or prediction writing.",
                            "Data rights, price accuracy and the protocol are not established by this measurement.",
                            "Changing source bytes or running on another machine requires a new measurement."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--frequencies", nargs="+", choices=("minute", "hour", "day", "month"),
                        default=["minute", "hour", "day", "month"])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    report = measure(args.admission, args.frequencies, args.repetitions)
    directory = OUT / "source_io_benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"status": report["status"], "path": str(path),
                      "frequencies": report["frequencies"], "model_trained": False}))


if __name__ == "__main__":
    main()
