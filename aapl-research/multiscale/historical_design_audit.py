"""Audit the original fold design and clock horizons using metadata alone.

The original training modules, protocols, and artifacts remain unchanged. NPZ
members containing closes, targets, or predictions are never materialized.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .historical_coverage import (ANNUAL, CONTROLS, CORE, MATRIX, OUT, TASKS,
                                  digest, in_tree, read_json, require, snapshot)
from .historical_minute_check import CUTOFF_NS, fold_sources, local_time
from .historical_results import ADMISSION, FREQUENCIES, checked_coverage


HOUR_NS = 3_600_000_000_000


@dataclass(frozen=True)
class Metadata:
    start: np.ndarray
    end: np.ndarray
    observed: np.ndarray
    sessions: np.ndarray

    def validate(self):
        n = len(self.end)
        require(n > 0 and all(a.ndim == 1 and len(a) == n for a in
                             (self.start, self.end, self.observed, self.sessions)),
                "Bar metadata must be nonempty, one-dimensional and aligned")
        require(self.start.dtype.kind == self.end.dtype.kind == "i" and
                self.observed.dtype == np.bool_ and self.sessions.dtype.kind == "U",
                "Unexpected metadata types")
        require(np.all(np.diff(self.start) > 0) and np.all(np.diff(self.end) > 0) and
                np.all(self.start < self.end) and np.all(self.start[1:] >= self.end[:-1]) and
                np.all(self.sessions[1:] >= self.sessions[:-1]),
                "Bar intervals or sessions are unordered or overlap")
        require(np.all(self.sessions <= "2026-06-30") and
                np.all(self.end[self.observed] < CUTOFF_NS),
                "Metadata crosses the protected cutoff")


def index_digest(values):
    return sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def load_metadata(directory, admission, frequency):
    path = in_tree(directory / f"bars_{frequency}.npz")
    require(digest(path) == admission["files"].get(path.name), "Admitted bar archive changed")
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {"start_ns", "end_ns", "observed", "session_date", "close"},
                "Unexpected admitted archive fields")
        result = Metadata(*(z[key].copy() for key in
                            ("start_ns", "end_ns", "observed", "session_date")))
    result.validate()
    return result, {"path": str(path), "sha256": digest(path),
                    "members_read": ["start_ns", "end_ns", "observed", "session_date"]}


def complete_origins(meta, lookback, horizon, same_session):
    require(lookback > 0 and horizon > 0, "Nonpositive history or horizon")
    origins = np.arange(lookback - 1, len(meta.end) - horizon, dtype=np.int64)
    missing = np.concatenate(([0], np.cumsum(~meta.observed, dtype=np.int64)))
    valid = missing[origins + horizon + 1] == missing[origins - lookback + 1]
    if same_session:
        valid &= meta.sessions[origins] == meta.sessions[origins + horizon]
    return origins[valid]


def partitions(meta, origins, horizon, validation):
    train = origins[meta.end[origins + horizon] < meta.end[validation[0]]]
    groups = np.unique(meta.sessions[train])
    tail = max(1, (15 * len(groups) + 99) // 100)
    stop = train[np.isin(meta.sessions[train], groups[-tail:])]
    fit = train[meta.end[train + horizon] < meta.end[stop[0]]] if len(stop) else train[:0]
    return {"train": train, "fit": fit, "stop": stop, "validation": validation}


def reconstruct_folds(meta, origins, horizon, strategy):
    """Reconstruct the count-only rules without importing the training splitter."""
    groups = np.unique(meta.sessions[origins])
    require(strategy in ("rolling", "warmup"), "Unknown recorded split rule")
    if strategy == "rolling":
        require(len(groups) >= 6, "Insufficient groups for the original six chunks")
        chunks = np.array_split(groups, 6)[1:]
    else:
        for warmup in range(1, len(groups) - 5 * 6 + 1):
            first = origins[meta.sessions[origins] == groups[warmup]][:1]
            part = partitions(meta, origins, horizon, first)
            if len(part["fit"]) >= 24 and len(part["stop"]) >= 4:
                break
        else:
            raise ValueError("Original count-only warmup cannot form five folds")
        chunks = np.array_split(groups[warmup:], 5)
    result = []
    for chunk in chunks:
        validation = origins[np.isin(meta.sessions[origins], chunk)]
        require(len(validation) > 0, "Empty reconstructed validation fold")
        part = partitions(meta, origins, horizon, validation)
        require(all(len(part[name]) > 0 for name in part), "Empty reconstructed partition")
        require(np.intersect1d(part["fit"], part["stop"]).size == 0 and
                meta.end[part["train"][-1] + horizon] < meta.end[validation[0]] and
                meta.end[part["fit"][-1] + horizon] < meta.end[part["stop"][0]],
                "Reconstructed labels overlap the next partition")
        result.append(part)
    return result


def clock_summary(meta, origins, horizon):
    origins = np.asarray(origins, dtype=np.int64)
    require(len(origins) > 0 and np.all(np.diff(origins) > 0) and origins[0] >= 0 and
            origins[-1] + horizon < len(meta.end), "Invalid timing origins")
    elapsed = (meta.end[origins + horizon] - meta.end[origins]) / HOUR_NS
    require(np.all(elapsed > 0), "Nonpositive clock horizon")
    quantiles = np.quantile(elapsed, [0, .5, .95, 1], method="linear")
    crossing = meta.sessions[origins + horizon] != meta.sessions[origins]
    return {"origins": len(origins), "elapsed_hours_min": float(quantiles[0]),
            "elapsed_hours_median": float(quantiles[1]), "elapsed_hours_p95": float(quantiles[2]),
            "elapsed_hours_max": float(quantiles[3]), "cross_session_origins": int(crossing.sum()),
            "cross_session_fraction": float(crossing.mean()),
            "first_origin_session": str(meta.sessions[origins[0]]),
            "last_origin_session": str(meta.sessions[origins[-1]]),
            "first_decision_ny": local_time(meta.end[origins[0]]),
            "last_decision_ny": local_time(meta.end[origins[-1]]),
            "first_target_session": str(meta.sessions[origins[0] + 1]),
            "last_target_session": str(meta.sessions[origins[-1] + horizon]),
            "last_completion_ny": local_time(meta.end[origins[-1] + horizon])}


def original_jobs(admission_hash):
    records, protocols = {}, {}

    def protocol(path, expected=None):
        path = in_tree(path)
        value, identity = read_json(path), digest(path)
        require(expected is None or expected == identity, "Original protocol changed")
        require(value.get("admission_sha256") == admission_hash and
                value.get("Q3_values_read") is False and value.get("independent_test") is False,
                "Protocol source or development scope differs")
        protocols[str(path)] = identity
        return value

    def add(key, job, lookback, path):
        require(key not in records, "Duplicated original task-fold")
        hashes = job.get("split_sha256") or {
            name: job.get(name + "_indices_sha256") for name in ("train", "fit", "stop", "validation")}
        require(all(hashes.get(name) for name in ("train", "fit", "stop", "validation")),
                "Missing original partition hashes")
        records[key] = {"job": job, "lookback": lookback, "hashes": hashes, "protocol": str(path)}

    for folder in (CORE, ANNUAL):
        path = folder / "PRIVATE_PROTOCOL.json"
        plan = protocol(path)
        for job in plan["jobs"]:
            annual = folder == ANNUAL
            key = (("annual", 12, job["target"], job["fold"]) if annual else
                   (job["profile"], job["horizon"], job["target"], job["fold"]))
            add(key, job, plan["lookback_months"] if annual else job["lookback"], path)
    path = MATRIX / "PRIVATE_PROTOCOL.json"
    plan = protocol(path)
    for job in plan["jobs"]:
        add(("minute", job["horizon"], job["target"], job["fold"]), job, plan["lookback"], path)
    sources = {str(Path(s["protocol"]).parent): s for s in plan["external_sources"]}
    for item in plan["excluded_external_jobs"]:
        key = ("minute", item["horizon"], item["target"], item["fold"])
        if item["source"] == "verified full-origin controls and neural first fold":
            path = CONTROLS / "PRIVATE_PROTOCOL.json"
            other = protocol(path)
            add(key, other, other["lookback"], path)
        else:
            source = sources[item["source"]]
            path = Path(source["protocol"])
            other = protocol(path, source["protocol_sha256"])
            jobs = [j for j in other["jobs"] if
                    (j["fold"] == item["fold"] if source["kind"] == "h60_return"
                     else j["key"] == item["worker"])]
            require(len(jobs) == 1, "Ambiguous original external fold")
            add(key, jobs[0], other["lookback"], path)
    require(set(records) == {(*task, f) for task in TASKS for f in range(1, 6)},
            "Original protocols do not cover the 100-fold design exactly")
    return records, protocols


def verify_saved_metadata(row, meta, validation, lookback, hashes):
    target_dir, _, _, _, target_hash = fold_sources(row)
    path = target_dir / "targets.npz"
    require(digest(path) == target_hash, "Saved target archive changed")
    with np.load(path, allow_pickle=False) as z:
        require(np.array_equal(z["origins"], validation) and
                np.array_equal(z["decision_ns"], meta.end[validation]) and
                np.array_equal(z["target_completion_ns"], meta.end[validation + row["horizon"]]),
                "Saved forecast timing differs from original admitted bar positions")
    if "fold_directory" in row["evidence"]:
        receipt = read_json(target_dir / "fold_receipt.json")
        require(receipt["config"]["lookback"] == lookback and
                all(receipt["identity_fields"].get(name + "_indices_sha256") == value
                    for name, value in hashes.items()), "Saved training partitions differ from design")
    return {"path": str(path), "sha256": target_hash,
            "members_read": ["origins", "decision_ns", "target_completion_ns"]}


def audit(coverage):
    rows = checked_coverage(coverage, snapshot(), allow_partial=True)
    admission = read_json(ADMISSION / "admission.json")
    admission_hash = digest(ADMISSION / "admission.json")
    jobs, protocols = original_jobs(admission_hash)
    metadata, data_sources = {}, {}
    for frequency in sorted(set(FREQUENCIES.values())):
        metadata[frequency], data_sources[frequency] = load_metadata(ADMISSION, admission, frequency)
    designs, records, by_task = {}, [], defaultdict(list)
    saved_sources = {}
    for row in rows:
        profile, horizon, target, number = (row[k] for k in ("profile", "horizon", "target", "fold"))
        key = (profile, horizon, target, number)
        original = jobs[key]
        lookback = original["lookback"]
        meta = metadata[FREQUENCIES[profile]]
        design_key = (profile, horizon, lookback)
        if design_key not in designs:
            origins = complete_origins(meta, lookback, horizon, profile == "minute")
            strategy = "warmup" if profile in ("annual", "quarterly") else "rolling"
            designs[design_key] = reconstruct_folds(meta, origins, horizon, strategy)
        part = designs[design_key][number - 1]
        hashes = {name: index_digest(values) for name, values in part.items()}
        require(hashes == original["hashes"], "Reconstructed split differs from original protocol")
        for name in ("fit", "stop", "validation"):
            require(len(part[name]) == original["job"][name + "_origins"],
                    "Reconstructed origin count differs from protocol")
        val = part["validation"]
        timing = clock_summary(meta, val, horizon)
        if profile == "minute":
            require(timing["cross_session_origins"] == 0 and
                    np.all(meta.end[val + horizon] - meta.end[val] == horizon * 60_000_000_000),
                    "Minute target crosses a session or omits an expected minute")
        saved = None
        if row["status"] == "completed":
            saved = verify_saved_metadata(row, meta, val, lookback, hashes)
            saved_sources[saved["path"]] = saved
        record = {"profile": profile, "horizon": horizon, "target": target, "fold": number,
                  "experiment_status_at_snapshot": row["status"], "lookback": lookback,
                  "frequency": FREQUENCIES[profile], "protocol": original["protocol"],
                  "split_sha256": hashes, "counts": {name: len(values) for name, values in part.items()},
                  "outer_label_purge_gap_hours": float((meta.end[val[0]] -
                      meta.end[part["train"][-1] + horizon]) / HOUR_NS),
                  "inner_label_purge_gap_hours": float((meta.end[part["stop"][0]] -
                      meta.end[part["fit"][-1] + horizon]) / HOUR_NS),
                  "last_fit_label_ny": local_time(meta.end[part["fit"][-1] + horizon]),
                  "first_stop_decision_ny": local_time(meta.end[part["stop"][0]]),
                  "last_train_label_ny": local_time(meta.end[part["train"][-1] + horizon]),
                  "saved_metadata_checked": saved is not None, **timing}
        records.append(record)
        by_task[(profile, horizon, target)].append((row, val))
    tasks = []
    for (profile, horizon, target), items in sorted(by_task.items()):
        meta = metadata[FREQUENCIES[profile]]
        planned = np.concatenate([indices for _, indices in items])
        completed = [indices for row, indices in items if row["status"] == "completed"]
        tasks.append({"profile": profile, "horizon": horizon, "target": target,
                      "planned_folds": 5, "completed_folds_at_snapshot": len(completed),
                      "planned_origin_timing": clock_summary(meta, planned, horizon),
                      "completed_origin_timing": clock_summary(meta, np.concatenate(completed), horizon)
                      if completed else None})
    paired = defaultdict(dict)
    for record in records:
        paired[(record["profile"], record["horizon"], record["fold"])][record["target"]] = record["split_sha256"]
    require(len(paired) == 50 and all(pair["future_close"] == pair["future_log_return"]
                                      for pair in paired.values()), "Price and return designs differ")
    return {"status": "historical_fold_design_and_clock_metadata_verified",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "planned_folds_verified": len(records), "completed_saved_fold_metadata_verified": len(saved_sources),
            "paired_price_return_designs_verified": len(paired), "Q3_values_read": False,
            "price_target_prediction_values_materialized": False, "new_training_updates": 0,
            "admission_sha256": admission_hash, "original_protocol_sha256": protocols,
            "bar_metadata_sources": data_sources, "saved_metadata_sources": saved_sources,
            "folds": records, "tasks": tasks,
            "limitations": ["Design timing for unfinished folds is not a completed experimental result.",
                            "Observed masks are inherited from the original admission; this does not reaudit price quality.",
                            "Clock horizons use recorded bar completion timestamps, not a historical vendor-availability guarantee.",
                            "All experimental periods remain exposed historical development."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    coverage_path = in_tree(args.coverage)
    coverage = read_json(coverage_path)
    result = audit(coverage)
    result["coverage"] = {"path": str(coverage_path), "sha256": digest(coverage_path),
                          "completed_folds_at_snapshot": coverage["completed_folds"]}
    result["source_sha256"] = {str(Path(__file__)): digest(Path(__file__))}
    destination = in_tree(args.destination)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "DESIGN_TIMING_AUDIT.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8")
    columns = ["profile", "horizon", "target", "fold", "experiment_status_at_snapshot", "lookback",
               "origins", "elapsed_hours_min", "elapsed_hours_median", "elapsed_hours_p95",
               "elapsed_hours_max", "cross_session_fraction", "first_decision_ny", "last_decision_ny",
               "last_completion_ny", "outer_label_purge_gap_hours", "inner_label_purge_gap_hours"]
    with (destination / "FOLD_TIMING.csv").open("x", newline="", encoding="utf8") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["folds"])
    with (destination / "COVERAGE.json").open("x", encoding="utf8") as stream:
        json.dump(coverage, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "planned_folds_verified",
                    "completed_saved_fold_metadata_verified", "paired_price_return_designs_verified")} |
                    {"destination": str(destination)}))


if __name__ == "__main__":
    main()
