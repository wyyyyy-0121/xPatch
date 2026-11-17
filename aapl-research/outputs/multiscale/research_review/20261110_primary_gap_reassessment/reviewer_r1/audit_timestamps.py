"""Admission-gated, fit-index/calendar-only audit. No price or label loader.

Only four independently stored calendar NPY files and fit.npy inside ten
registered split archives are decoded. The script deliberately does not import
the original Bars/Data classes, torch, model code or target constructors.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import zipfile

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[4]
STOP = False
MINUTE = 60_000_000_000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".atomic_", suffix=".json")
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encoded(value) + b"\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    except BaseException:
        try:
            os.close(fd)
        except OSError:
            pass
        raise
    finally:
        if temp.exists():
            temp.unlink()


def request_stop(*_):
    global STOP
    STOP = True


def check(deadline):
    if STOP or time.time() >= deadline:
        raise TimeoutError("Unchanged timestamp-audit deadline reached; completed chunks retained")


def install_guard(calendar, splits, output):
    allowed_inputs = {Path(p).resolve() for p in (*calendar, *splits)}
    output = output.resolve()

    def audit(event, args):
        if event in ("socket.connect", "socket.bind", "subprocess.Popen"):
            raise RuntimeError("Network/subprocess execution is forbidden in timestamp audit")
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any("holdout_sealed" in part.lower() for part in path.parts):
            raise RuntimeError("Sealed data is forbidden")
        mode, flags = args[1], args[2]
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(int(flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        if path in allowed_inputs:
            if writing:
                raise RuntimeError("Frozen calendar/split inputs are read-only")
            return
        if path.suffix.lower() in (".npy", ".npz", ".pt", ".pth", ".csv", ".zip", ".tar", ".pkl", ".pickle", ".parquet", ".h5", ".hdf5"):
            raise RuntimeError(f"Forbidden non-calendar/non-fit data path: {path}")
        if path.is_relative_to(output) and path.suffix.lower() != ".json":
            raise RuntimeError("Only JSON checkpoints/results may be opened inside the audit output tree")
        if writing and not path.is_relative_to(output):
            raise RuntimeError("Writes outside the timestamp audit output tree are forbidden")
    sys.addaudithook(audit)


def gate(args):
    protocol = read(args.protocol)
    if protocol.get("status") != "frozen_before_execution" or protocol.get("mode") != "timestamp_fit_only":
        raise RuntimeError("A separately frozen timestamp-only execution protocol is required")
    if float(args.deadline) != float(protocol["deadline_epoch"]):
        raise RuntimeError("CLI deadline differs from the frozen absolute deadline")
    check(args.deadline)
    if set(os.sched_getaffinity(0)) != {26}:
        raise RuntimeError("This audit is restricted to logical CPU26")
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    code = protocol.get("code_sha256", {})
    if code.get(str(Path(__file__).resolve())) != digest(__file__):
        raise RuntimeError("This exact audit source has not been frozen")
    for path, sha in code.items():
        if digest(path) != sha:
            raise RuntimeError("Frozen source changed")
    design_path = BASE / "AUDIT_DESIGN.json"
    if protocol.get("design_sha256") != digest(design_path):
        raise RuntimeError("Frozen fit-only design changed")
    design = read(design_path)
    for path, sha in protocol.get("metadata_sha256", {}).items():
        if digest(path) != sha:
            raise RuntimeError("Frozen source metadata changed")
    if (protocol.get("timestamp_arrays_admitted") is not True or protocol.get("price_labels_weights_admitted", False)
            or protocol.get("Q3_allowed", False) or protocol.get("network_allowed", False)):
        raise RuntimeError("Wrong timestamp execution admission")
    calendar = {row["path"]: row["expected_file_sha256"] for row in design["timestamp_input_allowlist"]}
    if len(calendar) != 4 or {Path(p).name for p in calendar} != {"timestamps_ns.npy", "dates.npy", "session_start.npy", "session_end.npy"}:
        raise RuntimeError("Exactly four original calendar NPY files are permitted")
    tasks = design["split_input_registration"]
    if len(tasks) != 2 or sum(len(t["folds"]) for t in tasks) != 10:
        raise RuntimeError("Exactly two tasks/five old fit states each are required")
    for task in tasks:
        if digest(task["TASK"]) != task["TASK_sha256"]:
            raise RuntimeError("Frozen TASK metadata changed")
    splits = [f["split_archive"] for task in tasks for f in task["folds"]]
    install_guard(calendar, splits, BASE / "timestamp_audit")
    return protocol, digest(args.protocol), design, calendar, tasks


def load_calendar(calendar, deadline):
    values, receipts = {}, []
    for path, expected in calendar.items():
        check(deadline)
        if digest(path) != expected:
            raise RuntimeError("Original calendar NPY identity changed")
        array = np.load(path, allow_pickle=False)
        values[Path(path).name] = array
        receipts.append({"path": path, "sha256": expected, "decoded": True})
    start, dates = values["timestamps_ns.npy"], values["dates.npy"]
    if (start.ndim != 1 or start.dtype.kind not in "iu" or dates.ndim != 1 or len(start) != len(dates) or dates.dtype.kind != "U"
            or np.any(dates > "2026-06-30") or not np.all(np.diff(start) > 0)):
        raise RuntimeError("Invalid original pre-Q3 timestamp/date arrays")
    if (values["session_start.npy"].shape != start.shape or values["session_start.npy"].dtype.kind != "b"
            or values["session_end.npy"].shape != start.shape or values["session_end.npy"].dtype.kind not in "iu"):
        raise RuntimeError("Invalid original session flag/end-index arrays")
    first = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    last = np.r_[first[1:] - 1, len(start) - 1]
    flags = np.zeros(len(start), dtype=bool)
    flags[first] = True
    if not np.array_equal(flags, values["session_start.npy"]):
        raise RuntimeError("Session-start flags differ from date runs")
    if not np.array_equal(np.repeat(last, last - first + 1), values["session_end.npy"]):
        raise RuntimeError("Session-end indices differ from date runs")
    if np.any((np.diff(start) != MINUTE) & (dates[1:] == dates[:-1])):
        raise RuntimeError("The expected within-session minute grid has a timestamp gap")
    local = pd.to_datetime(start, utc=True).tz_convert("America/New_York")
    if not np.array_equal(local.strftime("%Y-%m-%d").to_numpy(), dates):
        raise RuntimeError("Calendar session dates disagree with historical New York timezone")
    counts = last - first + 1
    hourly_counts = counts // 60
    hour_starts = np.concatenate([a + 60 * np.arange(n) for a, n in zip(first, hourly_counts)])
    hour_ends = hour_starts + 59
    metadata = {"sessions": len(first), "full_hour_bars": len(hour_starts),
                "hours_per_session_counts": {str(k): int(np.sum(hourly_counts == k)) for k in np.unique(hourly_counts)},
                "residual_minutes_total": int(np.sum(counts % 60)), "actual_provider_missingness_verified": False}
    expected = {"sessions": 1378, "full_hour_bars": 8238, "hours_per_session_counts": {"3": 10, "6": 1368}, "residual_minutes_total": 41340}
    if any(metadata[key] != value for key, value in expected.items()):
        raise RuntimeError("Timestamp-only reconstruction differs from frozen aggregation metadata")
    return {"minute": (start + MINUTE, dates), "hourly": (start[hour_ends] + MINUTE, dates[hour_ends])}, metadata, receipts


def fit_indices(fold):
    # Do not hash the whole archive or deserialize any other member.
    with zipfile.ZipFile(fold["split_archive"], "r") as archive:
        with archive.open("fit.npy", "r") as member:
            value = np.load(io.BytesIO(member.read()), allow_pickle=False)
    if value.ndim != 1 or value.dtype.kind not in "iu" or len(value) != fold["counts"]["fit"]:
        raise RuntimeError("Original fit index shape/count changed")
    if hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest() != fold["canonical_int64_sha256"]["fit"]:
        raise RuntimeError("Original canonical fit indices changed")
    return np.asarray(value, dtype=np.int64)


def timeline(end, dates, profile):
    first = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    last = np.r_[first[1:] - 1, len(end) - 1]
    sid = np.repeat(np.arange(len(first)), last - first + 1)
    local = pd.to_datetime(end, utc=True).tz_convert("America/New_York")
    clocks = local.strftime("%H:%M:%S").to_numpy()
    weekday = local.weekday.to_numpy()
    offsets = local.tz_localize(None).as_unit("ns").asi8 - end
    delta = np.r_[0, np.diff(end)]
    transitions = np.r_[False, dates[1:] != dates[:-1]]
    duration = MINUTE if profile == "minute" else 60 * MINUTE
    events = np.flatnonzero(transitions | ((delta != duration) & (np.arange(len(end)) > 0)))
    dst = np.flatnonzero(np.r_[False, offsets[1:] != offsets[:-1]])
    # These are calendar-derived descriptions. A skipped weekday is not
    # attributed to an exchange holiday rather than an omitted session.
    session_dates = np.asarray(dates[first], dtype="datetime64[D]")
    elapsed = (session_dates[1:] - session_dates[:-1]).astype(int)
    skipped_weekdays = np.busday_count(session_dates[:-1] + 1, session_dates[1:])
    weekend = first[1:][elapsed - 1 - skipped_weekdays > 0]
    weekday_gap = first[1:][skipped_weekdays > 0]
    return {"end": end, "dates": dates, "first": first, "last": last, "sid": sid,
            "clocks": clocks, "weekday": weekday, "delta": delta, "transitions": transitions,
            "events": events, "dst": dst, "weekend": weekend, "weekday_gap": weekday_gap,
            "counts": last - first + 1}


def origin_record(o, info, L, H, profile):
    low = o - L + 1
    if low < 0:
        raise RuntimeError("Original fit origin lacks its frozen lookback")
    events = info["events"][np.searchsorted(info["events"], low + 1):np.searchsorted(info["events"], o, side="right")]
    signature = ";".join(f"{o-j}:{int(info['delta'][j])}:{int(info['transitions'][j])}" for j in events)
    transitions = ";".join(str(o-j) for j in events if info["transitions"][j])
    current_session = int(info["sid"][o])
    slot = int(o - info["first"][current_session])

    def historical(lag):
        session = current_session - lag
        q = int(info["first"][session] + slot) if session >= 0 and slot < info["counts"][session] else None
        exists = q is not None and info["clocks"][q] == info["clocks"][o]
        if not exists:
            result = {"slot_exists": False, "mature_le": False, "mature_lt": False,
                    "maturity_equals_origin": False, "window_contains_both_endpoints": False,
                    "time_horizon_legal": False, "calendar_usable": False,
                    "fixed_nominal_same_slot_lag_shift_matches": False}
            if lag == 1:
                result["fixed_nominal_previous_session_shift_matches"] = False
            return result
        # Do not dereference a future completion timestamp to construct a past feature.
        completed = q + H <= o
        mature_le = completed and info["end"][q + H] <= info["end"][o]
        mature_lt = completed and info["end"][q + H] < info["end"][o]
        contained = q >= low and q + H <= o
        legal = profile != "minute" or (completed and info["dates"][q] == info["dates"][q + H])
        result = {"slot_exists": True, "mature_le": bool(mature_le), "mature_lt": bool(mature_lt),
                "maturity_equals_origin": bool(mature_le and not mature_lt),
                "window_contains_both_endpoints": bool(contained), "time_horizon_legal": bool(legal),
                "calendar_usable": bool(mature_le and contained and legal),
                "fixed_nominal_same_slot_lag_shift_matches": bool(o-q == lag * (390 if profile == "minute" else 6))}
        if lag == 1:
            result["fixed_nominal_previous_session_shift_matches"] = result["fixed_nominal_same_slot_lag_shift_matches"]
        return result

    nearest = None
    for lag in range(1, current_session + 1):
        session = current_session - lag
        if info["last"][session] < low:
            break
        if historical(lag)["calendar_usable"]:
            nearest = lag
            break
    completed_sessions = np.unique(info["sid"][events[info["transitions"][events]] - 1]) if len(events) else np.empty(0, dtype=int)
    normal = 390 if profile == "minute" else 6
    short = bool(np.any(info["counts"][completed_sessions] < normal))
    def has_event(key):
        return bool(np.searchsorted(info[key], o, side="right") > np.searchsorted(info[key], low + 1))
    return {"clock": f"{int(info['weekday'][o])}:{info['clocks'][o]}", "gap_signature": signature,
            "transition_signature": transitions, "previous1": historical(1), "previous2": historical(2),
            "nearest_matured_visible_session_lag": nearest,
            "historical_short_session": short, "historical_DST_offset_change": has_event("dst"),
            "historical_weekend_crossing": has_event("weekend"),
            "historical_weekday_non_session_gap": has_event("weekday_gap")}


def state(task, fold, info, identity, deadline, cache, seen):
    name = task["task"] + "_f" + str(fold["fold"])
    output = BASE / "timestamp_audit" / name
    output.mkdir(parents=True, exist_ok=True)
    final = output / "RESULT.json"
    indices = fit_indices(fold)
    if (np.any(indices < task["lookback"] - 1) or np.any(indices >= len(info["end"]))
            or not np.all(np.diff(indices) > 0)):
        raise RuntimeError("Fit origins are not the original sorted legal index sequence")
    seen.update(map(int, indices))
    if final.exists():
        old = read(final)
        if old.get("identity") != identity or old.get("origins") != len(indices):
            raise RuntimeError("Existing audit state belongs to another protocol/cohort")
        return old
    progress = output / "PROGRESS.json"
    stats = read(progress) if progress.exists() else {"identity": identity, "cursor": 0, "clock_groups": {},
                "previous1": {}, "previous2": {}, "nearest_lag_counts": {}, "boundary_counts": {},
                "failure_strata": {}, "bounded_timestamp_only_examples": {}}
    if stats.get("identity") != identity:
        raise RuntimeError("Audit checkpoint identity changed")
    cursor = stats.get("cursor")
    if (not isinstance(cursor, int) or cursor < 0 or cursor > len(indices)
            or (cursor != len(indices) and cursor % 4096 != 0)
            or sum(g["origins"] for g in stats["clock_groups"].values()) != cursor
            or sum(stats["nearest_lag_counts"].values()) != cursor
            or (cursor and any(stats[p].get("slot_exists", 0) > cursor for p in ("previous1", "previous2")))):
        raise RuntimeError("Audit checkpoint cursor/count closure failed")
    for start in range(stats["cursor"], len(indices), 4096):
        check(deadline)
        for o in indices[start:start + 4096]:
            o = int(o)
            if o not in cache:
                cache[o] = origin_record(o, info, task["lookback"], task["horizon"], task["profile"])
            row = cache[o]
            group = stats["clock_groups"].setdefault(row["clock"], {"origins": 0, "gap_signatures": {}, "transition_signatures": {}})
            group["origins"] += 1
            for source, destination in (("gap_signature", "gap_signatures"), ("transition_signature", "transition_signatures")):
                group[destination][row[source]] = group[destination].get(row[source], 0) + 1
            for previous in ("previous1", "previous2"):
                for key, value in row[previous].items():
                    stats[previous][key] = stats[previous].get(key, 0) + int(value)
            lag = str(row["nearest_matured_visible_session_lag"])
            stats["nearest_lag_counts"][lag] = stats["nearest_lag_counts"].get(lag, 0) + 1
            for key in ("historical_short_session", "historical_DST_offset_change", "historical_weekend_crossing", "historical_weekday_non_session_gap"):
                stats["boundary_counts"][key] = stats["boundary_counts"].get(key, 0) + int(row[key])
            strata = {key: row[key] for key in stats["boundary_counts"]}
            for previous in ("previous1", "previous2"):
                candidate = row[previous]
                strata[previous + "_missing_slot"] = not candidate["slot_exists"]
                strata[previous + "_unmatured"] = candidate["slot_exists"] and not candidate["mature_le"]
                strata[previous + "_mature_but_cross_session_illegal"] = candidate["mature_le"] and not candidate["time_horizon_legal"]
                strata[previous + "_mature_but_outside_lookback"] = candidate["mature_le"] and not candidate["window_contains_both_endpoints"]
                strata[previous + "_mature_equals_origin"] = candidate["maturity_equals_origin"]
            for key, value in strata.items():
                stats["failure_strata"][key] = stats["failure_strata"].get(key, 0) + int(value)
                examples = stats["bounded_timestamp_only_examples"].setdefault(key, [])
                if value and len(examples) < 3:
                    examples.append({"origin_index": o, "origin_end_ns": int(info["end"][o]),
                                     "session_date": str(info["dates"][o]), "clock": row["clock"],
                                     "previous1": row["previous1"], "previous2": row["previous2"]})
        stats["cursor"] = min(start + 4096, len(indices))
        atomic(progress, stats)
    groups = stats["clock_groups"].values()
    summary = {"clock_groups": len(stats["clock_groups"]),
        "clock_groups_with_multiple_gap_signatures": sum(len(g["gap_signatures"]) > 1 for g in groups),
        "origins_in_multiple_gap_signature_groups": sum(g["origins"] for g in groups if len(g["gap_signatures"]) > 1),
        "clock_groups_with_multiple_transition_signatures": sum(len(g["transition_signatures"]) > 1 for g in groups),
        "origins_in_multiple_transition_signature_groups": sum(g["origins"] for g in groups if len(g["transition_signatures"]) > 1)}
    result = {"status": "completed_calendar_only", "identity": identity, "task": task["task"], "fold": fold["fold"],
              "partition": "fit", "origins": len(indices), "summary": summary,
              "previous1_counts": stats["previous1"], "previous2_counts": stats["previous2"],
              "nearest_matured_visible_session_lag_counts": stats["nearest_lag_counts"],
              "boundary_counts": stats["boundary_counts"], "details": str(progress), "details_sha256": digest(progress),
              "failure_strata": stats["failure_strata"],
              "bounded_timestamp_only_examples": stats["bounded_timestamp_only_examples"],
              "weekday_gap_is_not_verified_exchange_holiday": True,
              "history_interval": "[o-L+1,o] inclusive", "maturity_rule": "end_ns[p+H]<=end_ns[o]",
              "observed_path_verified": False, "point_in_time_receipt_verified": False,
              "predictive_information_or_method_gain_established": False,
              "decoded_split_member": "fit.npy", "close_labels_weights_read": False, "Q3_read": False}
    atomic(final, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    protocol, ph, design, calendar, tasks = gate(args)
    results, totals = [], {}
    try:
        arrays, metadata, receipts = load_calendar(calendar, args.deadline)
        for task in tasks:
            end, dates = arrays[task["profile"]]
            info, cache, seen = timeline(end, dates, task["profile"]), {}, set()
            for fold in task["folds"]:
                check(args.deadline)
                identity = hashlib.sha256(encoded({"protocol": ph, "task": task, "fold": fold})).hexdigest()
                results.append(state(task, fold, info, identity, args.deadline, cache, seen))
            totals[task["profile"]] = {"unique_original_fit_origins": len(seen),
                "fit_fold_row_visits": sum(r["origins"] for r in results if r["task"] == task["task"]),
                "unique_origins_evaluated_this_invocation": len(cache),
                "overlapping_folds_are_not_independent_replications": True}
        check(args.deadline)
        result = {"status": "completed_timestamp_fit_only_audit", "protocol_sha256": ph,
                  "deadline_epoch": args.deadline, "states": results, "profile_origin_totals": totals,
                  "calendar_aggregation": metadata, "decoded_calendar_files": receipts,
                  "split_members_decoded": [{"path": f["split_archive"], "member": "fit.npy"} for t in tasks for f in t["folds"]],
                  "fold_fit_overlap_is_not_independent_replication": True,
                  "candidate_admitted": False, "model_or_loss_computed": False,
                  "actual_missing_bar_locations_verified": False, "Q3_read": False,
                  "research_goal_achieved": False}
        atomic(BASE / "timestamp_audit" / "RESULTS.json", result)
        atomic(BASE / "timestamp_audit" / "EXECUTION_STATUS.json", {"status": result["status"],
              "protocol_sha256": ph, "deadline_epoch": args.deadline, "completed_states": len(results)})
        print(json.dumps({"status": result["status"], "states": len(results)}, sort_keys=True))
    except BaseException as exc:
        atomic(BASE / "timestamp_audit" / "EXECUTION_STATUS.json", {
            "status": "paused_deadline_or_stop" if isinstance(exc, TimeoutError) else "failed",
            "protocol_sha256": ph, "deadline_epoch": args.deadline, "completed_states": len(results),
            "completed_state_identities": [r["identity"] for r in results],
            "error_type": type(exc).__name__, "error": str(exc),
            "completed_checkpoints_retained": True, "deadline_must_not_reset_on_resume": True,
            "market_model_or_loss_computed": False, "Q3_read": False, "research_goal_achieved": False})
        raise


if __name__ == "__main__":
    main()
