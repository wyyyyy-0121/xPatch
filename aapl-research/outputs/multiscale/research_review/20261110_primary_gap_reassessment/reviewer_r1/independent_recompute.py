"""Independent dense-window and vector-index verification; calendar/fit only."""
from pathlib import Path
import argparse
import collections
import hashlib
import importlib.util
import io
import json
import os
import time
import zipfile
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tag(delta, transitions):
    return hashlib.sha256(np.asarray(delta, dtype="<i8").tobytes() + np.packbits(transitions).tobytes()).hexdigest()


def transition_tag(transitions):
    return hashlib.sha256(np.packbits(transitions).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    assert protocol["status"] == "frozen_before_execution" and protocol["mode"] == "timestamp_fit_only"
    assert protocol["independent_recompute_admitted"] is True
    assert set(os.sched_getaffinity(0)) == {26}
    for path, expected in protocol["code_sha256"].items():
        assert sha(path) == expected
    def check():
        assert time.time() < protocol["deadline_epoch"], "Original execution deadline expired"
    check()
    design = json.loads((BASE / "AUDIT_DESIGN.json").read_text())
    assert sha(BASE / "AUDIT_DESIGN.json") == protocol["design_sha256"]
    source_path = BASE / "timestamp_audit/RESULTS.json"
    formal = json.loads(source_path.read_text())
    assert formal["status"] == "completed_timestamp_fit_only_audit" and len(formal["states"]) == 10
    assert formal["protocol_sha256"] == sha(args.protocol)
    spec = importlib.util.spec_from_file_location("access_guard_only", BASE / "audit_timestamps.py")
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    calendar = {row["path"]: row["expected_file_sha256"] for row in design["timestamp_input_allowlist"]}
    splits = [fold["split_archive"] for task in design["split_input_registration"] for fold in task["folds"]]
    guard.install_guard(calendar, splits, BASE / "timestamp_audit")
    arrays = {}
    for path, expected in calendar.items():
        assert sha(path) == expected
        arrays[Path(path).name] = np.load(path, allow_pickle=False)
    start, dates = arrays["timestamps_ns.npy"], arrays["dates.npy"]
    assert np.all(dates <= "2026-06-30") and np.all(np.diff(start) > 0)
    first = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    last = np.r_[first[1:] - 1, len(dates) - 1]
    minute = 60_000_000_000
    hour_indices = np.concatenate([np.arange(a + 59, z + 1, 60) for a, z in zip(first, last)])
    profiles = {"minute": (start + minute, dates), "hourly": (start[hour_indices] + minute, dates[hour_indices])}
    checks, totals = [], {}
    for task in design["split_input_registration"]:
        check()
        profile, L, H = task["profile"], task["lookback"], task["horizon"]
        end, labels = profiles[profile]
        begins = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        ends = np.r_[begins[1:] - 1, len(labels) - 1]
        counts = ends - begins + 1
        sid = np.searchsorted(begins, np.arange(len(end)), side="right") - 1
        local = pd.to_datetime(end, utc=True).tz_convert("America/New_York")
        clocks = local.strftime("%H:%M:%S").to_numpy()
        weekdays = (labels.astype("datetime64[D]").astype(np.int64) + 3) % 7
        clock_keys = np.asarray([str(w) + ":" + c for w, c in zip(weekdays, clocks)])
        fold_indices = {}
        for fold in task["folds"]:
            with zipfile.ZipFile(fold["split_archive"]) as archive:
                with archive.open("fit.npy") as member:
                    indices = np.load(io.BytesIO(member.read()), allow_pickle=False)
            assert hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest() == fold["canonical_int64_sha256"]["fit"]
            fold_indices[fold["fold"]] = indices
        union = np.unique(np.concatenate(list(fold_indices.values())))
        dense_records = {}
        for offset in range(0, len(union), 512):
            check()
            origins = union[offset:offset + 512]
            indices = origins[:, None] + np.arange(1 - L, 1)
            deltas = np.diff(end[indices], axis=1)
            transitions = np.diff(sid[indices], axis=1) != 0
            for row, origin in enumerate(origins):
                dense_records[int(origin)] = (tag(deltas[row], transitions[row]), transition_tag(transitions[row]))

        def past(origins, lag):
            previous_sid = sid[origins] - lag
            safe = np.maximum(previous_sid, 0)
            slot = origins - begins[sid[origins]]
            q = begins[safe] + slot
            exists = (previous_sid >= 0) & (slot < counts[safe])
            exists &= clocks[np.minimum(q, len(end) - 1)] == clocks[origins]
            mature = exists & (q + H <= origins)
            legal = exists.copy() if profile == "hourly" else np.zeros(len(origins), dtype=bool)
            if profile == "minute":
                legal[mature] = sid[q[mature]] == sid[(q + H)[mature]]
            contained = exists & (q >= origins - L + 1) & (q + H <= origins)
            values = {"slot_exists": exists, "mature_le": mature,
                      "mature_lt": mature & (q + H < origins),
                      "maturity_equals_origin": mature & (q + H == origins),
                      "window_contains_both_endpoints": contained, "time_horizon_legal": legal,
                      "calendar_usable": mature & contained & legal,
                      "fixed_nominal_same_slot_lag_shift_matches": exists & (origins - q == lag * (390 if profile == "minute" else 6))}
            if lag == 1:
                values["fixed_nominal_previous_session_shift_matches"] = values["fixed_nominal_same_slot_lag_shift_matches"]
            return values

        for fold in task["folds"]:
            check()
            origins = fold_indices[fold["fold"]]
            result = next(r for r in formal["states"] if r["task"] == task["task"] and r["fold"] == fold["fold"])
            progress_path = Path(result["details"])
            assert sha(progress_path) == result["details_sha256"]
            progress = json.loads(progress_path.read_text())
            for lag in (1, 2):
                vector_counts = {k: int(v.sum()) for k, v in past(origins, lag).items()}
                assert vector_counts == result[f"previous{lag}_counts"]
            nearest = np.zeros(len(origins), dtype=np.int64)
            for lag in range(1, L // int(counts.min()) + 3):
                usable = past(origins, lag)["calendar_usable"]
                nearest[(nearest == 0) & usable] = lag
            nearest_counts = dict(collections.Counter("None" if i == 0 else str(i) for i in nearest))
            assert nearest_counts == result["nearest_matured_visible_session_lag_counts"]
            independent_groups = {}
            for origin in origins:
                group = independent_groups.setdefault(clock_keys[origin], [collections.Counter(), collections.Counter()])
                a, b = dense_records[int(origin)]
                group[0][a] += 1
                group[1][b] += 1
            assert set(independent_groups) == set(progress["clock_groups"])
            for clock, source in progress["clock_groups"].items():
                converted_gap, converted_transition = collections.Counter(), collections.Counter()
                for signature, n in source["gap_signatures"].items():
                    delta = np.full(L - 1, minute if profile == "minute" else 60 * minute, dtype=np.int64)
                    transition = np.zeros(L - 1, dtype=bool)
                    for entry in signature.split(";") if signature else ():
                        lag, duration, flag = map(int, entry.split(":"))
                        delta[L - 2 - lag], transition[L - 2 - lag] = duration, bool(flag)
                    converted_gap[tag(delta, transition)] += n
                for signature, n in source["transition_signatures"].items():
                    transition = np.zeros(L - 1, dtype=bool)
                    for lag in signature.split(";") if signature else ():
                        transition[L - 2 - int(lag)] = True
                    converted_transition[transition_tag(transition)] += n
                assert independent_groups[clock] == [converted_gap, converted_transition]
            checks.append({"task": task["task"], "fold": fold["fold"], "origins": len(origins),
                           "previous1_previous2_counts_exact": True, "nearest_matured_lag_counts_exact": True,
                           "all_clock_groups_dense_gap_and_transition_histograms_exact": True,
                           "clock_groups_checked": len(independent_groups)})
        expected = formal["profile_origin_totals"][profile]
        assert len(union) == expected["unique_original_fit_origins"]
        assert sum(map(len, fold_indices.values())) == expected["fit_fold_row_visits"]
        totals[profile] = len(union)
    check()
    result = {"status": "passed_independent_dense_and_vector_recomputation", "states": checks,
              "unique_fit_origins_by_profile": totals, "main_results_sha256": sha(source_path),
              "protocol_sha256": sha(args.protocol), "algorithm": "Dense timestamp windows versus sparse gap events; vector session offsets versus per-origin enumeration.",
              "all_ten_fit_states_checked": True, "prices_labels_weights_read": False, "Q3_read": False,
              "predictive_gain_established": False, "new_candidate_admitted": False}
    guard.atomic(BASE / "timestamp_audit/INDEPENDENT_RECOMPUTATION.json", result)
    print(json.dumps({"status": result["status"], "states": len(checks)}))


if __name__ == "__main__":
    main()
