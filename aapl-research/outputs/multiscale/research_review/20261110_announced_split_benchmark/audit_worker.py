"""Zero-training, hash-bound announced-split benchmark audit.

Importing this file reads no protocol, market array or model. NumPy is imported
only after the CLI has checked the protocol and every frozen input identity.
The checkpoint contains completed fold records, so a missing derived fold file
can be recreated without reopening any completed fold's arrays.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo


BASE = Path(__file__).resolve().parent
NY = ZoneInfo("America/New_York")
ORIGINAL_NAMES = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")
TINY_NAME = "tiny_long_aligned"
SEEDS = (2026, 2027, 2028)
SHAPES = {"annual": ("month", 12), "quarterly": ("month", 3), "monthly": ("day", 21)}
TARGETS = ("future_close", "future_log_return")
RTOL = 1e-11
ATOL = 1e-14
STOP_REQUESTED = False


class PauseRequested(Exception):
    """A cooperative pause; committed folds remain authoritative."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def on_signal(_signum, _frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def ny_timestamp(value):
    seconds, nanoseconds = divmod(int(value), 1_000_000_000)
    return datetime.fromtimestamp(seconds, timezone.utc).astimezone(NY).isoformat() + f"; subsecond_ns={nanoseconds}"


def canonical_file(raw):
    require(isinstance(raw, str) and Path(raw).is_absolute(), "Every input path must be absolute")
    path = Path(raw)
    require(str(path.resolve()) == raw, "Input paths must already be canonical: " + raw)
    return path


def validate_protocol(protocol, protocol_path):
    require(isinstance(protocol, dict), "Protocol must be a JSON object")
    for key in ("inputs", "bars", "task_records", "old_results", "inherited_tasks", "known_ns",
                "effective_date", "ratio", "deadline_epoch"):
        require(key in protocol, "Missing protocol field: " + key)
    require(isinstance(protocol["inputs"], dict) and protocol["inputs"], "Frozen inputs map is required")
    for raw, identity in protocol["inputs"].items():
        canonical_file(raw)
        require(isinstance(identity, str) and len(identity) == 64 and
                all(c in "0123456789abcdef" for c in identity), "Invalid SHA-256 for " + raw)
    require(str(Path(__file__).resolve()) in protocol["inputs"], "Protocol must freeze this worker's source")
    require(str(protocol_path) not in protocol["inputs"], "Protocol cannot recursively hash itself")
    require(isinstance(protocol["known_ns"], int) and not isinstance(protocol["known_ns"], bool), "known_ns must be integer")
    expected_known = int(datetime(2020, 8, 1, tzinfo=NY).timestamp()) * 1_000_000_000
    require(protocol["known_ns"] == expected_known, "Only frozen Aug 1 2020 00:00 New York knowledge boundary is allowed")
    require(protocol["effective_date"] == "2020-08-31" and protocol["ratio"] == 4,
            "This audit admits only the reviewed Aug 31 2020 four-for-one event")
    deadline = protocol["deadline_epoch"]
    require(isinstance(deadline, (float, int)) and not isinstance(deadline, bool) and math.isfinite(deadline),
            "Finite absolute deadline required")
    require(set(protocol["bars"]) == {"day", "month"}, "Only day and month bar containers admitted")
    payloads = []
    for value in protocol["bars"].values():
        payloads.append(canonical_file(value))
    payloads.append(canonical_file(protocol["old_results"]))
    records = protocol["task_records"]
    expected_tasks = {f"{profile}_h{horizon}_{target}" for profile, (_, horizon) in SHAPES.items() for target in TARGETS}
    require(isinstance(records, list) and len(records) == 6, "Exactly six long-scale task records required")
    require({record["task"] for record in records} == expected_tasks, "Task identities differ from the six original tasks")
    for record in records:
        profile, target = record["profile"], record["target"]
        require(profile in SHAPES and target in TARGETS, "Unexpected long-scale profile/target")
        frequency, horizon = SHAPES[profile]
        require((record["frequency"], record["horizon"]) == (frequency, horizon), "Original task shape changed")
        require(record["task"] == f"{profile}_h{horizon}_{target}", "Task name/fields disagree")
        folds = record["folds"]
        require(isinstance(folds, list) and len(folds) == 5 and
                {fold["fold"] for fold in folds} == set(range(1, 6)), "All five original folds required")
        for fold in folds:
            payloads.append(canonical_file(fold["targets"]))
            models = fold["models"]
            expected = {(name, 2026) for name in ORIGINAL_NAMES} | {(TINY_NAME, seed) for seed in SEEDS}
            require(isinstance(models, list) and len(models) == len(expected) and
                    {(model["name"], model["seed"]) for model in models} == expected,
                    "Each fold requires the original eight single-seed methods and three frozen tiny seeds")
            payloads.extend(canonical_file(model["prediction"]) for model in models)
    require(all(str(path) in protocol["inputs"] for path in payloads), "A requested payload is not hash-bound")
    require(all(path.suffix == ".npz" for path in payloads if path != Path(protocol["old_results"])),
            "All array payloads must be explicit NPZ containers")
    return records


def verify_inputs(protocol):
    for raw, expected in sorted(protocol["inputs"].items()):
        if STOP_REQUESTED or time.time() >= protocol["deadline_epoch"]:
            raise PauseRequested("Stopped before input identity verification completed")
        require(digest(raw) == expected, "Frozen input SHA-256 mismatch: " + raw)


def work_deadline(protocol):
    reserve = protocol.get("reserve_seconds", 60)
    require(isinstance(reserve, (int, float)) and math.isfinite(reserve) and reserve >= 0,
            "Reserve must be finite and nonnegative")
    return protocol["deadline_epoch"] - reserve


def load_fields(np, path, fields, optional=()):
    # Every caller passes a role-specific path from the validated protocol.
    with np.load(path, allow_pickle=False) as archive:
        require(all(field in archive.files for field in fields), "Missing required arrays in " + str(path))
        result = {field: archive[field].copy() for field in fields}
        result.update({field: archive[field].copy() for field in optional if field in archive.files})
    require(all(value.dtype.kind != "O" for value in result.values()), "Object payload is forbidden")
    return result


def integer_vector(np, value, name):
    require(value.ndim == 1 and value.dtype.kind in "iu", name + " must be a one-dimensional integer array")
    if value.dtype.kind == "u":
        require(not len(value) or int(value.max()) <= 2**63 - 1, name + " exceeds int64")
    return value.astype(np.int64, copy=False)


def load_bars(np, path):
    arrays = load_fields(np, path, ("session_date", "end_ns", "close"))
    dates = arrays["session_date"]
    ends = integer_vector(np, arrays["end_ns"], "bar end_ns")
    prices = arrays["close"]
    require(dates.ndim == prices.ndim == 1 and dates.dtype.kind == "U" and
            len(dates) == len(ends) == len(prices) and len(dates) > 0, "Invalid bar shapes")
    require(prices.dtype.kind in "fiu" and np.isfinite(prices).all() and (prices > 0).all(), "Invalid bar closes")
    require((np.diff(ends) > 0).all() and (dates[1:] > dates[:-1]).all(), "Bar order/uniqueness failed")
    require(all(len(str(value)) == 10 and date.fromisoformat(str(value)).isoformat() == value for value in dates),
            "Invalid calendar session date")
    require(str(dates[-1]) <= "2026-06-30", "Post-development session in admitted bar container")
    cutoff_ns = int(datetime(2026, 7, 1, tzinfo=NY).timestamp()) * 1_000_000_000
    require(int(ends[-1]) <= cutoff_ns, "Post-development completion in admitted bar container")
    return {"session_date": dates, "end_ns": ends, "close": prices.astype(np.float64, copy=False)}


def announced_mask(np, decision_ns, origin_sessions, target_sessions, known_ns, effective_date):
    """Knowledge uses actual decision times; effective status uses session dates."""
    crosses = (origin_sessions[:, None] < effective_date) & (target_sessions >= effective_date)
    known = decision_ns >= known_ns
    return crosses, crosses & known[:, None]


def announced_benchmarks(np, anchor, decision_ns, origin_sessions, target_sessions,
                         known_ns, effective_date, ratio, target):
    """The shared market/synthetic baseline rule; it never changes labels."""
    require(target in TARGETS and math.isfinite(ratio) and ratio > 0, "Invalid benchmark definition")
    require(target_sessions.ndim == 2 and anchor.shape == decision_ns.shape == origin_sessions.shape ==
            (len(target_sessions),), "Benchmark metadata shapes disagree")
    crosses, mask = announced_mask(np, decision_ns, origin_sessions, target_sessions, known_ns, effective_date)
    if target == "future_close":
        old = np.broadcast_to(anchor[:, None], target_sessions.shape).astype(np.float64, copy=True)
        new = old.copy()
        new[mask] /= ratio
    else:
        old = np.zeros(target_sessions.shape, dtype=np.float64)
        new = old.copy()
        new[mask] = -math.log(ratio)
    return old, new, crosses, mask


def metric_sums(np, prediction, actual):
    error = prediction - actual
    endpoint = error[:, -1]
    values = {"endpoint_sse": float(np.square(endpoint).sum(dtype=np.float64)),
              "endpoint_absolute_sum": float(np.abs(endpoint).sum(dtype=np.float64)),
              "path_sse": float(np.square(error).sum(dtype=np.float64))}
    require(all(math.isfinite(value) and value >= 0 for value in values.values()), "Nonfinite/negative loss sum")
    return values


def finish_metrics(sums, count, horizon):
    return {**sums, "endpoint_mse": sums["endpoint_sse"] / count,
            "endpoint_mae": sums["endpoint_absolute_sum"] / count,
            "path_mse": sums["path_sse"] / (count * horizon)}


def gain_percent(value, reference):
    return 100.0 * (1.0 - value / reference) if reference > 0 else None


def add_comparisons(metrics, old, new):
    result = dict(metrics)
    for label, reference in (("old_benchmark", old), ("announced_benchmark", new)):
        for metric in ("endpoint_mse", "endpoint_mae", "path_mse"):
            result[metric + "_gain_pct_vs_" + label] = gain_percent(metrics[metric], reference[metric])
    return result


def audit_fold(np, protocol, task, fold, bars, identity):
    horizon, target = task["horizon"], task["target"]
    arrays = load_fields(np, fold["targets"], ("origins", "actual", "anchor", "decision_ns", "target_completion_ns"))
    origins = integer_vector(np, arrays["origins"], "target origins")
    require(len(origins) > 0 and (np.diff(origins) > 0).all() and (origins >= 0).all() and
            (origins < len(bars["close"]) - horizon).all(), "Invalid validation origins")
    count = len(origins)
    future = origins[:, None] + np.arange(1, horizon + 1, dtype=np.int64)
    actual = arrays["actual"]
    require(actual.shape == (count, horizon) and actual.dtype.kind in "fiu" and np.isfinite(actual).all(),
            "Invalid cached nominal target path")
    actual = actual.astype(np.float64, copy=False)
    anchor = arrays["anchor"]
    require(anchor.shape in ((count,), (count, 1)) and anchor.dtype.kind in "fiu", "Invalid cached anchor")
    anchor = anchor.reshape(count).astype(np.float64, copy=False)
    decisions = integer_vector(np, arrays["decision_ns"], "target decision_ns")
    completions = integer_vector(np, arrays["target_completion_ns"], "target completion_ns")
    require(np.array_equal(anchor, bars["close"][origins]), "Cached anchor differs from original nominal close")
    require(np.array_equal(decisions, bars["end_ns"][origins]) and
            np.array_equal(completions, bars["end_ns"][origins + horizon]), "Cached timestamps differ from original bars")
    rebuilt = bars["close"][future]
    if target == "future_log_return":
        rebuilt = np.log(rebuilt / anchor[:, None])
    require(np.array_equal(actual, rebuilt), "Cached labels differ from original unmodified nominal targets")
    sessions = bars["session_date"][origins]
    target_sessions = bars["session_date"][future]
    old_prediction, new_prediction, crosses, mask = announced_benchmarks(
        np, anchor, decisions, sessions, target_sessions, protocol["known_ns"],
        protocol["effective_date"], protocol["ratio"], target)
    old = finish_metrics(metric_sums(np, old_prediction, actual), count, horizon)
    new = finish_metrics(metric_sums(np, new_prediction, actual), count, horizon)
    models, seed_rows = {}, []
    tiny_endpoint, tiny_absolute, tiny_path = [], [], []
    for model in sorted(fold["models"], key=lambda value: (value["name"], value["seed"])):
        saved = load_fields(np, model["prediction"], ("prediction", "origins"), ("decision_ns",))
        require(np.array_equal(integer_vector(np, saved["origins"], "prediction origins"), origins),
                "Prediction/target origins differ")
        if "decision_ns" in saved:
            require(np.array_equal(integer_vector(np, saved["decision_ns"], "prediction decision_ns"), decisions),
                    "Prediction/target decisions differ")
        else:
            require(model["name"] == TINY_NAME, "Only frozen tiny files may omit decision_ns")
        prediction = saved["prediction"]
        require(prediction.shape == actual.shape and prediction.dtype.kind in "fiu" and np.isfinite(prediction).all(),
                "Invalid saved prediction path")
        prediction = prediction.astype(np.float64, copy=False)
        metrics = finish_metrics(metric_sums(np, prediction, actual), count, horizon)
        record = {"name": model["name"], "seed": model["seed"], "prediction": model["prediction"],
                  "prediction_sha256": protocol["inputs"][model["prediction"]],
                  "decision_ns_present_and_verified": "decision_ns" in saved,
                  "deterministic": model["name"] in ORIGINAL_NAMES[:4],
                  "seed_role": "source_recipe_label_not_random_replication" if model["name"] in ORIGINAL_NAMES[:4]
                               else "original_training_seed",
                  **add_comparisons(metrics, old, new)}
        seed_rows.append(record)
        if model["name"] == TINY_NAME:
            error = prediction - actual
            tiny_endpoint.append(np.square(error[:, -1]))
            tiny_absolute.append(np.abs(error[:, -1]))
            tiny_path.append(np.square(error).mean(axis=1, dtype=np.float64))
        else:
            models[model["name"]] = {"seeds": [2026], "seed_count": 1,
                                     "deterministic": model["name"] in ORIGINAL_NAMES[:4],
                                     "seed_role": record["seed_role"],
                                     **add_comparisons(metrics, old, new)}
    # Average losses at each origin, never predictions. All three frozen seeds
    # participate, including unfavorable seeds.
    tiny_sums = {"endpoint_sse": float(np.mean(tiny_endpoint, axis=0).sum(dtype=np.float64)),
                 "endpoint_absolute_sum": float(np.mean(tiny_absolute, axis=0).sum(dtype=np.float64)),
                 "path_sse": float(np.mean(tiny_path, axis=0).sum(dtype=np.float64) * horizon)}
    models[TINY_NAME] = {"seeds": list(SEEDS), "seed_count": 3,
                         "deterministic": False, "seed_role": "original_training_seeds",
                         "aggregation": "per_origin_mean_seed_loss_not_mean_prediction",
                         **add_comparisons(finish_metrics(tiny_sums, count, horizon), old, new)}
    require(set(models) == set(ORIGINAL_NAMES) | {TINY_NAME}, "Incomplete scored model set")
    require(all(models["last_close"][key] == old[key] for key in ("endpoint_mse", "endpoint_mae", "path_mse")),
            "Original last_close reference does not match persistence/zero")
    event_rows = []
    for index in np.flatnonzero(crosses.any(axis=1)):
        positions = np.flatnonzero(crosses[index]) + 1
        event_rows.append({"origin_index": int(origins[index]), "origin_session": str(sessions[index]),
                           "decision_ns": int(decisions[index]), "decision_new_york": ny_timestamp(decisions[index]),
                           "known_at_origin": bool(decisions[index] >= protocol["known_ns"]),
                           "endpoint_session": str(target_sessions[index, -1]),
                           "cross_event_path_cells": int(crosses[index].sum()),
                           "known_event_path_cells": int(mask[index].sum()),
                           "first_cross_event_step": int(positions[0]), "last_cross_event_step": int(positions[-1])})
    return {"schema": "announced_split_fold_v1", "protocol_sha256": identity,
            "task": task["task"], "profile": task["profile"], "frequency": task["frequency"],
            "horizon": horizon, "target": target, "fold": fold["fold"], "origins": count,
            "origins_sha256": hashlib.sha256(origins.astype("<i8", copy=False).tobytes()).hexdigest(),
            "first_origin_session": str(sessions[0]), "last_origin_session": str(sessions[-1]),
            "first_decision_ns": int(decisions[0]), "last_decision_ns": int(decisions[-1]),
            "last_target_completion_ns": int(completions[-1]),
            "cross_event_origins": int(crosses.any(axis=1).sum()),
            "known_event_origins": int(mask.any(axis=1).sum()),
            "cross_event_path_cells": int(crosses.sum()), "known_event_path_cells": int(mask.sum()),
            "adjusted_endpoint_origins": int(mask[:, -1].sum()), "event_origins": event_rows,
            "old_benchmark": old, "announced_benchmark": new, "models": models, "seed_metrics": seed_rows,
            "scope": "recomputed_preQ3_exposed_validation_saved_nominal_labels_and_predictions",
            "new_training": False, "new_inference": False, "independent_confirmation": False, "Q3_read": False}


def fold_key(record):
    return record["task"] + "/fold_" + str(record["fold"])


def write_checkpoint(output, identity, protocol, rows):
    atomic_json(output / "CHECKPOINT.json", {"schema": "announced_split_checkpoint_v1",
                "protocol_sha256": identity, "deadline_epoch": protocol["deadline_epoch"],
                "completed_folds": rows, "committed_utc": utc_now()})


def materialize_folds(output, identity, rows):
    for row in rows:
        atomic_json(output / "folds" / row["task"] / f"fold_{row['fold']}.json", row)
    atomic_json(output / "FOLD_RESULTS.json", {"protocol_sha256": identity, "folds": rows,
                                              "completed_folds": len(rows), "expected_folds": 30})


def pooled_results(task_records, rows):
    pooled, summaries = [], []
    sum_fields = ("endpoint_sse", "endpoint_absolute_sum", "path_sse")
    for task in sorted(task_records, key=lambda value: value["task"]):
        selected = sorted((row for row in rows if row["task"] == task["task"]), key=lambda row: row["fold"])
        require(len(selected) == 5 and {row["fold"] for row in selected} == set(range(1, 6)), "Missing pooled folds")
        count = sum(row["origins"] for row in selected)
        references = {}
        for name in ("old_benchmark", "announced_benchmark"):
            references[name] = finish_metrics({key: sum(row[name][key] for row in selected) for key in sum_fields},
                                               count, task["horizon"])
        method_metrics = {name: finish_metrics({key: sum(row["models"][name][key] for row in selected)
                                                   for key in sum_fields}, count, task["horizon"])
                          for name in (*ORIGINAL_NAMES, TINY_NAME)}
        for name, values in {**method_metrics, **references}.items():
            pooled.append({"task": task["task"], "profile": task["profile"], "horizon": task["horizon"],
                           "target": task["target"], "name": name, "origins": count, "completed_folds": 5,
                           "seed_count": 3 if name == TINY_NAME else 1,
                           "deterministic": name in ORIGINAL_NAMES[:4] or name.endswith("benchmark"),
                           "seed_role": "original_training_seeds" if name == TINY_NAME else
                                        ("original_training_seed" if name in ORIGINAL_NAMES[4:] else "no_random_replication"),
                           **add_comparisons(values, references["old_benchmark"], references["announced_benchmark"]),
                           "baseline_mse": references["old_benchmark"]["endpoint_mse"],
                           "announced_baseline_mse": references["announced_benchmark"]["endpoint_mse"],
                           "aggregation": "per_origin_mean_seed_loss" if name == TINY_NAME else "pooled_origin_loss",
                           "scope": "recomputed_six_long_scale_tasks_exposed_development"})
        tiny = next(value for value in pooled if value["task"] == task["task"] and value["name"] == TINY_NAME)
        summaries.append({"task": task["task"], "profile": task["profile"], "horizon": task["horizon"],
                          "target": task["target"], "origins": count, "completed_folds": 5,
                          **{key: sum(row[key] for row in selected) for key in
                             ("cross_event_origins", "known_event_origins", "cross_event_path_cells",
                              "known_event_path_cells", "adjusted_endpoint_origins")},
                          "old_benchmark": references["old_benchmark"],
                          "announced_benchmark": references["announced_benchmark"],
                          "tiny_long_aligned": tiny,
                          "scope": "recomputed_six_long_scale_tasks_exposed_development"})
    return pooled, summaries


def check_old_tiny(np, protocol, pooled):
    old = read_json(protocol["old_results"])
    require(isinstance(old.get("pooled"), list), "Original result needs pooled records")
    receipts = []
    for row in (value for value in pooled if value["name"] == TINY_NAME):
        matches = [value for value in old["pooled"] if value["task"] == row["task"] and value.get("variant") == "long_aligned"]
        require(len(matches) == 1, "Expected exactly one original long_aligned pooled row")
        expected = matches[0]
        require(expected["origins"] == row["origins"] and expected["seed_count"] == 3 and
                expected["completed_folds"] == 5, "Original pooled coverage differs")
        differences = {}
        for field in ("endpoint_mse", "baseline_mse", "endpoint_mae", "path_mse"):
            require(np.isclose(row[field], expected[field], rtol=RTOL, atol=ATOL),
                    "Original tiny scalar mismatch: " + row["task"] + "/" + field)
            differences[field] = row[field] - expected[field]
        receipts.append({"task": row["task"], "original_row": expected, "recomputed_minus_original": differences})
    require(len(receipts) == 6, "All six original tiny pooled rows must be checked")
    return {"status": "matched_original_saved_pooled_scalars", "rtol": RTOL, "atol": ATOL,
            "original_values_modified": False, "rows": receipts}


def inherited_overview(protocol, summaries):
    inherited = protocol["inherited_tasks"]
    scope = "inherited_only_not_recomputed; pre2020_event_absent_from_2021plus_intraday_history"
    if isinstance(inherited, dict):
        scope = inherited.get("inherited_scope", inherited.get("scope", scope))
        inherited = inherited.get("rows", inherited.get("tasks"))
    require(isinstance(inherited, list) and len(inherited) == 14, "Exactly fourteen inherited original task rows required")
    expected = {f"{profile}_h{h}_{target}" for profile, horizons in (("minute", (5, 15, 30, 60)), ("hourly", (1, 4, 8)))
                for h in horizons for target in TARGETS}
    require({value["task"] for value in inherited} == expected, "Inherited task identities differ")
    overview = list(summaries)
    for value in inherited:
        overview.append({"task": value["task"], "scope": value.get("inherited_scope", value.get("scope", scope)),
                         "recomputed_in_this_audit": False, "inherited_original_pooled_row": value,
                         "new_event_metrics_computed": False})
    require(len(overview) == 20 and len({row["task"] for row in overview}) == 20, "Twenty-task overview incomplete")
    return sorted(overview, key=lambda value: value["task"])


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, stream.getvalue())


def run_synthetic(args):
    """Necessary boundary tests using invented metadata; never reads a protocol."""
    os.sched_setaffinity(0, {24})
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    import numpy as np
    known = int(datetime(2020, 8, 1, tzinfo=NY).timestamp()) * 1_000_000_000
    decisions = np.asarray([known - 1, known, known + 28 * 86400 * 1_000_000_000,
                            known + 31 * 86400 * 1_000_000_000], dtype=np.int64)
    origins = np.asarray(["2020-07-31", "2020-07-31", "2020-08-28", "2020-08-31"])
    future = np.asarray([["2020-08-28", "2020-08-31", "2020-09-30"],
                         ["2020-08-28", "2020-08-31", "2020-09-30"],
                         ["2020-08-31", "2020-09-01", "2020-09-02"],
                         ["2020-09-01", "2020-10-01", "2020-11-01"]])
    expected_cross = np.asarray([[False, True, True], [False, True, True],
                                 [True, True, True], [False, False, False]])
    expected_mask = np.asarray([[False, False, False], [False, True, True],
                                [True, True, True], [False, False, False]])
    anchor = np.asarray([100.0, 200.0, 400.0, 50.0])
    price_old, price_new, cross, mask = announced_benchmarks(
        np, anchor, decisions, origins, future, known, "2020-08-31", 4.0, "future_close")
    return_old, return_new, return_cross, return_mask = announced_benchmarks(
        np, anchor, decisions, origins, future, known, "2020-08-31", 4.0, "future_log_return")
    expected_price = np.asarray([[100.0, 100.0, 100.0], [200.0, 50.0, 50.0],
                                 [100.0, 100.0, 100.0], [50.0, 50.0, 50.0]])
    expected_return = np.asarray([[0.0, 0.0, 0.0], [0.0, -math.log(4), -math.log(4)],
                                  [-math.log(4), -math.log(4), -math.log(4)], [0.0, 0.0, 0.0]])
    checks = {"all_cross_event_cells": np.array_equal(cross, expected_cross),
              "all_announcement_known_cells": np.array_equal(mask, expected_mask),
              "known_boundary_minus1ns_excluded_exact_boundary_included": not bool(mask[0].any()) and bool(mask[1, 1]),
              "target_effective_date_inclusive": bool(mask[1, 1]) and not bool(mask[1, 0]),
              "origin_effective_date_already_effective_no_second_split": not bool(mask[3].any()),
              "price_path_each_step_expected": np.array_equal(price_new, expected_price),
              "original_price_path_retained": np.array_equal(price_old, np.broadcast_to(anchor[:, None], future.shape)),
              "return_path_each_step_expected": np.array_equal(return_new, expected_return),
              "original_zero_path_retained": bool((return_old == 0).all()),
              "both_targets_same_masks": np.array_equal(return_cross, cross) and np.array_equal(return_mask, mask),
              "July31_session_with_Aug1_decision_admitted": bool(mask[1, 1]),
              "metadata_counts": int(cross.any(axis=1).sum()) == 3 and int(mask.any(axis=1).sum()) == 2 and
                                 int(cross.sum()) == 7 and int(mask.sum()) == 5}
    require(all(checks.values()), "Synthetic announced-event boundary qualification failed")
    output = args.output.resolve() if args.output else BASE
    require(output.is_relative_to(BASE), "Synthetic output must remain inside this separate stage")
    receipt = {"status": "passed_synthetic_only_boundary_acceptance", "checks": {key: bool(value) for key, value in checks.items()},
               "checks_passed": len(checks), "known_ns": known, "worker_sha256": digest(__file__),
               "market_inputs_or_protocol_read": False, "market_arrays_read": False, "Q3_read": False,
               "model_execution": False, "numeric_scope": "invented_four_origins_three_steps_both_targets",
               "CPU_affinity": sorted(os.sched_getaffinity(0)), "numeric_threads": 1, "created_utc": utc_now()}
    atomic_json(output / "SYNTHETIC_ACCEPTANCE.json", receipt)
    print(json.dumps(receipt), flush=True)
    return 0


def run(args):
    protocol_path = args.protocol.resolve()
    protocol = read_json(protocol_path)
    identity = digest(protocol_path)
    task_records = validate_protocol(protocol, protocol_path)
    output = args.output.resolve() if args.output else protocol_path.parent / "audit"
    require(output.is_relative_to(BASE), "Outputs must stay inside this separate audit stage")
    require(not any(Path(path).is_relative_to(output) for path in protocol["inputs"]),
            "Output directory cannot contain any frozen input")
    require(args.pause_after_folds is None or args.pause_after_folds >= 1, "pause-after-folds must be positive")
    os.sched_setaffinity(0, {24})
    require(os.sched_getaffinity(0) == {24}, "Worker must use CPU24 only")
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    signal.signal(signal.SIGUSR1, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "CHECKPOINT.json"
    rows = []
    if checkpoint_path.exists():
        require(args.resume, "Existing checkpoint requires --resume")
        checkpoint = read_json(checkpoint_path)
        require(checkpoint["protocol_sha256"] == identity and checkpoint["deadline_epoch"] == protocol["deadline_epoch"],
                "Resume cannot change protocol identity or absolute deadline")
        rows = checkpoint["completed_folds"]
    elif args.resume:
        require(not (output / "RESULTS.json").exists(), "Final result without authoritative checkpoint")
    expected = {(record["task"], fold["fold"]) for record in task_records for fold in record["folds"]}
    require(isinstance(rows, list) and len(rows) == len({fold_key(row) for row in rows}) and
            all((row["task"], row["fold"]) in expected and row["protocol_sha256"] == identity for row in rows),
            "Invalid/incompatible committed fold records")
    try:
        verify_inputs(protocol)
        import numpy as np
        bars = {frequency: load_bars(np, path) for frequency, path in protocol["bars"].items()}
        materialize_folds(output, identity, rows)
        completed = {fold_key(row) for row in rows}
        newly_completed = 0
        for task in sorted(task_records, key=lambda value: value["task"]):
            for fold in sorted(task["folds"], key=lambda value: value["fold"]):
                key = task["task"] + "/fold_" + str(fold["fold"])
                if key in completed:
                    continue
                if STOP_REQUESTED or time.time() >= work_deadline(protocol):
                    raise PauseRequested("Signal or unchanged deadline minus finalization reserve before next fold")
                row = audit_fold(np, protocol, task, fold, bars[task["frequency"]], identity)
                rows.append(row)
                completed.add(key)
                # Checkpoint is authoritative; all separate fold reports are
                # derived from it, including recovery after a write interruption.
                write_checkpoint(output, identity, protocol, rows)
                atomic_json(output / "folds" / task["task"] / f"fold_{fold['fold']}.json", row)
                atomic_json(output / "FOLD_RESULTS.json", {"protocol_sha256": identity, "folds": rows,
                                                          "completed_folds": len(rows), "expected_folds": 30})
                newly_completed += 1
                print(json.dumps({"status": "fold_committed", "task": task["task"], "fold": fold["fold"],
                                  "completed_folds": len(rows)}, ensure_ascii=False), flush=True)
                if STOP_REQUESTED or time.time() >= work_deadline(protocol) or (
                        args.pause_after_folds is not None and newly_completed >= args.pause_after_folds):
                    raise PauseRequested("Cooperative pause after committed fold")
        pooled, summaries = pooled_results(task_records, rows)
        old_check = check_old_tiny(np, protocol, pooled)
        overview = inherited_overview(protocol, summaries)
        write_csv(output / "POOLED.csv", pooled)
        atomic_json(output / "TWENTY_TASK_OVERVIEW.json", {"protocol_sha256": identity, "tasks": overview,
                    "recomputed_tasks": 6, "inherited_only_tasks": 14})
        result = {"status": "completed_descriptive_announced_split_benchmark_audit", "protocol_sha256": identity,
                  "deadline_epoch": protocol["deadline_epoch"], "completed_folds": len(rows), "recomputed_tasks": 6,
                  "inherited_only_tasks": 14, "known_ns": protocol["known_ns"],
                  "known_new_york": ny_timestamp(protocol["known_ns"]), "effective_date": protocol["effective_date"],
                  "ratio": protocol["ratio"], "pooled": pooled, "task_summaries": summaries,
                  "twenty_task_overview": overview, "original_tiny_scalar_checks": old_check,
                  "scientific_pass_rule": None, "method_candidate_admitted": False,
                  "independent_confirmation": False, "research_goal_achieved": False,
                  "new_training": False, "new_inference": False, "GPU": False, "Q3_read": False,
                  "scope": "single_announced_event_exposed_development; added_public_information_benchmark",
                  "not_same_information_architecture_comparison": True,
                  "original_intervals_not_reused_for_new_reference": True,
                  "completed_utc": utc_now()}
        atomic_json(output / "RESULTS.json", result)
        atomic_json(output / "STATE.json", {"status": "completed", "protocol_sha256": identity,
                    "deadline_epoch": protocol["deadline_epoch"], "completed_folds": len(rows), "updated_utc": utc_now()})
        print(json.dumps({"status": "completed", "completed_folds": len(rows), "results": str(output / "RESULTS.json")}), flush=True)
        return 0
    except PauseRequested as exc:
        write_checkpoint(output, identity, protocol, rows)
        materialize_folds(output, identity, rows)
        atomic_json(output / "STATE.json", {"status": "paused", "protocol_sha256": identity,
                    "deadline_epoch": protocol["deadline_epoch"], "completed_folds": len(rows),
                    "reason": str(exc), "updated_utc": utc_now()})
        print(json.dumps({"status": "paused", "completed_folds": len(rows), "reason": str(exc)}), flush=True)
        return 75
    except Exception as exc:
        atomic_json(output / "STATE.json", {"status": "failed", "protocol_sha256": identity,
                    "deadline_epoch": protocol["deadline_epoch"], "completed_folds": len(rows),
                    "error_type": type(exc).__name__, "error": str(exc), "updated_utc": utc_now()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=BASE / "PROTOCOL.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-after-folds", type=int)
    parser.add_argument("--synthetic-only", action="store_true")
    args = parser.parse_args()
    return run_synthetic(args) if args.synthetic_only else run(args)


if __name__ == "__main__":
    sys.exit(main())
