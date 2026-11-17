"""Inventory the approved development folds without reading market-data arrays."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/multiscale"
CORE = OUT / "private_three_scale_development/20261107T134128870773Z"
ANNUAL = OUT / "private_annual_short_development/20261107T143734746737Z"
MATRIX = OUT / "private_minute_matrix_development/20261107T151850900588Z"
CONTROLS = OUT / "private_minute_full_controls/20261107T140007359046Z"
NEURAL = OUT / "private_minute_full_neural/20261107T140317864464Z"
MODELS = {"last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch"}
TASKS = {(profile, horizon, target)
         for profile, horizons in (("annual", (12,)), ("quarterly", (3,)),
                                   ("monthly", (21,)), ("hourly", (1, 4, 8)),
                                   ("minute", (5, 15, 30, 60)))
         for horizon in horizons for target in ("future_close", "future_log_return")}


class EvidenceError(ValueError):
    pass


class StateChanged(EvidenceError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceError(message)


def digest_bytes(data):
    return sha256(data).hexdigest()


def digest(path):
    return digest_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def in_tree(path, root=OUT):
    result = Path(path).resolve()
    require(result.is_relative_to(root.resolve()), "Evidence path leaves the private output tree")
    return result


def verify_fold(entry, expected, job, standard, expected_folder, *, protocol_hash,
                project_root=ROOT, output_root=OUT):
    """Check completed-fold identities; no prediction or target arrays are read."""
    require(entry.get("status") == "completed" and entry.get("exit_code", 0) == 0,
            "Fold was not reported complete")
    expected_folder = in_tree(expected_folder, output_root)
    receipt_path = expected_folder / "fold_receipt.json"
    receipt = read_json(receipt_path)
    receipt_hash = digest(receipt_path)
    require(entry.get("receipt_sha256", receipt_hash) == receipt_hash, "Queue receipt digest differs")
    config = receipt.get("config", {})
    require(receipt.get("status") == "completed" and receipt.get("phase") == "development" and
            receipt.get("Q3_values_read") is False and receipt.get("independent_test") is False,
            "Fold scope or protected-period flag differs")
    require((config.get("profile"), config.get("horizon"), config.get("target")) == expected[:3] and
            config.get("protocol_identity") == protocol_hash and
            set(receipt.get("models", {})) == MODELS and
            receipt.get("validation_origins") == job["validation_origins"],
            "Fold task, models or validation count differs")
    split_hash = job.get("validation_indices_sha256", job.get("split_sha256", {}).get("validation"))
    require(split_hash and receipt.get("identity_fields", {}).get("validation_indices_sha256") == split_hash,
            "Fold validation origins differ from its protocol")

    standard = in_tree(standard, output_root)
    candidates = []
    if entry.get("verification"):
        candidates.append(in_tree(entry["verification"], output_root))
    candidates.extend((standard, standard.with_name(standard.stem + "_night_resumed.json")))
    errors = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            verification = read_json(path)
            verify_hash = digest(path)
            require(entry.get("verification_sha256", verify_hash) == verify_hash,
                    "Queue verification digest differs")
            require(verification.get("status") == "verified_saved_fold" and
                    in_tree(verification["fold_directory"], output_root) == expected_folder and
                    verification.get("fold_receipt_sha256") == receipt_hash and
                    verification.get("targets_sha256") == receipt.get("targets_sha256") and
                    verification.get("protocol_identity") == config.get("protocol_identity") and
                    verification.get("development_cutoff") == "2026-06-30" and
                    verification.get("independent_test") is False and
                    set(verification.get("models", {})) == MODELS,
                    "Verification does not match the saved fold")
            sources = receipt.get("source_sha256", {})
            matches = verification.get("current_source_matches_recorded", {})
            require(sources and set(matches) == set(sources) and all(matches.values()),
                    "Recorded source verification is incomplete")
            for name, source_hash in sources.items():
                source = (project_root / name).resolve()
                require(source.is_relative_to(project_root.resolve()) and source.is_file() and
                        digest(source) == source_hash, "Current source differs from the completed fold")
            return {"validation_origins": receipt["validation_origins"], "validation_sha256": split_hash,
                    "fold_directory": str(expected_folder), "receipt_sha256": receipt_hash,
                    "verification": str(path), "verification_sha256": verify_hash}
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"{path.name}: {exc}")
    raise EvidenceError("No matching completed-fold verification: " + "; ".join(errors))


def verify_first_minute_fold(controls=CONTROLS, neural=NEURAL, *, project_root=ROOT):
    docs = {}
    for label, folder in (("controls", controls), ("neural", neural)):
        protocol_path = folder / "PRIVATE_PROTOCOL.json"
        result_path = folder / "PRIVATE_RESULT.json"
        check_path = folder / "PRIVATE_INDEPENDENT_CHECK.json"
        protocol, result, check = (read_json(p) for p in (protocol_path, result_path, check_path))
        expected_status = ("private_full_minute_controls_recomputed" if label == "controls"
                           else "private_full_minute_neural_independently_recomputed")
        expected_result = ("private_full_origin_controls_complete" if label == "controls"
                           else "private_full_origin_neural_complete")
        require(protocol.get("task") == "minute H60 cumulative log return" and protocol.get("fold") == 1 and
                protocol.get("data_cutoff") == result.get("data_cutoff") == "2026-06-30" and
                result.get("status") == expected_result and check.get("status") == expected_status and
                result.get("protocol_sha256") == digest(protocol_path) and
                check.get("result_sha256") == digest(result_path) and
                protocol.get("Q3_values_read") is False and result.get("Q3_values_read") is False and
                protocol.get("independent_test") is False and result.get("independent_test") is False and
                check.get("independent_test") is False and check.get("Q3_targets_checked_before_cutoff") is True,
                "First minute fold result or protected-period evidence differs")
        expected_models = MODELS - {"mlp", "gru", "lstm", "xpatch"} if label == "controls" else {"mlp", "gru", "lstm", "xpatch"}
        require(set(protocol.get("models", [])) == expected_models and set(result.get("models", {})) == expected_models and
                set(check.get("models", {})) == expected_models and
                protocol.get("validation_origins") == result.get("validation_origins") == check.get("validation_origins"),
                "First minute fold model inventory or origin count differs")
        source_hashes = protocol.get("source_sha256", {})
        require(source_hashes, "First minute fold lacks source identities")
        for name, source_hash in source_hashes.items():
            source = (project_root / name).resolve()
            require(source.is_relative_to(project_root.resolve()) and source.is_file() and
                    digest(source) == source_hash, "First minute fold source differs")
        docs[label] = {"protocol": protocol, "result": result, "check": check,
                       "protocol_sha256": digest(protocol_path), "result_sha256": digest(result_path),
                       "check_sha256": digest(check_path)}
    c, n = docs["controls"], docs["neural"]
    require(n["protocol"].get("control_protocol_sha256") == c["protocol_sha256"] and
            n["protocol"].get("control_result_sha256") == c["result_sha256"] and
            n["protocol"].get("control_verification_sha256") == c["check_sha256"] and
            n["result"].get("controls_result_sha256") == c["result_sha256"] and
            n["check"].get("controls_verification_sha256") == c["check_sha256"] and
            n["protocol"].get("split_sha256") == c["protocol"].get("split_sha256") and
            n["protocol"].get("admission_sha256") == c["protocol"].get("admission_sha256") and
            n["protocol"].get("validation_origins") == c["protocol"].get("validation_origins"),
            "First minute fold controls and neural results are not aligned")
    return {"validation_origins": c["protocol"]["validation_origins"],
            "validation_sha256": c["protocol"]["split_sha256"]["validation"],
            "controls_check_sha256": c["check_sha256"], "neural_check_sha256": n["check_sha256"]}


def snapshot():
    states = {}

    def state(path):
        path = in_tree(path)
        if path not in states:
            data = path.read_bytes()
            states[path] = (json.loads(data), digest_bytes(data))
        return states[path][0]

    def protocol(folder, expected_hash=None):
        path = in_tree(folder / "PRIVATE_PROTOCOL.json")
        value = read_json(path)
        if expected_hash:
            require(digest(path) == expected_hash, "Queue protocol digest differs")
        require(value.get("Q3_values_read") is False and value.get("independent_test") is False,
                "Queue protocol scope differs")
        return value

    records = {}

    def record(task, fold, entry, queue_status, evidence=None):
        key = (*task, int(fold))
        require(task in TASKS and 1 <= int(fold) <= 5 and key not in records,
                "Unexpected or duplicated task-fold identity")
        item = {"profile": task[0], "horizon": task[1], "target": task[2], "fold": int(fold),
                "status": entry.get("status"), "queue_status": queue_status}
        if evidence:
            item["evidence"] = evidence
        records[key] = item

    for folder, annual in ((CORE, False), (ANNUAL, True)):
        queue = state(folder / "PRIVATE_QUEUE_STATE.json")
        plan = protocol(folder, queue.get("protocol_sha256"))
        require(set(queue["jobs"]) == {job["key"] for job in plan["jobs"]}, "Queue job inventory differs")
        for job in plan["jobs"]:
            task = (("annual", int(plan["horizon_months"]), job["target"]) if annual
                    else (job["profile"], int(job["horizon"]), job["target"]))
            entry = queue["jobs"][job["key"]]
            evidence = None
            if entry.get("status") == "completed":
                fold_dir = folder / job["key"]
                protocol_hash = digest(folder / "PRIVATE_PROTOCOL.json")
                if entry.get("reused_previously_viewed") is True:
                    require(job["key"] == "hourly_h8_future_log_return_f1", "Unexpected reused fold")
                    old_folder = OUT / "private_hourly_exploration/20261107T133537548946Z"
                    fold_dir = old_folder / "fold_01"
                    protocol_hash = digest(old_folder / "PRIVATE_PROTOCOL.json")
                require(in_tree(entry["fold_directory"]) == fold_dir.resolve(), "Unexpected fold directory")
                evidence = verify_fold(entry, (*task, int(job["fold"])), job,
                                       folder / (job["key"] + ".verification.json"), fold_dir,
                                       protocol_hash=protocol_hash)
            record(task, job["fold"], entry, queue["status"], evidence)

    matrix_state = state(MATRIX / "PRIVATE_QUEUE_STATE.json")
    matrix = protocol(MATRIX, matrix_state.get("protocol_sha256"))
    require(set(matrix_state["jobs"]) == {job["key"] for job in matrix["jobs"]}, "Minute matrix inventory differs")
    sources = {str(Path(x["protocol"]).parent.resolve()): x for x in matrix["external_sources"]}
    for external in matrix["excluded_external_jobs"]:
        task = ("minute", int(external["horizon"]), external["target"])
        fold = int(external["fold"])
        if external["source"] == "verified full-origin controls and neural first fold":
            require(task == ("minute", 60, "future_log_return") and fold == 1,
                    "Unexpected separate first-fold identity")
            record(task, fold, {"status": "completed"}, "completed", verify_first_minute_fold())
            continue
        folder = in_tree(external["source"])
        source = sources[str(folder)]
        plan = protocol(folder, source["protocol_sha256"])
        queue = state(folder / "PRIVATE_QUEUE_STATE.json")
        entry = queue["workers"][str(external["worker"])]
        if source["kind"] == "h60_return":
            jobs = [j for j in plan["jobs"] if int(j["fold"]) == fold]
            standard = folder / f"fold_{fold:02d}_verification.json"
            fold_dir = folder / f"fold_{fold:02d}"
        else:
            jobs = [j for j in plan["jobs"] if j["key"] == external["worker"]]
            standard = folder / (str(external["worker"]) + "_verification.json")
            fold_dir = folder / str(external["worker"])
        require(len(jobs) == 1 and int(jobs[0]["fold"]) == fold and
                (source["kind"] == "h60_return" or
                 (int(jobs[0]["horizon"]) == task[1] and jobs[0]["target"] == task[2])),
                "External minute job differs from the matrix plan")
        evidence = verify_fold(entry, (*task, fold), jobs[0], standard, fold_dir,
                               protocol_hash=source["protocol_sha256"]) if entry.get("status") == "completed" else None
        record(task, fold, entry, queue["status"], evidence)

    for job in matrix["jobs"]:
        task = ("minute", int(job["horizon"]), job["target"])
        entry = matrix_state["jobs"][job["key"]]
        evidence = verify_fold(entry, (*task, int(job["fold"])), job,
                               MATRIX / "verifications" / (job["key"] + ".json"),
                               MATRIX / "jobs" / job["key"],
                               protocol_hash=digest(MATRIX / "PRIVATE_PROTOCOL.json")) if entry.get("status") == "completed" else None
        record(task, job["fold"], entry, matrix_state["status"], evidence)

    require(set(records) == {(*task, fold) for task in TASKS for fold in range(1, 6)},
            "The planned 100 task-folds are not represented exactly once")
    paired = defaultdict(dict)
    for key, row in records.items():
        if row.get("evidence"):
            paired[(key[0], key[1], key[3])][key[2]] = row["evidence"]["validation_sha256"]
    comparable = [v for v in paired.values() if len(v) == 2]
    require(all(v["future_close"] == v["future_log_return"] for v in comparable),
            "Completed price and return tasks do not share a validation-origin identity")
    for path, (_, previous_hash) in states.items():
        if digest(path) != previous_hash:
            raise StateChanged("Queue changed while the snapshot was being assembled")
    ordered = [records[k] for k in sorted(records)]
    complete = [r for r in ordered if r["status"] == "completed"]
    per_task = Counter((r["profile"], r["horizon"], r["target"]) for r in complete)
    return {"status": "private_verified_coverage_snapshot",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "completed_folds": len(complete), "planned_folds": len(ordered),
            "five_fold_tasks": sum(per_task[t] == 5 for t in TASKS), "planned_tasks": len(TASKS),
            "status_counts": dict(Counter(r["status"] for r in ordered)),
            "paired_price_return_folds_checked": len(comparable),
            "queue_state_sha256": {str(p): h for p, (_, h) in states.items()},
            "script_sha256": digest(Path(__file__)), "folds": ordered,
            "limitations": ["This verifies recorded identities and queue status, not a new arithmetic re-evaluation.",
                            "All folds are exposed historical development; source-permission status is recorded separately.",
                            "Annual coverage uses the user-accepted short-history variant."]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="Write a new JSON snapshot instead of printing the full result")
    args = parser.parse_args()
    for attempt in range(3):
        try:
            result = snapshot()
            break
        except StateChanged:
            if attempt == 2:
                raise
            time.sleep(0.2)
    if args.output:
        path = in_tree(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        print(json.dumps({k: result[k] for k in ("status", "completed_folds", "planned_folds",
                                                   "five_fold_tasks", "planned_tasks", "status_counts",
                                                   "paired_price_return_folds_checked")} | {"path": str(path)},
                         ensure_ascii=False))
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
