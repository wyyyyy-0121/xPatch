"""Fail-closed acceptance; a plan or a finished training job is never delivery."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess
from statistics import median

from .evaluation import evaluate
from .io import ROOT, read_json, sha, stable_hash, write_json
from .jobs import receipt
from .provenance import verify_freeze
from .search import rank_trials
from .splits import Partition, inner_partitions


def search_coverage(directory, config, setup):
    expected_initial = {name: count * config["splits"]["inner_folds"] for name, count in config["search"]["counts"].items()}
    evidence, failures, errors = [], [], []
    registered = {name: {r["id"]: r["params"] for r in rows} for name, rows in setup["search"]["models"].items()}
    for split in setup["partitions"]:
        outer = Partition(**{key: value if key == "name" else tuple(value) for key, value in split.items()})
        inner = {p.name: p.dictionary() for p in inner_partitions(outer, config["splits"]["inner_folds"],
            config["splits"]["inner_validation_sessions"], config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"])}
        for panel in config["feature_sets"]:
            for name in config["models"]:
                identity = f"{split['name']}_{panel}_{name}"
                group = directory / identity
                entry = read_json(group / "experiment.json")
                if entry["status"] != "complete":
                    errors.append(identity + ": " + entry["status"])
                    continue
                selection = read_json(group / "selection.json")
                if not selection["hpo_complete"]:
                    errors.append(identity + ": incomplete selection")
                if name == "last_rv":
                    continue
                records = [receipt(ROOT / path) for path in selection["inner_jobs"]]
                for row in records:
                    if row is None:
                        errors.append(identity + ": missing trial receipt")
                    elif row["status"] != "complete":
                        failures.append({"directory": row["directory"], "error": row.get("error")})
                    if row is not None:
                        request = read_json(ROOT / row["directory"] / "request.json")
                        if (row["config_id"] not in registered[name] or
                                stable_hash(request["params"]) != stable_hash(registered[name].get(row["config_id"])) or
                                row["fold"] not in inner or stable_hash(request["partition"]) != stable_hash(inner.get(row["fold"])) or
                                request["phase"] != "inner" or request["signature"] != setup["signature"]):
                            errors.append(identity + ": trial differs from frozen space or dates")
                initial = [r for r in records if r is not None and "/initial/" in r["directory"]]
                expected_pairs = {(config_id, fold, 2026) for config_id in registered[name] for fold in inner}
                if len(initial) != expected_initial[name] or {(r["config_id"], r["fold"], r["seed"]) for r in initial} != expected_pairs:
                    errors.append(identity + ": initial search coverage differs")
                last_stage = initial
                if name in config["search"]["stochastic_models"]:
                    replicas = [r for r in records if r is not None and "/finalists/" in r["directory"]]
                    if len(replicas) != config["search"]["finalist_configs"] * 2 * config["splits"]["inner_folds"]:
                        errors.append(identity + ": finalist seed coverage differs")
                if name == "xpatch":
                    for epoch, keep in zip(config["search"]["xpatch"]["halving_epochs"][1:], config["search"]["xpatch"]["halving_keep"][1:]):
                        count = sum(r is not None and f"/halving_e{epoch}/" in r["directory"] for r in records)
                        if count != keep * config["splits"]["inner_folds"]:
                            errors.append(identity + f": halving {epoch} coverage differs")
                        stage_rows = [r for r in records if r is not None and f"/halving_e{epoch}/" in r["directory"]]
                        allowed = {r["config_id"] for r in rank_trials(last_stage)[:keep]}
                        if {(r["config_id"], r["fold"], r["seed"]) for r in stage_rows} != {(i, f, 2026) for i in allowed for f in inner}:
                            errors.append(identity + f": halving {epoch} differs from inner ranking")
                        last_stage = stage_rows
                final_rows = last_stage
                seeds = (2026,)
                if name in config["search"]["stochastic_models"]:
                    seeds = tuple(config["search"]["seeds"])
                    finalists = {r["config_id"] for r in rank_trials(last_stage)[:config["search"]["finalist_configs"]]}
                    final_rows = [r for r in last_stage if r["config_id"] in finalists] + replicas
                    if {(r["config_id"], r["fold"], r["seed"]) for r in replicas} != {(i, f, s) for i in finalists for f in inner for s in seeds[1:]}:
                        errors.append(identity + ": finalist identities differ from ranking")
                ranking = rank_trials(final_rows, expected_seeds=seeds)
                if not ranking or ranking[0]["config_id"] != selection["config_id"]:
                    errors.append(identity + ": final selection differs from inner scores")
                if name in ("xpatch", "lightgbm"):
                    key = "best_epoch" if name == "xpatch" else "best_iteration"
                    chosen_rows = [r for r in final_rows if r["config_id"] == selection["config_id"] and r["status"] == "complete"]
                    fixed = max(1, int(median(r["fitting"][key] for r in chosen_rows))) if chosen_rows else None
                    if fixed != selection["fixed_training_length"]:
                        errors.append(identity + ": outer length differs from inner median")
                evidence.append({"group": identity, "initial_trials": len(initial), "total_inner_jobs": len(records)})
    return {"passed": not errors, "groups": evidence, "failed_trials": failures, "errors": errors}


def artifact_index(directory, signature):
    """Root manifest -> every job receipt -> every hashed model/prediction file."""
    jobs = {}
    for path in sorted(Path(directory).rglob("result.json")):
        result = receipt(path.parent)
        if result["signature"] != signature or result["directory"] != str(path.parent.relative_to(ROOT)):
            raise ValueError("Artifact index found another run identity")
        request = read_json(path.parent / "request.json")
        if result["request_hash"] != stable_hash(request):
            raise ValueError("Artifact index found an altered job request")
        jobs[str(path.relative_to(ROOT))] = {"sha256": sha(path), "request_hash": result["request_hash"],
            "status": result["status"], "phase": result["phase"], "files": result["files"]}
    metadata = [p for p in Path(directory).rglob("*.json") if p.name in
        ("setup.json", "selection.json", "experiment.json", "summary.json", "run.json", "scheduler.json", "statistics.json", "statistics_failure.json")]
    return {"schema_version": 1, "signature": signature, "jobs": jobs, "job_count": len(jobs),
        "metadata": {str(path.relative_to(ROOT)): sha(path) for path in sorted(metadata)}}


def accept(config, run_directory):
    from .confirmation import confirmation_evidence
    from .inference import infer
    from .verify import require_verifications
    directory = Path(run_directory)
    checks, errors = {}, []
    manifest = None
    outcome = None
    try:
        freeze = verify_freeze(config)
        checks["signed_source_and_environment"] = True
        history = read_json(ROOT / "delivery/history.json")
        for path, digest in history["historical_file_hashes"].items():
            if sha(ROOT / path) != digest:
                raise ValueError(f"Historical evidence changed: {path}")
        checks["historical_files_preserved"] = True
        setup = read_json(directory / "setup.json")
        run_status = read_json(directory / "run.json")
        checks["all_model_groups_finished"] = run_status["finished_groups"] == 120 and run_status["search_finished"]
        checks["compute_budget_not_exhausted"] = not run_status["budget_exhausted"]
        coverage = search_coverage(directory, config, setup)
        checks["full_preregistered_search_coverage"] = coverage["passed"]
        write_json(directory / "search_coverage.json", coverage)
        before = read_json(directory / "evaluation.json")
        rebuilt = evaluate(directory, config)
        checks["metrics_and_gates_recomputed"] = stable_hash(before) == stable_hash(rebuilt)
        checks["all_twelve_model_panel_results"] = rebuilt["complete_model_panel_results"] == 12 and not rebuilt["errors"]
        checks["run_matches_signed_setup"] = (setup["signature"] == freeze["signature"] and setup["config"] == config and
            stable_hash(setup["search"]) == stable_hash(read_json(ROOT / "configs/rv30_search.json")))
        development_verifications = require_verifications(config, directory)
        checks["two_complete_development_verifications"] = len(development_verifications) == 2
        conditional_evidence = confirmation_evidence(config, directory, freeze["signature"])
        checks["conditional_confirmation_recomputed_and_verified"] = True
        confirmation = read_json(directory / "confirmation/evaluation.json")
        if confirmation["passed_candidates"]:
            # A passed flag or a backfilled prediction cannot qualify a prospective run.
            checks["qualified_prospective_60_sessions"] = False
            outcome = "waiting_prospective_acceptance"
        else:
            outcome = "completed_no_predictive_edge"
        if not rebuilt["passed_candidates"]:
            checks["conditional_stages_skipped_by_gate"] = confirmation["status"] == "skipped_no_development_gate_pass"
        model = read_json(ROOT / "delivery/model.json")
        checks["frozen_model_artifacts"] = (model["signature"] == freeze["signature"] and
            sha(ROOT / model["source_directory"] / "result.json") == model["source_result_sha256"] and
            model["prerequisites"] == conditional_evidence and model["role"] == "reference_har_baseline" and
            model["outcome"] == outcome and not model["active_predictive_edge_established"] and not model["prospective_qualified"])
        delivery_verifications = require_verifications(config, directory, "delivery")
        checks["two_complete_delivery_model_verifications"] = len(delivery_verifications) == 2
        example = infer(config)
        write_json(ROOT / "delivery/example_prediction.json", example)
        checks["runnable_inference_example"] = (example["rv_mean"] is not None and example["rv_mean"] > 0 and
            not example["direction_or_profit_prediction"] and not example["eligible_signal"])
        checks["report_and_documentation"] = all((ROOT / path).is_file() for path in ("delivery/REPORT.md", "delivery/README.md", "delivery/results.json"))
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        committed_files = dict(freeze["effective_source_files"])
        committed_files["delivery/rv30_freeze.json"] = sha(ROOT / "delivery/rv30_freeze.json")
        amendment = freeze["postprocessing_amendment"]
        committed_files.update(freeze["postprocessing_lock_files"])
        import hashlib
        matches = []
        for path, expected in committed_files.items():
            blob = subprocess.run(["git", "show", f"{git_commit}:{path}"], cwd=ROOT, capture_output=True)
            matches.append(blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == expected)
        checks["signed_code_locally_committed"] = all(matches)
        index = artifact_index(directory, freeze["signature"])
        write_json(directory / "artifact_index.json", index)
        checks["complete_job_artifact_index"] = bool(index["job_count"])
        if all(checks.values()):
            result_summary = read_json(ROOT / "delivery/results.json")
            result_summary.update(engineering_final_acceptance=True, outcome=outcome)
            write_json(ROOT / "delivery/results.json", result_summary)
            manifest = {"schema_version": 1, "signature": freeze["signature"], "outcome": outcome, "code_commit": git_commit,
                "postprocessing_signature": amendment["signature"] if amendment else None,
                "training_source_snapshot_commit": amendment["payload"]["base_code_commit"] if amendment else git_commit,
                "engineering_accepted": True, "active_predictive_edge_established": False,
                "created_utc": datetime.now(timezone.utc).isoformat(), "model": "delivery/model.json",
                "report": "delivery/REPORT.md", "readme": "delivery/README.md", "results": "delivery/results.json",
                "run_directory": str(directory.relative_to(ROOT)), "search_coverage": str((directory / "search_coverage.json").relative_to(ROOT)),
                "artifact_index": str((directory / "artifact_index.json").relative_to(ROOT)),
                "acceptance": str((directory / "acceptance.json").relative_to(ROOT)),
                "example_prediction": "delivery/example_prediction.json", "external_dependencies": "delivery/external_dependencies.json"}
            files = [ROOT / path for path in ("GOAL.md", "RESEARCH_PROTOCOL.md", "delivery/model.json", "delivery/REPORT.md",
                "delivery/README.md", "delivery/results.json", "delivery/rv30_freeze.json", "delivery/rv30_dependencies.json",
                "delivery/data.json", "delivery/history.json", "delivery/external_dependencies.json", "delivery/example_prediction.json",
                "configs/rv30.json", "configs/rv30_search.json", "configs/rv30_model_lock.json")]
            files.extend(ROOT / path for path in freeze["postprocessing_lock_files"])
            files += [directory / path for path in ("evaluation.json", "daily_losses.csv", "search_coverage.json", "artifact_index.json", "confirmation/evaluation.json")]
            for proofs in (development_verifications, conditional_evidence["verifications"], delivery_verifications):
                files.extend(ROOT / proof["path"] for proof in proofs.values())
            manifest["files"] = {str(path.relative_to(ROOT)): sha(path) for path in files}
    except (ValueError, KeyError, OSError, AssertionError, TypeError, IndexError, StopIteration) as error:
        errors.append(str(error))
    result = {"passed": bool(checks) and all(checks.values()) and not errors, "checks": checks, "errors": errors,
        "outcome": outcome, "goal_can_complete": bool(checks) and all(checks.values()) and not errors}
    write_json(directory / "acceptance.json", result)
    if result["passed"] and manifest is not None:
        manifest["files"][str((directory / "acceptance.json").relative_to(ROOT))] = sha(directory / "acceptance.json")
        write_json(ROOT / "delivery/manifest.json", manifest)
    return result
