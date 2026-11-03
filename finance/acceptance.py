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


def accept(config, run_directory):
    directory = Path(run_directory)
    checks, errors = {}, []
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
        for number in (1, 2):
            summary = read_json(directory / f"verify_{number}/summary.json")
            conditional = read_json(directory / "confirmation/evaluation.json")
            expected_jobs = 200 + len(conditional.get("job_directories", []))
            good = summary["passed"] and summary["signature"] == freeze["signature"] and summary["jobs"] == expected_jobs
            for path in summary["job_directories"]:
                evidence = receipt(ROOT / path)
                good &= evidence is not None and evidence["status"] == "complete" and evidence["verification"]["passed"]
            checks[f"independent_verify_{number}"] = bool(good)
        confirmation = read_json(directory / "confirmation/evaluation.json")
        if rebuilt["passed_candidates"]:
            checks["conditional_historical_confirmation"] = not confirmation.get("errors")
            passed = confirmation["passed_candidates"]
            if passed:
                # A manually supplied passed flag is not evidence of on-time forecasts.
                # Conditional prospective capture/evaluation requires its own frozen
                # issuance ledger and verifier before this stage can be admitted.
                checks["qualified_prospective_60_sessions"] = False
                outcome = "waiting_prospective_acceptance"
            else:
                outcome = "completed_no_predictive_edge"
        else:
            checks["conditional_stages_skipped_by_gate"] = confirmation["status"] == "skipped_no_development_gate_pass"
            outcome = "completed_no_predictive_edge"
        model = read_json(ROOT / "delivery/model.json")
        checks["frozen_model_artifacts"] = sha(ROOT / model["source_directory"] / "result.json") == model["source_result_sha256"]
        receipt(ROOT / model["source_directory"])
        from .inference import infer
        regenerated_example = infer(config)
        write_json(ROOT / "delivery/example_prediction.json", regenerated_example)
        example = read_json(ROOT / "delivery/example_prediction.json")
        checks["runnable_inference_example"] = example["rv_mean"] is not None and example["rv_mean"] > 0 and not example["direction_or_profit_prediction"]
        checks["report_and_documentation"] = all((ROOT / path).is_file() for path in ("delivery/REPORT.md", "delivery/README.md", "delivery/results.json"))
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        source_commit_matches = True
        import hashlib
        for path, expected in freeze["payload"]["source_files"].items():
            blob = subprocess.run(["git", "show", f"{git_commit}:{path}"], cwd=ROOT, capture_output=True)
            source_commit_matches &= blob.returncode == 0 and hashlib.sha256(blob.stdout).hexdigest() == expected
        checks["signed_code_locally_committed"] = bool(source_commit_matches)
        if all(checks.values()):
            manifest = {"schema_version": 1, "signature": freeze["signature"], "outcome": outcome, "code_commit": git_commit,
                "engineering_accepted": True, "active_predictive_edge_established": False,
                "created_utc": datetime.now(timezone.utc).isoformat(), "model": "delivery/model.json",
                "report": "delivery/REPORT.md", "readme": "delivery/README.md", "results": "delivery/results.json",
                "run_directory": str(directory.relative_to(ROOT)), "search_coverage": str((directory / "search_coverage.json").relative_to(ROOT)),
                "external_dependencies": "delivery/external_dependencies.json"}
            files = [ROOT / path for path in ("delivery/model.json", "delivery/REPORT.md", "delivery/README.md", "delivery/results.json", "delivery/rv30_freeze.json")]
            files += [directory / path for path in ("evaluation.json", "daily_losses.csv", "search_coverage.json", "verify_1/summary.json", "verify_2/summary.json", "confirmation/evaluation.json")]
            manifest["files"] = {str(path.relative_to(ROOT)): sha(path) for path in files}
            write_json(ROOT / "delivery/manifest.json", manifest)
            result_summary = read_json(ROOT / "delivery/results.json")
            result_summary["engineering_final_acceptance"] = True
            result_summary["outcome"] = outcome
            write_json(ROOT / "delivery/results.json", result_summary)
            manifest["files"]["delivery/results.json"] = sha(ROOT / "delivery/results.json")
            write_json(ROOT / "delivery/manifest.json", manifest)
    except (ValueError, KeyError, FileNotFoundError, AssertionError) as error:
        errors.append(str(error))
    result = {"passed": bool(checks) and all(checks.values()) and not errors, "checks": checks, "errors": errors,
        "goal_can_complete": bool(checks) and all(checks.values()) and not errors}
    write_json(directory / "acceptance.json", result)
    return result
