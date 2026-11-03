"""Conditional historical confirmation, using last-development configurations."""
from pathlib import Path

from .cache import load_data, prepare_data
from .evaluation import bundle, compare, describe, gates
from .io import ROOT, read_json, sha, stable_hash, write_json
from .jobs import receipt
from .provenance import verify_freeze
from .scheduler import Scheduler
from .splits import partition
from .verify import expected_sources, require_verifications


def development_prerequisites(config, directory, signature):
    development = read_json(Path(directory) / "evaluation.json")
    if (development["signature"] != signature or development["errors"] or
            development["complete_model_panel_results"] != 12):
        raise ValueError("Historical confirmation requires complete development evaluation")
    proofs = require_verifications(config, directory)
    return {"development_evaluation": {"path": str((Path(directory) / "evaluation.json").relative_to(ROOT)),
            "sha256": sha(Path(directory) / "evaluation.json")},
        "development_verifications": proofs,
        "not_before_utc": max(read_json(ROOT / proof["path"])["completed_utc"] for proof in proofs.values())}


def summarize(config, sources, allowed, signature, prerequisites):
    if not allowed:
        return {"status": "skipped_no_development_gate_pass", "passed_candidates": [], "signature": signature,
            "allowed_by_development": [], "errors": {}, "job_directories": [], "prerequisites": prerequisites}
    merged, descriptions, comparisons, errors = {}, {}, {}, {}
    for key, paths in sources.items():
        panel, name = key.split("/")
        seeds = config["search"]["seeds"] if name in config["search"]["stochastic_models"] else [2026]
        try:
            merged[key] = bundle(paths, seeds)
            descriptions[key] = describe(merged[key], confirmation=True)
        except ValueError as error:
            errors[key] = str(error)
    for key in allowed:
        baseline = key.split("/")[0] + "/har"
        if key in merged and baseline in merged:
            comparisons[key] = compare(merged[key], merged[baseline], config, confirmation=True)
    comparisons = gates(comparisons, config, confirmation=True)
    passed = [key for key in allowed if comparisons[key]["passed"]]
    return {"status": "waiting_prospective_eligibility" if passed else "no_candidate_passed_historical_confirmation",
        "signature": signature, "allowed_by_development": allowed, "passed_candidates": passed,
        "descriptions": descriptions, "comparisons": comparisons, "errors": errors,
        "job_directories": sorted(path for paths in sources.values() for path in paths), "prerequisites": prerequisites,
        "historical_exposure": True, "prospective_qualified": False,
        "next_required_stage": "received_at_capture_and_60_complete_sessions" if passed else "complete_negative_delivery"}


def confirmation_evidence(config, run_directory, signature):
    directory = Path(run_directory)
    prerequisites = development_prerequisites(config, directory, signature)
    development = read_json(directory / "evaluation.json")
    result = read_json(directory / "confirmation/evaluation.json")
    sources = {}
    if development["passed_candidates"]:
        for path, request in expected_sources(config, directory, "confirmation").items():
            if receipt(ROOT / path)["started_utc"] < prerequisites["not_before_utc"]:
                raise ValueError("Historical confirmation began before development verification completed")
            sources.setdefault(f"{request['panel']}/{request['model']}", []).append(path)
    rebuilt = summarize(config, sources, development["passed_candidates"], signature, prerequisites)
    if stable_hash(rebuilt) != stable_hash(result) or result["errors"]:
        raise ValueError("Historical confirmation or its development prerequisites could not be rebuilt")
    verifications = require_verifications(config, directory, "confirmation") if sources else {}
    return {"evaluation": {"path": str((directory / "confirmation/evaluation.json").relative_to(ROOT)),
            "sha256": sha(directory / "confirmation/evaluation.json")}, "verifications": verifications,
        "not_before_utc": max([prerequisites["not_before_utc"],
            *(read_json(ROOT / proof["path"])["completed_utc"] for proof in verifications.values())])}


def confirm(config, run_directory):
    freeze = verify_freeze(config)
    directory = Path(run_directory)
    prerequisites = development_prerequisites(config, directory, freeze["signature"])
    development = read_json(directory / "evaluation.json")
    allowed = development["passed_candidates"]
    destination = directory / "confirmation"
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "evaluation.json").exists() and read_json(destination / "evaluation.json").get("prerequisites") != prerequisites:
        raise ValueError("Historical confirmation prerequisites changed; preserve existing evidence")
    if not allowed:
        result = summarize(config, {}, allowed, freeze["signature"], prerequisites)
        write_json(destination / "evaluation.json", result)
        return result
    cache, identity = prepare_data(config)
    data = load_data(cache, identity)
    start, end = config["periods"]["historical_confirmation_start"], config["periods"]["historical_confirmation_end"]
    prior = [str(day) for day in data.days if day < start]
    evaluation_dates = [str(day) for day in data.days if start <= day <= end]
    split = partition("historical_confirmation", prior, evaluation_dates, config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"])
    requests, sources = [], {}
    needed = sorted(set(allowed) | {key.split("/")[0] + "/har" for key in allowed})
    for key in needed:
        panel, name = key.split("/")
        selection = read_json(directory / f"2025Q2_{panel}_{name}/selection.json")
        if name == "last_rv":
            params = read_json(ROOT / "configs/rv30_search.json")["models"]["last_rv"][0]["params"]
        else:
            params = selection["params"]
        seeds = config["search"]["seeds"] if name in config["search"]["stochastic_models"] else [2026]
        sources[key] = []
        for seed in seeds:
            job_id = f"confirmation/{panel}/{name}/s{seed}"
            request = {"job_id": job_id, "directory": str((destination / panel / name / f"s{seed}").relative_to(ROOT)),
                "signature": freeze["signature"], "phase": "confirmation", "partition": split.dictionary(), "model": name,
                "panel": panel, "config_id": selection["config_id"], "params": params, "seed": seed}
            if name == "xpatch":
                request.update(fixed_epochs=selection["fixed_training_length"], max_epochs=30)
            if name == "lightgbm":
                request["fixed_iterations"] = selection["fixed_training_length"]
            requests.append(request)
            sources[key].append(request["directory"])
    with Scheduler(config, cache, identity, destination) as scheduler:
        results = scheduler.map(requests)
    result = summarize(config, sources, allowed, freeze["signature"], prerequisites)
    write_json(destination / "evaluation.json", result)
    return result
