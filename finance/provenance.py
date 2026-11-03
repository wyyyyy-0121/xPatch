"""Freeze hashes only after engineering checks and pre-2023 resource calibration."""
from datetime import datetime, timezone
from importlib import metadata
import platform
import re

from .cache import prepare_data
from .io import ROOT, read_json, sha, stable_hash, write_json
from .search import configurations


FREEZE_PATH = ROOT / "delivery/rv30_freeze.json"
POSTPROCESS_PATH = ROOT / "delivery/rv30_postprocessing_lock.json"
# Postprocessing corrections cannot change model fitting, information, search,
# resources, calibration, metrics, or statistical gate definitions.
POSTPROCESS_ALLOWED = frozenset({"finance/provenance.py", "finance/delivery.py", "finance/verify.py",
    "finance/confirmation.py", "finance/inference.py", "finance/acceptance.py", "finance/cli.py"})
DEPENDENCIES = ("numpy", "pandas", "scipy", "scikit-learn", "lightgbm", "torch", "chronos-forecasting",
    "transformers", "safetensors", "exchange-calendars", "pytest")


def dependencies():
    return {"python": platform.python_version(), "packages": {name: metadata.version(name) for name in DEPENDENCIES},
        "installed_distributions": dict(sorted((distribution.metadata["Name"], distribution.version)
            for distribution in metadata.distributions()))}


def sources():
    paths = [*sorted((ROOT / "finance").glob("*.py")), *sorted((ROOT / "tests").glob("test_finance_rv_*.py")),
        ROOT / "tests/test_finance_runtime.py", ROOT / "tests/test_delivery_acquisition.py",
        ROOT / "configs/rv30.json", ROOT / "configs/rv30_model_lock.json", ROOT / "RESEARCH_PROTOCOL.md"]
    paths += [ROOT / path for path in ("models/xPatch.py", "layers/network.py", "layers/ema.py", "layers/dema.py", "layers/decomp.py", "layers/revin.py",
        "finance_v2/data.py", "finance_v2/io.py", "finance_v2/__init__.py")]
    return {str(path.relative_to(ROOT)): sha(path) for path in paths}


def freeze(config, smoke_directory, tests_log):
    if FREEZE_PATH.exists():
        return verify_freeze(config)
    settings = config["runtime"]
    if config["status"] != "frozen" or any(settings[key] is None for key in
            ("task_timeout_seconds", "total_cpu_core_hours", "total_gpu_hours")):
        raise ValueError("Set pre-outer budgets and frozen status before signing")
    for key in ("total_cpu_core_hours", "total_gpu_hours"):
        if settings[key] <= 0:
            raise ValueError("Compute budgets must be positive")
    if not isinstance(settings["task_timeout_seconds"], dict) or any(settings["task_timeout_seconds"].get(model, 0) <= 0 for model in config["models"]):
        raise ValueError("Each model needs a positive frozen job timeout")
    smoke_directory, tests_log = ROOT / smoke_directory, ROOT / tests_log
    summary = read_json(smoke_directory / "summary.json")
    smoke_setup = read_json(smoke_directory / "setup.json")
    native = read_json(smoke_directory / "native_checks.json")
    if not summary["passed"] or not native["passed"] or len(native["configurations"]) != 32:
        raise ValueError("Pre-2023 smoke and native configuration checks must pass")
    if any(day >= "2023-01-01" for day in smoke_setup["partition"]["evaluation_dates"]):
        raise ValueError("Smoke data extends into the registered outer evaluation")
    if stable_hash(smoke_setup["search"]) != stable_hash(configurations(config)):
        raise ValueError("Search changed since the passing resource smoke")
    for path, digest in smoke_setup["sources"].items():
        if path not in ("configs/rv30.json", "RESEARCH_PROTOCOL.md") and sha(ROOT / path) != digest:
            raise ValueError(f"Code changed since the passing smoke: {path}")
    if not summary.get("independent_replay_passed") or summary.get("independent_replay_jobs", 0) < 16:
        raise ValueError("Pre-outer smoke must pass independent saved-model replay")
    log = tests_log.read_text()
    counts = re.findall(r"(\d+) passed", log)
    if not counts or re.search(r"\d+ (failed|error)", log):
        raise ValueError("Full engineering tests must pass before signing")
    cache_directory, cache_identity = prepare_data(config)
    lock = read_json(ROOT / config["model_lock"])
    for entry in lock["models"].values():
        for filename, value in entry["files"].items():
            if sha(ROOT / entry["directory"] / filename) != value["sha256"]:
                raise ValueError("Frozen pretrained weights changed")
    dependency = dependencies()
    write_json(ROOT / "delivery/rv30_dependencies.json", dependency)
    search = configurations(config)
    write_json(ROOT / "configs/rv30_search.json", search)
    payload = {"schema_version": 1, "task": "rv30", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": sources(), "config": config, "search_sha256": sha(ROOT / "configs/rv30_search.json"),
        "dependencies": dependency, "dependencies_sha256": sha(ROOT / "delivery/rv30_dependencies.json"),
        "data_manifest_sha256": sha(ROOT / config["data_manifest"]), "cache_identity": cache_identity,
        "cache_directory": str(cache_directory.relative_to(ROOT)), "cache_metadata_sha256": sha(cache_directory / "data.json"),
        "smoke_directory": str(smoke_directory.relative_to(ROOT)), "smoke_summary_sha256": sha(smoke_directory / "summary.json"),
        "tests_log": str(tests_log.relative_to(ROOT)), "tests_log_sha256": sha(tests_log), "tests_passed": int(counts[-1]),
        "pretrained_lock": lock, "outer_scores_read_before_freeze": False,
        "replay_tolerance": {"rtol": 1e-6, "atol": 1e-12},
        "pretrained_replay_scope": "first_and_last_fixed_batches_of_each_calibration_and_outer_or_confirmation_job",
        "history": "retrospective_exposed_data_not_a_clean_blind_test"}
    signature = stable_hash(payload)
    result = {"signature": signature, "payload": payload}
    write_json(FREEZE_PATH, result)
    write_json(ROOT / "delivery/status.json", {"goal_status": "active", "protocol": "frozen", "signature": signature,
        "engineering": "passed_pre_outer_tests_and_smoke", "outer_validation": "not_started", "final_acceptance": "not_started",
        "external_items": "delivery/external_dependencies.json"})
    return result


def _postprocessing_sources(freeze, source_view=None, excluded_additions=()):
    base = freeze["payload"]["source_files"]
    actual = source_view if source_view is not None else {path: sha(ROOT / path) for path in base}
    changed = {path for path in base if actual[path] != base[path]}
    if not changed and not POSTPROCESS_PATH.exists():
        return base.copy(), None
    if not changed.issubset(POSTPROCESS_ALLOWED):
        raise ValueError(f"Frozen research implementation changed: {sorted(changed - POSTPROCESS_ALLOWED)}")
    if not POSTPROCESS_PATH.exists():
        raise ValueError("Postprocessing changes require a separately signed amendment")
    amendment = read_json(POSTPROCESS_PATH)
    extension_path = POSTPROCESS_PATH.with_name("rv30_statistics_lock.json")
    if extension_path.exists():
        import hashlib
        import subprocess
        extension = read_json(extension_path)
        extended = extension["payload"]
        previous = amendment["payload"]
        if (stable_hash(previous) != amendment["signature"] or previous["base_signature"] != freeze["signature"] or
                previous["base_freeze_sha256"] != sha(FREEZE_PATH) or
                extended["prior_postprocessing_lock_sha256"] != sha(POSTPROCESS_PATH) or
                extended["prior_postprocessing_signature"] != amendment["signature"]):
            raise ValueError("Independent-statistics amendment predecessor changed")
        snapshot = extended["prior_source_snapshot_commit"]
        for path, record in previous["changes"].items():
            if path not in POSTPROCESS_ALLOWED or record["before_sha256"] != base[path]:
                raise ValueError("Predecessor amendment includes a research change")
            blob = subprocess.check_output(["git", "show", f"{snapshot}:{path}"], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != record["after_sha256"]:
                raise ValueError("Predecessor amended source snapshot changed")
        for path, expected in previous.get("added_sources", {}).items():
            blob = subprocess.check_output(["git", "show", f"{snapshot}:{path}"], cwd=ROOT)
            if hashlib.sha256(blob).hexdigest() != expected:
                raise ValueError("Predecessor validation source snapshot changed")
        for path, expected in previous["validation_files"].items():
            if sha(ROOT / path) != expected:
                raise ValueError("Predecessor validation evidence changed")
        amendment = extension
    payload = amendment["payload"]
    if stable_hash(payload) != amendment["signature"] or payload["base_signature"] != freeze["signature"]:
        raise ValueError("Postprocessing amendment identity changed")
    if payload["base_freeze_sha256"] != sha(FREEZE_PATH) or set(payload["changes"]) != changed:
        raise ValueError("Postprocessing amendment differs from actual source changes")
    for path, record in payload["changes"].items():
        if record != {"before_sha256": base[path], "after_sha256": actual[path]}:
            raise ValueError(f"Unregistered postprocessing change: {path}")
    if payload["research_definitions_changed"] or payload["model_search_space_changed"] or payload["thresholds_changed"]:
        raise ValueError("A postprocessing amendment cannot revise research definitions")
    additions = payload.get("added_sources", {})
    discovered = {str(path.relative_to(ROOT)) for pattern, parent in (("*.py", "finance"), ("test_finance_rv_*.py", "tests"))
        for path in (ROOT / parent).glob(pattern)} - set(base) - set(excluded_additions)
    if set(additions) != discovered:
        raise ValueError("Added implementation or validation sources are not fully registered")
    for path, expected in additions.items():
        if not path.startswith("tests/test_finance_rv_") or not path.endswith(".py") or path in base or sha(ROOT / path) != expected:
            raise ValueError("Unregistered added validation source")
    for path, expected in payload["validation_files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Postprocessing validation changed: {path}")
    return {**actual, **additions}, amendment


def _acceleration_sources(freeze, source_view=None, excluded_additions=()):
    """Keep the original locks intact and verify the optional execution-only lock."""
    path = ROOT / "delivery/rv30_acceleration_lock.json"
    if not path.exists():
        return _postprocessing_sources(freeze, source_view, excluded_additions)
    import hashlib
    import subprocess
    amendment = read_json(path)
    payload = amendment["payload"]
    if (stable_hash(payload) != amendment["signature"] or
            payload["base_signature"] != freeze["signature"] or
            payload["base_freeze_sha256"] != sha(FREEZE_PATH) or
            payload["prior_statistics_lock_sha256"] != sha(ROOT / "delivery/rv30_statistics_lock.json")):
        raise ValueError("Acceleration amendment or predecessor identity changed")
    if (set(payload["changes"]) != {"finance/models.py", "finance/provenance.py"} or
            set(payload["added_sources"]) != {"finance/cuda_graph.py"} or
            set(payload["support_sources"]) != {"scripts/start_rv30_checked.py",
                "scripts/analyze_rv30_termination.py", "tests/test_rv30_acceleration.py"} or
            payload["research_definitions_changed"] or payload["model_search_space_changed"] or
            payload["thresholds_changed"] or not payload["bitwise_equivalence_required"]):
        raise ValueError("Acceleration lock cannot change research definitions or arbitrary source files")
    installed = source_view if source_view is not None else {p: sha(ROOT / p) for p in freeze["payload"]["source_files"]}
    view = installed.copy()
    for name, record in payload["changes"].items():
        if view[name] != record["after_sha256"]:
            raise ValueError("Unregistered acceleration source: " + name)
        previous = subprocess.check_output(["git", "show", f"{payload['prior_source_snapshot_commit']}:{name}"], cwd=ROOT)
        if hashlib.sha256(previous).hexdigest() != record["before_sha256"]:
            raise ValueError("Acceleration predecessor source changed: " + name)
        view[name] = record["before_sha256"]
    prior, postprocessing = _postprocessing_sources(freeze, view,
        (*payload["added_sources"], *excluded_additions))
    for name, record in payload["changes"].items():
        if prior[name] != record["before_sha256"]:
            raise ValueError("Acceleration does not extend the signed implementation")
    extra = {**payload["added_sources"], **payload["support_sources"]}
    for name, expected in {**extra, **payload["validation_files"]}.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Acceleration evidence changed: " + name)
    summary = read_json(ROOT / payload["validation_summary"])
    if (not summary["passed"] or summary["all_shape_cases"] != 192 or summary["fit_resume_cases"] != 6 or
            not summary["sustained"]["passed"] or summary["sustained"]["resumed_stages"] != [1, 5, 15, 30] or
            summary["candidate_sha256"] != sha(ROOT / "finance/cuda_graph.py")):
        raise ValueError("Full-shape, checkpoint and sustained acceleration checks must pass")
    integrated = read_json(ROOT / payload["integrated_validation"])
    if not integrated["passed"] or not integrated["cross_implementation_resume_bitwise_equal"]:
        raise ValueError("Installed accelerated implementation did not pass recovery validation")
    if integrated["source_files"] != {name: installed[name] if name in installed else sha(ROOT / name)
            for name in ("finance/models.py", "finance/cuda_graph.py")}:
        raise ValueError("Installed acceleration validation belongs to different source files")
    after = {name: record["after_sha256"] for name, record in payload["changes"].items()}
    return {**prior, **after, **extra}, postprocessing


def _throughput_sources(freeze, source_view=None, excluded_additions=()):
    """Extend execution evidence without overwriting any prior signed lock."""
    path = ROOT / "delivery/rv30_throughput_lock.json"
    if not path.exists():
        return _acceleration_sources(freeze, source_view, excluded_additions)
    import hashlib
    import subprocess
    amendment = read_json(path)
    payload = amendment["payload"]
    if (stable_hash(payload) != amendment["signature"] or
            payload["base_signature"] != freeze["signature"] or
            payload["base_freeze_sha256"] != sha(FREEZE_PATH) or
            payload["prior_acceleration_lock_sha256"] != sha(ROOT / "delivery/rv30_acceleration_lock.json")):
        raise ValueError("Throughput amendment or predecessor identity changed")
    if (set(payload["changes"]) != {"finance/models.py", "finance/provenance.py"} or
            set(payload["added_sources"]) != {"finance/throughput.py"} or
            set(payload["support_sources"]) != {"tests/test_rv30_throughput.py"} or
            payload["research_definitions_changed"] or payload["model_search_space_changed"] or
            payload["thresholds_changed"] or not payload["bitwise_equivalence_required"]):
        raise ValueError("Throughput lock cannot change research definitions or arbitrary source files")
    installed = source_view if source_view is not None else {name: sha(ROOT / name) for name in freeze["payload"]["source_files"]}
    view = installed.copy()
    for name, record in payload["changes"].items():
        if installed[name] != record["after_sha256"]:
            raise ValueError("Unregistered throughput source: " + name)
        previous = subprocess.check_output(["git", "show", f"{payload['prior_source_snapshot_commit']}:{name}"], cwd=ROOT)
        if hashlib.sha256(previous).hexdigest() != record["before_sha256"]:
            raise ValueError("Throughput predecessor source changed: " + name)
        view[name] = record["before_sha256"]
    prior, postprocessing = _acceleration_sources(freeze, view, (*payload["added_sources"], *excluded_additions))
    if any(prior[name] != record["before_sha256"] for name, record in payload["changes"].items()):
        raise ValueError("Throughput does not extend the signed implementation")
    extra = {**payload["added_sources"], **payload["support_sources"]}
    if not {payload["validation_summary"], payload["integrated_validation"]}.issubset(payload["validation_files"]):
        raise ValueError("Throughput evidence must bind both validation summaries")
    for name, expected in {**extra, **payload["validation_files"]}.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Throughput evidence changed: " + name)
    summary = read_json(ROOT / payload["validation_summary"])
    if (not summary["passed"] or summary["all_config_cases"] != 192 or summary["seed_resume_cases"] != 6 or
            [stage["to_epoch"] for stage in summary["stages"]] != [1, 5, 15, 30] or
            not summary["maximum_30_epoch_checkpoint_bitwise_equal"] or
            not summary["maximum_full_predictions_bitwise_equal"] or summary["outer_scores_read"]):
        raise ValueError("Throughput configuration, recovery and sustained validation must pass")
    integrated = read_json(ROOT / payload["integrated_validation"])
    if (not integrated["passed"] or not integrated["cross_implementation_resume_bitwise_equal"] or
            not integrated["resident_fallback_bitwise_equal"] or
            integrated["source_files"] != {name: installed[name] if name in installed else sha(ROOT / name)
                for name in ("finance/models.py", "finance/cuda_graph.py", "finance/throughput.py")}):
        raise ValueError("Installed throughput implementation did not pass recovery validation")
    after = {name: record["after_sha256"] for name, record in payload["changes"].items()}
    return {**prior, **after, **extra}, postprocessing


def _replay_sources(freeze, source_view=None, excluded_additions=()):
    """Bind a precision-only replay repair and retain every failed receipt."""
    path = ROOT / "delivery/rv30_replay_lock.json"
    if not path.exists():
        return _throughput_sources(freeze, source_view, excluded_additions)
    import hashlib
    import subprocess
    amendment = read_json(path)
    payload = amendment["payload"]
    if (stable_hash(payload) != amendment["signature"] or
            payload["base_signature"] != freeze["signature"] or
            payload["base_freeze_sha256"] != sha(FREEZE_PATH) or
            payload["prior_throughput_lock_sha256"] != sha(ROOT / "delivery/rv30_throughput_lock.json")):
        raise ValueError("Replay amendment or predecessor identity changed")
    if (set(payload["changes"]) != {"finance/models.py", "finance/verify.py", "finance/provenance.py"} or
            set(payload["support_sources"]) != {"tests/test_rv30_replay.py"} or
            payload["research_definitions_changed"] or payload["model_search_space_changed"] or
            payload["thresholds_changed"] or payload["training_changed"] or
            not payload["original_predictions_preserved"]):
        raise ValueError("Replay correction cannot revise fitting or research definitions")
    installed = source_view if source_view is not None else {name: sha(ROOT / name) for name in freeze["payload"]["source_files"]}
    view = installed.copy()
    for name, record in payload["changes"].items():
        if installed[name] != record["after_sha256"]:
            raise ValueError("Unregistered replay correction source: " + name)
        previous = subprocess.check_output(["git", "show", f"{payload['prior_source_snapshot_commit']}:{name}"], cwd=ROOT)
        if hashlib.sha256(previous).hexdigest() != record["before_sha256"]:
            raise ValueError("Replay predecessor source changed: " + name)
        view[name] = record["before_sha256"]
    prior, postprocessing = _throughput_sources(freeze, view, excluded_additions)
    if any(prior[name] != record["before_sha256"] for name, record in payload["changes"].items()):
        raise ValueError("Replay correction does not extend the signed implementation")
    if not {payload["validation_summary"], payload["verification_retry"]["failed_attempt_index"]}.issubset(payload["validation_files"]):
        raise ValueError("Replay correction must bind the full replay and preserved failures")
    for name, expected in {**payload["support_sources"], **payload["validation_files"]}.items():
        if sha(ROOT / name) != expected:
            raise ValueError("Replay correction evidence changed: " + name)
    validation = read_json(ROOT / payload["validation_summary"])
    if (not validation["passed"] or validation["models"] != 60 or
            not validation["all_full_predictions_bitwise_equal"] or validation["outer_scores_read"] or
            validation["source_files"] != {name: installed[name] for name in ("finance/models.py", "finance/verify.py")}):
        raise ValueError("Every saved xPatch must replay in full at the fitting precision")
    retry = payload["verification_retry"]
    original = ROOT / retry["run_directory"] / "verify_1"
    if (retry["scope"] != "development" or retry["round"] != 1 or
            retry["original_directory"] != str(original.relative_to(ROOT)) or
            retry["replacement_directory"] != str((original.parent / "verify_1_replay_correction").relative_to(ROOT)) or
            sha(original / "summary.json") != retry["original_summary_sha256"]):
        raise ValueError("Replay retry must preserve the original failed first round")
    index = read_json(ROOT / retry["failed_attempt_index"])
    actual = {str(p.relative_to(ROOT)): sha(p) for p in original.rglob("*") if p.is_file()}
    if (index["directory"] != retry["original_directory"] or actual != index["files"] or
            len(index["failed_receipts"]) != 42 or read_json(original / "summary.json")["passed"]):
        raise ValueError("Failed replay records were omitted, overwritten or substituted")
    after = {name: record["after_sha256"] for name, record in payload["changes"].items()}
    return {**prior, **after, **payload["support_sources"]}, postprocessing


def _prospective_sources(freeze, source_view=None, excluded_additions=()):
    """Add prospective collection without replacing any historical research lock."""
    path = ROOT / "delivery/rv30_prospective_execution_lock.json"
    if not path.exists():
        return _replay_sources(freeze, source_view, excluded_additions)
    import hashlib
    import subprocess
    lock = read_json(path)
    payload = lock["payload"]
    if (stable_hash(payload) != lock["signature"] or payload["base_signature"] != freeze["signature"] or
            payload["base_freeze_sha256"] != sha(FREEZE_PATH) or
            payload["prior_replay_lock_sha256"] != sha(ROOT / "delivery/rv30_replay_lock.json")):
        raise ValueError("Prospective execution amendment or predecessor changed")
    if (set(payload["changes"]) != {"finance/cli.py", "finance/inference.py", "finance/provenance.py"} or
            set(payload["added_sources"]) != {"finance/prospective.py", "finance/observation.py"} or
            payload["historical_research_changed"] or payload["model_search_space_changed"] or
            payload["historical_thresholds_changed"] or payload["confirmed_weights_changed"] or
            not payload["preregistered_before_live_observation"]):
        raise ValueError("Prospective execution cannot change the completed historical research")
    installed = source_view if source_view is not None else {name: sha(ROOT/name) for name in freeze["payload"]["source_files"]}
    view = installed.copy()
    for name, change in payload["changes"].items():
        previous = subprocess.check_output(["git", "show", f"{payload['prior_source_snapshot_commit']}:{name}"], cwd=ROOT)
        if installed[name] != change["after_sha256"] or hashlib.sha256(previous).hexdigest() != change["before_sha256"]:
            raise ValueError("Unregistered prospective integration: " + name)
        view[name] = change["before_sha256"]
    prior, amendment = _replay_sources(freeze, view, (*payload["added_sources"], *excluded_additions))
    if any(prior[name] != change["before_sha256"] for name, change in payload["changes"].items()):
        raise ValueError("Prospective integration does not extend the signed replay implementation")
    required = {payload["validation_summary"], payload["performance_summary"]}
    if not required.issubset(payload["validation_files"]):
        raise ValueError("Prospective execution must bind correctness and throughput evidence")
    extra = {**payload["added_sources"], **payload["support_sources"]}
    for name, digest in {**extra, **payload["validation_files"]}.items():
        if sha(ROOT/name) != digest:
            raise ValueError("Prospective code or validation changed: " + name)
    validation = read_json(ROOT/payload["validation_summary"])
    performance = read_json(ROOT/payload["performance_summary"])
    if (not validation["passed"] or not validation["timing_and_no_backfill_checks_passed"] or
            not performance["passed"] or performance["complete_session_origins"] != 346 or
            performance["models"] != 4 or performance["prospective_observations_created"] or
            performance["source_files"] != {name: sha(ROOT/name) for name in payload["added_sources"]}):
        raise ValueError("Prospective timing, integrity and complete-workload performance must pass")
    after = {name: change["after_sha256"] for name, change in payload["changes"].items()}
    return {**prior, **after, **extra}, amendment


def effective_sources(freeze):
    """Complete the forward acceptance path without changing the frozen plan."""
    path = ROOT / "delivery/rv30_completion_lock.json"
    if not path.exists():
        return _prospective_sources(freeze)
    import hashlib
    import subprocess
    lock = read_json(path)
    payload = lock["payload"]
    if (stable_hash(payload) != lock["signature"] or payload["base_signature"] != freeze["signature"] or
            payload["base_freeze_sha256"] != sha(FREEZE_PATH) or
            payload["prior_prospective_execution_lock_sha256"] != sha(ROOT / "delivery/rv30_prospective_execution_lock.json") or
            payload["prospective_plan_sha256"] != sha(ROOT / "delivery/prospective_lock.json")):
        raise ValueError("Forward completion amendment or predecessor changed")
    if (set(payload["changes"]) != {"finance/acceptance.py", "finance/delivery.py", "finance/provenance.py"} or
            set(payload["added_sources"]) != {"finance/completion.py", "finance/forward_runtime.py"} or
            set(payload["support_sources"]) != {"tests/test_rv30_completion.py", "tests/test_rv30_completion_lock.py", "tests/test_rv30_forward_runtime.py"} or
            payload["research_definitions_changed"] or payload["thresholds_changed"] or payload["model_fitting_changed"] or
            not payload["original_prospective_plan_preserved"]):
        raise ValueError("Completion cannot revise the registered research, models or forward plan")
    installed = {name: sha(ROOT/name) for name in freeze["payload"]["source_files"]}
    view = installed.copy()
    for name, record in payload["changes"].items():
        previous = subprocess.check_output(["git", "show", f"{payload['prior_source_snapshot_commit']}:{name}"], cwd=ROOT)
        if installed[name] != record["after_sha256"] or hashlib.sha256(previous).hexdigest() != record["before_sha256"]:
            raise ValueError("Unregistered completion source: " + name)
        view[name] = record["before_sha256"]
    prior, amendment = _prospective_sources(freeze, view, tuple(payload["added_sources"]))
    if any(prior[name] != record["before_sha256"] for name, record in payload["changes"].items()):
        raise ValueError("Completion does not extend the registered prospective implementation")
    if payload["validation_summary"] not in payload["validation_files"]:
        raise ValueError("Completion validation summary must be bound")
    extra = {**payload["added_sources"], **payload["support_sources"]}
    for name, digest in {**extra, **payload["validation_files"]}.items():
        if sha(ROOT/name) != digest:
            raise ValueError("Completion validation or source changed: " + name)
    validation = read_json(ROOT/payload["validation_summary"])
    if not validation["passed"] or not validation["early_acceptance_refused"] or not validation["positive_and_negative_selection_tested"]:
        raise ValueError("Forward acceptance must reject early claims and cover both outcomes")
    after = {name: record["after_sha256"] for name, record in payload["changes"].items()}
    return {**prior, **after, **extra}, amendment


def verify_freeze(config):
    freeze = read_json(FREEZE_PATH)
    payload = freeze["payload"]
    if stable_hash(payload) != freeze["signature"] or config != payload["config"] or config["status"] != "frozen":
        raise ValueError("Protocol/config freeze identity changed")
    source_files, amendment = effective_sources(freeze)
    if dependencies() != payload["dependencies"]:
        raise ValueError("Numerical dependencies differ from the signed environment")
    for path, key in ((config["data_manifest"], "data_manifest_sha256"), ("configs/rv30_search.json", "search_sha256"),
            ("delivery/rv30_dependencies.json", "dependencies_sha256"),
            (payload["cache_directory"] + "/data.json", "cache_metadata_sha256")):
        if sha(ROOT / path) != payload[key]:
            raise ValueError(f"Signed input changed: {path}")
    lock_paths = [POSTPROCESS_PATH, POSTPROCESS_PATH.with_name("rv30_statistics_lock.json"),
        ROOT / "delivery/rv30_acceleration_lock.json", ROOT / "delivery/rv30_throughput_lock.json",
        ROOT / "delivery/rv30_replay_lock.json", ROOT / "delivery/rv30_prospective_execution_lock.json",
        ROOT / "delivery/rv30_completion_lock.json"]
    return {**freeze, "effective_source_files": source_files, "postprocessing_amendment": amendment,
        "postprocessing_lock_files": {str(path.relative_to(ROOT)): sha(path) for path in lock_paths if path.exists()}}
