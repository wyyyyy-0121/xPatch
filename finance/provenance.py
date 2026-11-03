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


def effective_sources(freeze):
    base = freeze["payload"]["source_files"]
    actual = {path: sha(ROOT / path) for path in base}
    changed = {path for path in base if actual[path] != base[path]}
    if not changed and not POSTPROCESS_PATH.exists():
        return base.copy(), None
    if not changed.issubset(POSTPROCESS_ALLOWED):
        raise ValueError(f"Frozen research implementation changed: {sorted(changed - POSTPROCESS_ALLOWED)}")
    if not POSTPROCESS_PATH.exists():
        raise ValueError("Postprocessing changes require a separately signed amendment")
    amendment = read_json(POSTPROCESS_PATH)
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
        for path in (ROOT / parent).glob(pattern)} - set(base)
    if set(additions) != discovered:
        raise ValueError("Added implementation or validation sources are not fully registered")
    for path, expected in additions.items():
        if not path.startswith("tests/test_finance_rv_") or not path.endswith(".py") or path in base or sha(ROOT / path) != expected:
            raise ValueError("Unregistered added validation source")
    for path, expected in payload["validation_files"].items():
        if sha(ROOT / path) != expected:
            raise ValueError(f"Postprocessing validation changed: {path}")
    return {**actual, **additions}, amendment


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
    return {**freeze, "effective_source_files": source_files, "postprocessing_amendment": amendment}
