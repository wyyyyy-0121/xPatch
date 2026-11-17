"""Freeze a separately reviewed shared-horizon development comparison.

The gate checks evidence and identities; it cannot decide the legal meaning of
provider terms, create user authorization, rent hardware, or grant data rights.
Synthetic fixtures are isolated and cannot admit market data.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .bars import file_digest
from .matrix_executor import _job_key, _json_inside, load_reviewed
from .matrix_gate import (REQUIRED_SOURCE_FILES, _digest, _evidence, _reviewer,
                          locked_protocol_issues, profile_selection)
from .report_development import _training_record, stable_matrix
from .shared_development import SHARED_SOURCE_FILES
from .spec import DEV_END, OUT, PROFILES, ROOT
from .verify_development import check_fold


HORIZONS = (5, 15, 30, 60)
MECHANISM_SOURCE_FILES = tuple(sorted(set(REQUIRED_SOURCE_FILES) | set(SHARED_SOURCE_FILES) | {
    "multiscale/report_development.py", "multiscale/shared_gate.py", "multiscale/shared_executor.py",
    "multiscale/verify_shared_development.py",
}))


def _positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _projection_ok(item, root, review, base_review, admission_hash, base_protocol_hash, kind):
    if not _evidence(item, root, require_quote=False):
        return False
    try:
        projection = json.loads(Path(item["path"]).read_text(encoding="utf8"))
        status = "synthetic_mechanism_projection_only" if kind == "synthetic_test" else "reviewed_mechanism_projection"
        benchmark = base_review["compute"]["benchmark_evidence"]["sha256"]
        return (projection.get("status") == status and projection.get("data_kind") == kind and
                projection.get("admission_sha256") == admission_hash and
                projection.get("base_protocol_sha256") == base_protocol_hash and
                projection.get("benchmark_sha256") == benchmark and
                projection.get("seed_count") == 3 and projection.get("fold_count") == 5 and
                projection.get("models_per_fold") == 5 and projection.get("Q3_values_read") is False and
                projection.get("includes_io_scoring_and_verification") is True and
                projection.get("projected_wall_seconds") == review["compute"]["projected_wall_seconds"] and
                _reviewer(projection))
    except (OSError, UnicodeError, ValueError, KeyError, TypeError):
        return False


def mechanism_review_issues(review, *, admission_hash, base_protocol, base_protocol_hash, base_review,
                            base_review_hash, base_state_hash, evidence_root):
    """Check a human-reviewed, separately budgeted mechanism record."""
    issues = []
    review = review if isinstance(review, dict) else {}
    kind = base_protocol.get("data_kind")
    expected_status = "reviewed_synthetic_test" if kind == "synthetic_test" else "reviewed_and_approved"
    if (kind not in ("synthetic_test", "market") or review.get("schema_version") != 1 or
            review.get("status") != expected_status or review.get("data_kind") != kind or not _reviewer(review) or
            review.get("admission_sha256") != admission_hash or review.get("base_protocol_sha256") != base_protocol_hash or
            review.get("base_review_sha256") != base_review_hash or review.get("base_matrix_state_sha256") != base_state_hash or
            review.get("development_only") is not True or review.get("independent_test") is not False or
            review.get("Q3_sealed") is not True or review.get("selection_before_mechanism_scoring") is not True or
            review.get("prior_core_exposure_acknowledged") is not True or
            review.get("synthetic_fixture_only") is not (kind == "synthetic_test")):
        issues.append("mechanism_review_identity_or_exposure_invalid")
    if kind == "market" and not _digest(base_state_hash):
        issues.append("completed_core_state_missing")
    seeds = review.get("seeds")
    first_seed = base_protocol.get("training", {}).get("seed")
    if (not isinstance(seeds, list) or len(seeds) != 3 or
            any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds) or
            len(set(seeds)) != 3 or seeds[0] != first_seed):
        issues.append("three_distinct_predeclared_seeds_required")
    compute = review.get("compute") if isinstance(review.get("compute"), dict) else {}
    base_compute = base_review.get("compute", {})
    maximum, projected = compute.get("maximum_wall_seconds"), compute.get("projected_wall_seconds")
    if (not _positive(maximum) or maximum > 6 * 3600 or not _positive(projected) or projected > maximum or
            any(compute.get(key) != base_compute.get(key) for key in ("host", "device", "device_name", "location")) or
            compute.get("device") != "cuda" or
            not _projection_ok(compute.get("projection_evidence"), evidence_root, review, base_review,
                               admission_hash, base_protocol_hash, kind)):
        issues.append("mechanism_projection_or_budget_invalid")
    if kind == "market" and (compute.get("explicit_user_authorization") is not True or
            not _evidence(compute.get("authorization_evidence"), evidence_root, require_quote=True)):
        issues.append("separate_user_compute_authorization_missing")
    return sorted(set(issues))


def _core_state(base_protocol, base_protocol_hash, admission_hash, review_hash, run_dir, *, deep):
    if base_protocol.get("matrix_scope") != "full" or base_protocol.get("selected_profiles") != [p.name for p in PROFILES]:
        raise ValueError("Market mechanism work requires the completed full core matrix")
    expected = (OUT / "matrix_runs" / base_protocol_hash[:24]).resolve()
    if run_dir is None or Path(run_dir).resolve() != expected or not expected.is_dir():
        raise ValueError("Core matrix path does not match its frozen protocol")
    state_path = expected / "matrix_state.json"
    with stable_matrix(expected):
        state_hash = file_digest(state_path)
        state = json.loads(state_path.read_text(encoding="utf8"))
        jobs = base_protocol["jobs"]
        keys = {_job_key(job) for job in jobs}
        records = state.get("jobs")
        if (state.get("status") != "completed_development_only" or state.get("data_kind") != "market" or
                state.get("matrix_scope") != "full" or state.get("protocol_sha256") != base_protocol_hash or
                state.get("review_sha256") != review_hash or state.get("selected_profiles") != base_protocol["selected_profiles"] or
                state.get("active") is not None or not isinstance(records, dict) or set(records) != keys or
                any(not isinstance(record, dict) or record.get("status") != "completed" for record in records.values())):
            raise ValueError("The full core matrix has not completed")
        for job in jobs:
            record = records[_job_key(job)]
            relative = record.get("directory")
            if not isinstance(relative, str) or not relative:
                raise ValueError("Core fold directory is missing")
            folder = (expected / relative).resolve()
            if not folder.is_relative_to(expected):
                raise ValueError("Core fold directory escapes its run")
            receipt = folder / "fold_receipt.json"
            if not _digest(record.get("receipt_sha256")) or file_digest(receipt) != record["receipt_sha256"]:
                raise ValueError("Core fold receipt identity changed")
            if deep:
                _training_record(expected, job, record, base_protocol_hash, admission_hash, base_protocol)
                checked = check_fold(folder)
                if (checked["fold_receipt_sha256"] != record["receipt_sha256"] or
                        checked["protocol_identity"] != base_protocol_hash or
                        checked["data_identity"] != "market:" + admission_hash or
                        not all(checked["current_source_matches_recorded"].values())):
                    raise ValueError("Core fold independent verification failed")
        if file_digest(state_path) != state_hash:
            raise ValueError("Core state changed during review")
    return state_hash


def load_context(admission_dir: Path, base_review_path: Path, base_protocol_path: Path,
                 mechanism_review_path: Path, base_run_dir: Path | None, *, deep_core=False):
    base, base_hash, base_path = _json_inside(Path(base_protocol_path), OUT)
    selected = profile_selection(base.get("selected_profiles"))
    if "minute" not in selected:
        raise ValueError("The base protocol does not include the complete minute scale")
    admission, admission_hash, source, base_review, base_review_hash, base_review_path = load_reviewed(
        Path(admission_dir), Path(base_review_path), selected_profiles=selected)
    issues = locked_protocol_issues(admission, admission_hash, base, base_review_hash)
    if issues:
        raise ValueError("Base protocol blocked: " + ", ".join(issues))
    kind = admission.get("data_kind", "market")
    if kind == "market":
        state_hash = _core_state(base, base_hash, admission_hash, base_review_hash, base_run_dir, deep=False)
    elif kind == "synthetic_test":
        marker = source / "SYNTHETIC_ONLY.txt"
        if (base_run_dir is not None or not source.name.startswith("synthetic_test_") or
                admission.get("status") != "synthetic_test_only" or not marker.is_file() or
                "SYNTHETIC TEST ONLY" not in marker.read_text(encoding="utf8")):
            raise ValueError("Synthetic mechanism fixtures must remain separately labeled")
        state_hash = None
    else:
        raise ValueError("Unknown mechanism data kind")
    review, review_hash, review_path = _json_inside(Path(mechanism_review_path), OUT)
    issues = mechanism_review_issues(review, admission_hash=admission_hash, base_protocol=base,
                                     base_protocol_hash=base_hash, base_review=base_review,
                                     base_review_hash=base_review_hash, base_state_hash=state_hash, evidence_root=OUT)
    if issues:
        raise ValueError("Mechanism admission blocked: " + ", ".join(issues))
    if kind == "market" and deep_core:
        if _core_state(base, base_hash, admission_hash, base_review_hash, base_run_dir, deep=True) != state_hash:
            raise ValueError("Core matrix identity changed during mechanism review")
    return {"admission": admission, "admission_hash": admission_hash, "source": source,
            "base_protocol": base, "base_protocol_hash": base_hash, "base_protocol_path": base_path,
            "base_review": base_review, "base_review_hash": base_review_hash, "base_review_path": base_review_path,
            "mechanism_review": review, "mechanism_review_hash": review_hash, "mechanism_review_path": review_path,
            "base_state_hash": state_hash, "data_kind": kind}


def expected_payload(context):
    base, review = context["base_protocol"], context["mechanism_review"]
    folds = sorted((job for job in base["jobs"] if job["profile"] == "minute" and job["horizon"] == 60 and
                    job["target"] == "future_log_return"), key=lambda item: item["fold"])
    if [job["fold"] for job in folds] != list(range(1, 6)):
        raise ValueError("The base protocol does not contain five H60 return folds")
    jobs = [{"fold": job["fold"], "seed": seed, "base_job": job} for seed in review["seeds"] for job in folds]
    training = {name: base["training"][name] for name in ("max_epochs", "patience", "batch_size", "learning_rate")}
    kind = context["data_kind"]
    # Reusing the same market authorization must not create a fresh budget by
    # freezing an otherwise identical protocol under a new timestamp.
    budget_identity = (context["mechanism_review_hash"] if kind == "synthetic_test" else
                       review["compute"]["authorization_evidence"]["sha256"])
    return {"schema_version": 1, "status": "frozen_synthetic_mechanism_only" if kind == "synthetic_test" else
            "frozen_historical_development_mechanism", "data_kind": kind, "frozen": True,
            "independent_test": False, "Q3_values_read": False, "development_cutoff_new_york": DEV_END,
            "admission_sha256": context["admission_hash"], "base_protocol_sha256": context["base_protocol_hash"],
            "base_review_sha256": context["base_review_hash"], "base_matrix_state_sha256": context["base_state_hash"],
            "mechanism_review_sha256": context["mechanism_review_hash"], "budget_identity": budget_identity,
            "core_requirement": "synthetic_fixture_no_core_completion_claim" if kind == "synthetic_test" else
                                "full_core_matrix_verified_before_mechanism",
            "horizons": list(HORIZONS), "model": "xpatch", "target": "future_log_return",
            "seeds": review["seeds"], "training": training, "jobs": jobs,
            "source_code_sha256": {name: file_digest(ROOT / name) for name in MECHANISM_SOURCE_FILES}}


def freeze(admission_dir: Path, base_review_path: Path, base_protocol_path: Path,
           mechanism_review_path: Path, base_run_dir: Path | None = None):
    context = load_context(admission_dir, base_review_path, base_protocol_path, mechanism_review_path,
                           base_run_dir, deep_core=True)
    protocol = expected_payload(context)
    protocol["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    directory = (OUT / "synthetic_mechanism_validation" / "locked_protocols" if context["data_kind"] == "synthetic_test"
                 else OUT / "locked_mechanism_protocols")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    with target.open("x", encoding="utf8") as stream:
        json.dump(protocol, stream, indent=2, allow_nan=False)
    return target


def load_locked(admission_dir: Path, base_review_path: Path, base_protocol_path: Path,
                mechanism_review_path: Path, mechanism_protocol_path: Path, base_run_dir: Path | None = None,
                *, deep_core=False):
    context = load_context(admission_dir, base_review_path, base_protocol_path, mechanism_review_path,
                           base_run_dir, deep_core=deep_core)
    root = (OUT / "synthetic_mechanism_validation" / "locked_protocols" if context["data_kind"] == "synthetic_test"
            else OUT / "locked_mechanism_protocols")
    protocol, digest, path = _json_inside(Path(mechanism_protocol_path), root)
    expected = expected_payload(context)
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise ValueError("Frozen mechanism protocol, evidence, or source identity changed")
    return context, protocol, digest, path
