"""Check evidence and a frozen protocol before a development matrix can run.

This is a technical consistency gate, not an automatic legal determination.
It does not open price arrays, start training, or grant data-use rights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .bars import file_digest
from .spec import DEV_END, MODELS, OUT, PROFILES, ROOT, TARGETS


REQUIRED_SOURCE_FILES = (
    "multiscale/development.py", "multiscale/bars.py", "multiscale/baselines.py",
    "multiscale/evaluation.py", "multiscale/features.py", "multiscale/models.py",
    "multiscale/ridge.py", "multiscale/splits.py", "multiscale/spec.py",
    "multiscale/training.py", "multiscale/mechanism.py", "multiscale/matrix_gate.py",
    "multiscale/matrix_executor.py", "multiscale/verify_development.py", "multiscale/aggregate_development.py",
    "multiscale/source_io_benchmark.py",
    "aapl/models.py", "aapl/vendor/xpatch_network.py",
)
PERIODS = {"minute": 390, "hour": 6, "day": 252, "month": 12}
BENCHMARK_SHAPES = {
    "minute": (480, 60, 320000, 56000, 75000),
    "hourly": (96, 8, 5700, 1000, 1300),
    "monthly": (252, 21, 1600, 280, 390),
    "quarterly": (36, 3, 50, 10, 10),
    "annual": (60, 12, 24, 4, 6),
}
MINIMUM_BLOCK_GROUPS = {"annual": 12, "quarterly": 3, "monthly": 21, "hourly": 2, "minute": 1}


def profile_selection(value):
    """Validate a predeclared, ordered group of complete study scales."""
    order = [profile.name for profile in PROFILES]
    if (not isinstance(value, list) or not value or any(not isinstance(name, str) for name in value) or
            len(set(value)) != len(value) or any(name not in order for name in value) or
            value != [name for name in order if name in value]):
        raise ValueError("Profiles must be a nonempty unique selection in study order")
    return value


def task_inventory(selected, omitted):
    return [{"profile": p.name, "horizon": horizon, "target": target,
             "status": "included" if p.name in selected else "omitted_before_scoring",
             "reason": None if p.name in selected else omitted[p.name]}
            for p in PROFILES for horizon in p.horizons for target in TARGETS]


def _digest(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value))


def _reviewer(value):
    if not isinstance(value, dict):
        return False
    person, at = value.get("reviewed_by"), value.get("reviewed_at_utc")
    if not isinstance(person, str) or not person.strip() or not isinstance(at, str):
        return False
    try:
        instant = datetime.fromisoformat(at.replace("Z", "+00:00"))
        return instant.tzinfo is not None
    except ValueError:
        return False


def _evidence(item, workspace: Path, *, require_quote: bool):
    """Verify an explicit local evidence artifact; do not infer legal meaning."""
    if not isinstance(item, dict) or not _digest(item.get("sha256")):
        return False
    name = item.get("path")
    if not isinstance(name, str) or not name:
        return False
    path = Path(name).resolve()
    root = workspace.resolve()
    try:
        if not path.is_relative_to(root) or not path.is_file() or file_digest(path) != item["sha256"]:
            return False
    except OSError:
        return False
    if require_quote:
        quote = item.get("grant_quote")
        if not isinstance(quote, str) or len(quote.strip()) < 25:
            return False
        try:
            body = path.read_text(encoding="utf8")
        except (OSError, UnicodeError):
            return False
        if " ".join(quote.split()) not in " ".join(body.split()):
            return False
    return True


def _representative_benchmark(item, workspace: Path, expected_device: str, *, require_runtime_source=True):
    if not _evidence(item, workspace, require_quote=False):
        return False
    try:
        report = json.loads(Path(item["path"]).read_text(encoding="utf8"))
        elapsed = report.get("elapsed_seconds")
        if (report.get("status") != "completed_engineering_only" or report.get("device") != expected_device or
                report.get("small_smoke_mode") is not False or
                report.get("all_shapes_measured") is not True or report.get("market_data_read") is not False or
                report.get("formal_experiment") is not False or isinstance(elapsed, bool) or
                not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 900):
            return False
        if require_runtime_source:
            from .benchmark_matrix_synthetic import BENCHMARK_SOURCE_FILES
            sources = report.get("source_sha256_at_start")
            if (not isinstance(sources, dict) or set(sources) != set(BENCHMARK_SOURCE_FILES) or
                    sources != report.get("source_sha256_at_end") or
                    any(not _digest(digest) or file_digest(ROOT / name) != digest for name, digest in sources.items())):
                return False
        expected = {(profile.name, model) for profile in PROFILES for model in ("mlp", "gru", "lstm", "xpatch")}
        measured = set()
        for row in report.get("records", []):
            if not isinstance(row, dict) or row.get("profile") not in BENCHMARK_SHAPES:
                continue
            lookback, horizon, fit, stop, validation = BENCHMARK_SHAPES[row["profile"]]
            if (row.get("lookback") == lookback and row.get("horizon") == horizon and
                    type(row.get("fit_origins")) is int and row["fit_origins"] >= fit and
                    type(row.get("stop_origins")) is int and row["stop_origins"] >= stop and
                    type(row.get("validation_origins")) is int and row["validation_origins"] >= validation and
                    type(row.get("epochs")) is int and row["epochs"] >= 1 and
                    type(row.get("steps")) is int and row["steps"] >= 1):
                measured.add((row["profile"], row.get("model")))
        return measured == expected
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def _source_io_evidence(item, workspace: Path, admission_hash: str, selected, host: str, *, synthetic=False):
    if not _evidence(item, workspace, require_quote=False):
        return False
    if synthetic:
        return True  # Only the separately labeled synthetic self-test uses this fixture.
    try:
        from .source_io_benchmark import SOURCE_FILES
        report = json.loads(Path(item["path"]).read_text(encoding="utf8"))
        sources = report.get("source_sha256_at_start")
        needed = {p.frequency for p in PROFILES if p.name in selected}
        if (report.get("status") != "candidate_read_only_io_not_admitted" or
                report.get("admission_sha256") != admission_hash or report.get("host") != host or
                report.get("Q3_values_read") is not False or report.get("model_trained") is not False or
                report.get("experiment_scored") is not False or
                not isinstance(report.get("frequencies"), list) or not needed.issubset(set(report["frequencies"])) or
                not isinstance(sources, dict) or set(sources) != set(SOURCE_FILES) or
                sources != report.get("source_sha256_at_end") or
                any(not _digest(digest) or file_digest(ROOT / name) != digest for name, digest in sources.items())):
            return False
        measured = {row.get("frequency") for row in report.get("records", []) if isinstance(row, dict)}
        return needed.issubset(measured)
    except (OSError, UnicodeError, ValueError, TypeError):
        return False


def review_issues(admission, admission_sha256, review, evidence_root: Path, exposure_note_sha256,
                  *, selected_profiles=None):
    issues = []
    try:
        selected = profile_selection(selected_profiles if selected_profiles is not None else [p.name for p in PROFILES])
    except ValueError:
        selected = []
        issues.append("predeclared_profile_selection_invalid")
    if (not _digest(admission_sha256) or admission.get("Q3_values_read") is not False or
            admission.get("experiment_scored") is not False or
            admission.get("development_cutoff_new_york") != DEV_END):
        issues.append("candidate_admission_identity_or_cutoff_invalid")
    expected_tasks = {(p.name, h, t) for p in PROFILES for h in p.horizons for t in TARGETS}
    profiles = {profile.name: profile for profile in PROFILES}
    try:
        task_rows = admission.get("tasks", [])
        task_ids = [(row["profile"], row["horizon"], row["target"]) for row in task_rows]
        if (len(task_ids) != len(expected_tasks) or set(task_ids) != expected_tasks or
                any(row.get("frequency") != profiles[row["profile"]].frequency for row in task_rows)):
            issues.append("candidate_task_scope_invalid")
    except (KeyError, TypeError, AttributeError):
        issues.append("candidate_task_scope_invalid")
    review = review if isinstance(review, dict) else {}
    if review.get("admission_sha256") != admission_sha256:
        issues.append("review_not_bound_to_candidate_admission")
    if review.get("data_kind", "market") != admission.get("data_kind", "market"):
        issues.append("review_data_kind_mismatch")
    if review.get("selected_profiles") != selected:
        issues.append("review_not_bound_to_selected_profiles")
    rights = review.get("rights") if isinstance(review.get("rights"), dict) else {}
    if (not _reviewer(rights) or rights.get("decision") != "permitted" or
            rights.get("training_allowed") is not True or
            rights.get("publication_of_nonreconstructable_aggregate_results_allowed") is not True or
            rights.get("admission_sha256") != admission_sha256 or rights.get("selected_profiles") != selected):
        issues.append("training_and_publication_rights_not_reviewed")
    required_sources = set()
    if any(name in selected for name in ("annual", "quarterly", "monthly")):
        required_sources.add("daily")
    if any(name in selected for name in ("hourly", "minute")):
        required_sources.add("minute")
    for name in sorted(required_sources):
        if not _evidence(rights.get(name + "_evidence"), evidence_root, require_quote=True):
            issues.append(name + "_rights_evidence_missing_or_changed")
    quality = review.get("quality") if isinstance(review.get("quality"), dict) else {}
    if (not _reviewer(quality) or quality.get("decision") != "approved" or
            quality.get("admission_sha256") != admission_sha256 or
            quality.get("selected_profiles") != selected or
            quality.get("nominal_price_and_split_chain_checked") is not True or
            quality.get("material_discrepancies_disposition_fixed") is not True or
            not isinstance(quality.get("review_notes"), str) or len(quality["review_notes"].strip()) < 40):
        issues.append("source_price_and_corporate_action_review_incomplete")
    exposure = review.get("exposure") if isinstance(review.get("exposure"), dict) else {}
    ledger_record = admission.get("legacy_test_exposure")
    ledger = ledger_record.get("sha256") if isinstance(ledger_record, dict) else None
    if (not _reviewer(exposure) or exposure.get("decision") != "reviewed_for_development_only" or
            exposure.get("admission_sha256") != admission_sha256 or
            exposure.get("independent_test") is not False or exposure.get("Q3_sealed") is not True or
            not _digest(ledger) or exposure.get("legacy_ledger_sha256") != ledger or
            not _digest(exposure_note_sha256) or exposure.get("post_cutoff_note_sha256") != exposure_note_sha256):
        issues.append("legacy_and_post_cutoff_exposure_review_incomplete")
    budget = review.get("compute") if isinstance(review.get("compute"), dict) else {}
    maximum, projected = budget.get("maximum_wall_seconds"), budget.get("projected_wall_seconds")
    def positive_number(number):
        return not isinstance(number, bool) and isinstance(number, (int, float)) and math.isfinite(number) and number > 0
    if (not _reviewer(budget) or budget.get("decision") != "approved" or
            budget.get("admission_sha256") != admission_sha256 or budget.get("device") != "cuda" or
            budget.get("location") not in ("local", "explicitly_authorized_cloud") or
            not isinstance(budget.get("host"), str) or not budget.get("host", "").strip() or
            not isinstance(budget.get("device_name"), str) or not budget.get("device_name", "").strip() or
            not positive_number(maximum) or not positive_number(projected) or projected > maximum or
            budget.get("real_source_io_reviewed") is not True or
            not _representative_benchmark(budget.get("benchmark_evidence"), evidence_root, budget.get("device_name"),
                                          require_runtime_source=admission.get("data_kind", "market") != "synthetic_test") or
            not _source_io_evidence(budget.get("real_source_io_evidence"), evidence_root, admission_sha256,
                                    selected, budget.get("host"),
                                    synthetic=admission.get("data_kind", "market") == "synthetic_test") or
            not _evidence(budget.get("projection_evidence"), evidence_root, require_quote=False)):
        issues.append("representative_benchmark_or_resource_budget_incomplete")
    needs_authorization = (budget.get("location") == "explicitly_authorized_cloud" or
                           (budget.get("location") == "local" and positive_number(maximum) and maximum > 6 * 3600))
    if needs_authorization and (budget.get("explicit_user_authorization") is not True or
            not isinstance(budget.get("authorization_note"), str) or len(budget["authorization_note"].strip()) < 40):
        issues.append("expanded_compute_resource_authorization_missing")
    return issues


def locked_protocol_issues(admission, admission_sha256, protocol, review_sha256):
    issues = []
    if not isinstance(protocol, dict):
        return ["frozen_matrix_protocol_missing"]
    data_kind = admission.get("data_kind", "market")
    expected_status = "frozen_for_synthetic_test" if data_kind == "synthetic_test" else "frozen_for_historical_development"
    if (data_kind not in ("market", "synthetic_test") or protocol.get("data_kind") != data_kind or
            protocol.get("schema_version") != 1 or protocol.get("status") != expected_status or
            protocol.get("frozen") is not True or protocol.get("independent_test") is not False or
            protocol.get("Q3_values_read") is not False or protocol.get("development_cutoff_new_york") != DEV_END or
            protocol.get("admission_sha256") != admission_sha256 or protocol.get("review_sha256") != review_sha256 or
            not _digest(review_sha256) or not _digest(protocol.get("recipe_sha256")) or protocol.get("models") != list(MODELS)):
        issues.append("frozen_protocol_identity_or_scope_invalid")
    try:
        selected = profile_selection(protocol.get("selected_profiles"))
    except ValueError:
        selected = []
        issues.append("frozen_profile_selection_invalid")
    omitted = protocol.get("omitted_profiles")
    expected_omitted = {p.name for p in PROFILES if p.name not in selected}
    if (not isinstance(omitted, dict) or set(omitted) != expected_omitted or
            any(not isinstance(note, str) or len(note.strip()) < 20 for note in omitted.values())):
        issues.append("frozen_omitted_profile_reasons_missing")
    elif protocol.get("task_inventory") != task_inventory(selected, omitted):
        issues.append("frozen_full_task_inventory_invalid")
    scope = "full" if len(selected) == len(PROFILES) else "partial"
    if protocol.get("matrix_scope") != scope:
        issues.append("frozen_matrix_scope_invalid")
    expected = {(p.name, h, t, fold) for p in PROFILES if p.name in selected
                for h in p.horizons for t in TARGETS for fold in range(1, 6)}
    profiles = {profile.name: profile for profile in PROFILES}
    rows = {(row["profile"], row["horizon"], row["target"]): row for row in admission.get("tasks", [])}
    jobs = protocol.get("jobs")
    if not isinstance(jobs, list):
        return issues + ["frozen_matrix_jobs_missing"]
    identities, group_settings, paired_folds = [], {}, {}
    for job in jobs:
        if not isinstance(job, dict):
            issues.append("invalid_frozen_job")
            continue
        if (not isinstance(job.get("profile"), str) or not isinstance(job.get("target"), str) or
                type(job.get("horizon")) is not int or type(job.get("fold")) is not int):
            issues.append("invalid_frozen_job")
            continue
        key = (job["profile"], job["horizon"], job["target"], job["fold"])
        identities.append(key)
        if key not in expected:
            issues.append("unexpected_frozen_job")
            continue
        profile = profiles[key[0]]
        row = rows.get(key[:3], {})
        if (row.get("minimum_history_met") is not True or job.get("frequency") != profile.frequency or
                job.get("lookback") != max(profile.lookbacks) or job.get("origin_lookback") != max(profile.lookbacks) or
                job.get("seasonal_period") != PERIODS[profile.frequency]):
            issues.append("frozen_job_history_or_shape_ineligible")
        warmup = job.get("warmup")
        if (job.get("fold_strategy") not in ("rolling", "warmup") or not isinstance(warmup, dict) or
                set(warmup) != {"min_inner_fit", "min_inner_stop", "min_validation_groups"} or
                any(type(v) is not int or v < 1 for v in warmup.values())):
            issues.append("frozen_job_fold_strategy_invalid")
        else:
            settings = (job["lookback"], job["seasonal_period"], job["fold_strategy"],
                        tuple(sorted(warmup.items())))
            task_key = key[:2]
            if task_key in group_settings and group_settings[task_key] != settings:
                issues.append("frozen_job_settings_differ_across_folds_or_targets")
            group_settings[task_key] = settings
        counts = [job.get(k) for k in ("fit_origins", "stop_origins", "validation_origins")]
        if (any(isinstance(n, bool) or not isinstance(n, int) for n in counts) or
                counts[0] < 8 or counts[1] < 1 or counts[2] < 1):
            issues.append("frozen_job_insufficient_origins")
        if any(not _digest(job.get(key)) for key in ("fit_indices_sha256", "stop_indices_sha256",
                                                    "validation_indices_sha256", "train_indices_sha256")):
            issues.append("frozen_job_split_identity_missing")
        pair_key = (job["profile"], job["horizon"], job["fold"])
        split_signature = tuple(job.get(name) for name in ("fit_origins", "stop_origins", "validation_origins",
                                                             "fit_indices_sha256", "stop_indices_sha256",
                                                             "validation_indices_sha256", "train_indices_sha256"))
        if pair_key in paired_folds and paired_folds[pair_key] != split_signature:
            issues.append("frozen_targets_use_different_chronological_splits")
        paired_folds[pair_key] = split_signature
    if len(identities) != len(expected) or len(set(identities)) != len(expected) or set(identities) != expected:
        issues.append("frozen_selected_tasks_missing_or_duplicate_folds")
    training = protocol.get("training")
    if not isinstance(training, dict):
        issues.append("frozen_training_configuration_missing")
    else:
        integers = (training.get("max_epochs"), training.get("patience"), training.get("batch_size"), training.get("seed"))
        lr, alphas = training.get("learning_rate"), training.get("ridge_alphas")
        if (any(type(n) is not int or n < 1 for n in integers) or integers[2] < 2 or
                isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0 or
                not isinstance(alphas, list) or not alphas or
                any(isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a) or a <= 0 for a in alphas) or
                len(set(alphas)) != len(alphas)):
            issues.append("frozen_training_configuration_invalid")
    sources = protocol.get("source_code_sha256")
    if not isinstance(sources, dict) or set(sources) != set(REQUIRED_SOURCE_FILES):
        issues.append("frozen_source_code_identity_missing")
    else:
        try:
            if any(not _digest(digest) or file_digest(ROOT / name) != digest for name, digest in sources.items()):
                issues.append("frozen_source_code_identity_changed")
        except OSError:
            issues.append("frozen_source_code_identity_changed")
    analysis = protocol.get("analysis")
    if not isinstance(analysis, dict):
        issues.append("frozen_analysis_plan_missing")
    else:
        blocks, resamples, seed, confidence = (analysis.get("block_lengths_by_profile"), analysis.get("resamples"),
                                               analysis.get("seed"), analysis.get("confidence"))
        if (not isinstance(blocks, dict) or set(blocks) != set(profiles) or
                any(not isinstance(values, list) or not values or
                    any(type(value) is not int or value < 1 for value in values) or
                    values != sorted(set(values)) for values in blocks.values()) or
                type(resamples) is not int or resamples < 100 or type(seed) is not int or seed < 0 or
                isinstance(confidence, bool) or not isinstance(confidence, (float, int)) or
                not math.isfinite(confidence) or not 0 < confidence < 1 or
                analysis.get("development_p_values") is not False or analysis.get("annual_nonoverlap") is not True):
            issues.append("frozen_analysis_plan_invalid")
        elif any(values[0] < MINIMUM_BLOCK_GROUPS[name] for name, values in blocks.items()):
            issues.append("frozen_analysis_blocks_shorter_than_declared_horizon")
    return sorted(set(issues))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    admission_path = (args.admission.resolve() / "admission.json").resolve()
    if not admission_path.is_relative_to((OUT / "admission").resolve()):
        raise ValueError("Admission is outside the isolated workspace")
    admission = json.loads(admission_path.read_text(encoding="utf8"))
    admission_hash = file_digest(admission_path)
    def optional(path):
        if path is None:
            return {}, None
        actual = path.resolve()
        if not actual.is_relative_to(OUT.resolve()):
            raise ValueError("Review and protocol must be in the isolated workspace")
        return json.loads(actual.read_text(encoding="utf8")), file_digest(actual)
    review, review_hash = optional(args.review)
    protocol, _ = optional(args.protocol)
    note = OUT / "POST_CUTOFF_EXPOSURE.md"
    note_hash = file_digest(note) if note.is_file() else None
    selected = protocol.get("selected_profiles") if protocol else None
    reasons = review_issues(admission, admission_hash, review, OUT, note_hash, selected_profiles=selected)
    reasons.extend(locked_protocol_issues(admission, admission_hash, protocol, review_hash))
    result = {"status": "blocked" if reasons else "technical_checks_passed_not_automatic_legal_clearance",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "admission_sha256": admission_hash,
              "review_sha256": review_hash, "blocking_reasons": sorted(set(reasons)),
              "market_price_arrays_opened": False, "experiment_scored": False,
              "frozen_split_hashes_recomputed_from_bars": False,
              "limitations": ["Evidence quotations and reviewer attestations are consistency checks, not legal advice or automatic interpretation of terms.",
                              "Passing this gate does not establish an independent test or the correctness of the data.",
                              "This command does not execute a matrix or authorize cloud rental."]}
    directory = OUT / "gate_reports"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    print(json.dumps({"status": result["status"], "blocking_reasons": result["blocking_reasons"], "report": str(path)}))


if __name__ == "__main__":
    main()
