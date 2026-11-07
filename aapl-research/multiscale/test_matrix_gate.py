"""Exercise evidence binding and fail-closed matrix scope on synthetic records."""
from copy import deepcopy
import hashlib
import json

from .matrix_gate import (BENCHMARK_SHAPES, REQUIRED_SOURCE_FILES, _source_io_evidence,
                          locked_protocol_issues, review_issues, task_inventory)
from .spec import DEV_END, MODELS, PROFILES, ROOT, TARGETS
from .test_matrix_plan import synthetic_admission


def evidence(path, text, quote=None):
    path.write_text(text, encoding="utf8")
    record = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if quote is not None:
        record["grant_quote"] = quote
    return record


def test_review_requires_bound_written_evidence_and_resource_measurement(tmp_path):
    admission = synthetic_admission()
    admission["data_kind"] = "synthetic_test"
    admission["legacy_test_exposure"] = {"sha256": "c" * 64}
    admission_hash, note_hash = "a" * 64, "d" * 64
    assert "training_and_publication_rights_not_reviewed" in review_issues(
        admission, admission_hash, {}, tmp_path, note_hash)
    reviewer = {"reviewed_by": "synthetic reviewer", "reviewed_at_utc": "2026-01-01T00:00:00+00:00"}
    selected = [profile.name for profile in PROFILES]
    quote = "Synthetic provider explicitly permits this synthetic test and aggregate publication."
    daily = evidence(tmp_path / "daily.txt", quote, quote)
    minute = evidence(tmp_path / "minute.txt", quote, quote)
    records = [{"profile": profile.name, "model": model, "lookback": BENCHMARK_SHAPES[profile.name][0],
                "horizon": BENCHMARK_SHAPES[profile.name][1], "fit_origins": BENCHMARK_SHAPES[profile.name][2],
                "stop_origins": BENCHMARK_SHAPES[profile.name][3],
                "validation_origins": BENCHMARK_SHAPES[profile.name][4], "epochs": 1, "steps": 1}
               for profile in PROFILES for model in ("mlp", "gru", "lstm", "xpatch")]
    benchmark = {"status": "completed_engineering_only", "small_smoke_mode": False,
                 "all_shapes_measured": True, "market_data_read": False,
                 "formal_experiment": False, "elapsed_seconds": 901, "records": records, "device": "synthetic-gpu"}
    bench = evidence(tmp_path / "synthetic-benchmark.json", json.dumps(benchmark))
    io = evidence(tmp_path / "synthetic-io.txt", "synthetic test I/O evidence")
    projection = evidence(tmp_path / "synthetic-projection.txt", "synthetic test projection")
    review = {"data_kind": "synthetic_test", "admission_sha256": admission_hash, "selected_profiles": selected,
              "rights": {**reviewer, "decision": "permitted", "training_allowed": True,
                         "publication_of_nonreconstructable_aggregate_results_allowed": True,
                         "admission_sha256": admission_hash, "selected_profiles": selected,
                         "daily_evidence": daily, "minute_evidence": minute},
              "quality": {**reviewer, "decision": "approved", "admission_sha256": admission_hash,
                          "selected_profiles": selected,
                          "nominal_price_and_split_chain_checked": True, "material_discrepancies_disposition_fixed": True,
                          "review_notes": "Synthetic test only; the fictional data and splits were inspected."},
              "exposure": {**reviewer, "decision": "reviewed_for_development_only", "admission_sha256": admission_hash,
                           "independent_test": False, "Q3_sealed": True, "legacy_ledger_sha256": "c" * 64,
                           "post_cutoff_note_sha256": note_hash},
              "compute": {**reviewer, "decision": "approved", "admission_sha256": admission_hash,
                          "device": "cuda", "device_name": "synthetic-gpu", "location": "local", "host": "synthetic-host",
                          "maximum_wall_seconds": 2000, "projected_wall_seconds": 1500,
                          "real_source_io_reviewed": True, "benchmark_evidence": bench,
                          "real_source_io_evidence": io, "projection_evidence": projection}}
    assert review_issues(admission, admission_hash, review, tmp_path, note_hash) == []
    selected_partial = ["hourly", "minute"]
    partial_review = deepcopy(review)
    partial_review["selected_profiles"] = selected_partial
    partial_review["rights"]["selected_profiles"] = selected_partial
    partial_review["quality"]["selected_profiles"] = selected_partial
    partial_review["rights"]["daily_evidence"] = {}
    assert review_issues(admission, admission_hash, partial_review, tmp_path, note_hash,
                         selected_profiles=selected_partial) == []
    market_admission, market_review = deepcopy(admission), deepcopy(review)
    market_admission["data_kind"] = market_review["data_kind"] = "market"
    assert "representative_benchmark_or_resource_budget_incomplete" in review_issues(
        market_admission, admission_hash, market_review, tmp_path, note_hash)
    (tmp_path / "daily.txt").write_text("different synthetic text", encoding="utf8")
    assert "daily_rights_evidence_missing_or_changed" in review_issues(admission, admission_hash, review, tmp_path, note_hash)
    changed = deepcopy(review)
    changed["compute"]["projected_wall_seconds"] = 2001
    assert "representative_benchmark_or_resource_budget_incomplete" in review_issues(
        admission, admission_hash, changed, tmp_path, note_hash)
    changed = deepcopy(review)
    changed["compute"]["maximum_wall_seconds"] = 6 * 3600 + 1
    assert "expanded_compute_resource_authorization_missing" in review_issues(
        admission, admission_hash, changed, tmp_path, note_hash)


def test_locked_protocol_requires_exact_tasks_sources_and_eligible_folds():
    admission = synthetic_admission()
    jobs = []
    for profile in PROFILES:
        for horizon in profile.horizons:
            for target in TARGETS:
                for fold in range(1, 6):
                    jobs.append({"profile": profile.name, "frequency": profile.frequency, "horizon": horizon,
                                 "target": target, "fold": fold, "lookback": max(profile.lookbacks),
                                 "origin_lookback": max(profile.lookbacks),
                                 "seasonal_period": {"minute": 390, "hour": 6, "day": 252, "month": 12}[profile.frequency],
                                 "fold_strategy": "warmup",
                                 "warmup": {"min_inner_fit": 24, "min_inner_stop": 4, "min_validation_groups": 6},
                                 "fit_origins": 24,
                                 "stop_origins": 4, "validation_origins": 6,
                                 **{key: "f" * 64 for key in ("fit_indices_sha256", "stop_indices_sha256",
                                                                "validation_indices_sha256", "train_indices_sha256")}})
    sources = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in REQUIRED_SOURCE_FILES}
    protocol = {"schema_version": 1, "status": "frozen_for_historical_development", "data_kind": "market", "frozen": True,
                "independent_test": False, "Q3_values_read": False, "development_cutoff_new_york": DEV_END,
                "admission_sha256": "a" * 64, "review_sha256": "b" * 64, "recipe_sha256": "d" * 64, "models": list(MODELS),
                "selected_profiles": [p.name for p in PROFILES], "omitted_profiles": {}, "matrix_scope": "full",
                "task_inventory": task_inventory([p.name for p in PROFILES], {}),
                "jobs": jobs, "source_code_sha256": sources,
                "analysis": {"block_lengths_by_profile": {"annual": [12, 24], "quarterly": [3, 6],
                                                            "monthly": [21, 42], "hourly": [2, 5], "minute": [1, 5]}, "resamples": 100,
                             "seed": 2026, "confidence": .95, "development_p_values": False,
                             "annual_nonoverlap": True},
                "training": {"max_epochs": 20, "patience": 3, "batch_size": 128,
                             "seed": 2026, "learning_rate": .001, "ridge_alphas": [.1, 1.0, 10.0]}}
    assert locked_protocol_issues(admission, "a" * 64, protocol, "b" * 64) == []
    partial = deepcopy(protocol)
    partial["selected_profiles"] = ["hourly", "minute"]
    partial["omitted_profiles"] = {name: "Synthetic test source is not yet admitted." for name in ("annual", "quarterly", "monthly")}
    partial["matrix_scope"] = "partial"
    partial["task_inventory"] = task_inventory(partial["selected_profiles"], partial["omitted_profiles"])
    partial["jobs"] = [job for job in jobs if job["profile"] in partial["selected_profiles"]]
    assert len(partial["jobs"]) == 70
    assert locked_protocol_issues(admission, "a" * 64, partial, "b" * 64) == []
    partial["matrix_scope"] = "full"
    assert "frozen_matrix_scope_invalid" in locked_protocol_issues(admission, "a" * 64, partial, "b" * 64)
    changed = deepcopy(protocol); changed["jobs"].pop()
    assert "frozen_selected_tasks_missing_or_duplicate_folds" in locked_protocol_issues(
        admission, "a" * 64, changed, "b" * 64)
    changed = deepcopy(protocol); changed["Q3_values_read"] = True; changed["jobs"][0]["fit_origins"] = 7
    issues = locked_protocol_issues(admission, "a" * 64, changed, "b" * 64)
    assert "frozen_protocol_identity_or_scope_invalid" in issues and "frozen_job_insufficient_origins" in issues
    changed = deepcopy(protocol); changed["source_code_sha256"][REQUIRED_SOURCE_FILES[0]] = "0" * 64
    assert "frozen_source_code_identity_changed" in locked_protocol_issues(admission, "a" * 64, changed, "b" * 64)


def test_io_evidence_is_bound_to_source_host_and_selected_scales(tmp_path):
    from .source_io_benchmark import SOURCE_FILES
    source_hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    report = {"scope": "fictional test fixture", "status": "candidate_read_only_io_not_admitted",
              "admission_sha256": "a" * 64, "host": "synthetic-host", "frequencies": ["hour", "minute"],
              "Q3_values_read": False, "model_trained": False, "experiment_scored": False,
              "source_sha256_at_start": source_hashes, "source_sha256_at_end": source_hashes,
              "records": [{"frequency": "hour"}, {"frequency": "minute"}]}
    item = evidence(tmp_path / "synthetic-io.json", json.dumps(report))
    assert _source_io_evidence(item, tmp_path, "a" * 64, ["hourly", "minute"], "synthetic-host")
    assert not _source_io_evidence(item, tmp_path, "b" * 64, ["hourly", "minute"], "synthetic-host")
    assert not _source_io_evidence(item, tmp_path, "a" * 64, ["monthly", "minute"], "synthetic-host")
    assert not _source_io_evidence(item, tmp_path, "a" * 64, ["hourly", "minute"], "another-host")
