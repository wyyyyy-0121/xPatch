"""Test mechanism review rules using only isolated fictional metadata."""
from copy import deepcopy
import hashlib
import json

import pytest

from .shared_gate import _core_state, expected_payload, mechanism_review_issues


def _file(path, value, quote=None):
    path.write_text(value, encoding="utf8")
    result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    if quote is not None:
        result["grant_quote"] = quote
    return result


def _fixture(tmp_path, kind="synthetic_test"):
    admission_hash, base_hash, base_review_hash = "a" * 64, "b" * 64, "c" * 64
    state_hash = None if kind == "synthetic_test" else "d" * 64
    reviewer = {"reviewed_by": "fictional automated test reviewer",
                "reviewed_at_utc": "2026-01-01T00:00:00+00:00"}
    base = {"data_kind": kind, "training": {"seed": 2026, "max_epochs": 1, "patience": 1,
                                           "batch_size": 16, "learning_rate": 0.001}}
    base_review = {"compute": {"host": "fictional-host", "device": "cuda", "device_name": "fictional-gpu",
                               "location": "local", "benchmark_evidence": {"sha256": "e" * 64}}}
    projected = 120
    projection = {**reviewer, "status": "synthetic_mechanism_projection_only" if kind == "synthetic_test" else
                  "reviewed_mechanism_projection", "data_kind": kind, "admission_sha256": admission_hash,
                  "base_protocol_sha256": base_hash, "benchmark_sha256": "e" * 64,
                  "seed_count": 3, "fold_count": 5, "models_per_fold": 5,
                  "Q3_values_read": False, "includes_io_scoring_and_verification": True,
                  "projected_wall_seconds": projected}
    evidence = _file(tmp_path / f"{kind}-projection.json", json.dumps(projection))
    review = {**reviewer, "schema_version": 1, "status": "reviewed_synthetic_test" if kind == "synthetic_test" else
              "reviewed_and_approved", "data_kind": kind, "admission_sha256": admission_hash,
              "base_protocol_sha256": base_hash, "base_review_sha256": base_review_hash,
              "base_matrix_state_sha256": state_hash, "development_only": True, "independent_test": False,
              "Q3_sealed": True, "selection_before_mechanism_scoring": True,
              "prior_core_exposure_acknowledged": True, "synthetic_fixture_only": kind == "synthetic_test",
              "seeds": [2026, 2027, 2028],
              "compute": {"host": "fictional-host", "device": "cuda", "device_name": "fictional-gpu",
                          "location": "local", "maximum_wall_seconds": 180,
                          "projected_wall_seconds": projected, "projection_evidence": evidence}}
    args = {"admission_hash": admission_hash, "base_protocol": base, "base_protocol_hash": base_hash,
            "base_review": base_review, "base_review_hash": base_review_hash,
            "base_state_hash": state_hash, "evidence_root": tmp_path}
    return review, args


def test_synthetic_mechanism_review_requires_three_seeds_and_bounded_projection(tmp_path):
    review, args = _fixture(tmp_path)
    assert mechanism_review_issues(review, **args) == []
    changed = deepcopy(review); changed["seeds"] = [2026, 2027]
    assert "three_distinct_predeclared_seeds_required" in mechanism_review_issues(changed, **args)
    changed = deepcopy(review); changed["seeds"] = [2026, 2026, 2028]
    assert "three_distinct_predeclared_seeds_required" in mechanism_review_issues(changed, **args)
    changed = deepcopy(review); changed["compute"]["maximum_wall_seconds"] = 6 * 3600 + 1
    assert "mechanism_projection_or_budget_invalid" in mechanism_review_issues(changed, **args)
    changed = deepcopy(review); changed["compute"]["projected_wall_seconds"] = 181
    assert "mechanism_projection_or_budget_invalid" in mechanism_review_issues(changed, **args)
    changed = deepcopy(review); changed["synthetic_fixture_only"] = False
    assert "mechanism_review_identity_or_exposure_invalid" in mechanism_review_issues(changed, **args)
    (tmp_path / "synthetic_test-projection.json").write_text("changed", encoding="utf8")
    assert "mechanism_projection_or_budget_invalid" in mechanism_review_issues(review, **args)


def test_market_metadata_requires_separate_written_authorization(tmp_path):
    # These are fictional metadata, not a market-data review or authorization.
    review, args = _fixture(tmp_path, "market")
    assert "separate_user_compute_authorization_missing" in mechanism_review_issues(review, **args)
    quote = "SYNTHETIC FIXTURE ONLY: a fictional user approves this bounded example."
    review["compute"]["authorization_evidence"] = _file(tmp_path / "fictional-user.txt", quote, quote)
    review["compute"]["explicit_user_authorization"] = True
    assert mechanism_review_issues(review, **args) == []
    changed = deepcopy(review); changed["prior_core_exposure_acknowledged"] = False
    assert "mechanism_review_identity_or_exposure_invalid" in mechanism_review_issues(changed, **args)


def test_market_mechanism_rejects_partial_core_before_reading_results():
    with pytest.raises(ValueError, match="completed full core matrix"):
        _core_state({"matrix_scope": "partial", "selected_profiles": ["minute"]}, "a" * 64,
                    "b" * 64, "c" * 64, None, deep=False)


def test_same_authorization_reuses_budget_identity(tmp_path):
    review, args = _fixture(tmp_path, "market")
    quote = "SYNTHETIC FIXTURE ONLY: a fictional user approves this bounded example."
    review["compute"]["authorization_evidence"] = _file(tmp_path / "fictional-user.txt", quote, quote)
    review["compute"]["explicit_user_authorization"] = True
    base = args["base_protocol"]
    base["jobs"] = [{"profile": "minute", "horizon": 60, "target": "future_log_return", "fold": number}
                    for number in range(1, 6)]
    context = {"base_protocol": base, "mechanism_review": review, "data_kind": "market",
               "admission_hash": args["admission_hash"], "base_protocol_hash": args["base_protocol_hash"],
               "base_review_hash": args["base_review_hash"], "base_state_hash": args["base_state_hash"],
               "mechanism_review_hash": "f" * 64}
    first = expected_payload(context)
    context["mechanism_review_hash"] = "1" * 64
    second = expected_payload(context)
    assert first["budget_identity"] == second["budget_identity"] == review["compute"]["authorization_evidence"]["sha256"]
    assert len(first["jobs"]) == 15 and len(first["seeds"]) == 3
