"""Fictional metadata exercises historical provenance without training models."""
import json

import pytest

from . import historical_bundle_review as gate
from .historical_coverage import TASKS


@pytest.fixture
def fictional_evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "OUT", tmp_path)

    def save(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return path, gate.file_digest(path)

    admission, ah = save("admission/admission.json", {"Q3_values_read": False})
    bars = admission.parent / "bars_minute.npz"
    bars.write_bytes(b"FICTIONAL_ARCHIVE_IDENTITY_ONLY_NO_MARKET_VALUES")
    protocol, ph = save("protocol.json", {"admission_sha256": ah, "data_cutoff": "2026-06-30",
                                        "Q3_values_read": False, "independent_test": False})
    permission, permh = save("permission.json", {
        "status": "permitted_as_confirmed_by_user", "evidence_type": "direct_user_confirmation",
        "providers": ["Alpaca"], "training_allowed_as_confirmed": True,
        "publication_of_results_and_figures_allowed_as_confirmed": True,
        "user_reply_verbatim": "FICTIONAL UNIT TEST CONFIRMATION"})
    source, sh = save("source.json", {
        "status": "historical_development_data_provenance_documented",
        "permission_status": "permitted_as_confirmed_by_user",
        "permission_confirmation_sha256": permh, "admission_metadata_sha256": ah,
        "provider_used_in_reported_experiments": "Alpaca SIP/raw", "Q3_values_read": False,
        "independent_test": False, "derived_bars": {"directory": str(admission.parent),
            "sha256": {"bars_minute.npz": gate.file_digest(bars)}}})
    folder = tmp_path / "fold"
    folder.mkdir()
    receipt = {"data_identity": gate.IDENTITY_PREFIX + ah,
               "config": {"profile": "minute", "horizon": 5, "target": "future_close",
                          "protocol_identity": ph}, "validation_origins": 2,
               "identity_fields": {"validation_indices_sha256": "1" * 64},
               "targets_sha256": "2" * 64, "source_sha256": {"fictional.py": "3" * 64}}
    _, rh = save("fold/fold_receipt.json", receipt)
    verification, vh = save("verification.json", {
        "status": "verified_saved_fold", "fold_receipt_sha256": rh, "fold_directory": str(folder),
        "protocol_identity": ph, "targets_sha256": receipt["targets_sha256"],
        "development_cutoff": "2026-06-30", "independent_test": False,
        "current_source_matches_recorded": {"fictional.py": True}})
    rows = [{"profile": p, "horizon": h, "target": t, "fold": f, "status": "pending"}
            for p, h, t in sorted(TASKS) for f in range(1, 6)]
    row = next(r for r in rows if (r["profile"], r["horizon"], r["target"], r["fold"]) ==
               ("minute", 5, "future_close", 1))
    row.update(status="completed", evidence={"receipt_sha256": rh, "fold_directory": str(folder),
        "validation_origins": 2, "validation_sha256": "1" * 64,
        "verification": str(verification), "verification_sha256": vh})
    coverage, ch = save("coverage.json", {"status": "private_verified_coverage_snapshot", "folds": rows})
    review = {"schema_version": 1, "status": gate.REVIEW_STATUS,
              "independent_test": False, "Q3_values_read": False,
              "coverage": {"path": str(coverage), "sha256": ch},
              "permission": {"path": str(permission), "sha256": permh},
              "source_manifest": {"path": str(source), "sha256": sh}}
    review_path, _ = save("packaging.json", review)
    return {"folder": folder, "receipt": receipt, "rh": rh, "admission": admission.parent,
            "protocol": protocol, "review_path": review_path, "review": review,
            "permission": permission, "coverage": coverage, "bars": bars,
            "verification": verification, "save": save}


def check(example):
    return gate.check_provenance(example["folder"], example["receipt"], example["rh"],
        example["admission"], example["review_path"], example["protocol"])


def test_historical_provenance_preserves_original_exploration_identity(fictional_evidence):
    e = fictional_evidence
    original = e["receipt"]["data_identity"]
    result = check(e)
    assert result["scope"] == "previously_exposed_historical_exploration"
    assert result["permission_basis"] == "direct_user_confirmation"
    assert e["receipt"]["data_identity"] == original


@pytest.mark.parametrize("artifact", ["permission", "coverage", "verification", "bars"])
def test_changed_evidence_cannot_be_used_for_export(fictional_evidence, artifact):
    e = fictional_evidence
    e[artifact].write_bytes(e[artifact].read_bytes() + b" ")
    with pytest.raises(ValueError, match="changed"):
        check(e)


def test_unconfirmed_permission_is_rejected_even_after_rebinding_digest(fictional_evidence):
    e = fictional_evidence
    permission = json.loads(e["permission"].read_text())
    permission["publication_of_results_and_figures_allowed_as_confirmed"] = False
    path, digest = e["save"]("permission.json", permission)
    e["review"]["permission"] = {"path": str(path), "sha256": digest}
    e["save"]("packaging.json", e["review"])
    with pytest.raises(ValueError, match="permission is missing"):
        check(e)


def test_pending_fold_is_rejected_even_after_rebinding_digest(fictional_evidence):
    e = fictional_evidence
    coverage = json.loads(e["coverage"].read_text())
    next(row for row in coverage["folds"] if row["status"] == "completed")["status"] = "paused_night"
    path, digest = e["save"]("coverage.json", coverage)
    e["review"]["coverage"] = {"path": str(path), "sha256": digest}
    e["save"]("packaging.json", e["review"])
    with pytest.raises(ValueError, match="completed-coverage evidence"):
        check(e)


def test_exploration_provenance_cannot_claim_an_admitted_market_identity(fictional_evidence):
    result = check(fictional_evidence)
    with pytest.raises(ValueError, match="original identity differs"):
        gate.validate_provenance(result, data_identity="market:" + result["admission_sha256"],
                                 protocol_identity=result["protocol_sha256"])
