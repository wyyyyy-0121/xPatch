"""Bind an existing exploration fold to current permission and source evidence.

This preserves the original exploration identity. It neither reclassifies a
run as a prospectively admitted matrix nor authorizes further computation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .bars import file_digest
from .spec import OUT


IDENTITY_PREFIX = "private_exploration_not_publication_admitted:"
IDENTITY_PREFIXES = (IDENTITY_PREFIX, "private-minute-breadth:",
                     "private-minute-parallel:", "private-minute-matrix:")
REVIEW_STATUS = "historical_development_packaging_record"
PROVENANCE_FIELDS = {
    "admission_sha256", "protocol_sha256", "coverage_sha256",
    "permission_sha256", "source_manifest_sha256", "verification_sha256",
    "packaging_record_sha256",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def read_record(path, expected=None):
    path = Path(path).resolve()
    require(path.is_relative_to(OUT.resolve()) and path.is_file(),
            "Historical packaging evidence is outside the output tree or missing")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(expected is None or (sha256(expected) and digest == expected),
            "Historical packaging evidence changed: " + path.name)
    return json.loads(raw), digest


def validate_provenance(value, *, data_identity, protocol_identity):
    require(isinstance(value, dict) and set(value) == PROVENANCE_FIELDS | {
                "scope", "permission_basis", "permission_status"} and
            all(sha256(value.get(key)) for key in PROVENANCE_FIELDS) and
            value.get("scope") == "previously_exposed_historical_exploration" and
            value.get("permission_basis") == "direct_user_confirmation" and
            value.get("permission_status") == "permitted_as_confirmed_by_user" and
            data_identity in {prefix + value["admission_sha256"] for prefix in IDENTITY_PREFIXES} and
            protocol_identity == value["protocol_sha256"],
            "Historical bundle provenance or original identity differs")


def check_provenance(fold_directory, receipt, receipt_sha256, admission_dir,
                     review_path, protocol_path):
    """Check immutable historical evidence without opening market arrays."""
    require(admission_dir is not None and review_path is not None and protocol_path is not None,
            "Historical exploration bundles require admission, packaging record and original protocol")
    admission_dir = Path(admission_dir).resolve()
    admission, admission_hash = read_record(admission_dir / "admission.json")
    require(admission.get("Q3_values_read") is False,
            "Admission does not preserve the protected cutoff")
    review, review_hash = read_record(review_path)
    require(review.get("schema_version") == 1 and review.get("status") == REVIEW_STATUS and
            review.get("independent_test") is False and review.get("Q3_values_read") is False,
            "Expected an explicitly historical packaging record")
    evidence = {}
    for label in ("coverage", "permission", "source_manifest"):
        item = review.get(label, {})
        require(isinstance(item, dict) and set(item) == {"path", "sha256"},
                "Missing historical packaging reference: " + label)
        evidence[label] = read_record(item["path"], item["sha256"])
    coverage, coverage_hash = evidence["coverage"]
    permission, permission_hash = evidence["permission"]
    source, source_hash = evidence["source_manifest"]
    protocol, protocol_hash = read_record(protocol_path)
    require(receipt["data_identity"] in {prefix + admission_hash for prefix in IDENTITY_PREFIXES} and
            receipt["config"]["protocol_identity"] == protocol_hash and
            protocol.get("admission_sha256") == admission_hash and
            protocol.get("Q3_values_read") is False and protocol.get("independent_test") is False and
            protocol.get("data_cutoff") == "2026-06-30",
            "Original exploration protocol, source or cutoff differs")
    require(permission.get("status") == "permitted_as_confirmed_by_user" and
            permission.get("evidence_type") == "direct_user_confirmation" and
            "Alpaca" in permission.get("providers", []) and
            permission.get("training_allowed_as_confirmed") is True and
            permission.get("publication_of_results_and_figures_allowed_as_confirmed") is True and
            isinstance(permission.get("user_reply_verbatim"), str) and
            bool(permission["user_reply_verbatim"].strip()),
            "Current user-confirmed permission is missing")
    require(source.get("status") == "historical_development_data_provenance_documented" and
            source.get("permission_status") == permission["status"] and
            source.get("permission_confirmation_sha256") == permission_hash and
            source.get("admission_metadata_sha256") == admission_hash and
            source.get("provider_used_in_reported_experiments") == "Alpaca SIP/raw" and
            source.get("Q3_values_read") is False and source.get("independent_test") is False,
            "Historical source manifest does not bind the source and permission")
    derived = source.get("derived_bars", {})
    require(Path(derived.get("directory", "")).resolve() == admission_dir,
            "Historical source manifest refers to another bar directory")
    # Hash the relevant archive, without requesting any of its price arrays.
    from .spec import PROFILES
    profile = next(p for p in PROFILES if p.name == receipt["config"]["profile"])
    filename = "bars_" + profile.frequency + ".npz"
    require(sha256(derived.get("sha256", {}).get(filename)) and
            file_digest(admission_dir / filename) == derived["sha256"][filename],
            "Historical derived-bar archive changed")
    from .historical_coverage import TASKS
    rows = coverage.get("folds", [])
    keys = [(r["profile"], r["horizon"], r["target"], r["fold"]) for r in rows]
    require(coverage.get("status") == "private_verified_coverage_snapshot" and
            len(keys) == len(set(keys)) and
            set(keys) == {(*task, fold) for task in TASKS for fold in range(1, 6)},
            "Historical packaging coverage inventory differs")
    folder = Path(fold_directory).resolve()
    matches = [r for r in rows if r.get("status") == "completed" and
               r.get("evidence", {}).get("receipt_sha256") == receipt_sha256 and
               Path(r["evidence"].get("fold_directory", "")).resolve() == folder]
    require(len(matches) == 1, "Fold lacks unique completed-coverage evidence")
    row, cfg = matches[0], receipt["config"]
    require(all(row[k] == cfg[k] for k in ("profile", "horizon", "target")) and
            row["evidence"]["validation_origins"] == receipt["validation_origins"] and
            row["evidence"]["validation_sha256"] == receipt["identity_fields"]["validation_indices_sha256"],
            "Historical packaging task or validation origins differ")
    verification, verification_hash = read_record(row["evidence"]["verification"],
                                                  row["evidence"]["verification_sha256"])
    sources = receipt["source_sha256"]
    require(verification.get("status") == "verified_saved_fold" and
            verification.get("fold_receipt_sha256") == receipt_sha256 and
            Path(verification.get("fold_directory", "")).resolve() == folder and
            verification.get("protocol_identity") == protocol_hash and
            verification.get("targets_sha256") == receipt["targets_sha256"] and
            verification.get("development_cutoff") == "2026-06-30" and
            verification.get("independent_test") is False and
            verification.get("current_source_matches_recorded") == {key: True for key in sources},
            "Historical fold verification differs from its saved receipt")
    result = {"scope": "previously_exposed_historical_exploration",
              "permission_basis": "direct_user_confirmation", "permission_status": permission["status"],
              "admission_sha256": admission_hash, "protocol_sha256": protocol_hash,
              "coverage_sha256": coverage_hash, "permission_sha256": permission_hash,
              "source_manifest_sha256": source_hash, "verification_sha256": verification_hash,
              "packaging_record_sha256": review_hash}
    validate_provenance(result, data_identity=receipt["data_identity"], protocol_identity=protocol_hash)
    return result
