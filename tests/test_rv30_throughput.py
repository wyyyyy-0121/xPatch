"""Protect the chained execution amendment and bounded optional GPU data copy."""
import hashlib
import subprocess

import pytest

from finance import provenance
from finance.io import read_json, sha, stable_hash, write_json


@pytest.fixture
def throughput_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "FREEZE_PATH", tmp_path / "delivery/rv30_freeze.json")
    changes = ("finance/models.py", "finance/provenance.py")
    originals = {name: ("previous:" + name).encode() for name in changes}
    before = {name: hashlib.sha256(value).hexdigest() for name, value in originals.items()}
    for name in (*changes, "finance/cuda_graph.py", "finance/throughput.py", "tests/test_rv30_throughput.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("installed:" + name)
    freeze = {"signature": "research", "payload": {"source_files": before}}
    write_json(provenance.FREEZE_PATH, freeze)
    prior_lock = tmp_path / "delivery/rv30_acceleration_lock.json"
    write_json(prior_lock, {"signature": "preserved_acceleration"})
    summary = {"passed": True, "all_config_cases": 192, "seed_resume_cases": 6,
        "stages": [{"to_epoch": stage} for stage in (1, 5, 15, 30)],
        "maximum_30_epoch_checkpoint_bitwise_equal": True,
        "maximum_full_predictions_bitwise_equal": True, "outer_scores_read": False}
    integrated = {"passed": True, "cross_implementation_resume_bitwise_equal": True,
        "resident_fallback_bitwise_equal": True,
        "source_files": {name: sha(tmp_path / name) for name in
            ("finance/models.py", "finance/cuda_graph.py", "finance/throughput.py")}}
    write_json(tmp_path / "evidence/summary.json", summary)
    write_json(tmp_path / "evidence/installed.json", integrated)
    payload = {"base_signature": "research", "base_freeze_sha256": sha(provenance.FREEZE_PATH),
        "prior_acceleration_lock_sha256": sha(prior_lock), "prior_source_snapshot_commit": "preserved",
        "research_definitions_changed": False, "model_search_space_changed": False,
        "thresholds_changed": False, "bitwise_equivalence_required": True,
        "changes": {name: {"before_sha256": before[name], "after_sha256": sha(tmp_path / name)}
            for name in changes},
        "added_sources": {"finance/throughput.py": sha(tmp_path / "finance/throughput.py")},
        "support_sources": {"tests/test_rv30_throughput.py": sha(tmp_path / "tests/test_rv30_throughput.py")},
        "validation_summary": "evidence/summary.json", "integrated_validation": "evidence/installed.json",
        "validation_files": {name: sha(tmp_path / name) for name in ("evidence/summary.json", "evidence/installed.json")}}
    lock = tmp_path / "delivery/rv30_throughput_lock.json"
    write_json(lock, {"signature": stable_hash(payload), "payload": payload})
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: originals[args[-1].split(":", 1)[1]])
    def previous(frozen, source_view=None, excluded_additions=()):
        assert frozen is freeze
        assert source_view == before
        assert set(excluded_additions) == {"finance/throughput.py"}
        return before, {"signature": "preserved_statistics"}
    monkeypatch.setattr(provenance, "_acceleration_sources", previous)
    return tmp_path, freeze, payload, lock


def resign(lock, payload):
    write_json(lock, {"signature": stable_hash(payload), "payload": payload})


def test_throughput_verifies_predecessor_source_view(throughput_lock):
    root, freeze, payload, lock = throughput_lock
    effective, previous = provenance.effective_sources(freeze)
    assert previous["signature"] == "preserved_statistics"
    assert effective["finance/models.py"] == sha(root / "finance/models.py")
    assert effective["finance/throughput.py"] == sha(root / "finance/throughput.py")


@pytest.mark.parametrize("target", ["finance/models.py", "finance/throughput.py",
    "tests/test_rv30_throughput.py", "evidence/summary.json", "delivery/rv30_acceleration_lock.json"])
def test_throughput_rejects_changed_evidence_or_predecessor(throughput_lock, target):
    root, freeze, payload, lock = throughput_lock
    with (root / target).open("a") as stream:
        stream.write("tamper")
    with pytest.raises(ValueError):
        provenance.effective_sources(freeze)


@pytest.mark.parametrize("field", ["research_definitions_changed", "model_search_space_changed", "thresholds_changed"])
def test_throughput_cannot_change_research(throughput_lock, field):
    root, freeze, payload, lock = throughput_lock
    payload[field] = True
    resign(lock, payload)
    with pytest.raises(ValueError, match="cannot change research"):
        provenance.effective_sources(freeze)


@pytest.mark.parametrize("field,bad_value", [("passed", False), ("all_config_cases", 191),
    ("seed_resume_cases", 5), ("stages", [{"to_epoch": 1}]),
    ("maximum_30_epoch_checkpoint_bitwise_equal", False),
    ("maximum_full_predictions_bitwise_equal", False), ("outer_scores_read", True)])
def test_throughput_requires_full_validation(throughput_lock, field, bad_value):
    root, freeze, payload, lock = throughput_lock
    summary = read_json(root / payload["validation_summary"])
    summary[field] = bad_value
    write_json(root / payload["validation_summary"], summary)
    payload["validation_files"][payload["validation_summary"]] = sha(root / payload["validation_summary"])
    resign(lock, payload)
    with pytest.raises(ValueError, match="sustained validation"):
        provenance.effective_sources(freeze)


@pytest.mark.parametrize("field", ["cross_implementation_resume_bitwise_equal", "resident_fallback_bitwise_equal", "source_files"])
def test_throughput_requires_installed_recovery(throughput_lock, field):
    root, freeze, payload, lock = throughput_lock
    integrated = read_json(root / payload["integrated_validation"])
    integrated[field] = {} if field == "source_files" else False
    write_json(root / payload["integrated_validation"], integrated)
    payload["validation_files"][payload["integrated_validation"]] = sha(root / payload["integrated_validation"])
    resign(lock, payload)
    with pytest.raises(ValueError, match="recovery validation"):
        provenance.effective_sources(freeze)


def test_throughput_cannot_leave_summary_unsigned(throughput_lock):
    root, freeze, payload, lock = throughput_lock
    del payload["validation_files"][payload["validation_summary"]]
    resign(lock, payload)
    with pytest.raises(ValueError, match="bind both"):
        provenance.effective_sources(freeze)


@pytest.mark.parametrize("required,free,allowed", [(0, 4, False), (1, 2, True),
    (1, 1, False), (2, 3, False)])
def test_resident_admission_caps_copy_and_keeps_reserve(required, free, allowed):
    from finance.throughput import resident_admitted
    gib = 1024**3
    assert resident_admitted(required * gib, free * gib) is allowed
