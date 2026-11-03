"""Reject runtime changes unless their recovery/equivalence evidence is intact."""
from copy import deepcopy
import hashlib
import subprocess

import pytest

from finance import provenance
from finance.io import read_json, sha, stable_hash, write_json


@pytest.fixture
def execution_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "FREEZE_PATH", tmp_path / "delivery/rv30_freeze.json")
    names = ("finance/models.py", "finance/provenance.py")
    originals = {name: ("before:" + name).encode() for name in names}
    prior = {name: hashlib.sha256(value).hexdigest() for name, value in originals.items()}
    for name in (*names, "finance/cuda_graph.py", "scripts/start_rv30_checked.py",
            "scripts/analyze_rv30_termination.py", "tests/test_rv30_acceleration.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("installed:" + name)
    freeze = {"signature": "research", "payload": {"source_files": prior}}
    write_json(provenance.FREEZE_PATH, freeze)
    statistics = tmp_path / "delivery/rv30_statistics_lock.json"
    write_json(statistics, {"signature": "unchanged_statistics"})
    summary = {"passed": True, "all_shape_cases": 192, "fit_resume_cases": 6,
        "sustained": {"passed": True, "resumed_stages": [1, 5, 15, 30]},
        "candidate_sha256": sha(tmp_path / "finance/cuda_graph.py")}
    write_json(tmp_path / "evidence/summary.json", summary)
    integrated = {"passed": True, "cross_implementation_resume_bitwise_equal": True,
        "source_files": {name: sha(tmp_path/name) for name in ("finance/models.py", "finance/cuda_graph.py")}}
    write_json(tmp_path / "evidence/installed.json", integrated)
    payload = {"base_signature": "research", "base_freeze_sha256": sha(provenance.FREEZE_PATH),
        "prior_statistics_lock_sha256": sha(statistics), "prior_source_snapshot_commit": "preserved",
        "research_definitions_changed": False, "model_search_space_changed": False, "thresholds_changed": False,
        "bitwise_equivalence_required": True,
        "changes": {name: {"before_sha256": prior[name], "after_sha256": sha(tmp_path/name)} for name in names},
        "added_sources": {"finance/cuda_graph.py": sha(tmp_path/"finance/cuda_graph.py")},
        "support_sources": {name: sha(tmp_path/name) for name in ("scripts/start_rv30_checked.py",
            "scripts/analyze_rv30_termination.py", "tests/test_rv30_acceleration.py")},
        "validation_files": {name: sha(tmp_path/name) for name in ("evidence/summary.json", "evidence/installed.json")},
        "validation_summary": "evidence/summary.json", "integrated_validation": "evidence/installed.json"}
    lock = tmp_path / "delivery/rv30_acceleration_lock.json"
    write_json(lock, {"signature": stable_hash(payload), "payload": payload})
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: originals[args[-1].split(":",1)[1]])
    def predecessor(frozen, source_view=None, excluded_additions=()):
        assert frozen is freeze
        assert source_view == prior
        assert set(excluded_additions) == {"finance/cuda_graph.py"}
        return prior, {"signature": "unchanged_statistics"}
    monkeypatch.setattr(provenance, "_postprocessing_sources", predecessor)
    return tmp_path, freeze, payload, lock


def resign(lock, payload):
    write_json(lock, {"signature": stable_hash(payload), "payload": payload})


def test_execution_lock_keeps_predecessor_statistics_and_adds_exact_sources(execution_lock):
    root, freeze, payload, lock = execution_lock
    effective, previous = provenance.effective_sources(freeze)
    assert previous["signature"] == "unchanged_statistics"
    assert effective["finance/models.py"] == sha(root/"finance/models.py")
    assert effective["finance/cuda_graph.py"] == sha(root/"finance/cuda_graph.py")


@pytest.mark.parametrize("target", ["finance/models.py", "finance/cuda_graph.py", "evidence/summary.json",
    "delivery/rv30_statistics_lock.json", "scripts/start_rv30_checked.py"])
def test_execution_lock_rejects_modified_source_evidence_or_predecessor(execution_lock, target):
    root, freeze, payload, lock = execution_lock
    with (root/target).open("a") as stream:
        stream.write("tamper")
    with pytest.raises(ValueError):
        provenance.effective_sources(freeze)


@pytest.mark.parametrize("field", ["research_definitions_changed", "model_search_space_changed", "thresholds_changed"])
def test_execution_lock_cannot_redefine_research(execution_lock, field):
    root, freeze, payload, lock = execution_lock
    payload[field] = True
    resign(lock, payload)
    with pytest.raises(ValueError, match="cannot change research"):
        provenance.effective_sources(freeze)


def test_execution_lock_rejects_failed_or_wrong_source_recovery(execution_lock):
    root, freeze, payload, lock = execution_lock
    integrated = read_json(root/"evidence/installed.json")
    integrated["cross_implementation_resume_bitwise_equal"] = False
    write_json(root/"evidence/installed.json", integrated)
    payload["validation_files"]["evidence/installed.json"] = sha(root/"evidence/installed.json")
    resign(lock, payload)
    with pytest.raises(ValueError, match="recovery validation"):
        provenance.effective_sources(freeze)
