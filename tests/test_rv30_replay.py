"""Fresh inference must restore fitting precision; a failed round stays intact."""
import hashlib
import json
import os
import subprocess
import sys

import pytest

from finance import provenance, verify
from finance.io import read_json, sha, stable_hash, write_json


def test_load_checkpoint_restores_fitting_precision_before_model_construction(tmp_path):
    cpus = sorted(os.sched_getaffinity(0))[:2]
    code = """
import json,sys
from pathlib import Path
from finance.runtime import configure,configure_torch
configure(2,json.loads(sys.argv[2]));configure_torch()
import torch
from finance import models
path=Path(sys.argv[1]);torch.save({'seed':2026,'channels':1,'state_dict':{}},path/'model.pt')
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.benchmark=True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
class Model:
 def load_state_dict(self,values): assert values=={}
 def eval(self): return self
def construct(*args):
 assert torch.are_deterministic_algorithms_enabled()
 assert not torch.backends.cudnn.benchmark
 assert not torch.backends.cuda.matmul.allow_tf32
 assert not torch.backends.cudnn.allow_tf32
 return Model()
models.native_xpatch=construct
assert isinstance(models.load_model('xpatch',path,{}, {},'cpu'),Model)
print('passed')
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path), json.dumps(cpus)], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "passed"


@pytest.fixture
def replay_lock(monkeypatch, tmp_path):
    for module in (provenance, verify):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "FREEZE_PATH", tmp_path / "delivery/rv30_freeze.json")
    names = ("finance/models.py", "finance/verify.py", "finance/provenance.py")
    originals = {name: ("previous:" + name).encode() for name in names}
    before = {name: hashlib.sha256(value).hexdigest() for name, value in originals.items()}
    for name in (*names, "tests/test_rv30_replay.py"):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("installed:" + name)
    frozen = {"signature": "research", "payload": {"source_files": before}}
    write_json(provenance.FREEZE_PATH, frozen)
    write_json(tmp_path / "delivery/rv30_throughput_lock.json", {"signature": "preserved_throughput"})
    directory = tmp_path / "artifacts/rv30/research/development"
    original = directory / "verify_1"
    write_json(original / "summary.json", {"passed": False})
    write_json(original / "failed.json", {"status": "failed", "error": "original precision mismatch"})
    index = {"directory": str(original.relative_to(tmp_path)),
        "files": {str(p.relative_to(tmp_path)): sha(p) for p in original.rglob("*") if p.is_file()},
        "failed_receipts": [f"failed/{i}" for i in range(42)]}
    write_json(tmp_path / "evidence/failed_attempt_index.json", index)
    validation = {"passed": True, "models": 60, "all_full_predictions_bitwise_equal": True,
        "outer_scores_read": False, "source_files": {name: sha(tmp_path/name) for name in names[:2]}}
    write_json(tmp_path / "evidence/full_replay.json", validation)
    payload = {"base_signature": "research", "base_freeze_sha256": sha(provenance.FREEZE_PATH),
        "prior_throughput_lock_sha256": sha(tmp_path / "delivery/rv30_throughput_lock.json"),
        "prior_source_snapshot_commit": "preserved", "research_definitions_changed": False,
        "model_search_space_changed": False, "thresholds_changed": False, "training_changed": False,
        "original_predictions_preserved": True,
        "changes": {name: {"before_sha256": before[name], "after_sha256": sha(tmp_path/name)} for name in names},
        "support_sources": {"tests/test_rv30_replay.py": sha(tmp_path / "tests/test_rv30_replay.py")},
        "validation_summary": "evidence/full_replay.json",
        "validation_files": {name: sha(tmp_path/name) for name in ("evidence/full_replay.json", "evidence/failed_attempt_index.json")},
        "verification_retry": {"scope": "development", "round": 1, "run_directory": str(directory.relative_to(tmp_path)),
            "original_directory": str(original.relative_to(tmp_path)),
            "replacement_directory": str((directory/'verify_1_replay_correction').relative_to(tmp_path)),
            "original_summary_sha256": sha(original/'summary.json'), "failed_attempt_index": "evidence/failed_attempt_index.json"}}
    lock = tmp_path / "delivery/rv30_replay_lock.json"
    write_json(lock, {"signature": stable_hash(payload), "payload": payload})
    monkeypatch.setattr(subprocess, "check_output", lambda args, **kw: originals[args[-1].split(":",1)[1]])
    def previous(freeze, source_view=None, excluded_additions=()):
        assert freeze is frozen and source_view == before and not excluded_additions
        return before, {"signature": "preserved_statistics"}
    monkeypatch.setattr(provenance, "_throughput_sources", previous)
    return tmp_path, frozen, payload, lock, directory


def test_retry_preserves_failure_and_uses_distinct_job_identity(replay_lock):
    root, frozen, payload, lock, directory = replay_lock
    actual, prior = provenance.effective_sources(frozen)
    assert actual["finance/models.py"] == sha(root/"finance/models.py")
    assert prior["signature"] == "preserved_statistics"
    assert verify.verification_directory(directory, 1, "development") == directory/"verify_1_replay_correction"
    assert verify.verification_directory(directory, 2, "development") == directory/"verify_2"
    assert verify.verification_directory(directory, 1, "confirmation") == directory/"confirmation/verify_1"
    assert verify.verification_directory(directory, 1, "delivery") == directory/"delivery_verification/verify_1"
    assert verify.verification_directory(root/"another", 1, "development") == root/"another/verify_1"
    assert not read_json(directory/"verify_1/summary.json")["passed"]
    source = {**{k: k for k in ("signature", "model", "panel", "config_id", "params", "seed", "partition")}, "job_id": "outer/source"}
    write_json(root/"source/result.json", {})
    request = verify.verification_requests(directory, 1, "development", {"source":source})[0]
    assert request["job_id"].startswith("verify1_replay_correction/")
    assert request["directory"].startswith(str((directory/"verify_1_replay_correction").relative_to(root)))


@pytest.mark.parametrize("target", ["finance/models.py", "finance/verify.py", "evidence/full_replay.json",
    "delivery/rv30_throughput_lock.json", "artifacts/rv30/research/development/verify_1/failed.json"])
def test_replay_lock_rejects_changed_source_proof_or_failed_attempt(replay_lock, target):
    root, frozen, payload, lock, directory = replay_lock
    with (root/target).open("a") as stream: stream.write("tamper")
    with pytest.raises(ValueError): provenance.effective_sources(frozen)


@pytest.mark.parametrize("field,value", [("training_changed", True), ("thresholds_changed", True),
    ("model_search_space_changed", True), ("original_predictions_preserved", False)])
def test_replay_correction_cannot_change_research(replay_lock, field, value):
    root, frozen, payload, lock, directory = replay_lock
    payload[field] = value
    write_json(lock, {"signature":stable_hash(payload),"payload":payload})
    with pytest.raises(ValueError, match="cannot revise"): provenance.effective_sources(frozen)


def test_replay_requires_full_saved_predictions(replay_lock):
    root, frozen, payload, lock, directory = replay_lock
    validation = read_json(root/payload["validation_summary"])
    validation["models"] = 59
    write_json(root/payload["validation_summary"], validation)
    payload["validation_files"][payload["validation_summary"]] = sha(root/payload["validation_summary"])
    write_json(lock, {"signature":stable_hash(payload),"payload":payload})
    with pytest.raises(ValueError, match="Every saved xPatch"): provenance.effective_sources(frozen)


@pytest.mark.parametrize("evidence", ["evidence/full_replay.json", "evidence/failed_attempt_index.json"])
def test_replay_cannot_leave_validation_evidence_unsigned(replay_lock, evidence):
    root, frozen, payload, lock, directory = replay_lock
    del payload["validation_files"][evidence]
    write_json(lock, {"signature":stable_hash(payload),"payload":payload})
    with pytest.raises(ValueError, match="must bind"): provenance.effective_sources(frozen)


def test_failed_summary_cannot_be_rewritten_to_success(replay_lock):
    root, frozen, payload, lock, directory = replay_lock
    write_json(directory/"verify_1/summary.json", {"passed":True})
    with pytest.raises(ValueError, match="preserve"): provenance.effective_sources(frozen)
    with pytest.raises(ValueError, match="preserved"): verify.verification_directory(directory,1,"development")
