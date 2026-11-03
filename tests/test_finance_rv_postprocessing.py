"""Post-search corrections cannot bypass replay coverage or research fingerprints."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from finance import acceptance, confirmation, delivery, inference, provenance, verify
from finance.io import read_json, sha, stable_hash, write_json


@pytest.mark.parametrize("historical", [False, True])
def test_delivery_verifies_each_stage_before_its_dependents(monkeypatch, tmp_path, historical):
    events = []
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "evaluate", lambda *a: {"errors": {}, "complete_model_panel_results": 12})
    def replay(config, directory, number, scope):
        events.append(f"{scope}{number}")
        return {"passed": True}
    monkeypatch.setattr(delivery, "verify_run", replay)
    def confirm(*args):
        events.append("confirm")
        return {"passed_candidates": [], "job_directories": ["conditional"] if historical else []}
    monkeypatch.setattr(delivery, "confirm", confirm)
    monkeypatch.setattr(delivery, "report", lambda *a: events.append("report"))
    monkeypatch.setattr(delivery, "freeze_baseline", lambda *a: events.append("freeze"))
    monkeypatch.setattr(delivery, "infer", lambda *a: {"rv_mean": 1.})
    monkeypatch.setattr(delivery, "accept", lambda *a: events.append("accept") or {"passed": True})
    assert delivery.finish({}, tmp_path / "run")["passed"]
    assert events == ["development1", "development2", "confirm", *(["confirmation1", "confirmation2"] if historical else []),
        "report", "freeze", "delivery1", "delivery2", "accept"]


@pytest.mark.parametrize("failed_round", [1, 2])
def test_failed_development_replay_stops_confirmation(monkeypatch, tmp_path, failed_round):
    monkeypatch.setattr(delivery, "ROOT", tmp_path)
    monkeypatch.setattr(delivery, "evaluate", lambda *a: {"errors": {}, "complete_model_panel_results": 12})
    monkeypatch.setattr(delivery, "verify_run", lambda c, d, n, s: {"passed": n != failed_round})
    monkeypatch.setattr(delivery, "confirm", lambda *a: pytest.fail("confirmation ran before verification"))
    result = delivery.finish({}, tmp_path / "run")
    assert not result["passed"]
    assert result["stage"] == f"development_verification_{failed_round}_failed"


def test_direct_confirmation_requires_both_development_replays(monkeypatch, tmp_path):
    monkeypatch.setattr(confirmation, "ROOT", tmp_path)
    monkeypatch.setattr(confirmation, "verify_freeze", lambda c: {"signature": "test"})
    write_json(tmp_path / "evaluation.json", {"signature": "test", "errors": {},
        "complete_model_panel_results": 12, "passed_candidates": []})
    def missing(*a):
        raise ValueError("missing replay")
    monkeypatch.setattr(confirmation, "require_verifications", missing)
    monkeypatch.setattr(confirmation, "Scheduler", lambda *a: pytest.fail("confirmation scheduled"))
    with pytest.raises(ValueError, match="missing replay"):
        confirmation.confirm({}, tmp_path)
    assert not (tmp_path / "confirmation/evaluation.json").exists()


def test_final_baseline_cannot_bypass_conditional_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "verify_freeze", lambda c: {"signature": "test"})
    def missing(*a):
        raise ValueError("unverified confirmation")
    monkeypatch.setattr(confirmation, "confirmation_evidence", missing)
    monkeypatch.setattr(inference, "Scheduler", lambda *a: pytest.fail("baseline scheduled"))
    with pytest.raises(ValueError, match="unverified confirmation"):
        inference.freeze_baseline({}, tmp_path)


@pytest.fixture
def replay_run(monkeypatch, tmp_path):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    config = {"feature_sets": ["aapl"], "models": ["har", "xpatch"],
        "search": {"seeds": [2026, 2027, 2028], "stochastic_models": ["xpatch"]}}
    split = {"name": "2023Q1", "fit_dates": ["2022-01-03"], "calibration_dates": ["2022-12-01"],
        "evaluation_dates": ["2023-01-03"], "purge_before_calibration": ["2022-11-30"], "purge_before_evaluation": ["2022-12-30"]}
    search = {"models": {name: [{"id": name + "01", "params": {"alpha": 1.}}] for name in config["models"]}}
    directory = tmp_path / "run"
    write_json(directory / "setup.json", {"signature": "signed", "config": config, "search": search, "partitions": [split]})
    for name in config["models"]:
        group = directory / f"2023Q1_aapl_{name}"
        selection = {"config_id": name + "01", "params": {"alpha": 1.}, "fixed_training_length": 2}
        write_json(group / "selection.json", selection)
        paths = []
        for seed in config["search"]["seeds"] if name == "xpatch" else [2026]:
            path = group / "outer" / selection["config_id"] / "2023Q1" / f"s{seed}"
            relative = str(path.relative_to(tmp_path))
            request = {"signature": "signed", "job_id": relative.removeprefix("run/"), "directory": relative,
                "model": name, "panel": "aapl", "config_id": selection["config_id"], "params": selection["params"],
                "seed": seed, "phase": "outer", "partition": split}
            if name == "xpatch":
                request["fixed_epochs"] = 2
            write_json(path / "request.json", request)
            for filename in ("predictions.npz", "fit_state.json", "calibration.json"):
                (path / filename).write_text("fixture")
            files = {p.name: sha(p) for p in path.iterdir()}
            write_json(path / "result.json", {**request, "status": "complete", "request_hash": stable_hash(request),
                "files": files, "started_utc": "2026-11-03T01:00:00+00:00", "finished_utc": "2026-11-03T02:00:00+00:00"})
            paths.append(relative)
        write_json(group / "experiment.json", {"status": "complete", "outer_jobs": paths})
    return config, directory


def make_replay_round(config, directory, number):
    sources = verify.expected_sources(config, directory, "development")
    requests = verify.verification_requests(directory, number, "development", sources)
    for request in requests:
        path = verify.ROOT / request["directory"]
        write_json(path / "request.json", request)
        verification = {"passed": True, "source_directory": request["source_directory"],
            "source_result_sha256": request["source_result_sha256"], "verification_scope": "development", "round": number}
        write_json(path / "verification.json", verification)
        write_json(path / "result.json", {"status": "complete", "verification": verification,
            "request_hash": stable_hash(request), "started_utc": "2026-11-03T02:30:00+00:00",
            "finished_utc": "2026-11-03T02:31:00+00:00", "files": {p.name: sha(p) for p in path.iterdir() if p.name != "result.json"}})
    summary = {"signature": "signed", "scope": "development", "round": number, "passed": True, "jobs": len(requests),
        "completed_utc": "2026-11-03T02:31:00+00:00", "job_directories": [r["directory"] for r in requests],
        "sources": {r["source_directory"]: r["source_result_sha256"] for r in requests}}
    path = verify.verification_directory(directory, number, "development") / "summary.json"
    write_json(path, summary)
    return path, summary


def test_exact_source_coverage_rejects_same_count_duplicate_seed(replay_run):
    config, directory = replay_run
    assert len(verify.expected_sources(config, directory, "development")) == 4
    path = directory / "2023Q1_aapl_xpatch/experiment.json"
    value = read_json(path)
    value["outer_jobs"][1] = value["outer_jobs"][0]
    write_json(path, value)
    with pytest.raises(ValueError, match="omit or substitute"):
        verify.expected_sources(config, directory, "development")


@pytest.mark.parametrize("field", ["scope", "round", "sources", "job_directories", "completed_utc"])
def test_independent_summary_rejects_scope_round_digest_and_duplicate_coverage(replay_run, field):
    config, directory = replay_run
    path, summary = make_replay_round(config, directory, 1)
    assert verify.verification_evidence(config, directory, 1)["sha256"] == sha(path)
    if field == "sources":
        first = next(iter(summary[field]))
        summary[field][first] = "wrong-source-digest"
    elif field == "job_directories":
        summary[field][1] = summary[field][0]
    elif field == "round":
        summary[field] = 2
    else:
        summary[field] = "wrong"
    write_json(path, summary)
    with pytest.raises(ValueError, match="coverage or identity|completion time"):
        verify.verification_evidence(config, directory, 1)


def test_valid_replay_resume_preserves_prerequisite_summary_hash(replay_run, monkeypatch):
    config, directory = replay_run
    path, summary = make_replay_round(config, directory, 1)
    original = path.read_bytes()
    monkeypatch.setattr(provenance, "verify_freeze", lambda c: {"signature": "signed"})
    assert verify.verify_run(config, directory, 1) == summary
    assert path.read_bytes() == original


def test_source_result_change_invalidates_independent_evidence(replay_run):
    config, directory = replay_run
    make_replay_round(config, directory, 1)
    source = next(iter(verify.expected_sources(config, directory, "development")))
    path = verify.ROOT / source / "result.json"
    result = read_json(path)
    result["extra"] = "changed after replay"
    write_json(path, result)
    with pytest.raises(ValueError, match="coverage or identity"):
        verify.verification_evidence(config, directory, 1)


def test_artifact_index_links_every_receipt_and_rejects_changed_payload(replay_run, monkeypatch):
    config, directory = replay_run
    monkeypatch.setattr(acceptance, "ROOT", verify.ROOT)
    index = acceptance.artifact_index(directory, "signed")
    assert index["job_count"] == 4
    for relative, entry in index["jobs"].items():
        assert entry["sha256"] == sha(verify.ROOT / relative)
        assert "predictions.npz" in entry["files"]
    source = next(iter(verify.expected_sources(config, directory, "development")))
    (verify.ROOT / source / "predictions.npz").write_text("changed model predictions")
    with pytest.raises(ValueError, match="Job artifact changed"):
        acceptance.artifact_index(directory, "signed")


@pytest.mark.parametrize("field", ["origins", "calibration_origins", "dates", "timestamps"])
def test_replay_grid_rejects_partial_or_relabelled_samples(field):
    timestamps = pd.date_range("2022-01-03", periods=5, freq="min", tz="UTC")
    data = SimpleNamespace(timestamps=timestamps, dates=np.array(["2022-01-03"] * 5),
        origins=lambda dates, stride: np.array([0, 2, 4]) if dates == ["evaluation"] else np.array([1, 3]))
    config = {"sampling": {"outer_stride_minutes": 1, "inner_validation_stride_minutes": 5, "calibration_stride_minutes": 5}}
    request = {"phase": "delivery", "partition": {"evaluation_dates": ["evaluation"], "calibration_dates": ["calibration"]}}
    values = {"origins": np.array([0, 2, 4]), "calibration_origins": np.array([1, 3]),
        "dates": data.dates[[0, 2, 4]], "timestamps": timestamps[[0, 2, 4]].asi8}
    verify.validate_prediction_grid(data, config, request, values)
    values[field] = values[field][:-1]
    with pytest.raises(ValueError, match="registered prediction grid"):
        verify.validate_prediction_grid(data, config, request, values)


def test_postprocessing_amendment_never_allows_research_core_changes(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "POSTPROCESS_PATH", tmp_path / "lock.json")
    path = tmp_path / "finance/models.py"
    path.parent.mkdir()
    path.write_text("original")
    freeze = {"signature": "test", "payload": {"source_files": {"finance/models.py": sha(path)}}}
    path.write_text("modified")
    with pytest.raises(ValueError, match="Frozen research implementation changed"):
        provenance.effective_sources(freeze)


def test_postprocessing_amendment_binds_original_freeze_and_changed_files(monkeypatch, tmp_path):
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "FREEZE_PATH", tmp_path / "freeze.json")
    monkeypatch.setattr(provenance, "POSTPROCESS_PATH", tmp_path / "lock.json")
    path = tmp_path / "finance/delivery.py"
    path.parent.mkdir()
    path.write_text("original")
    before = sha(path)
    freeze = {"signature": "test", "payload": {"source_files": {"finance/delivery.py": before}}}
    write_json(provenance.FREEZE_PATH, freeze)
    path.write_text("corrected order")
    payload = {"base_signature": "test", "base_freeze_sha256": sha(provenance.FREEZE_PATH),
        "changes": {"finance/delivery.py": {"before_sha256": before, "after_sha256": sha(path)}},
        "research_definitions_changed": False, "model_search_space_changed": False, "thresholds_changed": False,
        "added_sources": {}, "validation_files": {}}
    write_json(provenance.POSTPROCESS_PATH, {"signature": stable_hash(payload), "payload": payload})
    effective, _ = provenance.effective_sources(freeze)
    assert effective["finance/delivery.py"] == sha(path)
    altered = deepcopy(payload)
    altered["thresholds_changed"] = True
    write_json(provenance.POSTPROCESS_PATH, {"signature": stable_hash(altered), "payload": altered})
    with pytest.raises(ValueError, match="cannot revise research"):
        provenance.effective_sources(freeze)
