"""Protect task completeness, chronological selection and registry routing."""
import json
from pathlib import Path

import pytest

from . import historical_bundle_review as review
from . import study_package as package
from .bars import file_digest


@pytest.fixture
def registry_case(tmp_path):
    folder = tmp_path / "package"
    folder.mkdir()
    record = folder / "PACKAGING_RECORD.json"
    record.write_text("{}")
    entries = [{"profile": p, "horizon": h, "target": t, "status": "unavailable",
                "completed_folds": [], "missing_folds": [1, 2, 3, 4, 5], "selected_fold": None}
               for p, h, t in sorted(package.TASK_KEYS)]
    value = {"schema_version": 1, "status": "partial_historical_development_study_package",
             "independent_test": False, "Q3_values_read": False, "tasks": entries,
             "planned_tasks": 20, "available_tasks": 0, "five_fold_tasks": 0,
             "core_comparison_complete": False, "package_source_sha256": file_digest(Path(package.__file__)),
             "packaging_record_sha256": file_digest(record)}

    def load():
        path = folder / "REGISTRY.json"
        path.write_text(json.dumps(value))
        return package.load_study_package(folder, expected_registry_sha256=file_digest(path))

    return folder, value, load


def test_unavailable_task_is_rejected_before_loading_any_model(registry_case):
    _, registry, load = registry_case
    predictor = load()
    e = registry["tasks"][0]
    with pytest.raises(ValueError, match="no completed saved fold"):
        predictor.predict(None, profile=e["profile"], horizon=e["horizon"], target=e["target"],
                          model="last_close", as_of_ns=0, history_identity="fictional")


def test_incomplete_registry_cannot_claim_the_full_comparison(registry_case):
    _, value, load = registry_case
    value["core_comparison_complete"] = True
    with pytest.raises(ValueError, match="completion claim"):
        load()


def test_duplicate_task_cannot_replace_a_missing_task(registry_case):
    _, value, load = registry_case
    value["tasks"][-1] = dict(value["tasks"][0])
    with pytest.raises(ValueError, match="task inventory"):
        load()


def test_an_earlier_fold_cannot_replace_the_latest_completed_fold(registry_case):
    _, value, load = registry_case
    value["tasks"][0].update(status="available", completed_folds=[1, 2],
                             missing_folds=[3, 4, 5], selected_fold=1)
    with pytest.raises(ValueError, match="fold selection"):
        load()


def test_bundle_path_cannot_escape_the_package(registry_case):
    folder, value, load = registry_case
    outside = folder.parent / "outside"
    outside.mkdir()
    manifest = outside / "manifest.json"
    manifest.write_text("{}")
    value["tasks"][0].update(status="available", completed_folds=[1], missing_folds=[2, 3, 4, 5],
        selected_fold=1, bundle="../outside", manifest_sha256=file_digest(manifest))
    value["available_tasks"] = 1
    with pytest.raises(ValueError, match="outside bundle"):
        load()


def test_in_memory_registry_mutation_is_rejected(registry_case):
    _, value, load = registry_case
    predictor = load()
    predictor.registry["core_comparison_complete"] = True
    e = value["tasks"][0]
    with pytest.raises(ValueError, match="identity changed"):
        predictor.predict(None, profile=e["profile"], horizon=e["horizon"], target=e["target"],
                          model="last_close", as_of_ns=0, history_identity="fictional")


def test_partial_export_is_rejected_before_creating_output(tmp_path, monkeypatch):
    monkeypatch.setattr(review, "OUT", tmp_path)
    monkeypatch.setattr(package, "OUT", tmp_path)
    coverage = tmp_path / "coverage.json"
    rows = [{"profile": p, "horizon": h, "target": t, "fold": f, "status": "pending"}
            for p, h, t in sorted(package.TASK_KEYS) for f in range(1, 6)]
    coverage.write_text(json.dumps({"status": "private_verified_coverage_snapshot", "folds": rows}))
    permission, source = tmp_path / "permission.json", tmp_path / "source.json"
    permission.write_text("{}")
    source.write_text("{}")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="partial export must be explicit"):
        package.export_study_package(coverage_path=coverage, permission_path=permission,
            source_manifest_path=source, admission_dir=tmp_path, output_directory=output)
    assert not output.exists()
