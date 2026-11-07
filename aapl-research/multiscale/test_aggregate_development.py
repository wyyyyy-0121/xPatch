"""Test pooled errors, partial-fold labeling and annual overlap on fake paths."""
import json

import numpy as np
import pandas as pd
import pytest

from . import aggregate_development as aggregate_module
from .spec import MODELS


def test_synthetic_fold_pooling_and_nonoverlap_do_not_claim_confirmation(tmp_path, monkeypatch):
    admission_hash, review_hash, protocol_hash = "a" * 64, "b" * 64, "c" * 64
    monkeypatch.setattr(aggregate_module, "OUT", tmp_path)
    jobs = [{"profile": "annual", "frequency": "month", "horizon": 12,
             "target": "future_log_return", "fold": number} for number in range(1, 6)]
    analysis = {"block_lengths_by_profile": {"annual": [12, 24]}, "resamples": 100, "seed": 2026,
                "confidence": .95, "development_p_values": False, "annual_nonoverlap": True}
    protocol = {"data_kind": "synthetic_test", "jobs": jobs, "analysis": analysis}
    monkeypatch.setattr(aggregate_module, "load_reviewed", lambda *args: ({}, admission_hash, tmp_path, {}, review_hash, None))
    monkeypatch.setattr(aggregate_module, "_json_inside", lambda *args: (protocol, protocol_hash, None))
    monkeypatch.setattr(aggregate_module, "locked_protocol_issues", lambda *args: [])
    labels = pd.date_range("2018-01-31", periods=50, freq="ME").strftime("%Y-%m-%d").to_numpy()
    monkeypatch.setattr(aggregate_module, "session_labels", lambda *args: labels)
    run = tmp_path / "synthetic_matrix_validation" / "runs" / protocol_hash[:24]
    run.mkdir(parents=True)
    state = {"protocol_sha256": protocol_hash, "review_sha256": review_hash, "data_kind": "synthetic_test", "jobs": {}}
    verified = {}
    for job, origins in zip(jobs[:2], ([0, 1, 2, 12], [13, 24, 25, 36]), strict=True):
        key = aggregate_module._job_key(job)
        directory = run / key
        directory.mkdir(parents=True)
        origins = np.asarray(origins)
        actual = np.repeat((origins[:, None] + 1).astype(float), 12, axis=1)
        anchor = np.full(len(origins), 100.0)
        decisions = origins * 100 + 1
        np.savez_compressed(directory / "targets.npz", actual=actual, anchor=anchor, origins=origins, decision_ns=decisions)
        metrics = {}
        for model in MODELS:
            prediction = actual * .5 if model == "mlp" else np.zeros_like(actual)
            np.savez_compressed(directory / f"prediction_{model}.npz", prediction=prediction)
            metrics[model] = {"metrics": {"endpoint_mse": float(np.square(prediction[:, -1] - actual[:, -1]).mean())}}
        receipt_hash = str(job["fold"]) * 64
        state["jobs"][key] = {"status": "completed", "directory": key, "receipt_sha256": receipt_hash}
        verified[str(directory.resolve())] = {"fold_receipt_sha256": receipt_hash, "protocol_identity": protocol_hash,
                                               "data_identity": "synthetic_test:" + admission_hash,
                                               "current_source_matches_recorded": {"synthetic": True}, "models": metrics}
    (run / "matrix_state.json").write_text(json.dumps(state))
    monkeypatch.setattr(aggregate_module, "check_fold", lambda folder: verified[str(folder.resolve())])
    with pytest.raises(ValueError, match="All five folds"):
        aggregate_module.aggregate(tmp_path, tmp_path, tmp_path, run, "annual", 12, "future_log_return")
    result = aggregate_module.aggregate(tmp_path, tmp_path, tmp_path, run, "annual", 12,
                                        "future_log_return", allow_partial=True)
    assert result["status"] == "synthetic_test_results_only" and result["partial"]
    assert result["completed_folds"] == [1, 2] and result["missing_folds"] == [3, 4, 5]
    assert result["independent_test"] is False and result["development_p_values_reported"] is False
    assert result["models"]["mlp"]["relative_improvement_vs_persistence"] == pytest.approx(.75)
    assert result["models"]["mlp"]["annual_nonoverlapping_origins"]["origins"] == 4
    assert result["models"]["last_close"]["direction_coverage"] == 0
    assert all("centered_one_sided_p_improvement" not in interval
               for interval in result["models"]["mlp"]["descriptive_block_intervals"])
