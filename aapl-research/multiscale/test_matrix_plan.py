"""Test that candidate task inventories cannot be mistaken for runnable work."""
from copy import deepcopy

import pytest

from .matrix_plan import GLOBAL_GATES, make_matrix
from .spec import DEV_END, PROFILES, TARGETS


def synthetic_admission():
    rows = []
    for profile in PROFILES:
        for horizon in profile.horizons:
            for target in TARGETS:
                folds = [{"fold": i, "status": "ok", "inner_fit_origins": i * 10,
                          "inner_stop_origins": 5, "validation_origins": 12,
                          "validation_first_decision": "2024-01-01T00:00:00-05:00",
                          "validation_last_decision": "2024-02-01T00:00:00-05:00",
                          "validation_last_target_completion": "2024-03-01T00:00:00-05:00"}
                         for i in range(1, 6)]
                rows.append({"profile": profile.name, "frequency": profile.frequency,
                             "horizon": horizon, "target": target,
                             "lookback_candidates": list(profile.lookbacks),
                             "common_max_lookback": max(profile.lookbacks),
                             "primary_endpoint": horizon == profile.primary_horizon,
                             "origins": 100, "status": "candidate",
                             "minimum_history_met": True, "minimum_inner_fit_origins": 10,
                             "known_legacy_overlap_origins": 0, "folds": folds,
                             "proposed_warmup_folds": None})
    return {"Q3_values_read": False, "experiment_scored": False,
            "development_cutoff_new_york": DEV_END, "tasks": rows}


def test_all_twenty_tasks_and_one_hundred_folds_remain_blocked():
    result = make_matrix(synthetic_admission(), "a" * 64)
    assert (result["tasks"], result["fold_jobs"], result["model_prediction_slots"]) == (20, 100, 800)
    assert (result["neural_fit_slots_one_seed"], result["ridge_fit_slots_one_seed"],
            result["baseline_slots_one_seed"]) == (400, 100, 300)
    assert len({job["job_id"] for job in result["jobs"]}) == 100
    assert all(job["status"] == "blocked_not_runnable" for job in result["jobs"])
    assert all(job["global_gates_not_yet_met"] == list(GLOBAL_GATES) for job in result["jobs"])
    assert not result["frozen"] and not result["experiment_scored"] and not result["market_price_arrays_opened"]
    first = result["jobs"][0]
    assert first["neural_update_upper_bound_draft"] == 80
    assert first["raw_prediction_payload_bytes_at_float64"] == 12 * 12 * 8 * 8


def test_missing_folds_and_insufficient_history_are_not_dropped():
    admission = synthetic_admission()
    row = admission["tasks"][0]
    row["minimum_history_met"] = False
    row["folds"][0]["inner_fit_origins"] = 2
    row["folds"].pop()
    result = make_matrix(admission, "b" * 64)
    group = [job for job in result["jobs"] if job["profile"] == row["profile"]
             and job["horizon"] == row["horizon"] and job["target"] == row["target"]]
    assert len(group) == 5
    assert all("original_minimum_history_not_met" in job["local_blockers"] for job in group)
    assert "fewer_than_eight_inner_fit_origins" in group[0]["local_blockers"]
    assert "candidate_fold_missing" in group[-1]["local_blockers"]


def test_rejects_identity_scope_duplicates_and_changed_task_shapes():
    admission = synthetic_admission()
    with pytest.raises(ValueError, match="SHA-256"):
        make_matrix(admission, "not-a-digest")
    modified = deepcopy(admission)
    modified["Q3_values_read"] = True
    with pytest.raises(ValueError, match="source admission state"):
        make_matrix(modified, "c" * 64)
    modified = deepcopy(admission)
    modified["tasks"][0]["folds"][1]["fold"] = 1
    with pytest.raises(ValueError, match="Duplicate candidate fold"):
        make_matrix(modified, "c" * 64)
    modified = deepcopy(admission)
    modified["tasks"][0]["common_max_lookback"] = 2
    with pytest.raises(ValueError, match="lookbacks"):
        make_matrix(modified, "c" * 64)
