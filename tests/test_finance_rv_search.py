"""Search decisions must use all prescribed folds/seeds and lock outer lengths."""
from datetime import date, timedelta

from finance.experiment import Experiment
from finance.io import ROOT, read_json
from finance.search import configurations
from finance.splits import partition


def setup(tmp_path, model):
    config = read_json(ROOT / "configs/rv30.json")
    dates = [(date(2021, 1, 1) + timedelta(days=i)).isoformat() for i in range(500)]
    outer = partition("2023Q1", dates, ["2023-01-03", "2023-01-04"])
    # Directories are within the repo because production requests use relative paths.
    directory = ROOT / "artifacts/rv30/test_search" / tmp_path.name
    experiment = Experiment(outer, "aapl", model, configurations(config), config, "test_signature", directory)
    return experiment, config


def finish(experiment, requests, score_function):
    for request in requests:
        result = {**{k: request[k] for k in ("job_id", "config_id", "model", "panel", "seed", "directory")},
            "fold": request["partition"]["name"], "status": "complete", "score": {"qlike": score_function(request)},
            "fitting": {"best_iteration": {2026: 11, 2027: 22, 2028: 33}[request["seed"]], "best_epoch": 3}}
        experiment.accept(result)
    return experiment.advance()


def test_stochastic_finalists_can_reverse_first_seed_ranking_and_lock_iterations(tmp_path):
    experiment, config = setup(tmp_path, "lightgbm")
    initial = experiment.initial_requests()
    assert len(initial) == 80 * 3
    finalists = finish(experiment, initial, lambda r: int(r["config_id"].split("_")[1]) + 1.)
    assert len(finalists) == 3 * 2 * 3
    assert {r["seed"] for r in finalists} == {2027, 2028}
    # First-seed winner gets poor replication; second-ranked config improves.
    outer = finish(experiment, finalists, lambda r: 10. if "_000_" in r["config_id"] else .1 if "_001_" in r["config_id"] else 5.)
    assert len(outer) == 3
    assert all("_001_" in r["config_id"] for r in outer)
    assert {r["fixed_iterations"] for r in outer} == {22}
    assert {r["seed"] for r in outer} == {2026, 2027, 2028}
    assert all(r["phase"] == "outer" and r["partition"]["name"] == "2023Q1" for r in outer)


def test_xpatch_all_three_inner_folds_precede_each_halving(tmp_path):
    experiment, _ = setup(tmp_path, "xpatch")
    initial = experiment.initial_requests()
    assert len(initial) == 32 * 3 and {r["max_epochs"] for r in initial} == {5}
    next_stage = finish(experiment, initial, lambda r: int(r["config_id"].split("_")[1]) + 1.)
    assert len(next_stage) == 16 * 3 and {r["max_epochs"] for r in next_stage} == {15}
    assert all("resume_directory" in r for r in next_stage)
    final_stage = finish(experiment, next_stage, lambda r: int(r["config_id"].split("_")[1]) + 1.)
    assert len(final_stage) == 8 * 3 and {r["max_epochs"] for r in final_stage} == {30}
    replicas = finish(experiment, final_stage, lambda r: int(r["config_id"].split("_")[1]) + 1.)
    assert len(replicas) == 3 * 2 * 3 and {r["max_epochs"] for r in replicas} == {30}
    assert all("resume_directory" not in r for r in replicas)
    outer = finish(experiment, replicas, lambda r: int(r["config_id"].split("_")[1]) + 1.)
    assert {r["fixed_epochs"] for r in outer} == {3}
    assert len(experiment.selection["selection_jobs"]) == 9


def test_deterministic_models_have_one_outer_fit_and_no_redundant_seed_replication(tmp_path):
    experiment, _ = setup(tmp_path, "har")
    initial = experiment.initial_requests()
    assert len(initial) == 12 * 3
    outer = finish(experiment, initial, lambda r: int(r["config_id"].split("_")[1]) + 1.)
    assert len(outer) == 1 and outer[0]["phase"] == "outer"
    assert experiment.selection["selected_before_outer_labels"]
