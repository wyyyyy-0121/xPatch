"""Independent aggregate calculations must recover the frozen evaluator and gates."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from finance import evaluation, verify
from finance.evaluation import bundle, compare, describe, gates
from finance.io import ROOT, read_json, sha, write_json
from test_finance_rv_postprocessing import make_replay_round, replay_run


def prediction_values(confirmation=False, constant=False):
    quarters = pd.period_range("2025Q3", "2026Q2", freq="Q") if confirmation else pd.period_range("2023Q1", "2025Q2", freq="Q")
    days = [day.strftime("%Y-%m-%d") for quarter in quarters for day in pd.bdate_range(quarter.start_time, periods=8)]
    # Unequal numbers of origins, including three time-of-day strata.
    dates, stamps = [], []
    for index, day in enumerate(days):
        for time in ("10:00", "12:00", "14:45")[:1 if index % 7 == 0 else 3]:
            dates.append(day)
            stamps.append(pd.Timestamp(day + " " + time, tz="America/New_York").tz_convert("UTC").value)
    rng = np.random.default_rng(3492)
    observed = np.exp(rng.normal(-12, .8, len(dates)))
    mean = observed * np.exp(rng.normal(0, .12, len(dates)))
    if constant:
        mean[:] = 6e-6
    return {"y": observed, "mean": mean, "dates": np.asarray(dates), "timestamps": np.asarray(stamps),
        "origins": np.arange(len(dates)), "levels": [.05, .1, .5, .9, .95],
        "quantiles": mean[:, None] * np.array([.5, .7, 1., 1.4, 2.]),
        "seeds": {str(seed): mean * factor for seed, factor in ((2026, .97), (2027, 1.), (2028, 1.03))}}


@pytest.mark.parametrize("confirmation", [False, True])
@pytest.mark.parametrize("constant", [False, True])
def test_independent_descriptions_recover_daily_weighting_intervals_segments_and_mz(confirmation, constant):
    values = prediction_values(confirmation, constant)
    verify.assert_reference_equal(verify.reference_description(values, confirmation), describe(values, confirmation))


@pytest.mark.parametrize("confirmation", [False, True])
def test_independent_hac_bootstrap_holm_and_all_stability_gates(confirmation):
    config = read_json(ROOT / "configs/rv30.json")
    candidate = prediction_values(confirmation)
    baseline = {**candidate, "mean": candidate["y"] * 2.}
    expected = gates({"aapl/xpatch": compare(candidate, baseline, config, confirmation)}, config, confirmation)
    reference = verify.reference_gates({"aapl/xpatch": verify.reference_comparison(candidate, baseline, config["statistics"], confirmation)}, config, confirmation)
    verify.assert_reference_equal(reference, expected)
    assert reference["aapl/xpatch"]["passed"]
    # A single bad seed must independently block the otherwise good ensemble.
    candidate["seeds"]["2028"] *= 20
    blocked = verify.reference_gates({"aapl/xpatch": verify.reference_comparison(candidate, baseline, config["statistics"], confirmation)}, config, confirmation)
    assert not blocked["aapl/xpatch"]["passed"]
    assert not blocked["aapl/xpatch"]["gate_checks"]["all_seed_pooled_and_segments_improve"]


@pytest.mark.parametrize("ratio", [1., .5, 2.])
def test_independent_zero_variance_dm_and_circular_block_bootstrap(ratio):
    config = read_json(ROOT / "configs/rv30.json")
    candidate = prediction_values()
    candidate.update(y=np.full(len(candidate["y"]), 1e-5), mean=np.full(len(candidate["y"]), 1e-5))
    candidate["seeds"] = {"2026": candidate["mean"]}
    baseline = {**candidate, "mean": candidate["mean"] * ratio}
    verify.assert_reference_equal(verify.reference_comparison(candidate, baseline, config["statistics"]), compare(candidate, baseline, config))


def test_statistics_hash_cannot_mask_wrong_scope_or_evaluation_digest(replay_run):
    config, directory = replay_run
    path, summary = make_replay_round(config, directory, 1)
    statistics_path = path.parent / "statistics.json"
    statistics = read_json(statistics_path)
    statistics["evaluation"]["sha256"] = "wrong evaluation"
    write_json(statistics_path, statistics)
    summary["statistics_sha256"] = sha(statistics_path)
    write_json(path, summary)
    with pytest.raises(ValueError, match="aggregate statistics identity"):
        verify.verification_evidence(config, directory, 1)


@pytest.mark.parametrize("corruption", ["holm", "seed", "quarter"])
def test_independent_tree_comparison_rejects_changed_statistic_or_gate(corruption):
    config = read_json(ROOT / "configs/rv30.json")
    candidate = prediction_values()
    baseline = {**candidate, "mean": candidate["y"] * 2}
    reference = verify.reference_gates({"aapl/xpatch": verify.reference_comparison(candidate, baseline, config["statistics"])}, config)
    altered = deepcopy(reference)
    row = altered["aapl/xpatch"]
    if corruption == "holm":
        row["holm_p"] = .99
    elif corruption == "seed":
        row["seed_differences"]["2028"]["pooled"] = .1
    else:
        row["gate_checks"]["quarter_stability"] = False
    with pytest.raises(ValueError, match="Independent statistic"):
        verify.assert_reference_equal(reference, altered)


def test_full_independent_aggregate_from_saved_predictions_catches_gate_tampering(monkeypatch, tmp_path):
    monkeypatch.setattr(verify, "ROOT", tmp_path)
    config = read_json(ROOT / "configs/rv30.json")
    candidate = prediction_values()
    baseline = {**candidate, "mean": candidate["y"] * 2, "seeds": {"2026": candidate["y"] * 2}}
    sources = {}
    for name, values in (("har", baseline), ("xpatch", candidate)):
        path = tmp_path / "run" / name
        path.mkdir(parents=True)
        np.savez_compressed(path / "predictions.npz", **{key: values[key] for key in ("origins", "timestamps", "dates", "y", "mean", "quantiles")},
            calibration_origins=np.arange(20), calibration_y=np.full(20, 1e-5), calibration_raw=np.full(20, 1e-5))
        write_json(path / "calibration.json", {"levels": values["levels"], "log_factor": 0})
        write_json(path / "result.json", {"fixture": True})
        sources[str(path.relative_to(tmp_path))] = {"model": name, "panel": "aapl", "seed": 2026, "partition": {"name": "all_ten_quarters"}}
        values["seeds"] = {"2026": values["mean"]}
    directory = tmp_path / "run"
    write_json(directory / "setup.json", {"signature": "test"})
    comparisons = gates({"aapl/xpatch": compare(candidate, baseline, config)}, config)
    evaluation = {"descriptions": {"aapl/har": describe(baseline), "aapl/xpatch": describe(candidate)},
        "comparisons": comparisons, "passed_candidates": [key for key, row in comparisons.items() if row["passed"]]}
    write_json(directory / "evaluation.json", evaluation)
    assert verify.verify_aggregate_statistics(config, directory, 1, "development", sources)["passed"]
    evaluation["comparisons"]["aapl/xpatch"]["holm_p"] = .99
    write_json(directory / "evaluation.json", evaluation)
    with pytest.raises(ValueError, match="Independent statistic differs"):
        verify.verify_aggregate_statistics(config, directory, 2, "development", sources)


def test_independent_three_seed_ensemble_reconstructs_saved_calibration_intervals(replay_run, monkeypatch):
    config, directory = replay_run
    monkeypatch.setattr(evaluation, "ROOT", verify.ROOT)
    sources = verify.expected_sources(config, directory, "development")
    base = prediction_values()
    paths = {}
    for relative, request in sources.items():
        path = verify.ROOT / relative
        scale = 1 + (request["seed"] - 2026) * .15
        mean = base["mean"] * scale
        np.savez_compressed(path / "predictions.npz", **{key: base[key] for key in ("origins", "timestamps", "dates", "y")},
            mean=mean, quantiles=base["quantiles"] * scale, calibration_origins=np.arange(30),
            calibration_y=np.linspace(1e-6, 6e-6, 30), calibration_raw=np.linspace(2e-6, 4e-6, 30) * scale)
        write_json(path / "calibration.json", {"dates": ["2022-12-01"], "log_factor": .1 * scale,
            "levels": base["levels"], "residual_quantiles": [-.7, -.3, 0., .3, .7], "samples": 30})
        result = read_json(path / "result.json")
        result["files"] = {name: sha(path / name) for name in result["files"]}
        write_json(path / "result.json", result)
        paths.setdefault(request["panel"] + "/" + request["model"], []).append(relative)
    reference = verify.reference_bundles(sources)
    for key, group in paths.items():
        original = bundle(group)
        for name in ("origins", "timestamps", "dates", "y", "mean", "quantiles"):
            if name == "dates":
                np.testing.assert_array_equal(reference[key][name], original[name])
            else:
                np.testing.assert_allclose(reference[key][name], original[name], rtol=1e-12, atol=1e-14)
        assert set(reference[key]["seeds"]) == set(original["seeds"])
