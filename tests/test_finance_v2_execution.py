"""Model-independent score checks and end-to-end date-isolated saved-artifact acceptance."""
import copy

import numpy as np
import pandas as pd
import pytest

from finance_v2.data import calendar_grid, load_market_data, training_dates
from finance_v2.io import ROOT, read_json
from finance_v2.metrics import calibrate, calibrated, forecast_metrics, holm, paired_block_bootstrap
from finance_v2.run import check_receipt, run_one
from finance_v2.training import load_neural, neural_predictions, predict_baseline


def test_cumulative_scores_are_not_summed_twice_and_calibration_is_ordered():
    y = np.array([[.01, -.02, .03], [-.01, .04, -.05]])
    pred = np.array([[.001, -.002, .003], [.002, .003, .004]])
    c = calibrate(y, pred)
    q = calibrated(pred, c)
    scores = forecast_metrics(y, pred, quantiles=q)
    assert scores["12"]["mse"] == pytest.approx(np.mean((y[:, 2] - pred[:, 2]) ** 2))
    assert scores["5"]["mse"] == pytest.approx(np.mean((y[:, 1] - pred[:, 1]) ** 2))
    assert np.all(np.diff(q, axis=2) >= 0)
    assert scores["12"]["pinball"]["0.5"] == pytest.approx(np.abs(y[:, 2] - q[:, 2, 1]).mean() / 2)


def test_day_and_moving_blocks_do_not_treat_intraday_replication_as_independent():
    dates = np.array([f"2024-03-{i+1:02}" for i in range(20)])
    delta = np.linspace(-.01, -.002, len(dates))
    for block in (1, 5):
        first = paired_block_bootstrap(delta, dates, repeats=500, block_days=block)
        repeated = paired_block_bootstrap(np.repeat(delta, 13), np.repeat(dates, 13), repeats=500, block_days=block)
        np.testing.assert_allclose(first["ci95"], repeated["ci95"], rtol=1e-13)
        assert first["p_value_one_sided_candidate_better"] == repeated["p_value_one_sided_candidate_better"]
        assert first["ci95"][1] < 0
    assert holm({"a": .01, "b": .04, "c": .03}) == {"a": .03, "c": .06, "b": .06}


@pytest.fixture(scope="module")
def synthetic_data(tmp_path_factory):
    directory = tmp_path_factory.mktemp("v2_execution")
    rng = np.random.default_rng(2026)
    grid = calendar_grid("2024-01-02", "2024-05-10")
    days = list(grid)[:85]
    stamps = grid[days[0]]
    for day in days[1:]:
        stamps = stamps.append(grid[day])
    innovation = rng.normal(0, .0002, len(stamps))
    returns = innovation.copy()
    for i in range(1, len(returns)):
        returns[i] += .25 * returns[i-1]
    close = 100 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"timestamp": [v.isoformat() for v in stamps], "open": close,
                          "high": close + .01, "low": close - .01, "close": close,
                          "volume": rng.integers(1000, 10000, len(stamps))})
    path = directory / "synthetic.csv"
    frame.to_csv(path, index=False)
    return path, days


@pytest.mark.parametrize("name", ["zero", "ridge", "lightgbm", "xpatch48"])
def test_train_calibrate_save_reload_and_tamper_detection(tmp_path, synthetic_data, name):
    path, days = synthetic_data
    data = load_market_data(path)
    config = copy.deepcopy(read_json(ROOT / "configs/aapl_v2.json"))
    config["runtime"]["device"] = "cpu"
    config["runtime"]["torch_threads"] = 2
    config["training"]["max_epochs"] = 1
    config["training"]["lightgbm"]["n_estimators"] = 5
    config["training"]["lightgbm"]["num_leaves"] = 7
    config["origin_stride_minutes"] = 5
    split = {"name": "synthetic", "kind": "independent_synthetic_acceptance", "train_dates": days[:79],
             "purged_dates": days[79:80], "eval_dates": days[80:85]}
    signature = "independent_synthetic_v1"
    result = run_one(name, 2026, config, data, split, tmp_path, signature)
    directory = tmp_path / "synthetic" / "close" / f"{name}_seed2026"
    assert check_receipt(directory, signature) == result
    partition = training_dates(split["train_dates"], config)
    assert max(partition["final_fit"]) < min(partition["calibration"]) < min(split["eval_dates"])
    scale = data.scaler(partition["final_fit"])
    origins = data.samples(split["eval_dates"], 5)
    with np.load(directory / "predictions.npz") as saved:
        y, prediction = saved["y"].copy(), saved["pred"].copy()
        close = data.bars.close.to_numpy()
        direct = np.column_stack([np.log(close[origins+h] / close[origins]) for h in (1, 5, 12)])
        np.testing.assert_allclose(y, direct, rtol=1e-10, atol=1e-15)
    if name == "xpatch48":
        model, state = load_neural(directory / "final.pt", "cpu")
        assert state["epoch"] == 1 and state["scaler"] == scale
        steps = neural_predictions(model, data.inputs(origins, scale), config["runtime"]["batch_size_inference"], scale["target_std"])
        reproduced = np.cumsum(steps, axis=1)[:, [0, 4, 11]]
    else:
        reproduced = predict_baseline(name, data, origins, scale, directory)
    np.testing.assert_allclose(prediction, reproduced, rtol=1e-12, atol=1e-15)
    with (directory / "calibration.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(RuntimeError, match="Changed or incomplete"):
        check_receipt(directory, signature)


def test_report_uses_seed_mean_loss_and_rejects_unpaired_origins(tmp_path):
    from finance_v2.data import quarterly_splits
    from finance_v2.io import write_json
    from finance_v2.metrics import calibrated
    from finance_v2.report import generate_report
    from finance_v2.run import run_path, seeds_for
    config = copy.deepcopy(read_json(ROOT / "configs/aapl_v2.json"))
    config["statistics"]["bootstrap_repeats"] = 500
    # Create synthetic scores with opposite seed errors. Their forecast mean
    # would look perfect; the prescribed mean of losses must remain positive.
    write_json(tmp_path / "manifest.json", {"signature": "synthetic", "identity": {"config": config}})
    for i, split in enumerate(quarterly_splits(config)):
        dates = np.repeat(np.array(split["eval_dates"][:5]), 2)
        origins = np.arange(10) + i * 100
        y = np.tile(np.array([.001, .003, .005]), (10, 1))
        for feature in config["feature_sets"]:
            for name in config["models"]:
                for seed in seeds_for(name, config):
                    directory = run_path(tmp_path, split, feature, name, seed)
                    directory.mkdir(parents=True)
                    pred = np.zeros_like(y)
                    if name == "xpatch48":
                        pred = .8 * y
                    elif name == "xpatch240":
                        pred = .7 * y
                    elif name == "kronos_small":
                        pred = y + (1 if seed == 2026 else -1) * .02
                    q = calibrated(pred, {"residual_quantiles": [[-.01, 0, .01]] * 3})
                    np.savez_compressed(directory / "predictions.npz", origins=origins,
                        dates=dates, timestamps=dates, y=y, pred=pred, quantiles=q)
                    write_json(directory / "result.json", {"total_seconds": 1., "peak_cuda_allocated_mib": None})
    report = generate_report(tmp_path)
    for panel in report["panels"].values():
        assert panel["selected_original_from_development"] == "xpatch48"
        assert panel["vs_zero"]["xpatch48"]["development_gate_passed"]
        assert not panel["vs_zero"]["kronos_small"]["development_gate_passed"]
        assert panel["models"]["kronos_small"]["metrics"]["12"]["mse"] == pytest.approx(.0004)
    bad = run_path(tmp_path, quarterly_splits(config)[0], "close", "ridge", 0) / "predictions.npz"
    with np.load(bad) as saved:
        values = {k: saved[k].copy() for k in saved.files}
    values["origins"][0] += 1
    np.savez_compressed(bad, **values)
    with pytest.raises(AssertionError, match="Unpaired forecasts"):
        generate_report(tmp_path)
