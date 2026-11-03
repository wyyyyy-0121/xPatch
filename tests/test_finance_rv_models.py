"""Model persistence, exact sequence horizon and deterministic halving recovery."""
import copy

import numpy as np
import pytest

from finance.cache import load_data, save_data
from finance.features import FeatureStore, FitState
from finance.models import LinearModel, fit_neural, native_xpatch, raw_predictions, sequence_labels
from finance.search import configurations, rank_trials
from finance.splits import partition
from test_finance_rv_data import locate
from test_finance_rv_features import example


def test_linear_model_reloads_exact_prediction_and_rejects_permuted_features(tmp_path):
    rng = np.random.default_rng(19)
    x = rng.normal(size=(100, 4))
    y = 3 + x @ np.array([1., 2., 3., 4.])
    model = LinearModel.fit(x, y, 0., list("abcd"))
    np.testing.assert_allclose(model.predict(x, list("abcd")), y, atol=1e-12)
    model.save(tmp_path / "linear.npz")
    loaded = LinearModel.load(tmp_path / "linear.npz")
    np.testing.assert_array_equal(model.predict(x, list("abcd")), loaded.predict(x, list("abcd")))
    with pytest.raises(ValueError, match="order"):
        loaded.predict(x, list("dcba"))


def test_exact_sequence_terminal_and_session_boundary(example):
    _, data, config, state = example
    origins = data.origins(dates=data.days[-1:])[::20]
    for representation in ("log_rv", "log_seasonal_ratio"):
        labels = sequence_labels(data, state, origins, representation, "market", config)
        transformed = labels[:, -1, 0].astype(float) * state.series_std[representation][0] + state.series_center[representation][0]
        np.testing.assert_allclose(state.inverse(data, origins, transformed, representation), data.targets(origins), rtol=2e-6)
        assert labels.shape == (len(origins), 31, 3)
    with pytest.raises(ValueError, match="gap|session"):
        sequence_labels(data, state, [locate(data, "2022-04-29 15:45")], "log_rv", "aapl", config)


def test_mmap_preserves_eligibility_and_refuses_tampered_cache(example, tmp_path):
    _, data, _, _ = example
    save_data(data, tmp_path, "example")
    restored = load_data(tmp_path, "example")
    np.testing.assert_array_equal(data.origins(), restored.origins())
    assert not restored.returns.flags.writeable
    with (tmp_path / "returns.npy").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(ValueError, match="changed"):
        load_data(tmp_path, "example")


def test_fixed_search_has_full_coverage_and_no_outer_inputs(example):
    _, _, config, _ = example
    search = configurations(config)
    assert search == configurations(copy.deepcopy(config))
    for model, count in config["search"]["counts"].items():
        rows = search["models"][model]
        assert len(rows) == count and len({row["id"] for row in rows}) == count
    for key in ("seq_len", "patch_len", "ma_type", "alpha", "revin", "batch_size"):
        assert {row["params"][key] for row in search["models"]["xpatch"]} == set(config["search"]["xpatch"][key])


def test_failed_or_missing_inner_trials_cannot_win():
    records = [{"config_id": "complete", "fold": f, "seed": 2026, "status": "complete", "score": {"qlike": 2.}}
        for f in (1, 2, 3)]
    records += [{"config_id": "missing", "fold": f, "seed": 2026, "status": "complete", "score": {"qlike": 0.}}
        for f in (1, 2)]
    records += [{"config_id": "failed", "fold": f, "seed": 2026, "status": "failed", "score": {"qlike": 0.}}
        for f in (1, 2, 3)]
    assert rank_trials(records) == [{"config_id": "complete", "inner_qlike": 2.}]


def test_neural_halving_resume_matches_uninterrupted_training(example, tmp_path):
    import torch
    torch.set_num_threads(2)
    _, data, config, _ = example
    split = partition("smoke", data.days[:50], data.days[50:52])
    state = FitState.fit(data, split.fit_dates, config["seasonality"])
    store = FeatureStore(data, config)
    origins = data.origins(split.fit_dates, stride=5)[-64:]
    assert len(origins) == 64
    params = {"seq_len": 64, "pred_len": 31, "patch_len": 8, "stride": 4, "ma_type": "ema",
        "alpha": .1, "beta": .1, "revin": True, "batch_size": 32, "learning_rate": .001,
        "weight_decay": .0001, "representation": "log_rv"}
    full, _ = fit_neural(data, store, state, split, origins, params, "aapl", config, 2026,
        tmp_path / "full", max_epochs=2, device="cpu")
    fit_neural(data, store, state, split, origins, params, "aapl", config, 2026,
        tmp_path / "resume", max_epochs=1, device="cpu")
    resumed, summary = fit_neural(data, store, state, split, origins, params, "aapl", config, 2026,
        tmp_path / "resume", max_epochs=2, device="cpu", resume=True)
    assert summary["completed_epochs"] == 2
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    evaluation = data.origins(split.evaluation_dates, stride=10)[:3]
    predictions = raw_predictions("xpatch", resumed, data, store, state, evaluation, params, "aapl", config, device="cpu")
    assert predictions.shape == (3,) and np.isfinite(predictions).all() and (predictions > 0).all()
