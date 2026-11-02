import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from finance_ls.data import chronological_splits, load_market_data
from finance_ls.metrics import forecast_metrics, paired_day_bootstrap
from finance_ls.models import NEURAL_MODELS, make_model
from layers.ema import EMA
from layers.network import Network


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/finance_ls.json").read_text())
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def prices(path, days=12, minutes=270):
    timestamps = np.concatenate([pd.date_range(f"2025-01-{d:02} 09:30", periods=minutes, freq="min").to_numpy()
                                 for d in range(1, days + 1)])
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(timestamps))))
    df = pd.DataFrame({"timestamp": timestamps, "close": close, "OBV": -999})
    df.to_csv(path, index=False)
    return df


def test_return_alignment_close_column_and_price_count(tmp_path):
    path = tmp_path / "prices.csv"
    df = prices(path, days=1, minutes=253)
    data = load_market_data(path)
    np.testing.assert_array_equal(data.origins, [240])
    x, y = data.arrays(data.origins, 1.0)
    actual = np.diff(np.log(df.close.to_numpy()))
    np.testing.assert_allclose(x[0, :, 0], actual[:240], rtol=1e-6)
    np.testing.assert_allclose(y[0, :, 0], actual[240:], rtol=1e-6)
    assert x.shape == (1, 240, 1) and y.shape == (1, 12, 1)


def test_gaps_and_overnight_cannot_enter_samples(tmp_path):
    path = tmp_path / "prices.csv"
    df = prices(path, days=2, minutes=300)
    df.drop(index=[100]).to_csv(path, index=False)
    data = load_market_data(path, lookback=20, horizon=12)
    timestamps = data.timestamps
    for origin in data.origins:
        segment = timestamps[origin - 20:origin + 13]
        assert (np.diff(segment) == np.timedelta64(1, "m")).all()
        assert len(np.unique(data.dates[origin - 20:origin + 13])) == 1
    assert data.audit["within_date_gaps"] == 1
    assert np.isnan(data.returns[0])
    assert np.isnan(data.returns[299])  # first row of day 2 after dropping day-1 row
    assert sum(d["excluded_for_gaps"] for d in data.audit["coverage_by_date"]) > 0


def test_scaler_and_training_windows_do_not_depend_on_future(tmp_path):
    path = tmp_path / "prices.csv"
    df = prices(path)
    first = load_market_data(path)
    splits = chronological_splits(first)
    train_dates = splits[0]["train_dates"]
    scale = first.scale(train_dates)
    mask = pd.to_datetime(df.timestamp).dt.strftime("%Y-%m-%d") > max(train_dates)
    df.loc[mask, "close"] *= np.exp(np.linspace(0, 5, mask.sum()))
    df.to_csv(path, index=False)
    second = load_market_data(path)
    assert second.scale(train_dates) == scale
    for a, b in zip(first.arrays(first.samples(train_dates), scale["std"]),
                    second.arrays(second.samples(train_dates), scale["std"])):
        np.testing.assert_array_equal(a, b)
    used = first.returns[np.isin(first.dates, train_dates)]
    assert scale["std"] == pytest.approx(np.nanstd(used), rel=1e-14)
    for split in splits:
        train = first.samples(split["train_dates"])
        evaluate = first.samples(split["eval_dates"])
        train_label_times = first.timestamps[train[:, None] + np.arange(1, 13)].ravel()
        eval_label_times = first.timestamps[evaluate[:, None] + np.arange(1, 13)].ravel()
        assert not np.intersect1d(train_label_times, eval_label_times).size
        assert max(train_label_times) < min(eval_label_times)


@pytest.mark.parametrize("defect", ["duplicate", "zero", "nan", "off_minute"])
def test_invalid_source_is_rejected(tmp_path, defect):
    path = tmp_path / "prices.csv"
    df = prices(path, days=1)
    if defect == "duplicate":
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"]
    elif defect == "off_minute":
        df.loc[1, "timestamp"] += pd.Timedelta(seconds=1)
    else:
        df.loc[1, "close"] = 0 if defect == "zero" else np.nan
    df.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_market_data(path)


@pytest.mark.parametrize("alpha,length", [(0.1, 240), (0.9, 1000), (1.0, 48), (0.001, 240)])
def test_ema_matches_independent_recurrence_is_causal_and_differentiable(alpha, length):
    torch.manual_seed(32)
    x = torch.randn(2, length, 2, dtype=torch.float64, requires_grad=True)
    actual = EMA(alpha)(x)
    value = x[:, 0, :]
    reference = [value]
    for i in range(1, length):
        value = alpha * x[:, i, :] + (1 - alpha) * value
        reference.append(value)
    expected = torch.stack(reference, dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-11)
    changed = x.detach().clone()
    changed[:, length // 2:] += 100
    torch.testing.assert_close(EMA(alpha)(changed)[:, :length // 2], actual[:, :length // 2])
    actual.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_default_network_matches_upstream_exactly():
    # Compare the unmodified historical implementation, including concatenation order.
    source = subprocess.check_output(["git", "show", "7fffeafdec8205c37f4bdd7e673ec0f869de6ea4:layers/network.py"],
                                     cwd=ROOT, text=True)
    namespace = {}
    exec(compile(source, "upstream_network.py", "exec"), namespace)
    upstream = namespace["Network"](48, 12, 12, 6, "end").eval()
    extended = Network(48, 12, 12, 6, "end").eval()
    extended.load_state_dict(upstream.state_dict())
    s, t = torch.randn(3, 48, 2), torch.randn(3, 48, 2)
    torch.testing.assert_close(upstream(s, t), extended(s, t), rtol=0, atol=0)


def test_ema_cache_rebuilds_when_precision_changes():
    ema = EMA(0.1)
    ema(torch.randn(2, 48, 1))
    ema = ema.double()
    x = torch.randn(2, 48, 1, dtype=torch.float64)
    torch.testing.assert_close(ema(x), EMA(0.1)(x), rtol=0, atol=0)


@pytest.mark.parametrize("name", sorted(NEURAL_MODELS))
def test_shapes_gradients_and_all_parameters_used(name):
    model = make_model(name, CONFIG)
    x = torch.randn(4, 240, 1)
    y = model(x)
    assert y.shape == (4, 12, 1)
    y.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_parameter_match_and_short_stream_ignores_old_history():
    a, b = make_model("ls", CONFIG), make_model("xpatch48", CONFIG)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    model = make_model("short_only", CONFIG).eval()
    x = torch.randn(2, 240, 1)
    changed = x.clone()
    changed[:, :-48] += 100
    torch.testing.assert_close(model(x), model(changed), rtol=0, atol=0)


def test_pooling_and_streams_end_at_same_forecast_origin():
    model = make_model("ls", CONFIG).eval()
    captured = {}
    hook = model.net.register_forward_pre_hook(lambda module, args: captured.update(short=args[0], long=args[1]))
    x = torch.arange(240, dtype=torch.float32).reshape(1, 240, 1)
    model(x)
    hook.remove()
    # Independent scalar recurrence + bins checks the exact history selected by each path.
    def ema(v):
        values = [v[0]]
        for value in v[1:]:
            values.append(0.1 * value + 0.9 * values[-1])
        return np.array(values)
    expected_long = ema(np.arange(240)).reshape(48, 5).mean(axis=1)
    expected_short = np.arange(192, 240) - ema(np.arange(192, 240))
    np.testing.assert_allclose(captured["long"].detach().numpy().ravel(), expected_long, rtol=2e-6)
    np.testing.assert_allclose(captured["short"].detach().numpy().ravel(), expected_short, atol=0.0001)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("name", sorted(NEURAL_MODELS))
def test_cpu_gpu_consistency(name):
    torch.manual_seed(42)
    cpu = make_model(name, CONFIG).eval()
    gpu = make_model(name, CONFIG).cuda().eval()
    gpu.load_state_dict(cpu.state_dict())
    x = torch.randn(8, 240, 1)
    with torch.no_grad():
        torch.testing.assert_close(cpu(x), gpu(x.cuda()).cpu(), rtol=2e-4, atol=2e-5)


def test_cumulative_metrics_and_day_bootstrap():
    y = np.ones((8, 12))
    zero = np.zeros_like(y)
    m = forecast_metrics(y, zero)
    assert m["12"]["mse"] == 144 and m["5"]["mae"] == 5
    assert m["12"]["skill_vs_zero"] == 0
    dates = np.repeat(["a", "b", "c", "d"], 2)
    ci = paired_day_bootstrap(y, y, zero, dates, repeats=100)
    assert ci["ci95"] == [-144, -144]
    # Repeating observations in the same date must not fabricate extra day blocks.
    repeated = paired_day_bootstrap(np.tile(y, (3, 1)), np.tile(y, (3, 1)), np.tile(zero, (3, 1)),
                                    np.tile(dates, 3), repeats=100)
    assert repeated["ci95"] == ci["ci95"] and repeated["days"] == 4


def test_cli_stages_checkpoint_reload_and_frozen_protocol(tmp_path):
    path = tmp_path / "prices.csv"
    prices(path)
    config = {**CONFIG, "data": str(path), "max_epochs": 2, "patience": 1,
              "folds": 1, "seeds": [42, 43], "device": "cpu", "threads": 2,
              "models": ["zero", "ridge", "xpatch48", "xpatch240", "ls"], "bootstrap_repeats": 100}
    cfg, out = tmp_path / "config.json", tmp_path / "output"
    cfg.write_text(json.dumps(config))
    command = [sys.executable, "-m", "finance_ls.run", "--config", str(cfg), "--output", str(out)]
    failed = subprocess.run(command + ["--stage", "test"], cwd=ROOT, capture_output=True, text=True)
    assert failed.returncode != 0 and "freeze" in failed.stderr
    done = subprocess.run(command + ["--stage", "all"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stdout + done.stderr
    frozen_bytes = (out / "frozen.json").read_bytes()
    checkpoint = out / "test" / "ls_seed42" / "checkpoint.pt"
    from finance_ls.run import load_checkpoint
    model, payload = load_checkpoint(checkpoint)
    data = load_market_data(path)
    split = json.loads((out / "splits.json").read_text())[-1]
    origins = data.samples(split["eval_dates"])
    x, _ = data.arrays(origins, payload["scaler"]["std"])
    with torch.no_grad():
        prediction = model(torch.from_numpy(x)).numpy()[..., 0].astype(np.float64) * payload["scaler"]["std"]
    with np.load(checkpoint.parent / "predictions.npz") as saved:
        np.testing.assert_array_equal(prediction, saved["pred"])
    resumed = subprocess.run(command + ["--stage", "all"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert (out / "frozen.json").read_bytes() == frozen_bytes
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["models"]) == set(config["models"])
    assert (out / "REPORT.md").exists()
    verified = subprocess.run([sys.executable, "-m", "finance_ls.verify", "--output", str(out)],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    acceptance = json.loads((out / "acceptance.json").read_text())
    assert acceptance["passed"] and acceptance["runs_checked"] == 16
    # Changing a frozen setting may not silently reuse old results.
    config["alpha"] = 0.2
    cfg.write_text(json.dumps(config))
    changed = subprocess.run(command + ["--stage", "test"], cwd=ROOT, capture_output=True, text=True)
    assert changed.returncode != 0 and "changed" in changed.stderr
