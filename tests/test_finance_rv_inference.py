"""Exercise real fitted inference, availability boundaries and live status."""
import numpy as np

from finance.calibration import Calibration
from finance.features import FeatureStore
from finance.inference import infer
from finance.io import ROOT, sha, write_json
from finance.models import LinearModel, raw_predictions
from finance import data as data_module, provenance
from test_finance_rv_features import example


def test_frozen_baseline_inference_respects_fit_availability_and_live_entitlement(example, monkeypatch, tmp_path):
    _, data, config, state = example
    directory = ROOT / "artifacts/rv30/test_inference" / tmp_path.parent.name / tmp_path.name
    directory.mkdir(parents=True, exist_ok=True)
    params = {"representation": "log_rv", "semivariance": False, "alpha": 1.}
    store = FeatureStore(data, config)
    training = data.origins(state.fit_dates, stride=5)
    x, names = store.matrix(training, state, "aapl", "har", params)
    model = LinearModel.fit(x, state.labels(data, training, params["representation"]), params["alpha"], names)
    model.save(directory / "model.npz")
    write_json(directory / "fit_state.json", state.dictionary())
    calibration_origins = data.origins(data.days[51:53], stride=10)
    raw = raw_predictions("har", model, data, store, state, calibration_origins, params, "aapl", config, "cpu")
    calibration = Calibration.fit(data.targets(calibration_origins), raw, data.dates[calibration_origins])
    write_json(directory / "calibration.json", calibration.dictionary())
    files = {p.name: sha(p) for p in directory.iterdir() if p.name in ("model.npz", "fit_state.json", "calibration.json")}
    write_json(directory / "result.json", {"status": "complete", "request_hash": "test", "files": files})
    manifest = {"model": "har", "panel": "aapl", "params": params, "config_id": "test_har", "signature": "test",
        "active_predictive_edge_established": False, "prospective_qualified": False,
        "source_directory": str(directory.relative_to(ROOT)), "source_result_sha256": sha(directory / "result.json")}
    write_json(directory / "manifest.json", manifest)
    monkeypatch.setattr(provenance, "verify_freeze", lambda config: {"signature": "test"})
    monkeypatch.setattr(data_module.RVData, "from_manifest", classmethod(lambda cls, path: data))
    manifest_path = str((directory / "manifest.json").relative_to(ROOT))
    result = infer(config, manifest_path, "2022-05-02T10:00:00-04:00")
    assert result["rv_mean"] > 0 and np.isfinite(result["rv_mean"])
    assert result["status"] == "historical_replay_baseline" and result["received_at"] is None
    assert not result["eligible_signal"] and not result["direction_or_profit_prediction"]
    earlier = infer(config, manifest_path, "2022-02-15T10:00:00-05:00")
    assert earlier["rv_mean"] is None and earlier["status"] == "before_frozen_model_availability"
    closed = infer(config, manifest_path, "2022-05-02T17:00:00-04:00")
    assert closed["rv_mean"] is None and closed["status"] == "outside_session_or_insufficient_history"
    live = infer(config, manifest_path, mode="live")
    assert live["status"] == "live_unavailable" and live["rv_mean"] is None
