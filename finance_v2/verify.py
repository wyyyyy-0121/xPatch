"""Independently reconstruct labels, date isolation, calibration, weights and saved scores."""
from pathlib import Path

import numpy as np

from .data import confirmation_split, load_market_data, quarterly_splits, training_dates
from .io import ROOT, digest, read_json, write_json
from .metrics import calibrate, calibrated, forecast_metrics
from .models import NEURAL, PRETRAINED, Pretrained, verify_model_lock
from .training import load_neural, neural_predictions, predict_baseline, setup_runtime


def direct_labels(data, origins):
    closes = data.bars.close.to_numpy()
    return np.column_stack([np.log(closes[origins + h] / closes[origins]) for h in data.horizons])


def verify(output, full_models=True):
    from .run import check_receipt, identity_for, run_path, seeds_for, signature_of
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    config = manifest["identity"]["config"]
    identity = identity_for(config)
    if identity != manifest["identity"] or signature_of(identity) != manifest["signature"]:
        raise AssertionError("Current experiment differs from signed protocol")
    verify_model_lock(config)
    checks = []
    device = setup_runtime(config)
    for feature in config["feature_sets"]:
        data = load_market_data(ROOT / config["data"], config["lookback_returns"],
                                tuple(config["horizons_minutes"]), feature,
                                config["periods"]["development_start"], config["periods"]["historical_confirmation_end"])
        splits = quarterly_splits(config)
        frozen = read_json(output / "frozen.json") if (output / "frozen.json").exists() else None
        if frozen and frozen["selected"].get(feature) is not None and (output / "historical_confirmation").exists():
            splits.append(confirmation_split(config))
        for split in splits:
            dates = training_dates(split["train_dates"], config)
            order = [dates["early_fit"], dates["selection_purged"], dates["epoch_selection"],
                     dates["calibration_purged"], dates["calibration"], split["purged_dates"], split["eval_dates"]]
            for earlier, later in zip(order, order[1:]):
                if max(earlier) >= min(later):
                    raise AssertionError("Date partition leakage")
            origins = data.samples(split["eval_dates"], config["origin_stride_minutes"])
            cal_origins = data.samples(dates["calibration"], config["origin_stride_minutes"])
            scale = data.scaler(dates["final_fit"])
            names = config["models"] if split["name"] != "historical_confirmation" else list(dict.fromkeys(
                ["zero", "ridge", frozen["selected_original"][feature], frozen["selected"][feature]]))
            for name in names:
                pretrained = None
                for seed in seeds_for(name, config):
                    directory = run_path(output, split, feature, name, seed)
                    result = check_receipt(directory, manifest["signature"])
                    if split["name"] == "historical_confirmation":
                        if result["started_at_utc"] < frozen["frozen_at_utc"] or result["freeze_sha256"] != digest(output / "frozen.json"):
                            raise AssertionError("Historical confirmation did not follow frozen decision")
                    if read_json(directory / "scaler.json") != scale:
                        raise AssertionError("Training scaler differs")
                    if read_json(directory / "split.json") != {**split, "training_partition": dates}:
                        raise AssertionError("Date split differs")
                    for filename, indices in [("predictions.npz", origins), ("calibration_predictions.npz", cal_origins)]:
                        with np.load(directory / filename) as saved:
                            np.testing.assert_array_equal(saved["origins"], indices)
                            np.testing.assert_array_equal(saved["dates"], data.dates[indices].astype(str))
                            np.testing.assert_array_equal(saved["timestamps"], data.timestamps[indices].astype(str))
                            np.testing.assert_allclose(saved["y"], direct_labels(data, indices), rtol=1e-10, atol=2e-15)
                            # Exact minute/session continuity is checked independently.
                            for shift in (-data.lookback, max(data.horizons)):
                                delta = data.timestamps[indices + shift] - data.timestamps[indices]
                                np.testing.assert_array_equal(delta, np.full(len(indices), np.timedelta64(shift, "m")))
                                np.testing.assert_array_equal(data.dates[indices + shift], data.dates[indices])
                            if filename == "predictions.npz":
                                y, pred, q = saved["y"].copy(), saved["pred"].copy(), saved["quantiles"].copy()
                            else:
                                cy, cp = saved["y"].copy(), saved["pred"].copy()
                    calibration = read_json(directory / "calibration.json")
                    calculated = calibrate(cy, cp, config["calibration"]["quantile_levels"])
                    for key in calculated:
                        if calculated[key] != calibration[key]:
                            raise AssertionError("Calibration not reconstructed from disjoint tail")
                    np.testing.assert_array_equal(q, calibrated(pred, calibration))
                    assert result["metrics"] == forecast_metrics(y, pred, data.horizons, q, calibration["quantile_levels"])
                    checked = False
                    if full_models:
                        if name in NEURAL:
                            model, state = load_neural(directory / "final.pt", device)
                            assert state["scaler"] == scale and state["seed"] == seed
                            steps = neural_predictions(model, data.inputs(origins, scale),
                                                       config["runtime"]["batch_size_inference"], scale["target_std"])
                            reproduced = np.cumsum(steps, axis=1)[:, np.asarray(data.horizons) - 1]
                            np.testing.assert_array_equal(pred, reproduced)
                            del model
                        elif name in PRETRAINED:
                            if pretrained is None:
                                pretrained = Pretrained(name, config, device)
                            settings = config["pretrained"]["kronos" if name == "kronos_small" else "chronos2"]
                            size = settings["batch_size"]
                            for start in dict.fromkeys([0, ((len(origins) - 1) // size) * size]):
                                reproduced = pretrained.predict(data, origins[start:start + size], seed + start)
                                np.testing.assert_allclose(pred[start:start + size], reproduced, rtol=1e-6, atol=1e-9)
                        else:
                            reproduced = predict_baseline(name, data, origins, scale, directory)
                            np.testing.assert_allclose(pred, reproduced, rtol=1e-12, atol=1e-15)
                        checked = True
                    checks.append({"run": str(directory.relative_to(output)), "targets_dates_scaler_calibration_metrics": True,
                                   "model_forecasts_reproduced": checked,
                                   "pretrained_reproduction_scope": "first and last fixed batches" if name in PRETRAINED else None})
                    print(f"VERIFIED {directory}", flush=True)
                del pretrained
    from .report import generate_report
    summary = generate_report(output, "development")
    result = {"passed": True, "signature": manifest["signature"], "runs_checked": len(checks),
              "all_model_checks_requested": full_models, "checks": checks,
              "any_development_gate_passed": summary["any_development_gate_passed"],
              "independent_test": False, "point_in_time_proven": False}
    write_json(output / "acceptance.json", result)
    return result
