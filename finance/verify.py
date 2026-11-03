"""Independent CSV label/metric recomputation and saved-model prediction replay."""
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibration
from .features import FitState
from .io import ROOT, read_json, sha, stable_hash, write_json
from .jobs import receipt
from .models import ChronosModel, load_model, raw_predictions


def reference_targets(closes, origin_timestamps):
    """31 actual source closes give 30 adjacent returns, independent of RVData."""
    if isinstance(origin_timestamps, pd.DatetimeIndex):
        if origin_timestamps.tz is None:
            raise ValueError("Independent target timestamps must be timezone aware")
        timestamps = origin_timestamps.as_unit("ns").asi8
    else:
        timestamps = np.asarray(origin_timestamps, dtype=np.int64)
    if np.any(timestamps < pd.Timestamp("2000-01-01", tz="UTC").value):
        raise ValueError("Integer target timestamps must use nanoseconds since the epoch")
    parts = []
    for start in range(0, len(timestamps), 4000):
        batch = timestamps[start:start + 4000]
        starts = batch[:, None] + np.arange(-1, 30) * 60 * 10**9
        prices = closes.reindex(pd.DatetimeIndex(starts.reshape(-1), tz="UTC")).to_numpy().reshape(-1, 31)
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("Independent target has missing/nonpositive source closes")
        parts.append(np.sum(np.diff(np.log(prices), axis=1) ** 2, axis=1))
    return np.concatenate(parts)


def independent_scores(y, p, dates):
    y = np.maximum(y, 1e-12)
    p = np.maximum(p, 1e-12)
    frame = pd.DataFrame({"date": dates, "qlike": y / p + np.log(p) - np.log(y) - 1,
        "log_rv_mse": (np.log(y) - np.log(p)) ** 2})
    grouped = frame.groupby("date", sort=True)[["qlike", "log_rv_mse"]].mean()
    return {key: float(grouped[key].mean()) for key in grouped}


def replay_indices(size, batch, pretrained):
    if not pretrained:
        return np.arange(size)
    # Preserve original batch boundaries, including the last partial batch.
    last = ((size - 1) // batch) * batch
    return np.unique(np.r_[np.arange(min(size, batch)), np.arange(last, size)])


def verify_one(context, request, directory):
    source = ROOT / request["source_directory"]
    original = receipt(source)
    if original is None or original["status"] != "complete":
        raise ValueError("Replay source is not a verified completed job")
    original_request = read_json(source / "request.json")
    if original["request_hash"] != stable_hash(original_request):
        raise ValueError("Replay source request fingerprint changed")
    if request.get("source_result_sha256", sha(source / "result.json")) != sha(source / "result.json"):
        raise ValueError("Replay source result changed")
    for key in ("signature", "model", "panel", "params", "config_id", "seed", "partition"):
        if stable_hash(request[key]) != stable_hash(original_request[key]):
            raise ValueError("Replay request differs from source model identity")
    config, data, store = context["config"], context["data"], context["store"]
    if original_request["phase"] not in ("outer", "confirmation", "delivery", "smoke"):
        raise ValueError("Replay only completed evaluation jobs")
    state_saved = FitState(**read_json(source / "fit_state.json"))
    state = FitState.fit(data, original_request["partition"]["fit_dates"], config["seasonality"])
    if state_saved.dictionary() != state.dictionary():
        raise ValueError("Fit-only transforms could not be rebuilt")
    if "reference_closes" not in context:
        entry = read_json(ROOT / config["data_manifest"])["assets"]["AAPL"]
        if sha(ROOT / entry["csv"]) != entry["csv_sha256"]:
            raise ValueError("Independent target source changed")
        frame = pd.read_csv(ROOT / entry["csv"], usecols=["timestamp", "close"])
        context["reference_closes"] = pd.Series(frame.close.to_numpy(), index=pd.DatetimeIndex(pd.to_datetime(frame.timestamp, utc=True)))
    with np.load(source / "predictions.npz", allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    origins, cal_origins = values["origins"], values["calibration_origins"]
    validate_prediction_grid(data, config, original_request, values)
    rebuilt_y = reference_targets(context["reference_closes"], values["timestamps"])
    rebuilt_cal_y = reference_targets(context["reference_closes"], data.timestamps[cal_origins].asi8)
    np.testing.assert_allclose(rebuilt_y, values["y"], rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(rebuilt_cal_y, values["calibration_y"], rtol=1e-8, atol=1e-14)
    recomputed = independent_scores(rebuilt_y, values["mean"], values["dates"])
    for key in recomputed:
        np.testing.assert_allclose(recomputed[key], original["score"][key], rtol=1e-8, atol=1e-10)
    name, params, panel = request["model"], request["params"], request["panel"]
    if name == "chronos2" and context["chronos"] is None:
        context["chronos"] = ChronosModel(config, context["device"])
    model = load_model(name, source, params, config, context["device"], context["chronos"])
    batch = config["search"]["chronos2"]["batch_size"] if name == "chronos2" else params.get("batch_size", 512)
    indices = replay_indices(len(origins), batch, name == "chronos2")
    cal_indices = replay_indices(len(cal_origins), batch, name == "chronos2")
    # First and last pretrained batches are called separately, as in original prediction.
    def replay(selected):
        if name != "chronos2":
            return raw_predictions(name, model, data, store, state, selected, params, panel, config, context["device"])
        return np.concatenate([raw_predictions(name, model, data, store, state, selected[start:start + batch], params, panel,
            config, context["device"]) for start in range(0, len(selected), batch)])
    raw, cal_raw = replay(origins[indices]), replay(cal_origins[cal_indices])
    np.testing.assert_allclose(raw, values["raw"][indices], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(cal_raw, values["calibration_raw"][cal_indices], rtol=1e-6, atol=1e-12)
    calibration = Calibration(**read_json(source / "calibration.json"))
    rebuilt_cal = Calibration.fit(rebuilt_cal_y, values["calibration_raw"], data.dates[cal_origins], calibration.levels)
    np.testing.assert_allclose(rebuilt_cal.log_factor, calibration.log_factor, rtol=1e-7, atol=1e-8)
    np.testing.assert_allclose(rebuilt_cal.residual_quantiles, calibration.residual_quantiles, rtol=1e-7, atol=1e-8)
    mean, quantiles = calibration.predict(raw)
    np.testing.assert_allclose(mean, values["mean"][indices], rtol=1e-6, atol=1e-12)
    np.testing.assert_allclose(quantiles, values["quantiles"][indices], rtol=1e-6, atol=1e-12)
    result = {"passed": True, "source_directory": request["source_directory"], "source_result_sha256": sha(source / "result.json"),
        "verification_scope": request.get("verification_scope", "smoke"), "round": request.get("verification_round"),
        "independent_labels": len(origins), "independent_calibration_labels": len(cal_origins),
        "replayed_origins": len(indices), "replayed_calibration_origins": len(cal_indices),
        "scope": "first_and_last_fixed_batches" if name == "chronos2" else "full",
        "max_raw_absolute_error": float(np.max(np.abs(raw - values["raw"][indices]))), "independent_scores": recomputed}
    write_json(Path(directory) / "verification.json", result)
    if name == "xpatch":
        import gc
        import torch
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def validate_prediction_grid(data, config, request, values):
    """A correct score on a truncated or substituted time grid is insufficient."""
    split, phase = request["partition"], request["phase"]
    stride = config["sampling"]["outer_stride_minutes"] if phase in ("outer", "confirmation", "delivery") else config["sampling"]["inner_validation_stride_minutes"]
    origins = data.origins(split["evaluation_dates"], stride=stride)
    calibration = data.origins(split["calibration_dates"], stride=config["sampling"]["calibration_stride_minutes"])
    if request.get("smoke_limit_prediction") is not None:
        if phase != "smoke":
            raise ValueError("Prediction limits cannot be applied to formal verification")
        origins, calibration = origins[:request["smoke_limit_prediction"]], calibration[:request["smoke_limit_prediction"]]
    for key, expected in (("origins", origins), ("calibration_origins", calibration),
            ("timestamps", data.timestamps[origins].asi8), ("dates", data.dates[origins])):
        if not np.array_equal(values[key], expected):
            raise ValueError(f"Stored {key} differs from the registered prediction grid")


SCOPES = ("development", "confirmation", "delivery")


def verification_directory(directory, round_number, scope):
    if round_number not in (1, 2) or scope not in SCOPES:
        raise ValueError("Verification needs a registered scope and round 1 or 2")
    parent = Path(directory) if scope == "development" else Path(directory) / ("confirmation" if scope == "confirmation" else "delivery_verification")
    correction = ROOT / "delivery/rv30_replay_lock.json"
    if scope == "development" and round_number == 1 and correction.exists():
        lock = read_json(correction)
        payload = lock["payload"]
        if stable_hash(payload) != lock["signature"]:
            raise ValueError("Replay correction signature changed")
        retry = payload["verification_retry"]
        if parent == ROOT / retry["run_directory"]:
            original = parent / "verify_1"
            replacement = parent / "verify_1_replay_correction"
            if (retry["scope"] != scope or retry["round"] != round_number or
                    retry["original_directory"] != str(original.relative_to(ROOT)) or
                    retry["replacement_directory"] != str(replacement.relative_to(ROOT)) or
                    sha(original / "summary.json") != retry["original_summary_sha256"]):
                raise ValueError("The preserved failed verification attempt changed")
            return replacement
    return parent / f"verify_{round_number}"


def expected_sources(config, run_directory, scope):
    """Derive exact model/seed coverage, rather than trusting a summary count."""
    directory = Path(run_directory)
    setup = read_json(directory / "setup.json")
    if setup["config"] != config:
        raise ValueError("Verification setup differs from the registered config")
    expected = {}

    def add(path, identity):
        relative = str(path.relative_to(ROOT))
        request = read_json(path / "request.json")
        result = receipt(path, stable_hash(request))
        if result is None or result["status"] != "complete":
            raise ValueError(f"Verification source is incomplete: {relative}")
        for key, value in {"signature": setup["signature"], "directory": relative, **identity}.items():
            if stable_hash(request.get(key)) != stable_hash(value):
                raise ValueError(f"Verification source {key} differs: {relative}")
        for key in ("signature", "model", "panel", "config_id", "params", "seed", "phase", "directory", "job_id"):
            if stable_hash(result.get(key)) != stable_hash(request[key]):
                raise ValueError(f"Source receipt has another {key}: {relative}")
        if not {"request.json", "predictions.npz", "fit_state.json", "calibration.json"}.issubset(result["files"]):
            raise ValueError("Source receipt omits required prediction artifacts")
        if relative in expected:
            raise ValueError("Duplicate verification source")
        expected[relative] = request

    if scope == "development":
        for split in setup["partitions"]:
            for panel in config["feature_sets"]:
                for name in config["models"]:
                    group = directory / f"{split['name']}_{panel}_{name}"
                    experiment = read_json(group / "experiment.json")
                    selection = read_json(group / "selection.json")
                    registered = {row["id"]: row["params"] for row in setup["search"]["models"][name]}
                    params = registered[selection["config_id"]]
                    seeds = config["search"]["seeds"] if name in config["search"]["stochastic_models"] else [2026]
                    paths = [group / "outer" / selection["config_id"] / split["name"] / f"s{seed}" for seed in seeds]
                    if experiment["status"] != "complete" or sorted(experiment["outer_jobs"]) != sorted(str(p.relative_to(ROOT)) for p in paths):
                        raise ValueError("Development sources omit or substitute a registered model/seed")
                    for path, seed in zip(paths, seeds):
                        identity = {"phase": "outer", "partition": split, "model": name, "panel": panel,
                            "config_id": selection["config_id"], "params": params, "seed": seed}
                        if name in ("xpatch", "lightgbm"):
                            identity["fixed_epochs" if name == "xpatch" else "fixed_iterations"] = selection["fixed_training_length"]
                        add(path, identity)
    elif scope == "confirmation":
        from .cache import load_data, prepare_data
        from .splits import partition
        development = read_json(directory / "evaluation.json")
        confirmation = read_json(directory / "confirmation/evaluation.json")
        allowed = development["passed_candidates"]
        if not allowed or confirmation.get("allowed_by_development") != allowed:
            raise ValueError("Historical confirmation scope is not admitted")
        cache, identity = prepare_data(config)
        data = load_data(cache, identity)
        start, end = config["periods"]["historical_confirmation_start"], config["periods"]["historical_confirmation_end"]
        split = partition("historical_confirmation", [str(d) for d in data.days if d < start],
            [str(d) for d in data.days if start <= d <= end], config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"]).dictionary()
        needed = sorted(set(allowed) | {key.split("/")[0] + "/har" for key in allowed})
        for key in needed:
            panel, name = key.split("/")
            selection = read_json(directory / f"2025Q2_{panel}_{name}/selection.json")
            params = next(row["params"] for row in setup["search"]["models"][name] if row["id"] == selection["config_id"])
            seeds = config["search"]["seeds"] if name in config["search"]["stochastic_models"] else [2026]
            for seed in seeds:
                source = directory / "confirmation" / panel / name / f"s{seed}"
                identity = {"phase": "confirmation", "partition": split, "model": name, "panel": panel,
                    "params": params, "config_id": selection["config_id"], "seed": seed}
                if name in ("xpatch", "lightgbm"):
                    identity["fixed_epochs" if name == "xpatch" else "fixed_iterations"] = selection["fixed_training_length"]
                add(source, identity)
        if sorted(confirmation["job_directories"]) != sorted(expected):
            raise ValueError("Confirmation sources omit or substitute a registered model/seed")
    elif scope == "delivery":
        from .cache import load_data, prepare_data
        from .splits import partition
        manifest = read_json(ROOT / "delivery/model.json")
        selection = read_json(directory / "2025Q2_aapl_har/selection.json")
        cache, identity = prepare_data(config)
        data = load_data(cache, identity)
        split = partition("delivery_baseline", data.days[:-1].tolist(), data.days[-1:].tolist(),
            config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"]).dictionary()
        path = directory / "delivery_baseline"
        if manifest["source_directory"] != str(path.relative_to(ROOT)) or manifest["source_result_sha256"] != sha(path / "result.json"):
            raise ValueError("Delivery model differs from its replay source")
        if receipt(path)["started_utc"] < manifest["prerequisites"]["not_before_utc"]:
            raise ValueError("Delivery fitting began before independent verification completed")
        add(path, {"phase": "delivery", "partition": split, "model": "har", "panel": "aapl",
            "config_id": selection["config_id"], "params": selection["params"], "seed": 2026})
    else:
        raise ValueError("Unknown verification scope")
    if not expected:
        raise ValueError("An empty verification scope cannot pass")
    return expected


def verification_requests(directory, round_number, scope, sources):
    destination = verification_directory(directory, round_number, scope)
    attempt = destination.name.removeprefix(f"verify_{round_number}")
    return [{**{key: source[key] for key in ("signature", "model", "panel", "config_id", "params", "seed", "partition")},
        "job_id": f"verify{round_number}{attempt}/{scope}/{source['job_id']}",
        "directory": str((destination / source["job_id"]).relative_to(ROOT)), "phase": "verify",
        "verification_scope": scope, "verification_round": round_number, "source_directory": path,
        "source_result_sha256": sha(ROOT / path / "result.json")} for path, source in sorted(sources.items())]


def verification_evidence(config, run_directory, round_number, scope="development"):
    sources = expected_sources(config, run_directory, scope)
    requests = verification_requests(run_directory, round_number, scope, sources)
    path = verification_directory(run_directory, round_number, scope) / "summary.json"
    summary = read_json(path)
    signature = read_json(Path(run_directory) / "setup.json")["signature"]
    source_digests = {r["source_directory"]: r["source_result_sha256"] for r in requests}
    if (summary.get("signature") != signature or summary.get("scope") != scope or summary.get("round") != round_number or
            summary.get("passed") is not True or summary.get("jobs") != len(requests) or summary.get("sources") != source_digests or
            sorted(summary.get("job_directories", [])) != sorted(r["directory"] for r in requests)):
        raise ValueError(f"Verification summary coverage or identity differs: {scope} round {round_number}")
    finished = []
    for request in requests:
        result = receipt(ROOT / request["directory"], stable_hash(request))
        if result is None or result["status"] != "complete":
            raise ValueError("Missing or failed independent verification receipt")
        finished.append(result["finished_utc"])
        if result["started_utc"] < receipt(ROOT / request["source_directory"])["finished_utc"]:
            raise ValueError("Replay began before its source was complete")
        verification = result["verification"]
        if (verification.get("passed") is not True or verification.get("source_directory") != request["source_directory"] or
                verification.get("source_result_sha256") != request["source_result_sha256"] or
                verification.get("verification_scope") != scope or verification.get("round") != round_number or
                read_json(ROOT / request["directory"] / "request.json") != request or
                "verification.json" not in result["files"] or
                read_json(ROOT / request["directory"] / "verification.json") != verification):
            raise ValueError("Independent verification receipt differs from its source/scope/round")
    statistics_path = verification_directory(run_directory, round_number, scope) / "statistics.json"
    statistics = read_json(statistics_path)
    evaluation_path = Path(run_directory) / ("confirmation/evaluation.json" if scope == "confirmation" else "evaluation.json")
    evaluation = None if scope == "delivery" else {"path": str(evaluation_path.relative_to(ROOT)), "sha256": sha(evaluation_path)}
    if (summary.get("statistics_file") != str(statistics_path.relative_to(ROOT)) or
            summary.get("statistics_sha256") != sha(statistics_path) or statistics.get("passed") is not True or
            statistics.get("signature") != signature or statistics.get("scope") != scope or statistics.get("round") != round_number or
            statistics.get("sources") != source_digests or statistics.get("evaluation") != evaluation):
        raise ValueError("Independent aggregate statistics identity or source coverage differs")
    if summary.get("completed_utc") != max([*finished, statistics["completed_utc"]]):
        raise ValueError("Independent verification completion time differs from its receipts")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def require_verifications(config, run_directory, scope="development"):
    return {str(number): verification_evidence(config, run_directory, number, scope) for number in (1, 2)}


def reference_daily(values, dates):
    series = pd.Series(np.asarray(values, dtype=float), index=np.asarray(dates))
    grouped = series.groupby(level=0, sort=True).mean()
    return grouped.index.to_numpy(), grouped.to_numpy()


def reference_loss(y, p):
    return np.maximum(y, 1e-12) / np.maximum(p, 1e-12) + np.log(np.maximum(p, 1e-12)) - np.log(np.maximum(y, 1e-12)) - 1


def reference_mz(y, p, dates, lags=5):
    days, observed = reference_daily(y, dates)
    _, predicted = reference_daily(p, dates)
    scale = max(float(np.mean(observed)), 1e-12)
    design = np.column_stack((np.ones(len(days)), predicted / scale))
    base = {"unit": "daily_mean_of_overlapping_intraday_RV_forecasts", "hac_lags": lags, "sessions": len(days)}
    if np.linalg.matrix_rank(design) < 2:
        return {**base, "identified": False, "reason": "constant_or_collinear_daily_forecast", "intercept": None,
            "slope": None, "hac_covariance": None}
    beta = np.linalg.lstsq(design, observed / scale, rcond=None)[0]
    scores = design * (observed / scale - design @ beta)[:, None]
    # Accumulate score outer products explicitly, independently of metrics.hac_meat.
    meat = sum(np.outer(row, row) for row in scores)
    for lag in range(1, min(lags, len(days) - 1) + 1):
        cross = sum(np.outer(scores[i], scores[i - lag]) for i in range(lag, len(days)))
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.pinv(design.T @ design)
    conversion = np.diag((scale, 1.))
    covariance = conversion @ bread @ meat @ bread @ conversion * len(days) / (len(days) - 2)
    return {**base, "identified": True, "intercept": float(beta[0] * scale), "slope": float(beta[1]),
        "hac_covariance": covariance.tolist()}


def reference_score(values, mask=None):
    if mask is None:
        mask = np.ones(len(values["y"]), dtype=bool)
    y, p, dates = (values[key][mask] for key in ("y", "mean", "dates"))
    days, losses = reference_daily(reference_loss(y, p), dates)
    result = {"qlike": float(np.mean(losses)),
        "log_rv_mse": float(np.mean(reference_daily((np.log(np.maximum(y, 1e-12)) - np.log(p)) ** 2, dates)[1])),
        "origins": len(y), "sessions": len(days)}
    levels = list(values["levels"])
    quantiles = values["quantiles"][mask]
    for low, high, name in ((.1, .9, "80"), (.05, .95, "90")):
        lower, upper = quantiles[:, levels.index(low)], quantiles[:, levels.index(high)]
        result["coverage_" + name] = float(np.mean(reference_daily((y >= lower) & (y <= upper), dates)[1]))
        result["width_" + name] = float(np.mean(reference_daily(upper - lower, dates)[1]))
    return result


def reference_segments(dates, confirmation):
    return np.asarray([("2025H2" if day < "2026-01-01" else "2026H1") if confirmation else
        (day[:4] if day[:4] != "2025" else "2025H1") for day in dates])


def reference_description(values, confirmation=False):
    result = reference_score(values)
    result["mincer_zarnowitz"] = reference_mz(values["y"], values["mean"], values["dates"])
    clock = pd.DatetimeIndex(values["timestamps"], tz="UTC").tz_convert("America/New_York")
    minutes = clock.hour * 60 + clock.minute
    strata = np.asarray(["open" if m <= 630 else "midday" if m <= 840 else "late" for m in minutes])
    quarters = np.asarray([f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}" for day in values["dates"]])
    for key, labels in (("clock", strata), ("quarter", quarters), ("year_segment", reference_segments(values["dates"], confirmation))):
        result[key] = {label: reference_score(values, labels == label) for label in sorted(set(labels))}
    result["seeds"] = {}
    for seed, prediction in values["seeds"].items():
        scores = independent_scores(values["y"], prediction, values["dates"])
        result["seeds"][seed] = {**scores, "sessions": result["sessions"], "origins": result["origins"]}
    return result


def reference_comparison(candidate, baseline, settings, confirmation=False):
    from math import erfc, sqrt
    for key in ("origins", "timestamps", "dates", "y"):
        if not np.array_equal(candidate[key], baseline[key]):
            raise ValueError("Independent comparison found unpaired " + key)
    dates, y = candidate["dates"], candidate["y"]
    baseline_loss = reference_loss(y, baseline["mean"])
    days, difference = reference_daily(reference_loss(y, candidate["mean"]) - baseline_loss, dates)
    n, average, lags = len(days), float(np.mean(difference)), settings["dm_hac_lags"]
    centered = difference - average
    variance = sum(float(x * x) for x in centered)
    for lag in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - lag / (lags + 1)) * float(np.dot(centered[lag:], centered[:-lag]))
    variance = max(variance / n ** 2, 0.)
    zero = variance <= 1e-30
    statistic = None if zero else average / sqrt(variance)
    dm = {"mean_difference": average, "statistic": statistic, "p_less": (0. if average < 0 else 1.) if zero else erfc(-statistic / sqrt(2)) / 2,
        "hac_lags": lags, "zero_variance": zero}
    bootstrap = []
    repeats, seed = settings["bootstrap_repeats"], settings["bootstrap_seed"]
    for length in settings["block_lengths_sessions"]:
        starts = np.random.default_rng(seed).integers(0, n, (repeats, (n + length - 1) // length))
        # Circular-block sums from a cumulative sum avoid sharing the evaluator's
        # sampled-index expansion and batch implementation.
        extended = np.r_[difference, np.resize(difference, length)]
        cumulative = np.r_[0., np.cumsum(extended)]
        whole, tail = divmod(n, length)
        totals = (cumulative[starts[:, :whole] + length] - cumulative[starts[:, :whole]]).sum(axis=1)
        if tail:
            totals += cumulative[starts[:, -1] + tail] - cumulative[starts[:, -1]]
        means = totals / n
        bootstrap.append({"block_sessions": length, "repeats": repeats, "seed": seed, "mean_difference": average,
            "ci95": np.quantile(means, [.025, .975]).tolist(),
            "p_less_centered_null": float((1 + np.count_nonzero(means - average <= average)) / (repeats + 1))})
    quarters = np.asarray([f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}" for day in days])
    segments = reference_segments(days, confirmation)
    seed_segments = reference_segments(dates, confirmation)
    seed_results = {}
    for seed, prediction in candidate["seeds"].items():
        losses = reference_loss(y, prediction) - baseline_loss
        seed_results[seed] = {"pooled": float(np.mean(reference_daily(losses, dates)[1]))}
        for segment in sorted(set(seed_segments)):
            mask = seed_segments == segment
            seed_results[seed][segment] = float(np.mean(reference_daily(losses[mask], dates[mask])[1]))
    return {"mean_difference": average, "sessions": n, "dm": dm, "bootstraps": bootstrap,
        "p_conservative": max(dm["p_less"], *(item["p_less_centered_null"] for item in bootstrap)),
        "quarter_differences": {q: float(np.mean(difference[quarters == q])) for q in sorted(set(quarters))},
        "year_segment_differences": {s: float(np.mean(difference[segments == s])) for s in sorted(set(segments))},
        "seed_differences": seed_results}


def reference_gates(comparisons, config, confirmation=False):
    settings = config["statistics"]
    family = [f"{panel}/{model}" for panel in config["feature_sets"] for model in settings["candidate_models"]]
    adjusted, running = {}, 0.
    ordered = sorted(enumerate(family), key=lambda item: comparisons.get(item[1], {}).get("p_conservative", 1.))
    for rank, (_, key) in enumerate(ordered):
        running = min(1., max(running, (len(family) - rank) * comparisons.get(key, {}).get("p_conservative", 1.)))
        adjusted[key] = running
    for key in family:
        row = comparisons.setdefault(key, {"status": "missing_or_failed", "p_conservative": 1.})
        row["holm_p"] = adjusted[key]
        if "mean_difference" not in row:
            row.update(passed=False, gate_checks={"complete_paired_predictions": False})
            continue
        required = settings["confirmation_required_segments"] if confirmation else settings["required_year_segments"]
        minimum = settings["confirmation_minimum_improving_quarters"] if confirmation else settings["minimum_improving_quarters"]
        checks = {"qlike_improved": row["mean_difference"] < 0, "holm_significant": adjusted[key] < settings["alpha"],
            "both_ci_upper_negative": max(item["ci95"][1] for item in row["bootstraps"]) < 0,
            "quarter_stability": len([v for v in row["quarter_differences"].values() if v < 0]) >= minimum,
            "all_planned_quarters_present": len(row["quarter_differences"]) == (4 if confirmation else 10),
            "year_segment_stability": all(row["year_segment_differences"].get(s, float("inf")) < 0 for s in required),
            "all_seed_pooled_and_segments_improve": all(result.get(s, float("inf")) < 0 for result in row["seed_differences"].values()
                for s in ["pooled", *required])}
        row.update(passed=all(checks.values()), gate_checks=checks)
    return comparisons


def reference_bundles(sources):
    grouped = {}
    for path, request in sources.items():
        key = (request["panel"] + "/" + request["model"], request["partition"]["name"])
        grouped.setdefault(key, []).append((request["seed"], path))
    pieces = {}
    for (key, fold), entries in grouped.items():
        predictions, calibration_means, seeds = [], [], []
        for seed, path in sorted(entries):
            with np.load(ROOT / path / "predictions.npz", allow_pickle=False) as stored:
                values = {name: stored[name] for name in stored.files}
            cal = read_json(ROOT / path / "calibration.json")
            calibration_means.append(np.maximum(values["calibration_raw"] * np.exp(cal["log_factor"]), 1e-12))
            predictions.append(values)
            seeds.append(str(seed))
        first = predictions[0]
        for values in predictions[1:]:
            for name in ("origins", "dates", "timestamps", "y", "calibration_origins", "calibration_y"):
                if not np.array_equal(first[name], values[name]):
                    raise ValueError("Independent ensemble has unpaired seed " + name)
        mean = np.mean([values["mean"] for values in predictions], axis=0)
        if len(predictions) == 1:
            quantiles = first["quantiles"]
        else:
            residuals = np.log(np.maximum(first["calibration_y"], 1e-12)) - np.log(np.mean(calibration_means, axis=0))
            quantiles = mean[:, None] * np.exp(np.quantile(residuals, cal["levels"]))
        pieces.setdefault(key, []).append({**{name: first[name] for name in ("origins", "dates", "timestamps", "y")},
            "mean": mean, "quantiles": quantiles, "levels": cal["levels"],
            "seeds": {seed: values["mean"] for seed, values in zip(seeds, predictions)}})
    merged = {}
    for key, parts in pieces.items():
        parts.sort(key=lambda part: int(part["timestamps"][0]))
        if any(set(part["seeds"]) != set(parts[0]["seeds"]) for part in parts):
            raise ValueError("Independent aggregate found a changing seed set")
        values = {name: np.concatenate([part[name] for part in parts]) for name in ("origins", "dates", "timestamps", "y", "mean", "quantiles")}
        if np.any(np.diff(values["timestamps"]) <= 0):
            raise ValueError("Independent aggregate has duplicate or unordered timestamps")
        values.update(levels=parts[0]["levels"], seeds={seed: np.concatenate([part["seeds"][seed] for part in parts]) for seed in parts[0]["seeds"]})
        merged[key] = values
    return merged


def assert_reference_equal(reference, reported, path="statistics"):
    if isinstance(reference, dict):
        if not isinstance(reported, dict) or set(reference) != set(reported):
            raise ValueError("Independent statistic keys differ: " + path)
        for key, value in reference.items():
            assert_reference_equal(value, reported[key], path + "/" + key)
    elif isinstance(reference, (list, tuple)):
        if not isinstance(reported, (list, tuple)) or len(reference) != len(reported):
            raise ValueError("Independent statistic list differs: " + path)
        for index, value in enumerate(reference):
            assert_reference_equal(value, reported[index], path + f"/{index}")
    elif isinstance(reference, bool) or reference is None or isinstance(reference, str):
        if reference != reported or type(reference) is not type(reported):
            raise ValueError("Independent statistic or gate differs: " + path)
    else:
        try:
            np.testing.assert_allclose(reference, reported, rtol=1e-7, atol=1e-12)
        except (AssertionError, TypeError) as error:
            raise ValueError("Independent statistic differs: " + path) from error


def verify_aggregate_statistics(config, directory, round_number, scope, sources):
    from datetime import datetime, timezone
    values = reference_bundles(sources)
    descriptions = {key: reference_description(item, scope == "confirmation") for key, item in values.items()}
    comparisons = {}
    evidence = None
    if scope != "delivery":
        evaluation_path = Path(directory) / ("confirmation/evaluation.json" if scope == "confirmation" else "evaluation.json")
        evaluation = read_json(evaluation_path)
        for key, item in values.items():
            if key.split("/")[1] in config["statistics"]["candidate_models"]:
                comparisons[key] = reference_comparison(item, values[key.split("/")[0] + "/har"], config["statistics"], scope == "confirmation")
        comparisons = reference_gates(comparisons, config, scope == "confirmation")
        assert_reference_equal(descriptions, evaluation["descriptions"], "descriptions")
        assert_reference_equal(comparisons, evaluation["comparisons"], "comparisons")
        passed = [f"{panel}/{name}" for panel in config["feature_sets"] for name in config["statistics"]["candidate_models"]
            if comparisons[f"{panel}/{name}"]["passed"]]
        assert_reference_equal(passed, evaluation["passed_candidates"], "passed_candidates")
        evidence = {"path": str(evaluation_path.relative_to(ROOT)), "sha256": sha(evaluation_path)}
    else:
        for path, request in sources.items():
            reported = receipt(ROOT / path)["score"]
            key = request["panel"] + "/" + request["model"]
            assert_reference_equal(reference_score(values[key]), reported, "delivery_model_score")
    return {"passed": True, "signature": read_json(Path(directory) / "setup.json")["signature"],
        "scope": scope, "round": round_number, "evaluation": evidence, "completed_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {path: sha(ROOT / path / "result.json") for path in sorted(sources)},
        "descriptions": descriptions, "comparisons": comparisons,
        "implementation": "independent_pandas_daily_losses_scalar_HAC_circular_block_sums_and_stepdown_Holm"}


def verify_run(config, run_directory, round_number, scope="development"):
    from .cache import prepare_data
    from .provenance import verify_freeze
    from .scheduler import Scheduler
    freeze = verify_freeze(config)
    directory = Path(run_directory)
    setup = read_json(directory / "setup.json")
    if setup["signature"] != freeze["signature"]:
        raise ValueError("Verification run belongs to another freeze")
    destination = verification_directory(directory, round_number, scope)
    if round_number == 2:
        verification_evidence(config, directory, 1, scope)
    # Preserve the exact hashes used as prerequisites by downstream stages.
    if (destination / "summary.json").exists():
        verification_evidence(config, directory, round_number, scope)
        return read_json(destination / "summary.json")
    cache, identity = prepare_data(config)
    sources = expected_sources(config, directory, scope)
    requests = verification_requests(directory, round_number, scope, sources)
    with Scheduler(config, cache, identity, destination) as scheduler:
        results = scheduler.map(requests)
    summary = {"signature": freeze["signature"], "scope": scope, "round": round_number,
        "passed": all(r["status"] == "complete" and r["verification"]["passed"] for r in results),
        "jobs": len(results), "job_directories": [r["directory"] for r in results],
        "completed_utc": max(r["finished_utc"] for r in results),
        "sources": {r["source_directory"]: r["source_result_sha256"] for r in requests},
        "pretrained_scope": "first_and_last_fixed_batches_of_each_calibration_and_evaluation_job",
        "supervised_scope": "full_calibration_and_evaluation_predictions"}
    if summary["passed"]:
        try:
            statistics = verify_aggregate_statistics(config, directory, round_number, scope, sources)
            write_json(destination / "statistics.json", statistics)
            summary["statistics_sha256"] = sha(destination / "statistics.json")
            summary["statistics_file"] = str((destination / "statistics.json").relative_to(ROOT))
            summary["completed_utc"] = max(summary["completed_utc"], statistics["completed_utc"])
        except (ValueError, KeyError, AssertionError) as error:
            write_json(destination / "statistics_failure.json", {"passed": False, "error": str(error)})
            summary.update(passed=False, aggregate_statistics_error=str(error))
    write_json(destination / "summary.json", summary)
    if summary["passed"]:
        verification_evidence(config, directory, round_number, scope)
    return summary
