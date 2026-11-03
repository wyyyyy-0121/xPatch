"""Predeclared equal-day comparisons from immutable outer prediction files."""
from pathlib import Path

import numpy as np
import pandas as pd

from .calibration import Calibration
from .io import ROOT, read_json, sha, write_json
from .jobs import receipt
from .metrics import block_comparison, daily_mean, dm_test, holm, mincer_zarnowitz, qlike, score


def year_segment(dates, confirmation=False):
    if confirmation:
        return np.where(np.asarray(dates) < "2026-01-01", "2025H2", "2026H1")
    return np.array([day[:4] if day[:4] != "2025" else "2025H1" for day in dates])


def bundle(job_directories, expected_seeds=None):
    """Arithmetic mean of calibrated RV means; empirical ensemble intervals."""
    jobs = [receipt(ROOT / directory) for directory in job_directories]
    if not jobs or any(job is None or job["status"] != "complete" for job in jobs):
        raise ValueError("Missing/failed outer seed; candidate comparison remains in family")
    if expected_seeds is not None and sorted(j["seed"] for j in jobs) != sorted(expected_seeds):
        raise ValueError("Incomplete or duplicate seed set")
    predictions, cal_means, seeds = [], [], []
    for job in sorted(jobs, key=lambda j: j["seed"]):
        directory = ROOT / job["directory"]
        with np.load(directory / "predictions.npz", allow_pickle=False) as saved:
            values = {key: saved[key] for key in saved.files}
        cal = Calibration(**read_json(directory / "calibration.json"))
        cal_means.append(cal.predict(values["calibration_raw"])[0])
        seeds.append(job["seed"])
        predictions.append(values)
    first = predictions[0]
    for other in predictions[1:]:
        for key in ("origins", "y", "timestamps", "dates", "calibration_origins", "calibration_y"):
            if not np.array_equal(first[key], other[key]):
                raise ValueError("Seed predictions use different paired rows/targets")
    mean = np.mean([values["mean"] for values in predictions], axis=0)
    levels = read_json(ROOT / jobs[0]["directory"] / "calibration.json")["levels"]
    if len(predictions) == 1:
        quantiles = first["quantiles"]
    else:
        cal_mean = np.mean(cal_means, axis=0)
        residual = np.log(np.maximum(first["calibration_y"], 1e-12)) - np.log(cal_mean)
        quantiles = mean[:, None] * np.exp(np.quantile(residual, levels))
    return {"origins": first["origins"], "dates": first["dates"], "timestamps": first["timestamps"], "y": first["y"],
        "mean": mean, "quantiles": quantiles, "levels": levels,
        "seeds": {str(seed): values["mean"] for seed, values in zip(seeds, predictions)}}


def concatenate(pieces):
    if not pieces:
        raise ValueError("No predictions to concatenate")
    keys = ("origins", "dates", "timestamps", "y", "mean", "quantiles")
    merged = {key: np.concatenate([p[key] for p in pieces]) for key in keys}
    if np.any(np.diff(merged["timestamps"]) <= 0):
        raise ValueError("Overlapping/unordered outer predictions")
    merged["levels"] = pieces[0]["levels"]
    if any(set(p["seeds"]) != set(pieces[0]["seeds"]) for p in pieces):
        raise ValueError("Outer seed set changed across quarters")
    merged["seeds"] = {seed: np.concatenate([p["seeds"][seed] for p in pieces]) for seed in pieces[0]["seeds"]}
    return merged


def describe(values, confirmation=False):
    y, p, dates = values["y"], values["mean"], values["dates"]
    overall = score(y, p, dates, values["quantiles"], values["levels"])
    overall["mincer_zarnowitz"] = mincer_zarnowitz(y, p, dates)
    clock = pd.DatetimeIndex(values["timestamps"], tz="UTC").tz_convert("America/New_York")
    minute = np.asarray(clock.hour * 60 + clock.minute)
    strata = np.where(minute <= 630, "open", np.where(minute <= 840, "midday", "late"))
    quarter = np.array([f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}" for day in dates])
    segments = year_segment(dates, confirmation)
    for name, groups in (("clock", strata), ("quarter", quarter), ("year_segment", segments)):
        overall[name] = {}
        for group in sorted(set(groups)):
            mask = groups == group
            overall[name][group] = score(y[mask], p[mask], dates[mask], values["quantiles"][mask], values["levels"])
    overall["seeds"] = {seed: score(y, pred, dates) for seed, pred in values["seeds"].items()}
    return overall


def compare(candidate, baseline, config, confirmation=False):
    for key in ("origins", "timestamps", "dates", "y"):
        if not np.array_equal(candidate[key], baseline[key]):
            raise ValueError(f"Candidate and HAR do not use paired {key}")
    dates, y = candidate["dates"], candidate["y"]
    days, difference = daily_mean(qlike(y, candidate["mean"]) - qlike(y, baseline["mean"]), dates)
    settings = config["statistics"]
    dm = dm_test(difference, settings["dm_hac_lags"])
    bootstraps = [block_comparison(difference, length, settings["bootstrap_repeats"], settings["bootstrap_seed"])
        for length in settings["block_lengths_sessions"]]
    quarter = np.array([f"{day[:4]}Q{(int(day[5:7]) - 1) // 3 + 1}" for day in days])
    segments = year_segment(days, confirmation)
    quarter_differences = {group: float(difference[quarter == group].mean()) for group in sorted(set(quarter))}
    yearly_differences = {group: float(difference[segments == group].mean()) for group in sorted(set(segments))}
    seed_differences = {}
    seed_segments = year_segment(dates, confirmation)
    for seed, predicted in candidate["seeds"].items():
        losses = qlike(y, predicted) - qlike(y, baseline["mean"])
        seed_differences[seed] = {"pooled": float(daily_mean(losses, dates)[1].mean()),
            **{group: float(daily_mean(losses[seed_segments == group], dates[seed_segments == group])[1].mean())
                for group in sorted(set(seed_segments))}}
    return {"mean_difference": float(difference.mean()), "sessions": len(days), "dm": dm, "bootstraps": bootstraps,
        "p_conservative": max(dm["p_less"], *(r["p_less_centered_null"] for r in bootstraps)),
        "quarter_differences": quarter_differences, "year_segment_differences": yearly_differences,
        "seed_differences": seed_differences}


def gates(comparisons, config, confirmation=False):
    settings = config["statistics"]
    family = [f"{panel}/{model}" for panel in config["feature_sets"] for model in settings["candidate_models"]]
    adjusted = holm([comparisons.get(key, {}).get("p_conservative", 1.) for key in family])
    for key, pvalue in zip(family, adjusted):
        row = comparisons.setdefault(key, {"status": "missing_or_failed", "p_conservative": 1.})
        row["holm_p"] = float(pvalue)
        if "mean_difference" not in row:
            row.update(passed=False, gate_checks={"complete_paired_predictions": False})
            continue
        minimum = settings["confirmation_minimum_improving_quarters"] if confirmation else settings["minimum_improving_quarters"]
        required = settings["confirmation_required_segments"] if confirmation else settings["required_year_segments"]
        checks = {"qlike_improved": row["mean_difference"] < 0,
            "holm_significant": pvalue < settings["alpha"],
            "both_ci_upper_negative": all(r["ci95"][1] < 0 for r in row["bootstraps"]),
            "quarter_stability": sum(v < 0 for v in row["quarter_differences"].values()) >= minimum,
            "all_planned_quarters_present": len(row["quarter_differences"]) == (4 if confirmation else 10),
            "year_segment_stability": all(row["year_segment_differences"].get(year, float("inf")) < 0 for year in required),
            "all_seed_pooled_and_segments_improve": all(all(values.get(segment, float("inf")) < 0 for segment in ["pooled", *required])
                for values in row["seed_differences"].values())}
        row.update(passed=all(checks.values()), gate_checks={key: bool(value) for key, value in checks.items()})
    return comparisons


def evaluate(run_directory, config):
    directory = Path(run_directory)
    status = read_json(directory / "run.json")
    if not status["search_finished"]:
        raise ValueError("All model groups must finish before aggregate evaluation")
    setup = read_json(directory / "setup.json")
    merged, descriptions, errors, evidence = {}, {}, {}, {}
    for panel in config["feature_sets"]:
        for name in config["models"]:
            key = f"{panel}/{name}"
            pieces = []
            try:
                for split in setup["partitions"]:
                    group = directory / f"{split['name']}_{panel}_{name}"
                    entry = read_json(group / "experiment.json")
                    if entry["status"] != "complete":
                        raise ValueError(entry["status"])
                    seeds = config["search"]["seeds"] if name in config["search"]["stochastic_models"] else [2026]
                    pieces.append(bundle(entry["outer_jobs"], seeds))
                    evidence[str((group / "experiment.json").relative_to(ROOT))] = sha(group / "experiment.json")
                merged[key] = concatenate(pieces)
                descriptions[key] = describe(merged[key])
            except (ValueError, FileNotFoundError) as error:
                errors[key] = str(error)
    comparisons = {}
    for panel in config["feature_sets"]:
        for model in config["statistics"]["candidate_models"]:
            key, base = f"{panel}/{model}", f"{panel}/har"
            if key in merged and base in merged:
                comparisons[key] = compare(merged[key], merged[base], config)
    comparisons = gates(comparisons, config)
    results = {"signature": status["signature"], "scope": "retrospective_development_2023Q1_2025Q2",
        "descriptions": descriptions, "comparisons": comparisons, "errors": errors,
        "passed_candidates": [key for key, value in comparisons.items() if value["passed"]],
        "complete_model_panel_results": len(descriptions), "planned_model_panel_results": 12,
        "family_size": 10, "evidence": evidence,
        "prospective_eligible": False, "historical_exposure": True,
        "ensemble_intervals": "calibration_log_residual_quantiles_around_mean_of_calibrated_seed_RV_means"}
    write_json(directory / "evaluation.json", results)
    rows = []
    for key, values in merged.items():
        days, losses = daily_mean(qlike(values["y"], values["mean"]), values["dates"])
        _, counts = np.unique(values["dates"], return_counts=True)
        rows.extend({"model_panel": key, "date": str(day), "origins": int(count), "qlike": float(loss)}
            for day, count, loss in zip(days, counts, losses))
    pd.DataFrame(rows).to_csv(directory / "daily_losses.csv", index=False)
    return results
