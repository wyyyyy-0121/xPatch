"""Frozen descriptive analysis; reads only after 24 fits and scores complete."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

BASE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def metrics(prediction, truth):
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(truth, dtype=np.float64)
    return {"endpoint_mse": float(np.mean(error[:, -1] ** 2)),
            "path_mse": float(np.mean(error ** 2)),
            "endpoint_mae": float(np.mean(np.abs(error[:, -1])))}


def paired_interval(ref, new, groups, length):
    # Complete sessions/months are resampling units, never individual rows.
    names = np.unique(groups)
    if len(names) < 2 * length:
        return {"status": "insufficient_groups", "groups": len(names), "block_length": length}
    sums = np.array([[np.sum(ref[groups == g]), np.sum(new[groups == g]),
                      np.count_nonzero(groups == g)] for g in names], dtype=np.float64)
    generator = np.random.default_rng(2026)
    draws = []
    for _ in range(5000):
        starts = generator.integers(0, len(names) - length + 1,
                                    size=(len(names) + length - 1) // length)
        indices = np.concatenate([np.arange(start, start + length) for start in starts])[:len(names)]
        total = sums[indices].sum(axis=0)
        draws.append(100.0 * (total[0] - total[1]) / total[0] if total[0] > 0 else np.nan)
    if not np.isfinite(draws).all():
        return {"status": "undefined_reference_loss", "groups": len(names), "block_length": length}
    return {"status": "descriptive_exposed_development_only", "groups": len(names),
            "block_length": length, "draws": 5000, "seed": 2026,
            "improvement_percent_95_interval": np.quantile(draws, [.025, .975]).tolist(),
            "multiplicity_adjusted": False, "numerical_ranking_certificate": False}


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    assert set(os.sched_getaffinity(0)) <= {24, 26}
    freeze = read(BASE / "MODEL_FREEZE.json")
    assert len(freeze["entries"]) == 24
    reconstruction = read(BASE / "independent/RECONSTRUCTION.json")
    assert reconstruction["status"] == "passed"
    assert len(reconstruction["jobs"]) == 24
    states, sources = {}, {}
    independent = {item["id"]: item for item in reconstruction["jobs"]}
    for entry in freeze["entries"]:
        name = entry["id"]
        folder = BASE / "scores" / name
        result = read(folder / "RESULT.json")
        assert result["status"] == "final20_stop_only_development_scored"
        assert result["model_freeze_sha256"] == sha(BASE / "MODEL_FREEZE.json")
        assert sha(folder / "stop_predictions.npz") == result["predictions_sha256"]
        assert independent[name]["qualified"] is True
        ipath = Path(independent[name]["arrays"])
        assert sha(ipath) == independent[name]["arrays_sha256"]
        with np.load(folder / "stop_predictions.npz", allow_pickle=False) as saved:
            p, y, origins = saved["prediction"], saved["actual"], saved["origins"]
        with np.load(ipath, allow_pickle=False) as other:
            assert np.array_equal(origins, other["origins"])
            assert np.array_equal(y, other["rawtruth"])
            groups, baseline = other["groups"].astype(str), other["baseline"]
        assert np.isfinite(p).all() and np.isfinite(y).all()
        values = metrics(p, y)
        for key, value in values.items():
            assert abs(value - result[key]) / max(1., abs(value), abs(result[key])) <= 1e-7
        cfg = result["cfg"]
        key = name.rsplit("_", 1)[0]
        state = states.setdefault(key, {"profile": cfg["profile"], "target": cfg["target"],
                                       "fold": cfg["fold"], "seed": cfg["seed"], "arms": {}})
        state["arms"][cfg["arm"]] = {"metrics": values, "prediction": p, "truth": y,
                                          "origins": origins, "groups": groups, "baseline": baseline}
        sources[name] = {"result_sha256": sha(folder / "RESULT.json"),
                         "predictions_sha256": result["predictions_sha256"],
                         "independent_arrays_sha256": independent[name]["arrays_sha256"]}
    assert len(states) == 8
    for state in states.values():
        arms = state["arms"]
        assert set(arms) == {"A", "B", "C"}
        for other in ("B", "C"):
            for array in ("truth", "origins", "groups", "baseline"):
                assert np.array_equal(arms["A"][array], arms[other][array])
        state["origins"] = len(arms["A"]["truth"])
        state["distinct_groups"] = len(np.unique(arms["A"]["groups"]))
        state["baseline"] = metrics(arms["A"]["baseline"], arms["A"]["truth"])
        state["contrasts"] = {}
        for new, ref in (("B", "A"), ("B", "C"), ("C", "A")):
            contrast = {}
            for loss in ("endpoint", "path"):
                errors = {arm: (arms[arm]["prediction"] - arms[arm]["truth"]) ** 2 for arm in (new, ref)}
                losses = {arm: value[:, -1] if loss == "endpoint" else value.mean(axis=1)
                          for arm, value in errors.items()}
                refmean, newmean = losses[ref].mean(), losses[new].mean()
                blocks = (2, 5) if state["profile"] == "hourly" else (3, 6)
                contrast[loss] = {"reference_mean": float(refmean), "new_mean": float(newmean),
                                  "improvement_percent": float(100 * (refmean - newmean) / refmean),
                                  "intervals": [paired_interval(losses[ref], losses[new],
                                                                arms["A"]["groups"], n) for n in blocks]}
            state["contrasts"][f"{new}_vs_{ref}"] = contrast
        checks = []
        for ref in ("A", "C"):
            b, r = arms["B"]["metrics"], arms[ref]["metrics"]
            endpoint_margin = 1e-10 * max(abs(b["endpoint_mse"]), abs(r["endpoint_mse"]), 1e-30)
            path_margin = 1e-10 * max(abs(b["path_mse"]), abs(r["path_mse"]), 1e-30)
            checks.append({"reference": ref, "endpoint_lower": b["endpoint_mse"] < r["endpoint_mse"] - endpoint_margin,
                           "path_nonhigher": b["path_mse"] <= r["path_mse"] + path_margin})
        state["direction_checks"] = checks
        state["limited_direction_pass"] = all(v["endpoint_lower"] and v["path_nonhigher"] for v in checks)
        for arm in arms.values():
            for array in ("prediction", "truth", "origins", "groups", "baseline"):
                del arm[array]
            arm["endpoint_improvement_vs_baseline_percent"] = 100 * (
                state["baseline"]["endpoint_mse"] - arm["metrics"]["endpoint_mse"]) / state["baseline"]["endpoint_mse"]
    result = {"status": "complete_descriptive_R2_analysis", "states": states,
              "states_passing_limited_direction": sum(s["limited_direction_pass"] for s in states.values()),
              "broad_limited_direction_pass": all(s["limited_direction_pass"] for s in states.values()),
              "all_states_independently_qualified": True, "source_sha256": sources,
              "independent_reconstruction_sha256": sha(BASE / "independent/RECONSTRUCTION.json"),
              "model_freeze_sha256": sha(BASE / "MODEL_FREEZE.json"),
              "scientific_rules_sha256": sha(BASE / "SCIENTIFIC_RULES.json"),
              "no_price_return_or_hour_quarter_loss_pooling": True,
              "historical_stop_exposed_development": True, "new_forecasting_candidate_admitted": False,
              "primary_H60_H8_success_achieved": False, "independent_market_confirmation": False,
              "Q3_read": False, "research_goal_achieved": False}
    path = BASE / "analysis/RESULTS.json"
    if path.exists():
        assert read(path) == result
    else:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        os.replace(temporary, path)
    print(json.dumps({"states": len(states), "direction_passed": result["states_passing_limited_direction"],
                      "broad_direction_pass": result["broad_limited_direction_pass"]}))


if __name__ == "__main__":
    main()
