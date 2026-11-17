"""Separate mean seed loss from loss of mean predictions using existing arrays.

This is post-exposure descriptive reporting of an already completed monthly
diagnostic. It trains nothing and does not change the twenty-task core study.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path("/home/ayaka/wslcode/aapl-research")
OUT = ROOT / "outputs/multiscale"
COMPARISON = OUT / "private_split_neutral_seed_replication/20261107T150345851174Z/PRIVATE_SEED_COMPARISON.json"
PERMISSION = OUT / "permission_confirmations/20261108_ALPACA_KIBOT_USER_CONFIRMED.json"
MODELS = ("last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch")
SEEDS = (2026, 2027, 2028)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), "Preserve earlier reporting receipts"
    comparison = load(COMPARISON)
    assert comparison["seeds"] == list(SEEDS) and comparison["same_targets_and_splits"]
    assert comparison["validation_origins_per_seed"] == 1970 and comparison["independent_test"] is False
    permission = load(PERMISSION)
    assert permission["status"] == "permitted_as_confirmed_by_user"
    cutoff = int(datetime(2026, 7, 1, tzinfo=ZoneInfo("America/New_York")).timestamp()) * 10**9
    sources, truth, prediction = {str(COMPARISON): digest(COMPARISON), str(PERMISSION): digest(PERMISSION)}, [], {m: [] for m in MODELS}
    fold_metadata = []
    for fold in range(1, 6):
        shared = None
        per_model = {m: [] for m in MODELS}
        for seed in SEEDS:
            source = Path(comparison["sources"][str(seed)])
            summary = source / "PRIVATE_RESULTS.json"
            assert digest(summary) == comparison["source_result_sha256"][str(seed)]
            sources[str(summary)] = digest(summary)
            folder = source / f"fold_{fold:02}"
            receipt_path = folder / "fold_receipt.json"
            receipt = load(receipt_path)
            assert receipt["status"] == "completed" and receipt["independent_test"] is False and receipt["Q3_values_read"] is False
            assert receipt["config"]["seed"] == seed and receipt["config"]["horizon"] == 21
            assert receipt["config"]["target"] == "future_log_return" and receipt["config"]["profile"] == "monthly"
            sources[str(receipt_path)] = digest(receipt_path)
            targets = folder / "targets.npz"
            assert digest(targets) == receipt["targets_sha256"]
            sources[str(targets)] = digest(targets)
            with np.load(targets, allow_pickle=False) as z:
                fields = {k: z[k].copy() for k in z.files}
            assert set(fields) == {"origins", "actual", "anchor", "decision_ns", "target_completion_ns"}
            assert np.all(fields["target_completion_ns"] < cutoff) and np.all(fields["decision_ns"] < fields["target_completion_ns"])
            assert fields["actual"].shape == (receipt["validation_origins"], 21) and np.isfinite(fields["actual"]).all()
            if shared is None:
                shared = fields
            else:
                assert all(np.array_equal(shared[k], fields[k]) for k in fields), "Seeds do not use identical labels or origins"
            for model in MODELS:
                path = folder / ("prediction_" + model + ".npz")
                assert digest(path) == receipt["models"][model]["prediction_sha256"]
                sources[str(path)] = digest(path)
                with np.load(path, allow_pickle=False) as z:
                    assert np.array_equal(z["origins"], fields["origins"]) and np.array_equal(z["decision_ns"], fields["decision_ns"])
                    p = np.asarray(z["prediction"], dtype=np.float64).copy()
                assert p.shape == fields["actual"].shape and np.isfinite(p).all()
                per_model[model].append(p)
        truth.append(shared["actual"])
        for model in MODELS:
            prediction[model].append(np.stack(per_model[model], axis=0))
        fold_metadata.append({"fold": fold, "origins": len(shared["origins"]),
                              "first_decision_ns": int(shared["decision_ns"][0]), "last_decision_ns": int(shared["decision_ns"][-1]),
                              "labels_and_origins_identical_across_seeds": True})
    y = np.concatenate(truth, axis=0)
    assert y.shape == (1970, 21)
    baseline = float(np.mean(y[:, -1] ** 2))
    rows, seed_rows = [], []
    for model in MODELS:
        p = np.concatenate(prediction[model], axis=1)
        assert p.shape == (3, 1970, 21)
        errors = p - y[None, :, :]
        endpoint_mse = np.mean(errors[:, :, -1] ** 2, axis=1)
        path_mse = np.mean(errors ** 2, axis=(1, 2))
        for index, seed in enumerate(SEEDS):
            improvement = 1 - float(endpoint_mse[index]) / baseline
            assert np.isclose(improvement, comparison["relative_mse_improvement_vs_zero_return"][model][str(seed)], rtol=1e-10, atol=1e-12)
            seed_rows.append({"model": model, "seed": seed, "origins": len(y), "endpoint_mse": float(endpoint_mse[index]),
                              "path_mse": float(path_mse[index]), "relative_improvement": improvement})
        ensemble_error = np.mean(p, axis=0) - y
        mean_loss, ensemble_loss = float(np.mean(endpoint_mse)), float(np.mean(ensemble_error[:, -1] ** 2))
        assert ensemble_loss <= mean_loss + 1e-14
        rows.append({"model": model, "origins": len(y), "seeds": 3,
                     "mean_seed_endpoint_mse": mean_loss, "mean_seed_relative_improvement": 1 - mean_loss / baseline,
                     "ensemble_endpoint_mse": ensemble_loss, "ensemble_relative_improvement": 1 - ensemble_loss / baseline,
                     "mean_seed_path_mse": float(np.mean(path_mse)), "ensemble_path_mse": float(np.mean(ensemble_error**2))})
    args.output.mkdir(parents=True)
    artifacts = {}
    for name, data in (("SEED_METRICS.csv", seed_rows), ("SEED_VERSUS_ENSEMBLE.csv", rows)):
        path = args.output / name
        with path.open("x", newline="", encoding="utf8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
        artifacts[name] = digest(path)
    audit = {"status": "posthoc_existing_monthly_seed_losses_and_averaged_predictions_recomputed",
             "created_at_utc": datetime.now(timezone.utc).isoformat(), "task": "monthly H21 split-neutral cumulative price log return",
             "seeds": list(SEEDS), "folds_per_seed": 5, "origins": len(y), "baseline_endpoint_mse": baseline,
             "rows": rows, "seed_metrics": seed_rows, "folds": fold_metadata, "source_sha256": sources,
             "script_sha256": digest(__file__), "artifacts": artifacts,
             "permission_status": permission["status"], "permission_record_sha256": digest(PERMISSION),
             "original_receipts_changed": False, "new_training_started": False, "new_seeds_started": False,
             "core_experiment_changed": False, "Q3_values_read": False, "independent_test": False,
             "limitations": ["Post-exposure descriptive aggregation of the separate monthly diagnostic.",
                             "Equal weight is one third per already completed seed; weights were not fitted.",
                             "Ensemble errors are not average seed errors and do not show an independent market replication.",
                             "No bootstrap interval or confirmation test was added for the ensemble.",
                             "Historical unconfirmed-permission fields predate and are superseded by the direct researcher confirmation."]}
    save(args.output / "SEED_AGGREGATION_AUDIT.json", audit)
    for row in rows:
        print(row["model"], "mean_seed_improvement", f"{100*row['mean_seed_relative_improvement']:.4f}%", "ensemble_improvement", f"{100*row['ensemble_relative_improvement']:.4f}%")
    print("SAVED", args.output.resolve())


if __name__ == "__main__":
    main()
