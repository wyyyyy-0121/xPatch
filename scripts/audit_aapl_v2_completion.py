"""Audit completed v2 receipts and independently recompute pooled forecast losses.

This does not train models or replace finance_v2.verify's recorded model replay.
Run with the experiment environment from the repository root.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/aapl_v2/main"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    manifest = read(OUTPUT / "manifest.json")
    identity = manifest["identity"]
    config = identity["config"]
    signature = manifest["signature"]
    require(hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() == signature,
            "Manifest signature differs")
    require(config == read(ROOT / "configs/aapl_v2.json"), "Current configuration differs")
    for name, expected in identity["sources"].items():
        require(sha(ROOT / name) == expected, f"Signed source changed: {name}")
    require({str(p.relative_to(ROOT)) for p in (ROOT / "finance_v2").rglob("*.py")} <= set(identity["sources"]),
            "Unsigned experiment module added")
    for path, expected in ((config["data"], identity["data_sha256"]),
                           ("PROTOCOL.md", identity["protocol_sha256"]),
                           (config["pretrained"]["lock"], identity["model_lock_sha256"]),
                           ("requirements-finance-v2.txt", identity["requirements_sha256"])):
        require(sha(ROOT / path) == expected, f"Signed input changed: {path}")
    require(platform.python_version() == identity["python_version"], "Python version differs")
    for package, expected in identity["runtime_versions"].items():
        require(importlib.metadata.version(package) == expected, f"Runtime version differs: {package}")
    lock = read(ROOT / config["pretrained"]["lock"])
    for entry in lock["models"].values():
        for filename, evidence in entry["files"].items():
            require(sha(ROOT / entry["directory"] / filename) == evidence["sha256"], "Weight changed")

    splits = read(OUTPUT / "splits.json")
    summary = read(OUTPUT / "development_summary.json")
    acceptance = read(OUTPUT / "acceptance.json")
    frozen = read(OUTPUT / "frozen.json")
    pipeline = read(OUTPUT / "pipeline_status.json")
    stochastic = {"xpatch48", "xpatch240", "kronos_small"}
    expected_runs, rows, files_checked = set(), [], 0
    reference = None
    for feature in config["feature_sets"]:
        for name in config["models"]:
            seed_losses = []
            for seed in config["seeds"] if name in stochastic else [0]:
                pieces = []
                for split in splits:
                    run = f"{split['name']}/{feature}/{name}_seed{seed}"
                    expected_runs.add(run)
                    directory = OUTPUT / run
                    receipt = read(directory / "result.json")
                    require(receipt["signature"] == signature, f"Receipt signature differs: {run}")
                    require((receipt["split"], receipt["feature_set"], receipt["model"], receipt["seed"]) ==
                            (split["name"], feature, name, seed), f"Receipt identity differs: {run}")
                    for filename, expected in receipt["files"].items():
                        require(sha(directory / filename) == expected, f"Artifact changed: {run}/{filename}")
                        files_checked += 1
                    with np.load(directory / "predictions.npz") as saved:
                        pieces.append({k: saved[k].copy() for k in ("origins", "dates", "timestamps", "y", "pred")})
                joined = {k: np.concatenate([piece[k] for piece in pieces]) for k in pieces[0]}
                if reference is None:
                    reference = {k: joined[k] for k in ("origins", "dates", "timestamps", "y")}
                require(all(np.array_equal(joined[k], v) for k, v in reference.items()), f"Unpaired data: {feature}/{name}")
                loss = (joined["pred"] - joined["y"]) ** 2
                seed_losses.append(loss)
                scores = summary["panels"][feature]["models"][name]["per_seed_metrics"][len(seed_losses) - 1]
                for j, horizon in enumerate(config["horizons_minutes"]):
                    np.testing.assert_allclose(loss[:, j].mean(), scores[str(horizon)]["mse"], rtol=1e-12)
            mse = np.mean(np.stack(seed_losses), axis=(0, 1))
            model = summary["panels"][feature]["models"][name]
            for j, horizon in enumerate(config["horizons_minutes"]):
                np.testing.assert_allclose(mse[j], model["metrics"][str(horizon)]["mse"], rtol=1e-12)
            baseline = np.mean(reference["y"][:, -1] ** 2)
            if name != "zero":
                comparison = summary["panels"][feature]["vs_zero"][name]
                require(mse[-1] > baseline and comparison["mse_difference"] > 0,
                        f"Audit conclusion needs review: {feature}/{name}")
                require(not comparison["development_gate_passed"], "Unexpected passing candidate")
            rows.append({"feature": feature, "model": name, "mse_12m": float(mse[-1]),
                         "mse_increase_vs_zero_percent": float(100 * (mse[-1] / baseline - 1))})

    actual_runs = {str(p.parent.relative_to(OUTPUT)) for p in OUTPUT.glob("20*Q*/*/*/result.json")}
    require(actual_runs == expected_runs and len(expected_runs) == 260, "Incomplete or unexpected runs")
    require(acceptance["signature"] == signature and acceptance["passed"] and
            acceptance["all_model_checks_requested"] and acceptance["runs_checked"] == 260, "Model acceptance absent")
    require({c["run"] for c in acceptance["checks"]} == expected_runs, "Acceptance coverage differs")
    require(all(c["targets_dates_scaler_calibration_metrics"] and c["model_forecasts_reproduced"]
                for c in acceptance["checks"]), "A recorded replay check failed")
    for i in (1, 5):
        log = OUTPUT / f"pipeline_{i:02}_verify.log"
        require(sum(line.startswith("VERIFIED ") for line in log.read_text().splitlines()) == 260,
                "A verification log does not cover all runs")
    require(frozen["signature"] == signature and all(x is None for x in frozen["selected"].values()), "Freeze differs")
    require(frozen["development_report_sha256"] == sha(OUTPUT / "development_summary.json"), "Frozen report changed")
    require(pipeline["state"] == "completed" and pipeline["independent_acceptance"], "Pipeline incomplete")
    require([s["stage"] for s in pipeline["stages"]] == ["verify", "freeze", "confirm", "report", "verify"] and
            all(s["state"] == "passed" and s["exit_code"] == 0 for s in pipeline["stages"]), "Stage order or exit differs")
    require(not (OUTPUT / "historical_confirmation").exists(), "Unexpected confirmation opened")
    log = (OUTPUT / "pipeline_03_confirm.log").read_text()
    require(all(f"GATE_STOP {f}:" in log for f in config["feature_sets"]), "Missing gate stop record")
    test_log = ROOT / "artifacts/aapl_v2/setup/tests_final.log"
    require("53 passed" in test_log.read_text() and "FAILED" not in test_log.read_text(), "Final tests did not pass")
    artifacts = ["manifest.json", "splits.json", "audit.json", "frozen.json", "acceptance.json",
                 "development_summary.json", "DEVELOPMENT_REPORT.md", "pipeline_status.json"]
    artifacts += [p.name for p in sorted(OUTPUT.glob("pipeline_*.log"))]
    evidence = {"passed": True, "signature": signature, "dataset_sha256": identity["data_sha256"],
                "validation_runs": len(expected_runs), "artifact_files_checked": files_checked,
                "samples_per_panel": len(reference["y"]), "sessions_per_panel": len(np.unique(reference["dates"])),
                "recorded_full_verification_passes": 2, "tests": 53, "independent_pooled_losses_recomputed": True,
                "replay_scope": "supervised evaluation forecasts fully reproduced; pretrained first and last fixed batches",
                "research_result": "no_candidate_passed", "selected": frozen["selected"],
                "historical_confirmation": "skipped by predeclared development gate",
                "transaction_cost_research": "skipped; predictive gate failed",
                "prospective_observation": "not opened; no eligible frozen candidate",
                "independent_test": False, "point_in_time_available": False, "trading_profitability_evaluated": False,
                "artifacts": {name: sha(OUTPUT / name) for name in artifacts}, "final_test_log_sha256": sha(test_log),
                "models": rows}
    target = OUTPUT / "completion_audit.json"
    target.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in evidence.items() if k not in {"artifacts", "models"}}, indent=2))


if __name__ == "__main__":
    main()
