"""Run the user-approved short-history annual task as local development.

The 12-month lookback and sample-count warmup are fixed before any scores are
calculated. Results are not a replacement for the original 15-year protocol.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from multiscale.bars import eligible_origins, file_digest
from multiscale.development import FoldConfig, atomic_json, run_development_fold, validate_split
from multiscale.matrix_executor import read_bars
from multiscale.spec import MODELS, OUT
from multiscale.splits import warmup_folds
from multiscale.training import digest_indices
from multiscale.verify_development import check_fold


def write_new(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    started = time.monotonic()
    source = OUT / "admission/alpaca_20261107T052228573242Z"
    admission_path = source / "admission.json"
    admission = json.loads(admission_path.read_text(encoding="utf8"))
    terms = OUT / "source_docs/alpaca_terms.txt"
    quote = "you agree to use the Services and Content solely for your own personal and non-commercial purposes"
    if (quote not in " ".join(terms.read_text(encoding="utf8").split()) or
            admission.get("Q3_values_read") is not False or admission.get("experiment_scored") is not False):
        raise ValueError("Candidate identity or recorded personal-use term is invalid")
    bars = read_bars(source, admission, "month")
    lookback, horizon = 12, 12
    origins = eligible_origins(bars, lookback, horizon, same_session=False)
    folds = warmup_folds(bars, origins, horizon, min_inner_fit=24, min_inner_stop=4,
                         min_validation_groups=6)
    if len(folds) != 5:
        raise ValueError("Cannot form five annual folds with the fixed sample-count warmup")
    targets = ("future_log_return", "future_close")
    jobs = []
    for target in targets:
        for fold in folds:
            train, fit, stop, val = validate_split(bars, fold, horizon)
            jobs.append({"key": f"annual_h12_{target}_f{fold['fold']}", "fold": fold["fold"],
                         "target": target, "outer_train_origins": len(train), "fit_origins": len(fit),
                         "stop_origins": len(stop), "validation_origins": len(val),
                         "validation_start": str(bars.session_date[val[0]]),
                         "validation_end": str(bars.session_date[val[-1]]),
                         "train_indices_sha256": digest_indices(train), "fit_indices_sha256": digest_indices(fit),
                         "stop_indices_sha256": digest_indices(stop),
                         "validation_indices_sha256": digest_indices(val)})
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parent = OUT / "private_annual_short_development"
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / stamp
    directory.mkdir(exist_ok=False)
    exposure = OUT / "POST_CUTOFF_EXPOSURE.md"
    previous = exposure.read_bytes()
    (directory / "EXPOSURE_BEFORE.md").write_bytes(previous)
    source_hash = file_digest(admission_path)
    data_identity = "private_exploration_not_publication_admitted:" + source_hash
    recipe = {"max_epochs": 20, "patience": 3, "batch_size": 128, "learning_rate": 0.001,
              "seed": 2026, "ridge_alphas": [0.1, 1.0, 10.0]}
    plan = {"status": "private_short_history_development_only_not_publication_admitted",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "user_choice": "10年即可; 直接做实验", "scope": "personal non-commercial local development",
            "personal_use_quote": quote, "alpaca_terms_sha256": file_digest(terms),
            "admission_sha256": source_hash, "account_specific_and_publication_rights": "not confirmed",
            "data_cutoff": "2026-06-30", "Q3_values_read": False, "independent_test": False,
            "frequency": "month", "horizon_months": horizon, "lookback_months": lookback,
            "original_minimum_history_years": 15,
            "short_history_deviation": "12-month lookback on about 10 years; exploratory, not the original protocol",
            "fold_strategy": "earliest count-only warmup with >=24 fit, >=4 stop, >=6 validation months per fold",
            "eligible_origins": len(origins), "recipe": recipe, "seasonal_period_months": 12,
            "models": list(MODELS), "jobs": jobs, "device": "cpu", "cpu_threads": 2,
            "maximum_wall_seconds": 600, "exposure_before_sha256": hashlib.sha256(previous).hexdigest(),
            "script_sha256": file_digest(Path(__file__)),
            "limitations": ["Observed development only; not an independent test or publication admission.",
                            "Monthly origins overlap in their 12-month target paths; sample count is not independent count.",
                            "A 12-month lookback and about 10 years are a departure from the original 15-year plan.",
                            "Nominal-price targets include splits; cash dividends are not included.",
                            "One seed and at most 20 epochs; completion does not establish convergence."]}
    plan_path = directory / "PRIVATE_PROTOCOL.json"
    write_new(plan_path, plan)
    protocol_hash = file_digest(plan_path)
    with exposure.open("a", encoding="utf8") as stream:
        stream.write("\n## 个人内部年度短历史开发：预先登记\n\n"
                     f"{stamp}：用户选择现有约 10 年日线并要求直接实验。年度 H12 采用 12 个月回看，"
                     "按样本数量预设热身后运行价格和累计收益各五个时间折；这偏离原定 15 年方案，"
                     "结果为已曝光、本机个人开发，账号和发表权待确认，2026Q3 保持封存。"
                     f"无论执行是否完成，后续冻结前均须检查 {plan_path}。\n")
    state = {"status": "running", "protocol_sha256": protocol_hash, "completed": 0,
             "total": len(jobs), "jobs": {}}
    state_path = directory / "PRIVATE_QUEUE_STATE.json"
    atomic_json(state_path, state)
    print(json.dumps({"status": "annual_short_queue_started", "directory": str(directory),
                      "jobs": len(jobs), "cpu_threads": 2, "maximum_seconds": 600,
                      "validation_origins_per_fold": [len(f["validation"]) for f in folds]},
                     ensure_ascii=False), flush=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    try:
        results = {}
        for job in jobs:
            if time.monotonic() - started >= 600:
                raise TimeoutError("Annual local queue reached its 600-second limit")
            key = job["key"]
            state["jobs"][key] = {"status": "running"}
            atomic_json(state_path, state)
            cfg = FoldConfig("annual", horizon, job["target"], lookback, 12,
                             tuple(recipe["ridge_alphas"]), recipe["max_epochs"], recipe["patience"],
                             recipe["batch_size"], recipe["learning_rate"], recipe["seed"], protocol_hash)
            fold_dir = directory / key
            result = run_development_fold(bars, folds[job["fold"] - 1], admission["split_actions"],
                                          cfg, fold_dir, data_identity, device="cpu")
            verified = check_fold(fold_dir)
            if not all(verified["current_source_matches_recorded"].values()):
                raise ValueError("Source changed during the annual queue")
            verification_path = directory / (key + ".verification.json")
            write_new(verification_path, verified)
            results[key] = result
            state["jobs"][key] = {"status": "completed", "fold_directory": str(fold_dir),
                                  "validation_origins": job["validation_origins"],
                                  "receipt_sha256": file_digest(fold_dir / "fold_receipt.json"),
                                  "verification_sha256": file_digest(verification_path)}
            state["completed"] += 1
            state["elapsed_seconds"] = time.monotonic() - started
            atomic_json(state_path, state)
            print(json.dumps({"status": "annual_fold_verified", "completed": state["completed"],
                              "total": len(jobs), "target": job["target"], "fold": job["fold"]}), flush=True)

        rows, task_summaries, aggregate_checks = [], {}, {}
        for target in targets:
            group = [j for j in jobs if j["target"] == target]
            count = sum(j["validation_origins"] for j in group)
            pooled = {name: sum(j["validation_origins"] * results[j["key"]]["models"][name]["metrics"]["endpoint_mse"]
                                for j in group) / count for name in MODELS}
            # Independently sum squared errors from saved targets and predictions.
            direct = {}
            for name in MODELS:
                total_sq = 0.0
                for job in group:
                    folder = directory / job["key"]
                    with np.load(folder / "targets.npz", allow_pickle=False) as data:
                        truth = np.asarray(data["actual"][:, -1], dtype=np.float64)
                    with np.load(folder / f"prediction_{name}.npz", allow_pickle=False) as data:
                        pred = np.asarray(data["prediction"][:, -1], dtype=np.float64)
                    total_sq += float(np.square(pred - truth).sum())
                direct[name] = total_sq / count
                if not np.isclose(direct[name], pooled[name], rtol=1e-12, atol=1e-18):
                    raise ValueError(f"Independent aggregate mismatch for {target} {name}")
            baseline = pooled["last_close"]
            metrics = {name: {"endpoint_mse": pooled[name],
                              "relative_improvement": 1 - pooled[name] / baseline if baseline else None}
                       for name in MODELS}
            task_summaries[target] = {"status": "five_short_history_development_folds_only",
                                      "validation_origins": count, "metrics": metrics,
                                      "independent_test": False, "publication_rights": "not confirmed"}
            aggregate_checks[target] = {"validation_origins": count, "direct_endpoint_mse": direct}
            for name in MODELS:
                rows.append({"scope": "short_history_personal_development_not_independent", "profile": "annual",
                             "horizon": horizon, "lookback": lookback, "target": target, "model": name,
                             "folds": 5, "validation_origins": count, **metrics[name]})
        summary_path = directory / "PRIVATE_ANNUAL_RESULTS.json"
        write_new(summary_path, task_summaries)
        write_new(directory / "PRIVATE_AGGREGATE_INDEPENDENT_CHECK.json",
                  {"status": "saved_arrays_recalculated", "targets": aggregate_checks,
                   "limitations": "Arithmetic verification only; not independent market confirmation."})
        with (directory / "PRIVATE_ANNUAL_RESULTS.csv").open("x", encoding="utf8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        state["status"] = "completed_short_history_development_only"
        state["elapsed_seconds"] = time.monotonic() - started
        state["results_sha256"] = file_digest(summary_path)
        atomic_json(state_path, state)
        with exposure.open("a", encoding="utf8") as stream:
            stream.write(f"\n{stamp}：年度短历史队列完成 {state['completed']} 折；结果位于 {directory}，"
                         "均为已曝光开发结果。\n")
        print(json.dumps({"status": state["status"], "directory": str(directory), "folds": state["completed"],
                          "elapsed_seconds": state["elapsed_seconds"]}, ensure_ascii=False), flush=True)
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["elapsed_seconds"] = time.monotonic() - started
        atomic_json(state_path, state)
        raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "annual_short_queue_failed", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        raise
