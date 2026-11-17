"""Run predeclared hourly, monthly and quarterly personal-development folds.

This bounded local queue does not modify or pass the publication matrix gate.
All results remain exploratory, local and tied to the actual data identity.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np
import torch

from multiscale.bars import eligible_origins, file_digest
from multiscale.development import FoldConfig, atomic_json, run_development_fold, validate_split
from multiscale.matrix_executor import read_bars
from multiscale.spec import MODELS, OUT, PROFILES, ROOT
from multiscale.splits import rolling_folds, warmup_folds
from multiscale.training import digest_indices
from multiscale.verify_development import check_fold


def write_new(path, content):
    with path.open("x", encoding="utf8") as stream:
        json.dump(content, stream, indent=2, ensure_ascii=False, allow_nan=False)
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
    source_hash = file_digest(admission_path)
    data_identity = "private_exploration_not_publication_admitted:" + source_hash
    profile_map = {item.name: item for item in PROFILES}
    # Primary H8 comes first; the first H8-return fold was already observed and
    # is explicitly reused only as prior development, never as confirmation.
    task_order = [("hourly", 8, "future_log_return"), ("hourly", 8, "future_close")]
    task_order += [("hourly", h, target) for h in (1, 4)
                   for target in ("future_log_return", "future_close")]
    task_order += [("monthly", 21, target) for target in ("future_log_return", "future_close")]
    task_order += [("quarterly", 3, target) for target in ("future_log_return", "future_close")]
    periods = {"hourly": 6, "monthly": 252, "quarterly": 12}
    cached = {}; jobs = []
    for name, horizon, target in task_order:
        profile = profile_map[name]; key = (name, horizon)
        if key not in cached:
            bars = read_bars(source, admission, profile.frequency)
            path = source / f"origins_{name}_h{horizon}.npy"
            if file_digest(path) != admission["files"][path.name]:
                raise ValueError("Candidate origin file changed")
            origins = np.load(path, allow_pickle=False)
            if not np.array_equal(origins, eligible_origins(bars, max(profile.lookbacks), horizon, same_session=False)):
                raise ValueError("Saved origins do not match the predefined task")
            if name == "quarterly":
                folds = warmup_folds(bars, origins, horizon, min_inner_fit=24, min_inner_stop=4,
                                     min_validation_groups=6)
            else:
                folds = rolling_folds(bars, origins, horizon)
            if len(folds) != 5:
                raise ValueError(f"Cannot form five predefined folds: {name} h{horizon}")
            cached[key] = (bars, folds)
        bars, folds = cached[key]
        for fold in folds:
            train, fit, stop, validation = validate_split(bars, fold, horizon)
            job = {"key": f"{name}_h{horizon}_{target}_f{fold['fold']}",
                   "profile": name, "frequency": profile.frequency, "horizon": horizon, "target": target,
                   "fold": fold["fold"], "lookback": max(profile.lookbacks), "seasonal_period": periods[name],
                   "fit_origins": len(fit), "stop_origins": len(stop), "validation_origins": len(validation),
                   "train_indices_sha256": digest_indices(train), "fit_indices_sha256": digest_indices(fit),
                   "stop_indices_sha256": digest_indices(stop), "validation_indices_sha256": digest_indices(validation)}
            jobs.append(job)
    if len(jobs) != 50 or len({job["key"] for job in jobs}) != 50:
        raise ValueError("Unexpected task matrix")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parent = OUT / "private_three_scale_development"; parent.mkdir(parents=True, exist_ok=True)
    directory = parent / stamp; directory.mkdir(exist_ok=False)
    exposure = OUT / "POST_CUTOFF_EXPOSURE.md"
    previous = exposure.read_bytes(); (directory / "EXPOSURE_BEFORE.md").write_bytes(previous)
    recipe = {"max_epochs":20,"patience":3,"batch_size":128,"learning_rate":0.001,
              "seed":2026,"ridge_alphas":[0.1,1.0,10.0]}
    plan = {"status":"private_personal_development_only_not_publication_admitted",
            "created_at_utc":datetime.now(timezone.utc).isoformat(),"user_request":"我要开始做实验",
            "scope":"personal non-commercial local development", "personal_use_quote":quote,
            "alpaca_terms_sha256":file_digest(terms), "admission_sha256":source_hash,
            "account_specific_and_publication_rights":"not confirmed; all results remain local",
            "data_cutoff":"2026-06-30", "Q3_values_read":False,"independent_test":False,
            "selected_profiles":["hourly","monthly","quarterly"],
            "omitted_profiles":{"annual":"available daily history is about 10 years; original minimum is 15",
                                "minute":"full minute matrix awaits its own compute plan; sampled pilot is separate"},
            "quarterly_strategy":"earliest feasible warmup: at least 24 fit, 4 stop and 6 validation months per fold",
            "recipe":recipe,"models":list(MODELS),"task_order":task_order,"jobs":jobs,
            "device":"cpu","cpu_threads":2,"maximum_wall_seconds":900,
            "prior_exposure":"The H8-return first-fold scores were already reviewed before this queue was frozen.",
            "exposure_before_sha256":hashlib.sha256(previous).hexdigest(), "script_sha256":file_digest(Path(__file__)),
            "limitations":["Historical development only; no independent test or publication admission.",
                           "Quarterly samples remain small even after warmup.",
                           "Daily close quality and point-in-time versions remain under review.",
                           "The previously viewed first fold is not independent."]}
    plan_path = directory / "PRIVATE_PROTOCOL.json"; write_new(plan_path,plan)
    protocol_hash = file_digest(plan_path)
    with exposure.open("a",encoding="utf8") as stream:
        stream.write("\n## 个人内部小时、月度和季度开发队列：预先登记\n\n"
                     f"{stamp}：用户要求开始实验。依原计划在已有 Alpaca 候选资料上预先登记"
                     "小时 H1/H4/H8、月度 H21、季度 H3 的价格与累计收益任务，各五个时间折；"
                     "季度采用样本数决定的热身方案，年度因不足 15 年未纳入，分钟另作抽样试跑。"
                     "此前小时 H8 收益首折已被查看，其余任务均为历史开发而非独立确认；"
                     "账号具体协议及论文发表权仍待确认，2026Q3 保持封存。"
                     f"无论执行是否完成，后续冻结前均须检查 {plan_path}。\n")
    state = {"status":"running", "protocol_sha256":protocol_hash, "completed":0, "total":len(jobs), "jobs":{}}
    state_path = directory / "PRIVATE_QUEUE_STATE.json"; atomic_json(state_path,state)
    print(json.dumps({"status":"private_three_scale_queue_started","directory":str(directory),
                      "folds":len(jobs),"tasks":len(task_order),"cpu_threads":2,"maximum_seconds":900},
                     ensure_ascii=False),flush=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    completed_results = {}
    old_dir = OUT / "private_hourly_exploration/20261107T133537548946Z/fold_01"
    for job in jobs:
        key = job["key"]; state["jobs"][key] = {"status":"running"}; atomic_json(state_path,state)
        bars, folds = cached[(job["profile"],job["horizon"])]; fold = folds[job["fold"] - 1]
        cfg = FoldConfig(job["profile"],job["horizon"],job["target"],job["lookback"],job["seasonal_period"],
                         tuple(recipe["ridge_alphas"]),recipe["max_epochs"],recipe["patience"],recipe["batch_size"],
                         recipe["learning_rate"],recipe["seed"],protocol_hash)
        use_old = key == "hourly_h8_future_log_return_f1"
        if use_old:
            saved = json.loads((old_dir / "fold_receipt.json").read_text(encoding="utf8"))
            expected_cfg = asdict(cfg); previous_cfg = dict(saved["config"])
            expected_cfg["ridge_alphas"] = list(expected_cfg["ridge_alphas"])
            previous_cfg.pop("protocol_identity",None); expected_cfg.pop("protocol_identity",None)
            if (previous_cfg != expected_cfg or saved["data_identity"] != data_identity or
                    any(saved["identity_fields"][name] != job[name] for name in
                        ("train_indices_sha256","fit_indices_sha256","stop_indices_sha256","validation_indices_sha256"))):
                raise ValueError("Prior exposed first fold differs from the new development plan")
            fold_dir = old_dir; verified = check_fold(fold_dir); result = saved
        else:
            fold_dir = directory / key
            result = run_development_fold(bars,fold,admission["split_actions"],cfg,fold_dir,data_identity,device="cpu")
            verified = check_fold(fold_dir)
        if not all(verified["current_source_matches_recorded"].values()):
            raise ValueError("Source changed during the development queue")
        verified["scope"] = plan["status"]
        verification_path = directory / (key + ".verification.json"); write_new(verification_path,verified)
        record = {"status":"completed", "fold_directory":str(fold_dir), "reused_previously_viewed":use_old,
                  "receipt_sha256":file_digest(fold_dir / "fold_receipt.json"),
                  "verification_sha256":file_digest(verification_path), "validation_origins":job["validation_origins"]}
        state["jobs"][key] = record; state["completed"] += 1; state["elapsed_seconds"] = time.monotonic()-started
        completed_results[key] = result; atomic_json(state_path,state)
        if job["fold"] == 5:
            group = [j for j in jobs if (j["profile"],j["horizon"],j["target"]) ==
                     (job["profile"],job["horizon"],job["target"])]
            count = sum(j["validation_origins"] for j in group)
            pooled = {name:sum(j["validation_origins"] * completed_results[j["key"]]["models"][name]["metrics"]["endpoint_mse"]
                               for j in group) / count for name in MODELS}
            base = pooled["last_close"]
            task_result = {"status":"five_historical_development_folds_only", "profile":job["profile"],
                           "horizon":job["horizon"], "target":job["target"], "validation_origins":count,
                           "metrics":{name:{"endpoint_mse":pooled[name],
                                            "relative_improvement":1-pooled[name]/base if base else None}
                                      for name in MODELS},
                           "independent_test":False,"publication_rights":"not confirmed"}
            task_path = directory / f"PRIVATE_TASK_{job['profile']}_h{job['horizon']}_{job['target']}.json"
            write_new(task_path,task_result)
            print(json.dumps({"status":"private_task_five_folds_complete", "profile":job["profile"],
                              "horizon":job["horizon"], "target":job["target"], "validation_origins":count,
                              "completed_folds":state["completed"],"total_folds":len(jobs),
                              "task_result":str(task_path)},ensure_ascii=False),flush=True)
    state["status"] = "completed_private_development_only"; state["elapsed_seconds"] = time.monotonic()-started
    atomic_json(state_path,state)
    rows=[]
    for name,horizon,target in task_order:
        path=directory / f"PRIVATE_TASK_{name}_h{horizon}_{target}.json"
        task=json.loads(path.read_text(encoding="utf8"))
        for model in MODELS:
            rows.append({"scope":"personal_development_not_independent", "profile":name,"horizon":horizon,
                         "target":target,"model":model,"folds":5,"validation_origins":task["validation_origins"],
                         **task["metrics"][model]})
    with (directory / "PRIVATE_THREE_SCALE_RESULTS.csv").open("x",encoding="utf8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with exposure.open("a",encoding="utf8") as stream:
        stream.write(f"\n{stamp}：上述个人开发队列已完成 {state['completed']} 个折；"
                     f"结果和状态保存在 {directory}，后续均视为已曝光开发结果。\n")
    print(json.dumps({"status":state["status"],"directory":str(directory),"folds":state["completed"],
                      "elapsed_seconds":state["elapsed_seconds"]},ensure_ascii=False),flush=True)


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"status":"private_three_scale_queue_failed", "error":f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False),file=sys.stderr,flush=True)
        raise
