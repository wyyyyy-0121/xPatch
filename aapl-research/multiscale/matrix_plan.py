"""Inventory every requested fold without opening prices or starting training.

This is a planning artifact, not a frozen protocol or permission to execute a
forecast. Counts and storage estimates use a candidate admission receipt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from .bars import file_digest
from .protocol_draft import make_draft
from .spec import DEV_END, MODELS, OUT, PROFILES, TARGETS


GLOBAL_GATES = (
    "provider_terms_and_academic_result_publication_review",
    "source_quality_and_split_convention_review",
    "post_cutoff_and_legacy_exposure_review",
    "immutable_task_fold_and_training_protocol",
    "representative_throughput_and_resource_budget",
)


def make_matrix(admission: dict, admission_sha256: str):
    if (not isinstance(admission_sha256, str) or len(admission_sha256) != 64 or
            any(char not in "0123456789abcdef" for char in admission_sha256)):
        raise ValueError("Admission identity must be a SHA-256 digest")
    draft = make_draft(admission, admission_sha256)
    profiles = {profile.name: profile for profile in PROFILES}
    rows = {(r["profile"], r["horizon"], r["target"]): r for r in admission["tasks"]}
    jobs = []
    for profile in PROFILES:
        for horizon in profile.horizons:
            for target in TARGETS:
                row = rows[(profile.name, horizon, target)]
                if row["frequency"] != profile.frequency:
                    raise ValueError("Task frequency differs from its requested profile")
                if (row["lookback_candidates"] != list(profile.lookbacks) or
                        row["common_max_lookback"] != max(profile.lookbacks)):
                    raise ValueError("Candidate lookbacks differ from the requested profile")
                candidate_folds = row.get("folds")
                if not isinstance(candidate_folds, list):
                    raise ValueError("Candidate folds are missing")
                fold_map = {}
                for entry in candidate_folds:
                    number = entry.get("fold")
                    if isinstance(number, bool) or not isinstance(number, int) or number not in range(1, 6):
                        raise ValueError("Invalid candidate fold number")
                    if number in fold_map:
                        raise ValueError("Duplicate candidate fold number")
                    fold_map[number] = entry
                for number in range(1, 6):
                    fold = fold_map.get(number)
                    local = []
                    if not row.get("minimum_history_met", False):
                        local.append("original_minimum_history_not_met")
                    if fold is None:
                        local.append("candidate_fold_missing")
                        fit = stop = validation = 0
                    else:
                        counts = [fold.get(key) for key in
                                  ("inner_fit_origins", "inner_stop_origins", "validation_origins")]
                        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in counts):
                            raise ValueError("Invalid candidate fold counts")
                        fit, stop, validation = counts
                        if fold.get("status") != "ok" or not stop or not validation:
                            local.append("candidate_fold_insufficient")
                        if fit < 8:
                            local.append("fewer_than_eight_inner_fit_origins")
                    job_id = f"{profile.name}/h{horizon}/{target}/fold{number:02d}/seed2026"
                    jobs.append({"job_id": job_id, "profile": profile.name,
                                 "frequency": profile.frequency, "horizon": horizon,
                                 "target": target, "fold": number, "seed": 2026,
                                 "candidate_lookback": row["common_max_lookback"],
                                 "candidate_origin_count": row["origins"],
                                 "fit_origins": fit, "stop_origins": stop,
                                 "validation_origins": validation,
                                 "validation_first_decision": fold.get("validation_first_decision") if fold else None,
                                 "validation_last_decision": fold.get("validation_last_decision") if fold else None,
                                 "validation_last_target_completion": fold.get("validation_last_target_completion") if fold else None,
                                 "models": list(MODELS), "task_status": row["status"],
                                 "fold_status": fold.get("status") if fold else "missing",
                                 "local_blockers": local, "global_gates_not_yet_met": list(GLOBAL_GATES),
                                 "status": "blocked_not_runnable",
                                 "neural_update_upper_bound_draft": math.ceil(fit / 128) * 20 * 4,
                                 "raw_prediction_payload_bytes_at_float64": validation * horizon * len(MODELS) * 8})
    expected = len(TARGETS) * sum(len(profile.horizons) for profile in profiles.values()) * 5
    if len(jobs) != expected or len({job["job_id"] for job in jobs}) != expected:
        raise AssertionError("The fold inventory is incomplete or has duplicate identities")
    counts = Counter(blocker for job in jobs for blocker in job["local_blockers"])
    return {"schema_version": 1, "status": "draft_blocked_not_runnable", "frozen": False,
            "phase": "historical_development_planning", "independent_test": False,
            "admission_sha256": admission_sha256, "development_cutoff_new_york": DEV_END,
            "Q3_values_read": False, "market_price_arrays_opened": False, "experiment_scored": False,
            "models": list(MODELS), "tasks": len(draft["tasks"]), "fold_jobs": len(jobs),
            "model_prediction_slots": len(jobs) * len(MODELS),
            "neural_fit_slots_one_seed": len(jobs) * 4, "ridge_fit_slots_one_seed": len(jobs),
            "baseline_slots_one_seed": len(jobs) * 3,
            "global_gates_not_yet_met": list(GLOBAL_GATES),
            "local_blocker_counts": dict(sorted(counts.items())),
            "jobs_with_any_local_blocker": sum(bool(job["local_blockers"]) for job in jobs),
            "neural_update_upper_bound_draft": sum(job["neural_update_upper_bound_draft"] for job in jobs),
            "raw_prediction_payload_bytes_at_float64": sum(job["raw_prediction_payload_bytes_at_float64"] for job in jobs),
            "assumptions_and_limits": [
                "All counts include blocked candidate folds; they are not completed or authorized jobs.",
                "Updates assume one seed, 20 epochs, batch 128 and four neural models; they are not a runtime projection.",
                "Raw storage assumes float64 prediction values for eight models; excludes compression, indices, targets, checkpoints, retries and other overhead.",
                "Candidate rolling folds are not frozen; proposed warm-up folds are separate and may change the counts.",
                "Fewer than eight inner-fit examples is an engineering rejection, not evidence that eight are statistically sufficient.",
                "The inventory contains historical development only and does not establish source rights or independent confirmation."
            ], "jobs": jobs}


def write_report(result: dict, directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    path = directory / "matrix_plan.json"
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf8")
    lines = ["# 多尺度实验矩阵：候选清单", "",
             "此清单只读取候选准入回执元数据；没有读取行情数组、启动模型或评分，也不是冻结协议。", "",
             f"- 任务：{result['tasks']}；五折任务：{result['fold_jobs']}；八模型预测槽位：{result['model_prediction_slots']}。",
             f"- 神经模型拟合槽位：{result['neural_fit_slots_one_seed']}；Ridge：{result['ridge_fit_slots_one_seed']}；无训练基线：{result['baseline_slots_one_seed']}。",
             f"- 草案神经更新次数上限：{result['neural_update_upper_bound_draft']:,}；不能据此推算运行时间。",
             f"- 八模型预测值按 float64 的未压缩载荷约 {result['raw_prediction_payload_bytes_at_float64'] / 2**30:.3f} GiB；不含索引、标签、权重、检查点、重试和压缩影响。",
             f"- 局部样本／折障碍涉及 {result['jobs_with_any_local_blocker']} 个折任务；全部任务仍有共同准入障碍。", "",
             "| 尺度 | 任务 | 五折任务 | 局部样本／折障碍 | 草案神经更新上限 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for profile in PROFILES:
        group = [job for job in result["jobs"] if job["profile"] == profile.name]
        lines.append(f"| {profile.name} | {len(profile.horizons) * len(TARGETS)} | {len(group)} | "
                     f"{sum(bool(job['local_blockers']) for job in group)} | "
                     f"{sum(job['neural_update_upper_bound_draft'] for job in group):,} |")
    lines.extend(["", "共同准入条件：数据使用及论文发表权、来源与拆股口径、曝光审计、任务／折／配置冻结、代表性吞吐和资源预算。",
                  "上述条件全部记录为未通过；本清单不提供执行开关。", "",
                  "候选五折和提出的热身五折尚未冻结；`matrix_plan.json` 保存逐折数量及独立路径。",
                  "该统计包含被阻塞的任务，不代表它们可以运行；原先独立测试曝光不会因更换数据或跨度消失。", ""])
    (directory / "MATRIX_PLAN.md").write_text("\n".join(lines), encoding="utf8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True, help="candidate admission directory")
    args = parser.parse_args()
    source = args.admission.resolve()
    if not source.is_relative_to((OUT / "admission").resolve()):
        raise ValueError("Admission path is outside the isolated workspace")
    receipt = source / "admission.json"
    if not receipt.resolve().is_relative_to((OUT / "admission").resolve()):
        raise ValueError("Admission receipt resolves outside the isolated workspace")
    value = json.loads(receipt.read_text(encoding="utf8"))
    result = make_matrix(value, file_digest(receipt))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = write_report(result, OUT / "matrix_plans" / stamp)
    print(json.dumps({"path": str(path), "status": result["status"], "fold_jobs": result["fold_jobs"],
                      "neural_update_upper_bound_draft": result["neural_update_upper_bound_draft"],
                      "raw_prediction_payload_bytes_at_float64": result["raw_prediction_payload_bytes_at_float64"]}))


if __name__ == "__main__":
    main()
