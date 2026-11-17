"""One predeclared, sampled minute-scale development fold, kept local."""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

from multiscale.bars import eligible_origins, file_digest
from multiscale.development import FoldConfig, run_development_fold, validate_split
from multiscale.matrix_executor import read_bars
from multiscale.spec import MODELS, OUT, PROFILES
from multiscale.splits import rolling_folds
from multiscale.verify_development import check_fold


def save_new(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    source = OUT / "admission/alpaca_20261107T052228573242Z"
    receipt_path = source / "admission.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    terms = OUT / "source_docs/alpaca_terms.txt"
    quote = "you agree to use the Services and Content solely for your own personal and non-commercial purposes"
    if (quote not in " ".join(terms.read_text(encoding="utf8").split()) or
            receipt.get("Q3_values_read") is not False or receipt.get("experiment_scored") is not False):
        raise ValueError("Candidate identity or recorded personal-use term is invalid")
    profile = next(item for item in PROFILES if item.name == "minute")
    bars = read_bars(source, receipt, "minute")
    origin_path = source / "origins_minute_h60.npy"
    if file_digest(origin_path) != receipt["files"][origin_path.name]:
        raise ValueError("Candidate origins changed")
    origins = np.load(origin_path, allow_pickle=False)
    if not np.array_equal(origins, eligible_origins(bars, max(profile.lookbacks), 60, same_session=True)):
        raise ValueError("Saved minute origins do not match the task definition")
    folds = rolling_folds(bars, origins, 60)
    if len(folds) != 5:
        raise ValueError("Cannot form five chronological folds")
    full = folds[0]
    def sampled(values):
        values = np.asarray(values, dtype=np.int64)
        return values[values % 30 == 0]
    fold = {"fold": 1, "train": sampled(full["train"]), "validation": sampled(full["validation"]),
            "inner": {"fit": sampled(full["inner"]["fit"]), "stop": sampled(full["inner"]["stop"]),
                      "status": "ok"}}
    train, fit, stop, validation = validate_split(bars, fold, 60)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parent = OUT / "private_minute_exploration"; parent.mkdir(parents=True, exist_ok=True)
    directory = parent / stamp; directory.mkdir(exist_ok=False)
    exposure = OUT / "POST_CUTOFF_EXPOSURE.md"
    previous = exposure.read_bytes(); (directory / "EXPOSURE_BEFORE.md").write_bytes(previous)
    cfg = FoldConfig("minute", 60, "future_log_return", 480, 390, (0.1, 1.0, 10.0),
                     20, 3, 128, 0.001, 2026, "pending_protocol_hash")
    plan = {
        "status": "private_personal_development_only_not_publication_admitted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "user_request": "我要开始做实验",
        "scope": "personal non-commercial local development",
        "entitlement_note": "Existing user-obtained Alpaca SIP candidate; account-specific terms and publication rights remain unconfirmed.",
        "personal_use_quote": quote, "alpaca_terms_sha256": file_digest(terms),
        "admission_sha256": file_digest(receipt_path), "bar_sha256": receipt["files"]["bars_minute.npz"],
        "origins_sha256": receipt["files"][origin_path.name],
        "task": "minute H60 cumulative log return", "models": list(MODELS), "fold": 1,
        "sampling": "retain only bar indices divisible by 30 in all partitions; fixed without inspecting labels",
        "full_fit_origins": len(full["inner"]["fit"]), "sampled_fit_origins": len(fit),
        "full_stop_origins": len(full["inner"]["stop"]), "sampled_stop_origins": len(stop),
        "full_validation_origins": len(full["validation"]), "sampled_validation_origins": len(validation),
        "config_before_protocol_hash": asdict(cfg), "device": "cpu", "cpu_threads": 2,
        "maximum_wall_seconds": 600, "data_cutoff": "2026-06-30", "Q3_values_read": False,
        "independent_test": False, "publication_allowed": "not established; keep results local",
        "exposure_before_sha256": hashlib.sha256(previous).hexdigest(), "script_sha256": file_digest(Path(__file__)),
        "limitations": ["An intentionally sampled first fold, not the full-minute matrix.",
                        "Different origins from the full planned comparison; do not present as its results.",
                        "One historical development fold is not independent confirmation."]}
    plan_path = directory / "PRIVATE_PROTOCOL.json"; save_new(plan_path, plan)
    protocol_hash = file_digest(plan_path)
    with exposure.open("a", encoding="utf8") as stream:
        stream.write("\n## 个人内部分钟抽样实验：预先登记\n\n"
                     f"{stamp}：用户要求开始实验。将已有 SIP 分钟候选首折中柱索引能被 30 整除的起点"
                     "用于 H60 收益的八模型内部探索；抽样规则在观察该实验评分之前固定，"
                     f"起点数为拟合 {len(fit)}、早停 {len(stop)}、验证 {len(validation)}。"
                     "它不代表完整分钟起点矩阵，也不是独立确认；发表权待核实，2026Q3 仍封存。"
                     f"无论运行是否成功，后续冻结前均须检查 {plan_path}。\n")
    print(json.dumps({"status":"private_minute_sampled_protocol_recorded", "directory":str(directory),
                      "fit_origins":len(fit), "validation_origins":len(validation), "cpu_threads":2,
                      "maximum_seconds":600},ensure_ascii=False),flush=True)
    cfg = FoldConfig(**{**asdict(cfg), "protocol_identity": protocol_hash})
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    result = run_development_fold(bars, fold, receipt["split_actions"], cfg, directory / "fold_01",
                                  "private_exploration_not_publication_admitted:" + file_digest(receipt_path), device="cpu")
    verified = check_fold(directory / "fold_01")
    if not all(verified["current_source_matches_recorded"].values()):
        raise ValueError("Source changed during the fold")
    verified["scope"] = plan["status"]; save_new(directory / "PRIVATE_VERIFICATION.json", verified)
    summary = {"status":"one_private_sampled_minute_fold_completed", "directory":str(directory),
               "scope":"internal sampled exploration; not full-minute results or independent confirmation",
               "validation_origins":len(validation),
               "scores":{name:{"endpoint_mse":result["models"][name]["metrics"]["endpoint_mse"],
                                "relative_improvement":result["models"][name]["against_last_close_or_zero"]["relative_improvement"]}
                         for name in MODELS},
               "fold_receipt_sha256":file_digest(directory / "fold_01/fold_receipt.json"),
               "verification_sha256":file_digest(directory / "PRIVATE_VERIFICATION.json")}
    save_new(directory / "PRIVATE_RESULT_SUMMARY.json",summary)
    with exposure.open("a",encoding="utf8") as stream:
        stream.write(f"\n{stamp}：上述分钟抽样首折已完成，开发误差保存在 {directory}/PRIVATE_RESULT_SUMMARY.json；"
                     "这些结果不能当作独立确认或完整分钟矩阵。\n")
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == "__main__":
    try: main()
    except Exception as exc:
        print(json.dumps({"status":"private_minute_exploration_failed","error":f"{type(exc).__name__}: {exc}"},ensure_ascii=False),
              file=sys.stderr,flush=True)
        raise
