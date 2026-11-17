"""One user-requested, personal-only historical development fold.

This is not the publication-reviewed matrix. It reads only the already saved
Alpaca hour candidate, records a protocol before scoring, and keeps all output
inside the local ignored outputs directory. Publication rights remain pending.
"""
from __future__ import annotations

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


def save_new(path: Path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def main():
    source = OUT / "admission/alpaca_20261107T052228573242Z"
    receipt_path = source / "admission.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf8"))
    if (receipt.get("Q3_values_read") is not False or receipt.get("experiment_scored") is not False or
            receipt.get("provider") != "Alpaca SIP/raw daily and previously admitted SIP/raw minutes"):
        raise ValueError("The expected historical candidate identity is unavailable")
    terms = OUT / "source_docs/alpaca_terms.txt"
    terms_text = terms.read_text(encoding="utf8")
    personal_quote = "you agree to use the Services and Content solely for your own personal and non-commercial purposes"
    if personal_quote not in " ".join(terms_text.split()):
        raise ValueError("The recorded personal-use term could not be verified")
    profile = next(item for item in PROFILES if item.name == "hourly")
    item = next(row for row in receipt["tasks"] if row["profile"] == "hourly" and
                row["horizon"] == 8 and row["target"] == "future_log_return")
    cfg = FoldConfig("hourly", 8, "future_log_return", 96, 6, (0.1, 1.0, 10.0),
                     20, 3, 128, 0.001, 2026, "pending_protocol_hash")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    parent = OUT / "private_hourly_exploration"
    parent.mkdir(parents=True, exist_ok=True)
    directory = parent / stamp
    directory.mkdir(exist_ok=False)
    source_hash = file_digest(receipt_path)
    exposure = OUT / "POST_CUTOFF_EXPOSURE.md"
    prior_exposure = exposure.read_bytes()
    (directory / "EXPOSURE_BEFORE.md").write_bytes(prior_exposure)
    plan = {
        "status": "private_personal_development_only_not_publication_admitted",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "user_request": "我要开始做实验", "scope": "personal non-commercial local development",
        "entitlement_note": "Existing user-obtained Alpaca SIP candidate; public personal-use term recorded. "
                            "Account-specific terms and publication of derived results are not confirmed.",
        "alpaca_terms_path": str(terms), "alpaca_terms_sha256": file_digest(terms),
        "personal_use_quote": personal_quote,
        "publication_allowed": "not established; do not publish data, predictions or results",
        "admission_sha256": source_hash, "bar_sha256": receipt["files"]["bars_hour.npz"],
        "origins_sha256": receipt["files"]["origins_hourly_h8.npy"],
        "first_fold_as_previously_calculated": item["folds"][0],
        "config_before_protocol_hash": asdict(cfg), "models": list(MODELS),
        "chronological_folds": 5, "folds_to_run": [1], "device": "cpu", "cpu_threads": 2,
        "maximum_wall_seconds": 900, "selection": "first chronological fold and primary hourly H8 return task selected before scoring",
        "data_cutoff": "2026-06-30", "Q3_values_read": False, "independent_test": False,
        "exposure_note_before_sha256": hashlib.sha256(prior_exposure).hexdigest(),
        "script_sha256": file_digest(Path(__file__)),
        "limitations": ["One historical development fold; not an independent or complete five-scale experiment.",
                        "Formal data and rights review remains pending; no publication claim.",
                        "The original annual task has insufficient daily history.",
                        "Existing post-cutoff exposure and historically viewed periods remain in the exposure ledger."],
    }
    plan_path = directory / "PRIVATE_PROTOCOL.json"
    save_new(plan_path, plan)
    plan_hash = file_digest(plan_path)
    with exposure.open("a", encoding="utf8") as stream:
        stream.write("\n## 个人内部多尺度开发实验：预先登记\n\n"
                     f"{stamp}：用户要求开始实验。拟在现有 Alpaca SIP 小时候选数据上，"
                     "按首个时间顺序折运行 H8 累计收益的八模型个人内部探索；"
                     f"验证决策时间 {item['folds'][0]['validation_first_decision']} 至 "
                     f"{item['folds'][0]['validation_last_decision']}，最末目标完成时间 "
                     f"{item['folds'][0]['validation_last_target_completion']}。"
                     "此区间及选择仅为历史开发，可能与旧研究曝光重叠，不是独立确认；"
                     "正式论文发表权仍待确认。2026Q3 保持封存。"
                     f"无论运行是否成功，后续冻结前均须检查此记录与 {directory}/PRIVATE_PROTOCOL.json。\n")
    print(json.dumps({"status": "private_protocol_recorded", "directory": str(directory),
                      "task": "hourly H8 future_log_return", "fold": 1,
                      "device": "cpu", "threads": 2, "maximum_seconds": 900}, ensure_ascii=False), flush=True)

    bars = read_bars(source, receipt, "hour")
    origins_path = source / "origins_hourly_h8.npy"
    if file_digest(origins_path) != receipt["files"][origins_path.name]:
        raise ValueError("Saved origins differ from the candidate receipt")
    origins = np.load(origins_path, allow_pickle=False)
    expected = eligible_origins(bars, max(profile.lookbacks), 8, same_session=False)
    if not np.array_equal(origins, expected):
        raise ValueError("Saved hourly origins do not match the task definition")
    folds = rolling_folds(bars, origins, 8)
    if len(folds) != 5:
        raise ValueError("The candidate does not contain the expected five folds")
    fold = folds[0]
    train, fit, stop, validation = validate_split(bars, fold, 8)
    expected_counts = item["folds"][0]
    if (len(train), len(fit), len(stop), len(validation)) != (
            expected_counts["train_origins"], expected_counts["inner_fit_origins"],
            expected_counts["inner_stop_origins"], expected_counts["validation_origins"]):
        raise ValueError("The actual first fold differs from the precomputed candidate")
    cfg = FoldConfig(**{**asdict(cfg), "protocol_identity": plan_hash})
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    result = run_development_fold(bars, fold, receipt["split_actions"], cfg, directory / "fold_01",
                                  "private_exploration_not_publication_admitted:" + source_hash, device="cpu")
    verification = check_fold(directory / "fold_01")
    if not all(verification["current_source_matches_recorded"].values()):
        raise ValueError("Source changed during the development fold")
    verification["scope"] = plan["status"]
    save_new(directory / "PRIVATE_VERIFICATION.json", verification)
    scores = {name: {"endpoint_mse": result["models"][name]["metrics"]["endpoint_mse"],
                     "relative_improvement": result["models"][name]["against_last_close_or_zero"]["relative_improvement"]}
              for name in MODELS}
    summary = {"status": "one_private_historical_fold_completed", "directory": str(directory),
               "user_visible": "internal exploratory results only; not a publication admission or independent confirmation",
               "phase": "development", "validation_origins": len(validation), "scores": scores,
               "fold_receipt_sha256": file_digest(directory / "fold_01/fold_receipt.json"),
               "verification_sha256": file_digest(directory / "PRIVATE_VERIFICATION.json")}
    save_new(directory / "PRIVATE_RESULT_SUMMARY.json", summary)
    with exposure.open("a", encoding="utf8") as stream:
        stream.write(f"\n{stamp}：上述个人内部首折已完成并开始披露开发误差；"
                     f"本地摘要及独立复算身份为 {directory}/PRIVATE_RESULT_SUMMARY.json。"
                     "不能将该已见结果视为独立确认。\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "private_exploration_failed", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False), file=sys.stderr, flush=True)
        raise
