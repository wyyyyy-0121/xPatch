"""Audit the original plan against the completed study without training.

Readiness mode records outstanding dependencies. Final mode additionally
requires the reviewed documents and portable release. This script never
signals workers, changes budgets, publishes, or marks the goal complete.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET
import zipfile

from release_complete_study_parallel8 import verify_source, verify_visual

ROOT = Path("/home/ayaka/wslcode/aapl-research")
BASE = ROOT / "outputs/multiscale"
TOOLS = Path(__file__).resolve().parent
RUN = BASE / "remaining_continuations/20261108T040346048516Z-parallel8"
EVIDENCE = BASE / "completed_core_evidence/20261108_parallel8_continuation"
SUPPLEMENT = BASE / "completed_manuscripts/20261108_parallel8_full_core"
PAPER = BASE / "double_column_manuscripts/20261108_parallel8_full_core"
RELEASE = BASE / "complete_study_releases/20261108_parallel8_full_core"
ADMISSION = BASE / "admission/alpaca_20261107T052228573242Z"
COMMIT = "f70b62e3dd4036614998d2499b2c1d554a08338a"
PLAN_SHA = "6665141f0e482b4892c789a424c9257d3d85d44a69ebd4152419f4ae8e17ddf1"
DEADLINE = 1794143421.9147575
HORIZONS = (("annual", 12), ("quarterly", 3), ("monthly", 21),
            ("hourly", 1), ("hourly", 4), ("hourly", 8),
            ("minute", 5), ("minute", 15), ("minute", 30), ("minute", 60))
TARGETS = ("future_close", "future_log_return")
TASKS = {(p, h, t) for p, h in HORIZONS for t in TARGETS}
MODELS = {"last_close", "drift", "seasonal", "ridge", "mlp", "gru", "lstm", "xpatch"}
CHECKED = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def bind(path, expected=None):
    path = Path(path).resolve()
    digest = sha(path)
    require(expected is None or digest == expected, "Artifact changed: " + str(path))
    CHECKED[str(path)] = digest
    return json.loads(path.read_text(encoding="utf8")) if path.suffix == ".json" else digest


def table(path):
    bind(path)
    with Path(path).open(newline="", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def task_key(item):
    return tuple(item[k] for k in ("profile", "horizon", "target"))


def sources_and_data():
    bind(ROOT / "MULTISCALE_RESEARCH_PLAN.md", PLAN_SHA)
    snapshot, source = verify_source()
    require(source["research_commit"] == COMMIT and source["files_verified"] == 125,
            "Preserved source scope differs")
    proof = source["recorded_remote_verification"]
    require(proof["remote_sha"] == COMMIT, "Recorded remote branch differs")
    for relative, expected in source["source_sha256"].items():
        bind(snapshot / relative, expected)
    permission = bind(BASE / "permission_confirmations/20261108_ALPACA_KIBOT_USER_CONFIRMED.json")
    require(permission["evidence_type"] == "direct_user_confirmation" and
            set(permission["providers"]) == {"Alpaca", "Kibot"} and
            permission["training_allowed_as_confirmed"] and
            permission["publication_of_results_and_figures_allowed_as_confirmed"],
            "User-confirmed permission differs")
    provenance = bind(BASE / "data_provenance/20261108_CURRENT_DEVELOPMENT_SOURCE_MANIFEST.json")
    bind(provenance["permission_confirmation"], provenance["permission_confirmation_sha256"])
    admitted = bind(ADMISSION / "admission.json", provenance["admission_metadata_sha256"])
    require(admitted["development_cutoff_new_york"] == "2026-06-30" and
            admitted["Q3_values_read"] is False and
            provenance["provider_used_in_reported_experiments"] == "Alpaca SIP/raw",
            "Data source or protected boundary differs")
    for relative, expected in admitted["files"].items():
        bind(ADMISSION / relative, expected)
    return {"source_files": 125, "commit": COMMIT, "remote_check_at_utc": proof["at_utc"],
            "remote_check_is_recorded_not_refreshed": True, "provider": "Alpaca SIP/raw",
            "annual_variant": "User-accepted approximately ten-year history; not the original longer minimum",
            "permission_basis": "Direct user confirmation; not independent inspection of a supplier contract"}


def author_information():
    record = bind(TOOLS / "AUTHOR_INFORMATION.json")
    require(len(record["authors"]) == 1 and record["authors"][0]["name_en"] == "Wang Yang",
            "Author identity differs")
    author, affiliation = record["authors"][0], record["affiliations"][0]
    require(author["email_completion"] == "author_will_fill" and author["email"] is None,
            "User-authorized email placeholder differs")
    require((affiliation["school"], affiliation["city"], affiliation["postal_code"], affiliation["country"]) ==
            ("High School Affiliated to Shanghai Jiao Tong University", "Shanghai", "200439", "China"),
            "Confirmed affiliation differs")
    require(affiliation["department"] is None, "An unconfirmed department was added")
    return {"author": author["name_en"], "email_completion": "author_will_fill",
            "affiliation": affiliation["school"], "postal_code": affiliation["postal_code"]}


def full_results():
    receipt = bind(EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json")
    require(receipt["completed_folds"] == 100 and receipt["complete_tasks"] == 20 and
            receipt["restored_neural_predictors"] == 400 and receipt["Q3_values_read"] is False,
            "Full evidence has a different scope")
    for relative, expected in receipt["artifacts"].items():
        bind(EVIDENCE / relative, expected)
    result = bind(EVIDENCE / "results/RESULTS.json")
    require(result["status"] == "complete_historical_development_results" and
            result["completed_folds"] == 100 and result["complete_tasks"] == 20 and
            result["independent_test"] is False and result["p_values_reported"] is False,
            "Results are incomplete or claim independent confirmation")
    require(len(result["tasks"]) == 20 and {task_key(t) for t in result["tasks"]} == TASKS,
            "Task inventory differs")
    for task in result["tasks"]:
        require(task["completed_folds"] == [1, 2, 3, 4, 5] and not task["missing_folds"] and
                len(task["folds"]) == 5 and set(task["pooled"]["models"]) == MODELS,
                "A task lacks a fold or method")
    protocol = bind(EVIDENCE / "results/COMPILATION_PROTOCOL.json", result["protocol_sha256"])
    require(protocol["allow_partial"] is False, "Partial compilation cannot be a final release")
    counts = {}
    for name, expected in result["csv_sha256"].items():
        bind(EVIDENCE / "results" / name, expected)
        counts[name] = len(table(EVIDENCE / "results" / name))
    for name, count in {"METRICS_POOLED.csv": 160, "METRICS_BY_FOLD.csv": 800,
                        "METRICS_PER_STEP.csv": 2544, "DESCRIPTIVE_INTERVALS.csv": 320,
                        "TRAINING_MODELS.csv": 400}.items():
        require(counts[name] == count, "Unexpected result table scope: " + name)
    rows = table(EVIDENCE / "results/TRAINING_MODELS.csv")
    for row in rows:
        require(row["status"] == "completed" and int(row["seed"]) == 2026 and
                int(row["max_epochs"]) == 20 and int(row["patience"]) == 3,
                "Training recipe or completion differs")
        bind(row["training_summary"], row["training_summary_sha256"])
    return {"tasks": 20, "folds": 100, "methods": 8, "neural_fits": 400, "csv_rows": counts}


def design_and_restoration():
    design = bind(EVIDENCE / "design/DESIGN_TIMING_AUDIT.json")
    require(design["planned_folds_verified"] == design["completed_saved_fold_metadata_verified"] == 100 and
            design["paired_price_return_designs_verified"] == 50 and design["Q3_values_read"] is False,
            "Design audit lacks the complete matrix")
    restoration = bind(EVIDENCE / "restoration/ALL100_RESTORATION.json")
    require(restoration["completed_folds"] == 100 and restoration["neural_models"] == 400 and
            restoration["all_arrays_exact"] and restoration["max_absolute_difference"] == 0 and
            restoration["saved_target_arrays_opened"] is False and len(restoration["checks"]) == 100,
            "Saved predictor restoration is incomplete")
    for entry in restoration["checks"]:
        value = bind(entry["check"], entry["sha256"])
        require(len(value["models"]) == 4 and all(m["exact_array_equal"] and
                m["max_absolute_difference"] == 0 for m in value["models"]),
                "A saved predictor failed exact reproduction")
    return {"original_designs": 100, "paired_designs": 50, "exact_neural_predictors": 400,
            "origin_model_paths": restoration["origin_model_paths"], "original_prediction_batch_size": 512}


def package_and_input_contract():
    from multiscale.bars import Bars
    from multiscale.matrix_executor import read_bars
    from multiscale.spec import PROFILES
    from multiscale.study_package import load_study_package
    import numpy as np
    import torch

    registry_path = EVIDENCE / "inference_package/REGISTRY.json"
    registry = bind(registry_path)
    audit = bind(EVIDENCE / "package_audit/PACKAGE_AUDIT.json")
    require(registry["core_comparison_complete"] and registry["five_fold_tasks"] == 20 and
            audit["core_comparison_complete"] and audit["tasks_verified"] == 20 and
            audit["models_verified"] == 160 and audit["fresh_processes"] == 20,
            "All-task package check is incomplete")
    require(audit["registry_sha256"] == sha(registry_path), "Package audit identity differs")
    require(len(registry["tasks"]) == 20 and {task_key(t) for t in registry["tasks"]} == TASKS,
            "Packaged task inventory differs")
    for entry in registry["tasks"]:
        require(entry["selected_fold"] == 5 and entry["completed_folds"] == [1, 2, 3, 4, 5],
                "Package does not use the latest completed fold")
        bind(EVIDENCE / "inference_package" / entry["bundle"] / "manifest.json", entry["manifest_sha256"])
    predictor = load_study_package(EVIDENCE / "inference_package",
                                   expected_registry_sha256=sha(registry_path), device="cpu")
    admitted, cache, short_checks = bind(ADMISSION / "admission.json"), {}, []
    expected_guards = {"unknown_model", "wrong_horizon", "wrong_decision", "unobserved_input", "omitted_bar"}
    for entry in audit["results"]:
        value = bind(entry["check"], entry["check_sha256"])
        require(set(value["input_guards_checked"]) == expected_guards and
                len(value["models"]) == 8 and value["training_started"] is False and
                value["saved_target_arrays_opened"] is False and value["Q3_values_read"] is False,
                "Real package input guards are incomplete")
        frequency = next(p.frequency for p in PROFILES if p.name == value["profile"])
        if frequency not in cache:
            cache[frequency] = read_bars(ADMISSION, admitted, frequency)
        bars = cache[frequency]
        decision = value["past_only_input_last_decision_ns"]
        index = int(np.searchsorted(bars.end_ns, decision))
        require(index < len(bars.end_ns) and int(bars.end_ns[index]) == decision,
                "Recorded package decision is absent from admitted bars")
        part = slice(index, index + 1)
        short = Bars(frequency, bars.start_ns[part], bars.end_ns[part], bars.close[part],
                     bars.observed[part], bars.session_date[part])
        before_rng = torch.get_rng_state().clone()
        try:
            predictor.predict(short, profile=value["profile"], horizon=value["horizon"],
                              target=value["target"], model="last_close", as_of_ns=decision,
                              history_identity="read-only-short-input-contract-audit")
        except ValueError as exc:
            require(str(exc) == "Not enough completed historical bars",
                    "Short input failed for a different reason: " + str(exc))
        else:
            raise ValueError("Insufficient historical input was accepted")
        require(torch.equal(before_rng, torch.get_rng_state()), "Input rejection changed caller RNG")
        short_checks.append({"task": task_key(value), "input_bars": 1,
                             "rejection": "Not enough completed historical bars"})
    require(len(short_checks) == 20 and {tuple(c["task"]) for c in short_checks} == TASKS,
            "Short-input audit does not cover every task")
    return {"past_only_model_calls": 160, "fresh_task_processes": 20,
            "input_guards_per_task": sorted(expected_guards), "short_input_checks": short_checks,
            "new_forecast_scores": 0, "training_updates": 0}


def archived_restart_and_seeds():
    restart = bind(BASE / "historical_training_recovery/20261108_real_mid_epoch_verified/RESTART_AUDIT.json")
    require(restart["fresh_processes"] == 6 and restart["total_optimizer_updates_in_audit"] == 8 and
            len(restart["checks"]) == 2 and restart["Q3_values_read"] is False and
            restart["original_workers_or_queues_modified"] is False, "Archived restart scope differs")
    for entry in restart["checks"]:
        bind(entry["source_checkpoint"], entry["source_checkpoint_sha256"])
        require(entry["all_checkpoint_fields_exactly_equal"] and entry["source_unchanged"],
                "Archived restart state differs")
    seeds_dir = BASE / "posthoc_seed_aggregation/20261108_complete_reporting"
    seeds = bind(seeds_dir / "SEED_AGGREGATION_AUDIT.json")
    require(seeds["seeds"] == [2026, 2027, 2028] and seeds["origins"] == 1970 and
            len(seeds["rows"]) == 8 and seeds["new_training_started"] is False and
            seeds["independent_test"] is False, "Post-hoc seed scope differs")
    for name, expected in seeds["artifacts"].items():
        bind(seeds_dir / name, expected)
    return {"archived_restart_models": [c["model"] for c in restart["checks"]],
            "restart_limit": "Two compared updates from each of two real archived states; not a full-fold GPU restart",
            "posthoc_seed_count": 3, "mean_seed_and_prediction_ensemble_losses_separated": True}


def reviewed_documents():
    main = bind(PAPER / "MAIN_PAPER_CONTENT_VERIFICATION.json")
    supplement = bind(SUPPLEMENT / "DOCUMENT_CONTENT_VERIFICATION.json")
    require(main["complete_core"] and not main["layout_preview"] and main["completed_folds"] == 100 and
            4 <= main["pdf_pages"] <= 6 and main["columns"] == [1, 2, 1, 2, 2],
            "Main paper is partial or outside the requested format")
    require(main["baseline_rows_checked"] == 10 and main["chart_ratio_cells_checked"] == 140 and
            main["initial_plan_principal_task_count"] == 10 and main["auxiliary_task_count"] == 10 and
            main["cited_references"] == 12 and main["author_name_confirmed"] and main["email_self_fill"],
            "Main paper content scope differs")
    for name, expected in main["artifacts"].items():
        bind(PAPER / name, expected)
    bind(PAPER / "MAIN_PAPER_SOURCE_AUDIT.json", main["source_audit_sha256"])
    require(supplement["completed_folds"] == 100 and supplement["complete_tasks"] == 20 and
            supplement["checks"]["tables"] == 37 and supplement["checks"]["editable_equations"] == 30,
            "Supplement lacks complete content")
    for key, name in (("markdown_sha256", "MANUSCRIPT_EN.md"), ("raw_docx_sha256", "MANUSCRIPT_RAW.docx"),
                      ("formatted_docx_sha256", "MANUSCRIPT_EN.docx"), ("pdf_sha256", "MANUSCRIPT_EN.pdf")):
        bind(SUPPLEMENT / name, supplement[key])
    for folder, pdf_name, pages, verification in (
            (PAPER, "MAIN_PAPER_EN.pdf", main["pdf_pages"], "MAIN_PAPER_CONTENT_VERIFICATION.json"),
            (SUPPLEMENT, "MANUSCRIPT_EN.pdf", supplement["checks"]["pdf_pages"], "DOCUMENT_CONTENT_VERIFICATION.json")):
        verify_visual(folder, pdf_name, pages, verification)
        bind(folder / "VISUAL_REVIEW.json")
        info = subprocess.run(["pdfinfo", str(folder / pdf_name)], capture_output=True,
                              text=True, check=True, timeout=20).stdout
        require(int(re.search(r"^Pages:\s+(\d+)", info, re.M)[1]) == pages, "PDF page count changed")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    with zipfile.ZipFile(PAPER / "MAIN_PAPER_EN.docx") as archive:
        xml = ET.fromstring(archive.read("word/document.xml"))
        actual_columns = [int(c.get("{" + ns["w"] + "}num", "1")) for c in xml.findall(".//w:sectPr/w:cols", ns)]
    require(actual_columns == [1, 2, 1, 2, 2], "Word column sections changed")
    body = (PAPER / "MAIN_PAPER_EN.md").read_text(encoding="utf8")
    full = (SUPPLEMENT / "MANUSCRIPT_EN.md").read_text(encoding="utf8")
    require("Principal horizons in the initial plan" in body and
            "annual H12, quarterly H3, monthly H21, hourly H8 and minute H60" in body and
            "[Email to be added by the author]" in body, "Principal endpoints or author placeholder disappeared")
    for text in (body, full):
        require(re.search(r"shared.horizon.{0,160}(?:was not run|was not performed|not run)", text, re.S),
                "Unperformed shared-horizon experiment is not disclosed")
        require("2026Q3" in text and "sealed" in text and "post-hoc" in text,
                "Historical evidence restrictions are not disclosed")
    return {"main_pages_reviewed": main["pdf_pages"],
            "supplement_pages_reviewed": supplement["checks"]["pdf_pages"],
            "main_columns": actual_columns, "email_completion": "author_will_fill",
            "mechanism": "Conditional experiment not run and not claimed",
            "independent_confirmation": "Not available; historical development only"}


def final_compute():
    state = bind(RUN / "STATE.json")
    approval = bind(RUN / "APPROVAL.json")
    require(state["phase"] == "completed" and len(state["jobs"]) == 15 and
            all(j["phase"] == "completed" for j in state["jobs"].values()),
            "Approved continuation is not complete")
    require(state["maximum_workers"] == 8 and state["deadline_epoch"] == DEADLINE and
            approval["maximum_wall_seconds"] == 36000, "Approved resource boundary changed")
    lineage = bind(EVIDENCE / "training/COMPUTE_LINEAGE.json")
    require(lineage["user_amended_limit"] == 8 and lineage["cpus_per_worker"] == 2 and
            lineage["additional_wall_seconds"] == 0 and lineage["gpu_training"] is False and
            lineage["new_seeds"] == 0 and lineage["new_configurations"] == 0 and
            lineage["deadline_epoch"] == DEADLINE, "Continuation exceeded its amended scope")
    note = bind(SUPPLEMENT / "COMPUTE_ACCOUNTING_NOTE.json")
    require(note["configured_maximum_training_processes"] == 8 and note["original_deadline_preserved"] and
            0 <= note["continuation_wall_envelope_seconds"] <= 36000 and
            note["exact_total_study_cpu_seconds"] is None and
            note["maximum_simultaneously_running_jobs_observed"] <= 8,
            "Compute envelope or cost interpretation differs")
    require((RUN / "STATE.json").stat().st_mtime <= DEADLINE,
            "Continuation completion was recorded after its original deadline")
    return {"training_workers": 8, "cpus_per_worker": 2, "gpu_training": False,
            "deadline_epoch": DEADLINE, "wall_envelope_seconds": note["continuation_wall_envelope_seconds"],
            "active_cpu_time_not_claimed": True, "additional_budget_granted_by_audit": 0}


def portable_release():
    receipt = bind(RELEASE.parent / (RELEASE.name + "_DELIVERY_RECEIPT.json"))
    release = bind(RELEASE / "RELEASE.json")
    require(release["completed_folds"] == 100 and release["complete_tasks"] == 20 and
            release["restored_neural_predictors"] == 400 and release["source_git_commit"] == COMMIT and
            release["maximum_training_processes"] == 8 and release["original_deadline_preserved"] and
            release["externally_published_or_submitted"] is False, "Portable release scope differs")
    inventory = bind(RELEASE / "FILE_SHA256.json", release["file_inventory_sha256"])
    for relative, expected in inventory.items():
        bind(RELEASE / relative, expected)
    example = bind(SUPPLEMENT / "package_example/EXAMPLE_VERIFICATION.json")
    require(example["core_comparison_complete"] and example["CLI_matches_checked_API_exactly"] and
            example["package_registry_sha256"] == sha(EVIDENCE / "inference_package/REGISTRY.json"),
            "Full-package CLI example differs")
    forecast = bind(RELEASE / "PORTABLE_FORECAST_CHECK.json")
    require(forecast["prediction_exactly_matches_original_example"] and forecast["forecast_path_steps"] == 60 and
            forecast["selected_fold"] == 5 and forecast["new_training_started"] is False,
            "Relocated-source forecast is not verified")
    archive = Path(receipt["archive"])
    bind(archive, receipt["archive_sha256"])
    require(archive.stat().st_size == receipt["archive_bytes"], "Archive size differs")
    files = {str(p.relative_to(RELEASE)) for p in RELEASE.rglob("*") if p.is_file()}
    require(files == set(inventory) | {"FILE_SHA256.json", "RELEASE.json"},
            "Release has missing or unrecorded files")
    with zipfile.ZipFile(archive) as zipped:
        require(zipped.testzip() is None, "Archive CRC failed")
        require(set(zipped.namelist()) == {str(Path(RELEASE.name) / p) for p in files},
                "Archive file inventory differs")
        for relative in files:
            require(hashlib.sha256(zipped.read(str(Path(RELEASE.name) / relative))).hexdigest() ==
                    sha(RELEASE / relative), "Archived file differs: " + relative)
    return {"archive": str(archive), "archive_bytes": archive.stat().st_size,
            "archive_files": len(files), "crc_and_all_file_hashes_verified": True,
            "portable_forecast_steps": 60, "publication": "Local delivery only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(os.sched_getaffinity(0) == {24, 26} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1",
            "Audit must use the two analysis CPUs with CUDA hidden")
    gates = [
        ("1,2,3,9,10.tasks,10.git", "Sources, admitted data, permission and preserved committed code",
         [], sources_and_data),
        ("User author-format request", "Confirmed byline, affiliation and authorized email placeholder",
         [], author_information),
        ("2,4,5,8,10.results", "All twenty tasks, eight methods, five folds and four hundred training histories",
         [EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json"], full_results),
        ("5,10.protocol,10.reproducibility", "Original chronology and exact full-array saved-predictor restoration",
         [EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json"], design_and_restoration),
        ("2,10.package,10.reproducibility", "Complete inference package, past-only calls and invalid-input checks",
         [EVIDENCE / "ALL100_EVIDENCE_RECEIPT.json"], package_and_input_contract),
        ("4,5.9,8,10.package", "Real archived process restart and separately reported post-hoc seed aggregation",
         [], archived_restart_and_seeds),
        ("5.8,5.10,6,10.report,User document format", "Reviewed English main paper and full supplement with conditional limits",
         [PAPER / "MAIN_PAPER_CONTENT_VERIFICATION.json", PAPER / "VISUAL_REVIEW.json",
          SUPPLEMENT / "DOCUMENT_CONTENT_VERIFICATION.json", SUPPLEMENT / "VISUAL_REVIEW.json"], reviewed_documents),
        ("7,8,9,User eight-worker amendment", "Completed approved continuation within the unchanged absolute deadline",
         [SUPPLEMENT / "COMPUTE_ACCOUNTING_NOTE.json"], final_compute),
        ("10.package,10.reproducibility,10.report", "Portable source, package, forecast and checked final archive",
         [RELEASE.parent / (RELEASE.name + "_DELIVERY_RECEIPT.json")], portable_release),
    ]
    rows = []
    for section, requirement, dependencies, action in gates:
        missing = [str(p) for p in dependencies if not p.is_file()]
        row = {"plan": section, "requirement": requirement}
        if missing:
            row.update(status="pending", missing=missing)
        else:
            try:
                row.update(status="verified", evidence=action())
            except Exception as exc:
                row.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        rows.append(row)
    all_verified = all(r["status"] == "verified" for r in rows)
    state = json.loads((RUN / "STATE.json").read_text(encoding="utf8"))
    report = {
        "status": "all_original_deliverables_verified" if all_verified and not args.readiness
                  else "readiness_snapshot_study_not_yet_complete",
        "at_utc": datetime.now(timezone.utc).isoformat(), "readiness_only": args.readiness,
        "requirements": rows, "artifact_files_hash_checked": len(CHECKED), "artifacts": CHECKED,
        "live_run_phase_at_snapshot": state["phase"],
        "completed_core_folds_at_snapshot": 85 + sum(j["phase"] == "completed" for j in state["jobs"].values()),
        "core_study_complete": all_verified and not args.readiness,
        "goal_marked_complete_by_script": False, "new_training_updates": 0, "process_signals_sent": False,
        "Q3_values_read": False, "budget_changed": False, "shared_checkout_changed": False,
        "script_sha256": sha(__file__),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": report["status"], "core_study_complete": report["core_study_complete"],
                      "gates": {r["requirement"]: r["status"] for r in rows},
                      "artifact_files_hash_checked": len(CHECKED)}))
    return 0 if args.readiness or all_verified else 2


if __name__ == "__main__":
    sys.exit(main())
