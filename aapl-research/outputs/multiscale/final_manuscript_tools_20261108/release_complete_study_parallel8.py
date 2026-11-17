"""Deliver the checked two-column paper, supplement and portable CPU inference.

The source is verified against the preserved research commit without changing
the shared checkout. Both PDFs require a recorded visual review. This utility
does not train, change the compute budget, publish, or complete the goal.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile

from release_complete_study import FORECAST_SCRIPT


ROOT = Path("/home/ayaka/wslcode/aapl-research")
RECOVERY = ROOT / "outputs/multiscale/source_recovery/20261108_after_external_branch_switch"
RESEARCH_COMMIT = "f70b62e3dd4036614998d2499b2c1d554a08338a"
SOURCE_SCOPES = ["aapl", "multiscale", "MULTISCALE_RESEARCH_PLAN.md", "environment_lock.json", ".gitignore"]
REMOTE_PROOF = Path(__file__).with_name("SOURCE_REMOTE_BRANCH_VERIFICATION_PARALLEL8.json")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def command(arguments, cwd=ROOT, timeout=120, binary=False):
    result = subprocess.run([str(x) for x in arguments], cwd=cwd, env=os.environ,
                            capture_output=True, text=not binary, timeout=timeout)
    if result.returncode:
        error = result.stderr.decode("utf8", errors="replace") if binary else result.stderr
        raise RuntimeError(str(arguments[0]) + " failed: " + error[-2500:])
    return result.stdout


def archive_files(raw):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            assert member.isfile(), "Source archive contains a non-regular file"
            assert member.name.startswith("aapl-research/")
            relative = Path(member.name).relative_to("aapl-research")
            assert ".." not in relative.parts and str(relative) not in result
            result[str(relative)] = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
    return result


def verify_source():
    recovery = read(RECOVERY / "SOURCE_RECOVERY_RECEIPT.json")
    assert recovery["known_source_commit"] == RESEARCH_COMMIT
    archive = RECOVERY / "PRESERVED_SOURCE.tar"
    assert sha(archive) == recovery["source_archive_sha256"]
    preserved_files = archive_files(archive.read_bytes())
    fresh_archive = command(["git", "archive", "--format=tar", RESEARCH_COMMIT, "--",
                             *["aapl-research/" + p for p in SOURCE_SCOPES]],
                            cwd=ROOT.parent, timeout=30, binary=True)
    assert archive_files(fresh_archive) == preserved_files, "Source capsule differs from the recorded research commit"
    snapshot = RECOVERY / "repository_snapshot/aapl-research"
    for relative, expected in preserved_files.items():
        path = snapshot / relative
        assert path.resolve().is_relative_to(snapshot.resolve()) and not path.is_symlink()
        assert sha(path) == expected, "Preserved source changed: " + relative
    assert len(preserved_files) == recovery["restored_files"] == 125
    receipt = {
        "status": "preserved_source_capsule_and_research_commit_verified",
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "research_commit": RESEARCH_COMMIT, "source_snapshot": str(snapshot),
        "source_archive_sha256": sha(archive),
        "source_recovery_receipt_sha256": sha(RECOVERY / "SOURCE_RECOVERY_RECEIPT.json"),
        "files_verified": len(preserved_files), "source_sha256": preserved_files,
        "current_shared_checkout_head": command(["git", "rev-parse", "HEAD"]).strip(),
        "current_shared_checkout_branch": command(["git", "branch", "--show-current"]).strip(),
        "shared_checkout_modified": False, "fresh_remote_verification_performed_by_release": False,
        "new_training_started": False, "Q3_values_read": False,
    }
    if REMOTE_PROOF.exists():
        proof = read(REMOTE_PROOF)
        if proof["status"] == "remote_research_branch_matches_preserved_commit":
            assert proof["http_status"] == 200 and proof["remote_sha"] == RESEARCH_COMMIT
            assert proof["remote_ref"] == "refs/heads/codex/aapl-return12-implementation"
            assert proof["network_write_performed"] is False
            receipt["recorded_remote_verification"] = {"at_utc": proof["at_utc"],
                "remote_sha": proof["remote_sha"], "remote_ref": proof["remote_ref"],
                "transport": proof["transport"], "evidence_sha256": sha(REMOTE_PROOF)}
    return snapshot, receipt


def verify_visual(folder, pdf_name, page_count, verification_name):
    review = read(folder / "VISUAL_REVIEW.json")
    assert review["status"] == "all_rendered_pages_reviewed"
    assert review["pdf_sha256"] == sha(folder / pdf_name)
    assert review["page_count"] == page_count
    assert review["pages_reviewed"] == list(range(1, page_count + 1))
    assert not review["issues_remaining"]
    if "content_verification_sha256" in review:
        assert review["content_verification_sha256"] == sha(folder / verification_name)
    for relative, expected in review.get("preview_sha256", {}).items():
        assert sha(folder / relative) == expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-source-only", action="store_true")
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--supplement", type=Path)
    parser.add_argument("--main-paper", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    assert os.sched_getaffinity(0) == {24, 26} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    snapshot, source = verify_source()
    if args.check_source_only:
        if args.source_receipt:
            save(args.source_receipt, source)
        print(json.dumps({k: source[k] for k in ("status", "research_commit", "files_verified", "current_shared_checkout_branch")}))
        return
    assert all((args.evidence, args.supplement, args.main_paper, args.output)), "Provide the complete study and both reviewed documents"
    evidence, supplement, paper, output = [p.resolve() for p in (args.evidence, args.supplement, args.main_paper, args.output)]
    archive = output.with_suffix(".zip")
    assert not output.exists() and not archive.exists(), "Preserve previous deliveries"
    assert source["recorded_remote_verification"]["remote_sha"] == RESEARCH_COMMIT

    full = read(evidence / "ALL100_EVIDENCE_RECEIPT.json")
    assert full["completed_folds"] == 100 and full["complete_tasks"] == 20
    assert full["restored_neural_predictors"] == 400 and full["Q3_values_read"] is False
    for relative, expected in full["artifacts"].items():
        assert sha(evidence / relative) == expected
    lineage = read(evidence / "training/COMPUTE_LINEAGE.json")
    assert lineage["user_amended_limit"] == 8 and lineage["cpus_per_worker"] == 2
    assert lineage["additional_wall_seconds"] == 0 and lineage["gpu_training"] is False
    assert lineage["source_recovery_receipt_sha256"] == source["source_recovery_receipt_sha256"]

    document = read(supplement / "DOCUMENT_CONTENT_VERIFICATION.json")
    assert document["completed_folds"] == 100 and document["complete_tasks"] == 20
    for key, name in (("markdown_sha256", "MANUSCRIPT_EN.md"), ("formatted_docx_sha256", "MANUSCRIPT_EN.docx"),
                      ("pdf_sha256", "MANUSCRIPT_EN.pdf")):
        assert sha(supplement / name) == document[key]
    verify_visual(supplement, "MANUSCRIPT_EN.pdf", document["checks"]["pdf_pages"], "DOCUMENT_CONTENT_VERIFICATION.json")
    main_check = read(paper / "MAIN_PAPER_CONTENT_VERIFICATION.json")
    assert main_check["complete_core"] and not main_check["layout_preview"] and main_check["completed_folds"] == 100
    assert 4 <= main_check["pdf_pages"] <= 6 and main_check["author_name_confirmed"]
    assert main_check["email_self_fill"] and main_check["Q3_values_read"] is False
    for name, expected in main_check["artifacts"].items():
        assert sha(paper / name) == expected
    assert sha(paper / "MAIN_PAPER_SOURCE_AUDIT.json") == main_check["source_audit_sha256"]
    verify_visual(paper, "MAIN_PAPER_EN.pdf", main_check["pdf_pages"], "MAIN_PAPER_CONTENT_VERIFICATION.json")

    example = read(supplement / "package_example/EXAMPLE_VERIFICATION.json")
    assert example["core_comparison_complete"] and example["CLI_matches_checked_API_exactly"]
    assert not example["new_training_started"] and not example["Q3_values_read"]
    assert sha(evidence / "inference_package/REGISTRY.json") == example["package_registry_sha256"]
    output.mkdir(parents=True)
    # Retain both raw conversions: the document receipts bind their hashes.
    ignore = shutil.ignore_patterns("previews", "contact_sheets")
    shutil.copytree(supplement, output / "manuscript", ignore=ignore)
    shutil.copytree(paper, output / "paper", ignore=ignore)
    shutil.copytree(evidence / "inference_package", output / "inference_package")
    for relative, expected in source["source_sha256"].items():
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(snapshot / relative, destination)
        assert sha(destination) == expected
    provenance = output / "provenance"
    provenance.mkdir()
    for relative in ("ALL100_EVIDENCE_RECEIPT.json", "training/COMPUTE_LINEAGE.json", "package_audit/PACKAGE_AUDIT.json"):
        shutil.copyfile(evidence / relative, provenance / Path(relative).name)
    shutil.copyfile(RECOVERY / "SOURCE_RECOVERY_RECEIPT.json", provenance / "SOURCE_RECOVERY_RECEIPT.json")
    shutil.copyfile(REMOTE_PROOF, provenance / REMOTE_PROOF.name)
    shutil.copyfile(__file__, provenance / Path(__file__).name)
    shutil.copyfile(Path(__file__).with_name("release_complete_study.py"), provenance / "release_complete_study.py")
    save(output / "SOURCE_GIT.json", source)
    (output / "RUN_FORECAST.py").write_text(FORECAST_SCRIPT, encoding="utf8")
    (output / "INFERENCE_REGISTRY_SHA256.txt").write_text(example["package_registry_sha256"] + "\n", encoding="utf8")
    packages = {}
    for name in ("numpy", "pandas", "torch", "exchange_calendars", "scipy", "python-docx", "matplotlib", "lxml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    save(output / "ENVIRONMENT.json", {"python": sys.version, "platform": platform.platform(),
         "machine": platform.machine(), "packages": packages, "cpu_affinity": sorted(os.sched_getaffinity(0)),
         "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
         "numerical_thread_limits": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
         "new_training_started": False, "Q3_values_read": False})
    portable = json.loads(command([sys.executable, output / "RUN_FORECAST.py"], cwd=output))
    original = read(supplement / "package_example/H60_RETURN_XPATCH_FORECAST.json")
    assert portable["prediction"] == original["prediction"]
    assert portable["core_comparison_complete"] and portable["selected_fold"] == 5 and not portable["independent_test"]
    save(output / "PORTABLE_FORECAST_CHECK.json", {"status": "relocated_source_and_complete_package_forecast_verified",
         "at_utc": datetime.now(timezone.utc).isoformat(), "forecast_path_steps": len(portable["prediction"]),
         "prediction_exactly_matches_original_example": True, "core_comparison_complete": True,
         "selected_fold": 5, "source_commit": RESEARCH_COMMIT, "new_training_started": False, "Q3_values_read": False})
    compute = read(supplement / "COMPUTE_ACCOUNTING_NOTE.json")
    assert compute["configured_maximum_training_processes"] == 8 and compute["original_deadline_preserved"]
    text = (
        "# AAPL complete core study — local delivery\n\n"
        "Read paper/MAIN_PAPER_EN.pdf for the four-to-six-page English two-column paper; "
        "the adjacent Word file is editable. The author is Wang Yang, High School Affiliated "
        "to Shanghai Jiao Tong University, Shanghai, 200439, China. The author will fill the "
        "authorized email placeholder. No venue-specific acceptance or external submission is claimed.\n\n"
        "The manuscript folder supplies the exhaustive supplementary report, all twenty task "
        "tables, fold and step errors, descriptive intervals, training records and source receipts. "
        "The core study has twenty tasks, eight methods, 100 chronological folds and seed 2026. "
        "All market periods are exposed historical development. No independent confirmation, "
        "trading evaluation or shared-horizon mechanism experiment was performed.\n\n"
        "The inference_package supplies all twenty tasks and eight methods using fold 5. "
        "With the dependencies in ENVIRONMENT.json available, run `python RUN_FORECAST.py`. "
        "It performs CPU inference on the supplied 480-bar past-only history, producing a "
        "sixty-step return path. The relocated source and package reproduced the original "
        "checked forecast exactly. Full licensed source prices and all training arrays remain "
        "in the original workspace and are required for complete training replication.\n\n"
        "SOURCE_GIT.json binds the included source files to the preserved research commit "
        f"{RESEARCH_COMMIT}. The shared checkout was left on its current branch. This release "
        "includes the recorded public HTTPS API verification of the matching remote research "
        "branch, with its check time and evidence hash; it does not claim a fresh SSH check. "
        "Upstream licences are retained in source/aapl/vendor.\n\n"
        "The annual variant uses approximately ten years of history and a twelve-month lookback, "
        "with 47 overlapping validation origins. Neural training preserves the 20-epoch cap and "
        "patience 3. The separate monthly three-seed and ensemble calculations are post-hoc. "
        "Protected 2026Q3 data remained sealed.\n\n"
        f"The original fifteen-fold continuation finished in a recorded wall envelope of {compute['continuation_wall_envelope_seconds']/3600:.3f} "
        "hours within its approved ten-hour maximum. The process limit changed from six to eight, "
        "with two logical CPUs each, the same absolute deadline and no GPU training. This wall "
        "envelope and the original overlapping stage receipts are not measured active CPU time.\n")
    (output / "README.md").write_text(text, encoding="utf8")
    inventory = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "FILE_SHA256.json", inventory)
    release = {"status": "complete_core_two_column_paper_supplement_and_portable_inference_delivered_locally",
        "at_utc": datetime.now(timezone.utc).isoformat(), "completed_folds": 100, "complete_tasks": 20,
        "models_per_task": 8, "restored_neural_predictors": full["restored_neural_predictors"],
        "restored_origin_model_paths": full["restored_origin_model_paths"], "core_seed": 2026,
        "main_paper_pages_reviewed": main_check["pdf_pages"], "supplement_pages_reviewed": document["checks"]["pdf_pages"],
        "main_paper_directory": str(paper), "supplement_directory": str(supplement), "evidence_directory": str(evidence),
        "source_git_commit": RESEARCH_COMMIT, "source_capsule_files_verified": source["files_verified"],
        "fresh_remote_verification_performed_by_release": False, "shared_checkout_modified": False,
        "source_remote_verified_commit": source["recorded_remote_verification"]["remote_sha"],
        "source_remote_verification_at_utc": source["recorded_remote_verification"]["at_utc"],
        "file_inventory_sha256": sha(output / "FILE_SHA256.json"),
        "main_visual_review_sha256": sha(paper / "VISUAL_REVIEW.json"),
        "supplement_visual_review_sha256": sha(supplement / "VISUAL_REVIEW.json"),
        "maximum_training_processes": 8, "cpus_per_training_process": 2, "additional_wall_seconds_granted": 0,
        "original_deadline_preserved": True, "email_completion": "author_will_fill",
        "independent_confirmation": False, "Q3_values_read": False, "new_training_started_by_release": False,
        "externally_published_or_submitted": False, "goal_marked_complete_by_script": False, "script_sha256": sha(__file__)}
    save(output / "RELEASE.json", release)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for relative, expected in inventory.items():
            assert hashlib.sha256(z.read(str(Path(output.name) / relative))).hexdigest() == expected
    receipt = {**release, "archive": str(archive), "archive_sha256": sha(archive),
               "archive_bytes": archive.stat().st_size, "archive_crc_and_inventory_check_passed": True}
    save(output.parent / (output.name + "_DELIVERY_RECEIPT.json"), receipt)
    print(json.dumps({k: receipt[k] for k in ("status", "completed_folds", "complete_tasks", "archive", "archive_bytes")}))


if __name__ == "__main__":
    main()
