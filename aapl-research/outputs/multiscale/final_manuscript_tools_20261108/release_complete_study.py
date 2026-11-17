"""Package a visually reviewed complete study and check portable inference.

This creates local delivery artifacts only. It does not train, publish, send
messages, change the experiment budget, or mark the goal complete.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

ROOT = Path("/home/ayaka/wslcode/aapl-research")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")


def run(arguments, cwd=ROOT, environment=None, timeout=120):
    result = subprocess.run([str(v) for v in arguments], cwd=cwd, env=environment or os.environ,
                            capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(str(arguments[0]) + " failed: " + result.stderr[-2500:])
    return result.stdout


FORECAST_SCRIPT = '''"""Re-run the supplied past-only forecast; no training is performed."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
example = root / "manuscript/package_example"
receipt = json.loads((example / "EXAMPLE_VERIFICATION.json").read_text())
history = example / "H60_RETURN_PAST_480_BARS.npz"
assert hashlib.sha256(history.read_bytes()).hexdigest() == receipt["input_sha256"]
registry_sha = (root / "INFERENCE_REGISTRY_SHA256.txt").read_text().strip()
environment = dict(os.environ, PYTHONPATH=str(root / "source"), CUDA_VISIBLE_DEVICES="-1",
                   OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
command = [sys.executable, "-m", "multiscale.study_package", "forecast",
           "--package", str(root / "inference_package"), "--registry-sha256", registry_sha,
           "--bars", str(history), "--profile", "minute", "--horizon", "60",
           "--target", "future_log_return", "--model", "xpatch",
           "--as-of-ns", str(receipt["decision_ns"]),
           "--history-identity", "supplied-past-only-example:" + receipt["input_sha256"], "--device", "cpu"]
subprocess.run(command, cwd=root / "source", env=environment, check=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence, manuscript, output = (p.resolve() for p in (args.evidence, args.manuscript, args.output))
    assert not output.exists(), "Preserve previous delivery releases"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and os.sched_getaffinity(0) == {24, 26}
    full = load(evidence / "ALL100_EVIDENCE_RECEIPT.json")
    assert full["completed_folds"] == 100 and full["complete_tasks"] == 20 and full["Q3_values_read"] is False
    for name, expected in full["artifacts"].items():
        assert digest(evidence / name) == expected
    document = load(manuscript / "DOCUMENT_CONTENT_VERIFICATION.json")
    visual = load(manuscript / "VISUAL_REVIEW.json")
    assert document["completed_folds"] == 100 and document["complete_tasks"] == 20
    for key, name in (("markdown_sha256", "MANUSCRIPT_EN.md"), ("formatted_docx_sha256", "MANUSCRIPT_EN.docx"),
                      ("pdf_sha256", "MANUSCRIPT_EN.pdf")):
        assert digest(manuscript / name) == document[key]
    pages = document["checks"]["pdf_pages"]
    assert visual["status"] == "all_rendered_pages_reviewed" and visual["pdf_sha256"] == document["pdf_sha256"]
    assert visual["page_count"] == pages and visual["pages_reviewed"] == list(range(1, pages + 1)) and not visual["issues_remaining"]
    example = load(manuscript / "package_example/EXAMPLE_VERIFICATION.json")
    assert example["core_comparison_complete"] and example["CLI_matches_checked_API_exactly"]
    assert example["new_training_started"] is False and example["Q3_values_read"] is False
    assert digest(evidence / "inference_package/REGISTRY.json") == example["package_registry_sha256"]
    assert not run(["git", "status", "--porcelain", "--untracked-files=no"]).strip(), "Tracked working tree changed"
    commit = run(["git", "rev-parse", "HEAD"]).strip()
    branch = run(["git", "branch", "--show-current"]).strip()
    assert branch.startswith("codex/")
    remote = run(["git", "ls-remote", "--heads", "origin", "refs/heads/" + branch], timeout=30).split()
    assert remote == [commit, "refs/heads/" + branch], "Remote branch is not the checked local commit"
    output.mkdir(parents=True)
    shutil.copytree(manuscript, output / "manuscript", ignore=shutil.ignore_patterns("previews", "contact_sheets", "MANUSCRIPT_RAW.docx"))
    shutil.copytree(evidence / "inference_package", output / "inference_package")
    sources = output / "source"
    sources.mkdir()
    for folder in ("multiscale", "aapl"):
        shutil.copytree(ROOT / folder, sources / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy2(ROOT / "MULTISCALE_RESEARCH_PLAN.md", sources / "MULTISCALE_RESEARCH_PLAN.md")
    (output / "RUN_FORECAST.py").write_text(FORECAST_SCRIPT, encoding="utf8")
    (output / "INFERENCE_REGISTRY_SHA256.txt").write_text(example["package_registry_sha256"] + "\n")
    packages = {}
    for name in ("numpy", "pandas", "torch", "exchange_calendars", "scipy", "python-docx", "matplotlib", "lxml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cpu_models = sorted({line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")})
    runtime = {"python": sys.version, "python_executable_used_for_checks": sys.executable,
               "platform": platform.platform(), "machine": platform.machine(), "cpu_models": cpu_models,
               "logical_cpus_available_on_host": os.cpu_count(), "release_check_cpu_affinity": sorted(os.sched_getaffinity(0)),
               "packages": packages, "cuda_visible_devices_during_release": os.environ["CUDA_VISIBLE_DEVICES"],
               "numerical_thread_limits": {n: os.environ.get(n) for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
               "training_performed_during_release": False, "Q3_values_read": False}
    save(output / "ENVIRONMENT.json", runtime)
    git = {"status": "source_commit_and_remote_branch_verified", "at_utc": datetime.now(timezone.utc).isoformat(),
           "commit": commit, "branch": branch, "remote": "origin", "remote_commit": remote[0],
           "tracked_working_tree_clean": True, "manuscript_or_market_data_pushed_by_this_release": False}
    save(output / "SOURCE_GIT.json", git)
    # A relocated source tree and package must work without importing the original checkout.
    portable = json.loads(run([sys.executable, output / "RUN_FORECAST.py"], cwd=output))
    original = load(manuscript / "package_example/H60_RETURN_XPATCH_FORECAST.json")
    assert portable["prediction"] == original["prediction"] and portable["core_comparison_complete"]
    assert portable["selected_fold"] == 5 and portable["independent_test"] is False
    save(output / "PORTABLE_FORECAST_CHECK.json", {"status": "relocated_source_and_complete_package_forecast_verified",
         "at_utc": datetime.now(timezone.utc).isoformat(), "forecast_path_steps": len(portable["prediction"]),
         "prediction_exactly_matches_original_example": True, "core_comparison_complete": True, "selected_fold": 5,
         "new_training_started": False, "Q3_values_read": False, "source_root": str(sources)})
    build = load(manuscript / "MANUSCRIPT_BUILD_AUDIT.json")
    compute = load(manuscript / "COMPUTE_ACCOUNTING_NOTE.json")
    readme = (
        "# AAPL complete core study — local delivery\n\n"
        "Open manuscript/MANUSCRIPT_EN.pdf to read the checked rendering, or the Word file for editing. "
        "The Markdown and two figures are included. The manuscript reports all twenty tasks, eight "
        "methods and 100 chronological development folds. The core study uses seed 2026.\n\n"
        "The all_task_results folder contains the complete pooled, fold, step, uncertainty and training "
        "tables. The timing, split-group, seed-aggregation, restoration and package receipts retain "
        "their distinct scopes. The study remains exposed historical development; no independent "
        "confirmation, trading evaluation or shared-horizon mechanism experiment was performed.\n\n"
        "The complete inference_package folder supplies twenty task interfaces and all eight methods "
        "using the latest chronological fold, fold 5. The included 480-bar past-only example produces "
        "a sixty-step cumulative-return path. With the recorded Python dependencies available, run "
        "`python RUN_FORECAST.py` from this folder. The release check ran this command from the relocated "
        "source tree and matched the original example exactly. It performs inference only.\n\n"
        "ENVIRONMENT.json records the checked runtime. SOURCE_GIT.json identifies the matching code "
        "commit and verified remote. Vendored upstream licence files remain in source/aapl/vendor. "
        "The compact delivery includes the example input and packaged predictors. Full licensed "
        "source prices and all fold-level training/prediction arrays remain in the original workspace; "
        "they are required for a complete training replication.\n\n"
        "The annual variant used approximately ten years of history and a twelve-month lookback, "
        "with 47 overlapping validation origins. Core neural training used the fixed 20-epoch cap and "
        "patience 3. Three-seed and ensemble summaries concern only the separate post-hoc monthly "
        "diagnostic. Protected 2026Q3 data remained sealed.\n\n"
        f"The approved continuation finished fifteen original folds in approximately {compute['continuation_wall_envelope_seconds']/3600:.3f} "
        "wall-clock hours, with at most six training processes and two logical CPUs each. Its maximum "
        "was ten hours and it used no GPU training. Original raw stage timers contain suspensions; "
        "neither their sums nor the night-only adjustment measure total study CPU time.\n\n"
        "The English manuscript is a complete generic manuscript with verified numerical content. "
        "A target venue and author/title-page details have not been supplied, so this release makes "
        "no claim of venue-specific formatting or submission. No material was submitted externally.\n")
    (output / "README.md").write_text(readme, encoding="utf8")
    inventory = {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()}
    save(output / "FILE_SHA256.json", inventory)
    release = {"status": "complete_core_manuscript_and_portable_inference_delivered_locally",
               "created_at_utc": datetime.now(timezone.utc).isoformat(), "completed_folds": 100, "complete_tasks": 20,
               "models_per_task": 8, "core_neural_fits": 400, "restored_neural_predictors": 400,
               "restored_origin_model_paths": full["restored_origin_model_paths"], "core_training_seed": 2026,
               "pdf_pages_visually_reviewed": pages, "manuscript_directory": str(manuscript),
               "evidence_directory": str(evidence), "source_git_commit": commit, "source_remote_verified": True,
               "file_inventory_sha256": digest(output / "FILE_SHA256.json"),
               "document_verification_sha256": digest(manuscript / "DOCUMENT_CONTENT_VERIFICATION.json"),
               "visual_review_sha256": digest(manuscript / "VISUAL_REVIEW.json"),
               "portable_forecast_check_sha256": digest(output / "PORTABLE_FORECAST_CHECK.json"),
               "neural_price_comparisons_below_baseline": build["neural_price_comparisons_below_baseline"],
               "neural_return_comparisons_below_baseline": build["neural_return_comparisons_below_baseline"],
               "continuation_wall_envelope_seconds": compute["continuation_wall_envelope_seconds"],
               "optional_mechanism_run": False, "independent_confirmation": False, "Q3_values_read": False,
               "new_training_started_by_release": False, "externally_published_or_submitted": False,
               "goal_marked_complete_by_script": False, "script_sha256": digest(__file__)}
    save(output / "RELEASE.json", release)
    archive = output.with_suffix(".zip")
    assert not archive.exists(), "Preserve previous delivery archive"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                z.write(path, arcname=str(Path(output.name) / path.relative_to(output)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    final = {**release, "archive": str(archive), "archive_sha256": digest(archive),
             "archive_bytes": archive.stat().st_size, "archive_crc_check_passed": True}
    save(output.parent / (output.name + "_DELIVERY_RECEIPT.json"), final)
    print(json.dumps({k: final[k] for k in ("status", "completed_folds", "complete_tasks", "source_git_commit", "archive", "archive_bytes")}))


if __name__ == "__main__":
    main()
