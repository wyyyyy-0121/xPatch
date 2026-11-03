"""Verify a restored environment and replay pre-2023 CPU checkpoints only."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
from importlib import metadata
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    output = ROOT / args.output_directory
    output.mkdir(parents=True, exist_ok=False)
    from finance.io import read_json, sha, write_json
    from finance.runtime import configure
    configure(2, [0, 1])
    commands = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
    (output / "pip_check.log").write_text(commands.stdout + commands.stderr)
    if commands.returncode:
        raise RuntimeError("Restored environment pip check failed")
    from finance.provenance import verify_freeze
    config = read_json(ROOT / "configs/rv30.json")
    freeze = verify_freeze(config)
    prefix = Path(sys.prefix).resolve()
    locations = {distribution.metadata["Name"]: str(Path(distribution.locate_file("")).resolve())
        for distribution in metadata.distributions()}
    if any(prefix not in Path(location).parents for location in locations.values()):
        raise ValueError("Restoration verification must use packages inside the new environment")
    from finance.jobs import initialize, run_job
    initialize(config, ROOT / freeze["payload"]["cache_directory"], freeze["payload"]["cache_identity"], [0, 1], False)
    smoke = ROOT / freeze["payload"]["smoke_directory"]
    results = []
    for panel in config["feature_sets"]:
        for name in ("last_rv", "har", "ridge", "lightgbm"):
            source = smoke / f"{panel}_{name}_smoke"
            original = read_json(source / "request.json")
            if max(original["partition"]["evaluation_dates"]) >= "2023-01-01":
                raise ValueError("Environment replay must stay before formal outer evaluation")
            identity = "environment_replay/" + original["job_id"]
            destination = output / original["job_id"]
            request = {**{k: original[k] for k in ("signature", "model", "panel", "config_id", "params", "seed", "partition")},
                "job_id": identity, "directory": str(destination.relative_to(ROOT)), "phase": "verify",
                "source_directory": str(source.relative_to(ROOT))}
            row = run_job(request)
            results.append(row)
            print(f"{panel}/{name}: {row['status']}", flush=True)
    summary = {"passed": len(results) == 8 and all(r["status"] == "complete" for r in results),
        "scope": "restored_environment_exact_dependencies_and_eight_pre2023_saved_CPU_model_replays",
        "base_signature": freeze["signature"], "executable": sys.executable, "python": sys.version,
        "restored_dependencies": freeze["payload"]["dependencies"],
        "package_locations": locations, "all_packages_inside_new_prefix": True,
        "source_effective_files": freeze["effective_source_files"], "gpu_execution_tested": False,
        "pip_check_log_sha256": sha(output / "pip_check.log"),
        "jobs": [{k: r.get(k) for k in ("job_id", "model", "panel", "status", "directory", "wall_seconds", "error")} for r in results],
        "completed_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__)}
    write_json(output / "summary.json", summary)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
