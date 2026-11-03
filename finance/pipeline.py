"""Resume a complete frozen experiment without assistant polling or score edits."""
from datetime import datetime, timezone

from .cache import load_data, prepare_data
from .experiment import Experiment
from .io import ROOT, read_json, stable_hash, write_json
from .scheduler import Scheduler
from .search import configurations
from .splits import outer_partitions


def run(config, signature, run_name="development"):
    from .provenance import verify_freeze
    freeze = verify_freeze(config)
    if signature != freeze["signature"]:
        raise ValueError("Requested experiment signature differs from the verified freeze")
    cache, cache_identity = prepare_data(config)
    data = load_data(cache, cache_identity)
    splits = outer_partitions(data.days.tolist(), config["periods"]["validation_start"], config["periods"]["validation_end"],
        config["splits"]["calibration_sessions"], config["splits"]["purge_sessions"])
    search = configurations(config)
    directory = ROOT / "artifacts/rv30" / signature / run_name
    directory.mkdir(parents=True, exist_ok=True)
    setup = {"signature": signature, "config": config, "search": search, "partitions": [p.dictionary() for p in splits]}
    if (directory / "setup.json").exists() and stable_hash(read_json(directory / "setup.json")) != stable_hash(setup):
        raise ValueError("Resume setup changed; prior artifacts must be preserved")
    write_json(directory / "setup.json", setup)
    experiments = [Experiment(split, panel, model, search, config, signature, directory)
        for split in splits for panel in config["feature_sets"] for model in config["models"]]
    owner, finished = {}, 0
    with Scheduler(config, cache, cache_identity, directory) as scheduler:
        def submit(experiment, requests):
            for request in requests:
                owner[request["job_id"]] = experiment
            scheduler.submit(requests)

        for experiment in experiments:
            submit(experiment, experiment.initial_requests())
        while finished < len(experiments):
            for result in scheduler.poll():
                experiment = owner.pop(result["job_id"])
                if experiment.accept(result):
                    requests = experiment.advance()
                    if requests:
                        submit(experiment, requests)
                    elif experiment.stage == "done":
                        finished += 1
                        # Progress only; outer scores are withheld from human decisions.
                        print(f"Completed model groups {finished}/{len(experiments)}: {experiment.identity}", flush=True)
            write_json(directory / "progress.json", {"updated_utc": datetime.now(timezone.utc).isoformat(),
                "signature": signature, "finished_groups": finished, "total_groups": len(experiments),
                "budget_exhausted": scheduler.budget_exhausted,
                "groups": {e.identity: {"stage": e.stage, "waiting": len(e.waiting)} for e in experiments}})
    status = {"signature": signature, "run_name": run_name, "directory": str(directory.relative_to(ROOT)),
        "finished_groups": finished, "total_groups": len(experiments), "search_finished": True,
        "budget_exhausted": scheduler.budget_exhausted, "outer_used_for_search": False,
        "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(directory / "run.json", status)
    return status
