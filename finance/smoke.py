"""Pre-2023 integration/resource checks; never opens formal outer scores."""
from datetime import datetime, timezone
import time

from .cache import load_data, prepare_data
from .io import ROOT, read_json, sha, stable_hash, write_json
from .scheduler import Scheduler
from .search import configurations
from .splits import partition


def native_checks(config, search, directory):
    import gc
    import torch
    from .models import native_xpatch, seed_torch
    from .runtime import configure_torch
    configure_torch()
    rows = []
    for trial in search["models"]["xpatch"]:
        seed_torch(2026)
        params = trial["params"]
        started = time.monotonic()
        model = native_xpatch(params, 3, "cuda")
        x = torch.randn(2, params["seq_len"], 3, device="cuda", requires_grad=True)
        output = model(x)
        if output.shape != (2, 31, 3) or not torch.isfinite(output).all():
            raise ValueError(f"Illegal native configuration: {trial['id']}")
        output.square().mean().backward()
        if not torch.isfinite(x.grad).all() or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError(f"Illegal native gradients: {trial['id']}")
        torch.cuda.synchronize()
        rows.append({"config_id": trial["id"], "passed": True, "forward_backward_seconds": time.monotonic() - started,
            "parameters": sum(p.numel() for p in model.parameters())})
        del model, x, output
    gc.collect()
    torch.cuda.empty_cache()
    evidence = {"passed": True, "configurations": rows,
        "native_sources": {path: sha(ROOT / path) for path in ("models/xPatch.py", "layers/network.py", "layers/ema.py", "layers/dema.py", "layers/decomp.py", "layers/revin.py")},
        "notes": ["Preserved EMA uses triangular causal weights; tiny powers do not appear in a divisor.",
            "DEMA float scalars are tensors for its preserved CUDA constructor; original recurrence is unchanged.",
            "All 32 registered configurations passed float32 forward and backward; none excluded for numerical legality."]}
    write_json(directory / "native_checks.json", evidence)
    return evidence


def smoke(config, run_name=None):
    cache, cache_identity = prepare_data(config)
    data = load_data(cache, cache_identity)
    dates = [str(day) for day in data.days if day < "2023-01-01"]
    split = partition("pre2023_smoke", dates[:-20], dates[-20:])
    search = configurations(config)
    run_name = run_name or datetime.now(timezone.utc).strftime("smoke_%Y%m%dT%H%M%SZ")
    directory = ROOT / "artifacts/rv30/setup" / run_name
    directory.mkdir(parents=True, exist_ok=True)
    from .provenance import sources
    source_hashes = sources()
    identity = stable_hash({"config": config, "search": search, "partition": split.dictionary(), "sources": source_hashes})
    setup = {"signature": identity, "config": config, "search": search, "partition": split.dictionary(), "sources": source_hashes}
    if (directory / "setup.json").exists() and stable_hash(read_json(directory / "setup.json")) != stable_hash(setup):
        raise ValueError("Smoke identity changed; use a new run name and preserve the previous smoke")
    write_json(directory / "setup.json", setup)
    print(f"Smoke {run_name}: validating all 32 native xPatch configurations", flush=True)
    native_checks(config, search, directory)
    requests = []
    for panel in config["feature_sets"]:
        for name in config["models"]:
            trial = search["models"][name][0]
            if name == "xpatch":
                trial = max(search["models"][name], key=lambda t: (t["params"]["seq_len"], t["params"]["patch_len"]))
            if name == "chronos2":
                trial = search["models"][name][-1]
            job_id = f"{panel}_{name}_smoke"
            request = {"job_id": job_id, "directory": str((directory / job_id).relative_to(ROOT)),
                "signature": identity, "phase": "smoke", "partition": split.dictionary(),
                "model": name, "panel": panel, "config_id": trial["id"], "params": trial["params"], "seed": 2026,
                "smoke_limit_training": 4096, "smoke_limit_prediction": 128}
            if name == "xpatch":
                request["max_epochs"] = 1
            requests.append(request)
        trial = max((row for row in search["models"]["xpatch"] if row["params"]["ma_type"] == "dema"),
            key=lambda row: (row["params"]["seq_len"], -row["params"]["batch_size"]))
        job_id = f"{panel}_xpatch_dema_smoke"
        requests.append({"job_id": job_id, "directory": str((directory / job_id).relative_to(ROOT)),
            "signature": identity, "phase": "smoke", "partition": split.dictionary(), "model": "xpatch", "panel": panel,
            "config_id": trial["id"], "params": trial["params"], "seed": 2026,
            "smoke_limit_training": 4096, "smoke_limit_prediction": 128, "max_epochs": 1})
    # GPU models run serially while up to 18 CPUs handle independent tabular jobs.
    print("Smoke: starting both panels for all six adapters, using pre-2023 dates only", flush=True)
    with Scheduler(config, cache, cache_identity, directory) as scheduler:
        results = scheduler.map(requests)
        resumed = []
        for request in requests:
            if "dema_smoke" in request["job_id"]:
                job_id = request["job_id"] + "_resumed"
                resumed.append({**request, "job_id": job_id, "directory": str((directory / job_id).relative_to(ROOT)),
                    "resume_directory": request["directory"], "max_epochs": 2})
        results += scheduler.map(resumed)
        all_forecasts = requests + resumed
        replays = [{**{key: request[key] for key in ("signature", "model", "panel", "config_id", "params", "seed", "partition")},
            "job_id": "replay/" + request["job_id"], "directory": str((directory / "replay" / request["job_id"]).relative_to(ROOT)),
            "phase": "verify", "source_directory": request["directory"]} for request in all_forecasts]
        replay_results = scheduler.map(replays)
    summary = {"schema_version": 1, "signature": identity, "scope": "pre2023_smoke_only_not_formal_outer_search",
        "passed": all(row["status"] == "complete" for row in results + replay_results), "created_utc": datetime.now(timezone.utc).isoformat(),
        "independent_replay_jobs": len(replay_results), "independent_replay_passed": all(row["status"] == "complete" for row in replay_results),
        "replay_directories": [row["directory"] for row in replay_results],
        "jobs": [{k: row.get(k) for k in ("job_id", "status", "model", "panel", "wall_seconds", "cpu_seconds",
            "allocated_cpu_seconds", "fitting", "error", "directory")} for row in results],
        "scheduler": read_json(directory / "scheduler.json")}
    write_json(directory / "summary.json", summary)
    print(f"Smoke complete: {sum(row['status'] == 'complete' for row in results)}/{len(results)} forecast jobs and "
        f"{sum(row['status'] == 'complete' for row in replay_results)}/{len(replay_results)} replays passed. {directory / 'summary.json'}", flush=True)
    return summary
