"""Estimate compute scenarios using statuses and timings, never forecast scores."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from finance.runtime import configure
    configure(1)
    from finance.cache import load_data
    from finance.io import read_json, write_json
    from finance.splits import outer_partitions, inner_partitions
    freeze = read_json(ROOT / "delivery/rv30_freeze.json")
    config = freeze["payload"]["config"]
    directory = ROOT / "artifacts/rv30" / freeze["signature"] / "development"
    evidence = ROOT / "artifacts/rv30/setup/acceleration_20261103"
    data = load_data(ROOT / freeze["payload"]["cache_directory"], freeze["payload"]["cache_identity"])
    trials = read_json(ROOT / "configs/rv30_search.json")["models"]["xpatch"]
    rows, counts, known = [], Counter(), set()
    # Select timing/status/shape metadata only; do not inspect score/ranking fields.
    for path in directory.rglob("result.json"):
        result = read_json(path)
        counts[(result["model"], result["phase"], result["status"])]+=1
        known.add(result["job_id"])
        if result["model"] == "xpatch" and result["phase"] == "inner" and result["status"] == "complete":
            fitting = result["fitting"]
            rows.append({"id": result["config_id"], "panel": result["panel"], "fold": result["fold"],
                "samples": fitting["training_origins"], "epochs": fitting["completed_epochs"],
                "seconds": result["wall_seconds"], "ma_type": result["params"]["ma_type"]})
    rates = defaultdict(list)
    for row in rows:
        rates[(row["id"], row["panel"])].append(row["seconds"] / row["epochs"] / row["samples"])
    rate = {key: median(values) for key, values in rates.items()}
    outer = outer_partitions(data.days.tolist())
    scenarios = []
    for label, selection in (("cheaper_retained", "cheap"), ("middle_cost_retained", "middle"), ("expensive_retained", "expensive")):
        native_initial = native_later = accelerated_initial = accelerated_later = 0.
        jobs_initial = jobs_later = 0
        for panel in config["feature_sets"]:
            for partition in outer:
                inner = inner_partitions(partition, 3, 20, 20, 1)
                sizes = [len(data.origins(p.fit_dates, stride=5)) for p in inner]
                outer_size = len(data.origins(partition.fit_dates, stride=5))
                costs = []
                for trial in trials:
                    identity, params = trial["id"], trial["params"]
                    per_sample = rate[(identity,panel)]
                    # Only DEMA changes. A representative measured fit speedup is
                    # an approximation across configurations, not a guarantee.
                    speed = 3.744 if params["ma_type"] == "dema" else 1.
                    initial = sum(5*per_sample*n for p,n in zip(inner,sizes)
                        if f"{partition.name}_{panel}_xpatch/initial/{identity}/{p.name}/s2026" not in known)
                    remaining = sum(1 for p in inner if f"{partition.name}_{panel}_xpatch/initial/{identity}/{p.name}/s2026" not in known)
                    native_initial += initial
                    accelerated_initial += initial / speed + (0.5*remaining if speed>1 else 0)
                    jobs_initial += remaining
                    costs.append((per_sample*sum(sizes),identity,per_sample,speed))
                costs.sort()
                if selection == "cheap":
                    retained15,retained30,finalists,chosen=costs[:16],costs[:8],costs[:3],costs[0]
                elif selection == "middle":
                    retained15,retained30,finalists,chosen=costs[8:24],costs[12:20],costs[14:17],costs[15]
                else:
                    retained15,retained30,finalists,chosen=costs[-16:],costs[-8:],costs[-3:],costs[-1]
                for configurations, epochs, seeds in ((retained15,10,1),(retained30,15,1),(finalists,30,2)):
                    for cost,_,_,speed in configurations:
                        native_later += cost*epochs*seeds
                        accelerated_later += cost*epochs*seeds/speed+(0.5*3*seeds if speed>1 else 0)
                        jobs_later += 3*seeds
                _,_,per_sample,speed=chosen
                native_later += per_sample*outer_size*30*3
                accelerated_later += per_sample*outer_size*30*3/speed+(0.5*3 if speed>1 else 0)
                jobs_later += 3
        scenarios.append({"selection_cost_scenario":label,
            "remaining_xpatch_native_hours":(native_initial+native_later)/3600,
            "remaining_xpatch_accelerated_hours":(accelerated_initial+accelerated_later)/3600,
            "remaining_initial_native_hours":native_initial/3600,
            "remaining_initial_accelerated_hours":accelerated_initial/3600,
            "remaining_initial_jobs":jobs_initial,"remaining_later_jobs":jobs_later})
    scheduler=read_json(evidence/"scheduler.json")
    result={"created_utc":datetime.now(timezone.utc).isoformat(),"scope":"timing_status_metadata_only_no_forecast_scores_or_rankings",
        "counts":[{"model":k[0],"phase":k[1],"status":k[2],"count":v} for k,v in sorted(counts.items())],
        "completed_jobs":sum(counts.values()),"failed_jobs":sum(v for k,v in counts.items() if k[2]!="complete"),
        "xpatch_max_completed_task_seconds":max(r["seconds"] for r in rows),
        "xpatch_task_timeout_seconds":config["runtime"]["task_timeout_seconds"]["xpatch"],
        "gpu_hours_before_validation":scheduler["gpu_wall_hours"],"gpu_hours_limit":config["runtime"]["total_gpu_hours"],
        "cpu_core_hours_before_validation":scheduler["allocated_cpu_core_hours"],"cpu_core_hours_limit":config["runtime"]["total_cpu_core_hours"],
        "artifact_gib_before_validation":scheduler["trial_artifact_bytes"]/1024**3,"artifact_gib_limit":config["runtime"]["max_artifact_gib"],
        "search_duration_scenarios":scenarios,
        "method":"Median completed initial task seconds per actual epoch and training origin; fixed 5/15/30 halving and full 30-epoch finalist/outer ceilings. Retain configurations solely by measured cost; use nested cheaper/middle/expensive subsets. Apply representative 3.744x DEMA fit acceleration and 0.5-second graph setup per job; EMA unchanged.",
        "limitations":["Scenarios are not confidence intervals, probabilities or deadlines.",
            "All-stage epochs are conservatively assumed to run to ceilings; early stopping and selected epochs are unknown.",
            "A single longest-context maximum-scale speedup cannot establish every configuration's speedup.",
            "Initial task totals include calibration and setup; per-sample scaling also scales these fixed costs.",
            "Chronos remaining work, full outer prediction workloads, both independent replays, statistics and conditional confirmation are additional costs.",
            "No predictive-effectiveness or strategy-profitability termination probability can be inferred from timing metadata."]}
    write_json(evidence/"termination_risk.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ("counts","method","limitations")},indent=2))


if __name__=="__main__":
    main()
