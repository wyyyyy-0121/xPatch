"""Read-only bounded observation of the approved eight-worker continuation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import torch

RUN = Path("/home/ayaka/wslcode/aapl-research/outputs/multiscale/remaining_continuations/20261108T040346048516Z-parallel8")
NEURAL = ("mlp", "gru", "lstm", "xpatch")
BOOT = Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def identity(ident):
    proc = Path("/proc") / str(ident["pid"])
    try:
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        matches = (BOOT == ident["boot_id"] and int(fields[19]) == ident["start_ticks"] and
                   hashlib.sha256((proc / "cmdline").read_bytes()).hexdigest() == ident["argv_sha256"])
        if not matches:
            raise ValueError("Process identity changed")
        return {"pid": ident["pid"], "start_ticks": ident["start_ticks"], "identity_matches": True,
                "live_state": fields[0],
                "cpu_seconds": (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")}
    except FileNotFoundError:
        return {"pid": ident["pid"], "transition_during_observation": True}


def observe():
    state = json.loads((RUN / "STATE.json").read_text())
    manifest = json.loads(Path(state["manifest"]).read_text())
    folders = {j["key"]: Path(j["folder"]) for j in manifest["jobs"]}
    fits, running = 340, []
    for key, job in state["jobs"].items():
        folder = folders[key]
        completed = []
        path = folder / "fold_state.json"
        if path.is_file():
            fold = json.loads(path.read_text())
            for model in NEURAL:
                summary = folder / model / "training_summary.json"
                recorded = fold.get("models", {}).get(model, {}).get("training_summary_sha256")
                if summary.is_file() and hashlib.sha256(summary.read_bytes()).hexdigest() == recorded:
                    completed.append(model)
                    fits += 1
        if job["phase"] != "running":
            continue
        row = {"job": key, "slot": job["slot"], "completed_neural_models": completed, **identity(job["process"])}
        candidates = [p for p in folder.glob("*/latest.pt") if p.parent.name not in completed]
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            checkpoint = torch.load(latest, map_location="cpu", weights_only=True)
            if checkpoint["schema"] != 1:
                raise ValueError("Unknown checkpoint schema")
            progress = checkpoint["state"]
            row.update(checkpoint_model=latest.parent.name,
                       checkpoint_identity=checkpoint["identity"],
                       checkpoint_modified_epoch=latest.stat().st_mtime,
                       epoch=progress["epoch"], optimizer_steps=progress["steps"],
                       cursor=progress["cursor"], stale_epochs=progress["stale_epochs"])
        running.append(row)
    if state["maximum_workers"] != 8 or state["deadline_epoch"] != 1794143421.9147575 or len(running) > 8:
        raise ValueError("Resource scope differs")
    return {"at_utc": datetime.now(timezone.utc).isoformat(), "phase": state["phase"],
            "completed_core_folds": 85 + sum(j["phase"] == "completed" for j in state["jobs"].values()),
            "completed_hash_checked_neural_training_summaries": fits, "required_neural_fits": 400,
            "running": running, "pending": sum(j["phase"] == "pending" for j in state["jobs"].values()),
            "maximum_workers": 8, "deadline_epoch": state["deadline_epoch"],
            "controller": identity(state["controller"]), "watchdog": identity(state["watchdog"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=40, choices=range(1, 56))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.sched_getaffinity(0) != {24, 26} or os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("Observer requires the two reserved CPUs and hidden CUDA")
    before = observe()
    print(json.dumps({"stage": "before", "folds": before["completed_core_folds"],
                      "neural_summaries": before["completed_hash_checked_neural_training_summaries"],
                      "running": len(before["running"])}), flush=True)
    started = time.monotonic()
    time.sleep(args.seconds)
    after = observe()
    old = {(r["job"], r["pid"]): r for r in before["running"]}
    for row in after["running"]:
        prev = old.get((row["job"], row["pid"]))
        if prev and "cpu_seconds" in row and "cpu_seconds" in prev:
            row["cpu_seconds_during_wait"] = round(row["cpu_seconds"] - prev["cpu_seconds"], 2)
            if (row.get("checkpoint_model") == prev.get("checkpoint_model") and
                    "optimizer_steps" in row and "optimizer_steps" in prev):
                row["optimizer_steps_during_wait"] = row["optimizer_steps"] - prev["optimizer_steps"]
    report = {"status": "verified_live_wait_with_schema1_checkpoint_and_cpu_progress",
              "before": before, "after": after, "wait_seconds": round(time.monotonic() - started, 2),
              "training_started_by_observer": False, "signals_sent": False, "Q3_values_read": False,
              "goal_marked_complete": False,
              "limitations": ["Training-summary counts are progress markers, not full restoration checks.",
                             "Checkpoint steps and process CPU deltas are sampled, not total-study CPU accounting."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"stage": "after", "folds": after["completed_core_folds"],
                      "neural_summaries": after["completed_hash_checked_neural_training_summaries"],
                      "running": len(after["running"]), "pending": after["pending"],
                      "optimizer_step_increments": {r["job"]: r.get("optimizer_steps_during_wait") for r in after["running"]},
                      "receipt": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
