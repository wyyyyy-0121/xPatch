"""Meter the unchanged forward collector; idle sleeps do not lease CPU time.

Completed actions charge active wall time times their four-CPU allocation.
Interrupted actions retain their entire immutable reservation. Unknown usage is
never counted as zero. This extends the original active-allocation budget.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import os
import math
import resource
import threading
import time
from types import SimpleNamespace
import uuid

from .io import ROOT, read_json, sha


ACTIVE = None
ACTION_LIMITS = {"observe": 7 * 86400, "verify_1": 12 * 3600,
                 "verify_2": 12 * 3600, "finalize": 3600, "accept": 3600}


def now():
    return datetime.now(timezone.utc).isoformat()


def historical_usage(signature):
    directory = ROOT / "artifacts/rv30" / signature
    cpu, gpu = 0., 0.
    for path in directory.rglob("result.json"):
        result = read_json(path)
        if result["signature"] != signature:
            raise ValueError("Another signature appeared in the resource ledger")
        cpu += result["allocated_cpu_seconds"]
        if result["model"] in ("xpatch", "chronos2"):
            gpu += result["wall_seconds"]
    charges = read_json(directory / "interrupted_resource_charges.json")
    cpu += sum(row["allocated_cpu_seconds"] for row in charges.values())
    gpu += sum(row["gpu_wall_seconds"] for row in charges.values())
    if not all(math.isfinite(value) and value >= 0 for value in (cpu, gpu)):
        raise ValueError("Negative resource charge")
    return cpu, gpu


def resource_records(lock):
    from .prospective import observation_directory
    directory = observation_directory(lock)
    entries, files = [], {}
    for path in sorted((directory / "resources").glob("*/start.json")):
        start = read_json(path)
        if (start["plan_signature"] != lock["signature"] or start["allocation"] != 4 or
                len(start["cpu_ids"]) > 4 or len(set(start["cpu_ids"])) != len(start["cpu_ids"]) or
                start["action"] not in ACTION_LIMITS or start["reserved_cpu_seconds"] <= 0 or
                start["reserved_cpu_seconds"] > 4 * ACTION_LIMITS[start["action"]]):
            raise ValueError("Unknown forward allocation or reservation")
        end_path = path.with_name("end.json")
        end = read_json(end_path) if end_path.exists() else None
        charged = start["reserved_cpu_seconds"]
        if end is not None:
            if (end["start_sha256"] != sha(path) or
                    not 0 <= end["allocated_cpu_seconds"] <= charged or
                    end["gpu_wall_seconds"] != 0 or not end["within_reserved_limit"] or
                    end["finished_utc"] < start["started_utc"]):
                raise ValueError("Forward resource receipt differs from its reservation")
            charged = end["allocated_cpu_seconds"]
            files[str(end_path.relative_to(ROOT))] = sha(end_path)
        files[str(path.relative_to(ROOT))] = sha(path)
        entries.append({"start": start, "end": end, "charged_cpu_seconds": charged,
                        "path": str(path.relative_to(ROOT))})
    return entries, files


def recover_interrupted(lock):
    """Close a dead process at its full reservation, retaining the original start."""
    from .prospective import observation_directory
    from .observation import Journal, immutable_json
    directory=observation_directory(lock);journal=Journal(directory,lock['signature'])
    rows,_=resource_records(lock)
    for row in rows:
        if row['end'] is not None:continue
        start=row['start'];process=Path(f"/proc/{start['pid']}/stat")
        if process.exists() and process.read_text().split()[21]==start['kernel_start_ticks']:
            raise RuntimeError('A forward process is still alive; do not replace its resource reservation')
        value={'start_sha256':sha(ROOT/row['path']),'finished_utc':now(),'success':False,
            'allocated_cpu_seconds':start['reserved_cpu_seconds'],'gpu_wall_seconds':0.,
            'within_reserved_limit':True,'event_count':len(journal.events),'last_event_hash':journal.last_hash,
            'interrupted':True,'accounting':'full immutable reservation, not assumed measured usage'}
        proof=directory/f"{start['action']}.json"
        if start['action'].startswith('verify_') and proof.exists():
            checked=read_json(proof)
            if (checked.get('passed') is True and checked.get('plan_signature')==lock['signature'] and
                    checked.get('last_event_hash')==journal.last_hash and checked['completed_utc']>=start['started_utc']):
                value['success']=True
                value['verification']={'path':str(proof.relative_to(ROOT)),'sha256':sha(proof)}
        immutable_json((ROOT/row['path']).with_name('end.json'),value)


def resource_evidence(config, lock, journal, verifications):
    entries, files = resource_records(lock)
    covered, rounds = set(), set()
    for row in entries:
        start, end = row["start"], row["end"]
        if end is None:
            if (ACTIVE is None or ACTIVE.start != start or
                    start["action"] not in ("finalize", "accept")):
                raise ValueError("Unresolved forward resource reservation; do not assume zero usage")
            continue
        first, last = start["event_count"], end["event_count"]
        if not 0 <= first <= last <= len(journal.events):
            raise ValueError("Resource receipt has an invalid observation range")
        prefix = lambda count: journal.events[count - 1]["hash"] if count else None
        if (prefix(first) != start["last_event_hash"] or prefix(last) != end["last_event_hash"]):
            raise ValueError("Resource receipt does not bind the actual observation chain")
        if start["action"] == "observe":
            covered.update(range(first + 1, last + 1))
        if start["action"].startswith("verify_") and end["success"]:
            number = start["action"].split("_")[1]
            if (end.get("verification") == verifications[number] and
                    start["full_replay_required"] and first == last == len(journal.events)):
                rounds.add(number)
    required = {event["sequence"] for event in journal.events
                if event["kind"] in ("prediction", "issuance_commit") or
                (event["kind"] == "bars" and event["payload"]["purpose"] in ("warmup", "live"))}
    if not required.issubset(covered) or rounds != {"1", "2"}:
        raise ValueError("Actual observation and both full replay rounds require metered receipts")
    cpu, gpu = historical_usage(lock["payload"]["base_signature"])
    cpu += sum(row["charged_cpu_seconds"] for row in entries)
    from .prospective import observation_directory
    roots=(ROOT/'artifacts/rv30'/lock['payload']['base_signature'],observation_directory(lock))
    artifact_bytes=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
    passed = (cpu < config["runtime"]["total_cpu_core_hours"] * 3600 and
              gpu < config["runtime"]["total_gpu_hours"] * 3600 and
              artifact_bytes < config['runtime']['max_artifact_gib']*1024**3)
    if not passed:
        raise ValueError("The original cumulative compute budget is exhausted")
    return {"passed": True, "allocated_cpu_core_hours": cpu / 3600,
            "gpu_wall_hours": gpu / 3600, "files": files,
            "cumulative_artifact_bytes":artifact_bytes,
            "metered_prediction_and_commit_events": len(required),
            "unclosed_finalizer_charged_at_full_reservation": ACTIVE is not None,
            "accounting": "active_allocated_wall_seconds; idle sleeps excluded; unresolved reservations charged in full"}


class Meter:
    def __init__(self, config, lock, action):
        from .observation import Journal, immutable_json
        from .prospective import observation_directory
        self.config, self.lock, self.action = config, lock, action
        if len(os.sched_getaffinity(0))>4:
            raise ValueError('Forward metering requires an inherited allocation of at most four CPUs')
        self.directory = observation_directory(lock)
        self.path = self.directory / "resources" / uuid.uuid4().hex
        self.guard, self.done = threading.RLock(), threading.Event()
        self.active_seconds, self.segment = 0., time.monotonic()
        self.limit = ACTION_LIMITS[action]
        journal = Journal(self.directory, lock["signature"])
        existing, _ = resource_records(lock)
        cpu, gpu = historical_usage(lock["payload"]["base_signature"])
        remaining = config["runtime"]["total_cpu_core_hours"] * 3600 - cpu - sum(r["charged_cpu_seconds"] for r in existing)
        if remaining <= self.limit * 4 or gpu >= config["runtime"]["total_gpu_hours"] * 3600:
            raise ValueError("Insufficient original budget for the bounded forward action")
        self.start = {"plan_signature": lock["signature"], "action": action, "allocation": 4,
                      "reserved_cpu_seconds": self.limit * 4, "started_utc": now(),
                      "pid": os.getpid(), "kernel_start_ticks": Path("/proc/self/stat").read_text().split()[21],
                      "cpu_ids":sorted(os.sched_getaffinity(0)),
                      "event_count": len(journal.events), "last_event_hash": journal.last_hash,
                      "full_replay_required": action.startswith("verify_") and not (self.directory / f"{action}.json").exists()}
        immutable_json(self.path / "start.json", self.start)

    def elapsed(self):
        return self.active_seconds + (time.monotonic() - self.segment if self.segment is not None else 0.)

    def sleep(self, seconds):
        # Only the collector's explicit idle sleeps release its allocation.
        with self.guard:
            self.active_seconds = self.elapsed()
            self.segment = None
        try:
            time.sleep(seconds)
        finally:
            with self.guard:
                self.segment = time.monotonic()

    def watch(self):
        while not self.done.wait(5):
            with self.guard:
                exhausted = (self.elapsed() >= self.limit - 5 or
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 >= self.config['runtime']['memory_budget_gib']*1024**3)
            if exhausted:
                # An interrupted action retains its full reservation. Do not
                # pretend to have measured a clean completion after a hard stop.
                os._exit(75)

    def finish(self, success):
        from .observation import Journal, immutable_json
        self.done.set()
        journal = Journal(self.directory, self.lock["signature"])
        value = {"start_sha256": sha(self.path / "start.json"), "finished_utc": now(),
                 "success": bool(success), "allocated_cpu_seconds": min(self.limit, self.elapsed()) * 4,
                 "within_reserved_limit":self.elapsed()<self.limit,
                 "gpu_wall_seconds": 0., "event_count": len(journal.events), "last_event_hash": journal.last_hash}
        proof = self.directory / f"{self.action}.json"
        if self.action.startswith("verify_") and proof.exists():
            value["verification"] = {"path": str(proof.relative_to(ROOT)), "sha256": sha(proof)}
        immutable_json(self.path / "end.json", value)


@contextmanager
def metered(config, lock, action):
    global ACTIVE
    from . import prospective
    if ACTIVE is not None:
        raise RuntimeError("Forward actions cannot nest their resource allocations")
    meter = Meter(config, lock, action)
    original = prospective.time
    prospective.time = SimpleNamespace(sleep=meter.sleep)
    ACTIVE = meter
    watcher = threading.Thread(target=meter.watch, daemon=True)
    watcher.start()
    success = False
    try:
        yield
        success = True
    finally:
        prospective.time = original
        meter.finish(success)
        ACTIVE = None


def main():
    import argparse
    from .runtime import configure, exclusive_run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("observe", "verify", "finalize", "accept"))
    parser.add_argument("--round", type=int, choices=(1, 2))
    args = parser.parse_args()
    configure(4, cpu_ids=sorted(os.sched_getaffinity(0))[:4])
    from .prospective import read_plan, observe, verify_observation
    from .completion import finalize_forward, accept_forward
    config = read_json(ROOT / "configs/rv30.json")
    with exclusive_run(ROOT / "artifacts/rv30/driver.lock"):
        lock = read_plan(config)
        recover_interrupted(lock)
        if args.action == "verify" and args.round is None:
            parser.error("verify requires --round")
        action = f"verify_{args.round}" if args.action == "verify" else args.action
        with metered(config, lock, action):
            if action == "observe":
                result = observe(config)
            elif action.startswith("verify_"):
                result = verify_observation(config, args.round)
            else:
                directory = ROOT / lock["payload"]["run_directory"]
                function = finalize_forward if action == "finalize" else accept_forward
                result = function(config, directory)
        print(__import__("json").dumps(result))
        return 0 if result.get("passed") else 2


if __name__ == "__main__":
    # Use the canonical module instance so acceptance sees the active meter.
    from finance.forward_runtime import main as run
    raise SystemExit(run())
