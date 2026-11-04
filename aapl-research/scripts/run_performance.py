"""Sequential bounded admission queue; never launches HPO or outer scoring."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aapl.common import write_json, utc_now

if __name__ == "__main__":
    started = time.perf_counter()
    queue = [(n, 'maximum', 600 if n in ('xpatch', 'patchtst') else 0)
             for n in ('xpatch', 'patchtst', 'chronos2_frozen', 'chronos2_lora')]
    record = {"started_at_utc": utc_now(), "queue": queue, "completed": [], "status": "running"}
    for name, mode, sustain in queue:
        if time.perf_counter() - started > 90 * 60:
            record["status"] = "pilot_budget_exceeded"
            break
        key = name + "_" + mode
        record["current"] = key
        write_json(ROOT / "outputs/implementation_performance_queue.json", record)
        print("START", key, utc_now(), flush=True)
        with (ROOT / "outputs" / ("profile_" + key + '_implementation.log')).open("w") as log:
            result = subprocess.run(["bash", str(ROOT / "scripts/python.sh"), "-m", "aapl.profile", name,
                                     "--mode", mode, "--sustain-seconds", str(sustain), '--tag', '_implementation', '--optimized'], cwd=ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        record["completed"].append({"job": key, "exit_code": result.returncode})
        print("END", key, "exit", result.returncode, flush=True)
        if result.returncode:
            record["status"] = "failed"
            break
    else:
        record["status"] = "completed"
    record["wall_seconds"] = time.perf_counter() - started
    write_json(ROOT / "outputs/implementation_performance_queue.json", record)
