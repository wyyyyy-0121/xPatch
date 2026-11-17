"""One bounded public retrieval; the immutable source spec starts the window."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import urllib.error
import urllib.request

BASE = Path(__file__).resolve().parent


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def main():
    os.sched_setaffinity(0, {26})
    spec = json.loads((BASE / "SOURCE_SPEC.json").read_text())
    manifest = BASE / "FETCH_RECEIPTS.json"
    saved = json.loads(manifest.read_text()) if manifest.exists() else {"records": [], "response_bytes": 0}
    before = time.time()
    if len(saved["records"]) >= 6 or saved["response_bytes"] >= 24 * 1024 * 1024:
        raise RuntimeError("Frozen retrieval budget exhausted")
    if before >= spec["deadline_epoch"] - .5:
        raise RuntimeError("Original shared source deadline elapsed")
    url, purpose = sys.argv[1:3]
    number = len(saved["records"]) + 1
    record = {"number": number, "url": url, "purpose": purpose, "started_epoch": before,
              "source_spec_sha256": hashlib.sha256((BASE / "SOURCE_SPEC.json").read_bytes()).hexdigest(),
              "CPU": list(os.sched_getaffinity(0))}
    chunks, response = [], None

    def alarm(*args):
        raise TimeoutError("Per-request wall limit or original source deadline")

    signal.signal(signal.SIGALRM, alarm)
    budget = min(20, spec["deadline_epoch"] - before - .25)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (public academic source verification)"})
        opener = urllib.request.build_opener(NoRedirect)
        try:
            response = opener.open(request, timeout=budget)
        except urllib.error.HTTPError as error:
            response = error
        record.update(status=int(response.code), final_url=response.geturl(), headers=dict(response.headers.items()))
        remaining = min(8 * 1024 * 1024, 24 * 1024 * 1024 - saved["response_bytes"])
        while remaining:
            block = response.read(min(remaining, 65536))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        if remaining == 0:
            record["body_at_byte_cap"] = True
    except BaseException as error:
        record["error"] = repr(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        if response is not None:
            response.close()
        data = b"".join(chunks)
        filename = f"REQUEST_{number}.body"
        (BASE / filename).write_bytes(data)
        record.update(body_file=filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                      ended_epoch=time.time())
        record["elapsed_seconds"] = record["ended_epoch"] - before
        saved["records"].append(record)
        saved["response_bytes"] += len(data)
        saved["last_network_finished_epoch"] = record["ended_epoch"]
        manifest.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
