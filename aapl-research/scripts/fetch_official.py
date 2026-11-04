"""Fetch small official source files through GitHub's public contents API."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "sources"
ITEMS = [
    ("xpatch_model", "stitsyuk/xPatch", "models/xPatch.py"),
    ("xpatch_network", "stitsyuk/xPatch", "layers/network.py"),
    ("xpatch_ema", "stitsyuk/xPatch", "layers/ema.py"),
    ("xpatch_license", "stitsyuk/xPatch", "LICENSE"),
    ("patchtst_license", "yuqinie98/PatchTST", "LICENSE"),
    ("chronos_license", "amazon-science/chronos-forecasting", "LICENSE"),
    ("kronos_license", "shiyu-coder/Kronos", "LICENSE"),
]


def fetch(item):
    key, repo, path = item
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    response = DEST / f"{key}_api.json"
    r = subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location",
                        "--connect-timeout", "8", "--max-time", "25", url,
                        "--output", str(response)], capture_output=True, text=True)
    record = {"key": key, "url": url, "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
              "exit_code": r.returncode}
    if r.returncode:
        record["error"] = r.stderr.strip()
    else:
        d = json.loads(response.read_text())
        content = base64.b64decode(d["content"])
        output = DEST / (key + (".py" if path.endswith(".py") else ".txt"))
        output.write_bytes(content)
        record.update(file=output.name, git_blob_sha=d["sha"], sha256=sha256(content).hexdigest(), bytes=len(content))
    print(json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(fetch, ITEMS))
    (DEST / "official_retrieval_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
