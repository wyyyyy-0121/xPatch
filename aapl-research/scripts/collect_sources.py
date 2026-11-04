"""Collect small public method documents only; never download weights or market data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "sources"
DEST.mkdir(exist_ok=True)
SOURCES = {
    "xpatch_readme.md": "https://raw.githubusercontent.com/stitsyuk/xPatch/main/README.md",
    "xpatch_paper.pdf": "https://arxiv.org/pdf/2412.17323",
    "chronos_readme.md": "https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/README.md",
    "kronos_readme.md": "https://raw.githubusercontent.com/shiyu-coder/Kronos/master/README.md",
    "kronos_paper.pdf": "https://arxiv.org/pdf/2508.02739",
    "timesfm_readme.md": "https://raw.githubusercontent.com/google-research/timesfm/master/README.md",
    "patchtst_paper.pdf": "https://arxiv.org/pdf/2211.14730",
}


def collect(item):
    name, url = item
    output = DEST / name
    result = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location",
         "--connect-timeout", "8", "--max-time", "35", "--max-filesize", "16000000",
         url, "--output", str(output)], capture_output=True, text=True)
    record = {"file": name, "url": url, "exit_code": result.returncode,
              "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    if result.returncode:
        record["error"] = result.stderr.strip()
    else:
        content = output.read_bytes()
        record.update(bytes=len(content), sha256=sha256(content).hexdigest())
        if name.endswith(".pdf"):
            conversion = subprocess.run(["pdftotext", "-layout", str(output),
                                         str(output.with_suffix(".txt"))], capture_output=True, text=True)
            record["pdftotext_exit_code"] = conversion.returncode
    print(json.dumps(record), flush=True)
    return record


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(collect, SOURCES.items()))
    (DEST / "retrieval_manifest.json").write_text(
        json.dumps({"scope": "Small public method documents; no checkpoints or market data",
                    "sources": records}, indent=2) + "\n")
