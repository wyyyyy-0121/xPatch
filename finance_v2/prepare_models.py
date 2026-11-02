"""Fetch immutable public checkpoints with LFS SHA256 verification; no credentials."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .io import ROOT, digest, write_json


REVISIONS = {
    "kronos_small": ("NeoQuasar/Kronos-small", "901c26c1332695a2a8f243eb2f37243a37bea320"),
    "kronos_tokenizer": ("NeoQuasar/Kronos-Tokenizer-base", "0e0117387f39004a9016484a186a908917e22426"),
    "chronos2": ("amazon/chronos-2", "29ec3766d36d6f73f0696f85560a422f50e8498c"),
}


def fetch(url, path, timeout=300):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".download")
    subprocess.run(["curl", "-L", "--fail", "--retry", "1", "--connect-timeout", "8",
                    "--max-time", str(timeout), "-sS", url, "-o", str(temporary)], check=True)
    temporary.replace(path)


def prepare(output, endpoint):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = {"schema_version": 1, "transport_endpoint": endpoint,
            "scope": "public checkpoints downloaded at pinned repository revisions; mirror metadata is not an independent provenance proof",
            "models": {}}
    for name, (repo, revision) in REVISIONS.items():
        directory = output / name
        directory.mkdir(exist_ok=True)
        metadata = directory / "metadata.json"
        if not metadata.exists():
            fetch(f"{endpoint}/api/models/{repo}?blobs=true", metadata, 20)
        info = json.loads(metadata.read_text())
        if info["sha"] != revision or info["id"] != repo:
            raise ValueError(f"Wrong repository/revision for {name}")
        files = {}
        for entry in info["siblings"]:
            filename = entry["rfilename"]
            if filename not in {"config.json", "model.safetensors", "README.md"}:
                continue
            path = directory / filename
            expected = entry.get("lfs", {}).get("sha256")
            if not path.exists() or (expected and digest(path) != expected):
                print(f"FETCH {name}/{filename}", flush=True)
                fetch(f"{endpoint}/{repo}/resolve/{revision}/{filename}?download=true", path)
            actual = digest(path)
            if expected and actual != expected:
                raise ValueError(f"LFS hash mismatch: {name}/{filename}")
            if not expected:
                content = path.read_bytes()
                git_blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
                if git_blob != entry["blobId"]:
                    raise ValueError(f"Git blob identity mismatch: {name}/{filename}")
            if filename.endswith("safetensors") and not expected:
                raise ValueError(f"No LFS digest supplied: {name}")
            files[filename] = {"sha256": actual, "bytes": path.stat().st_size,
                               "lfs_hash_verified": bool(expected)}
        if set(files) != {"config.json", "model.safetensors", "README.md"}:
            raise ValueError(f"Missing checkpoint files: {name}")
        lock["models"][name] = {"repository": repo, "revision": revision,
                                "license": info.get("cardData", {}).get("license"),
                                "directory": str(directory.resolve().relative_to(ROOT)), "files": files}
        write_json(output / "model_lock.json", lock)
        print(f"VERIFIED {name} revision={revision}", flush=True)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "artifacts/aapl_v2/weights"))
    parser.add_argument("--endpoint", default="https://huggingface.co")
    args = parser.parse_args()
    prepare(args.output, args.endpoint.rstrip("/"))
