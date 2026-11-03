"""Resolve exact locked wheels on PyPI and preserve their published SHA256s."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import email
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from urllib.parse import quote, urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def retry(action):
    for attempt in range(3):
        try:
            return action()
        except (OSError, ValueError) as error:
            if attempt == 2:
                raise error
            time.sleep(attempt + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--evidence-directory", required=True)
    args = parser.parse_args()
    wheelhouse = Path(args.wheelhouse).resolve()
    evidence = Path(args.evidence_directory).resolve()
    wheelhouse.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    from packaging.tags import sys_tags
    from packaging.utils import parse_wheel_filename
    from packaging.version import Version
    from finance.io import read_json, write_json
    locked = read_json(ROOT / "delivery/rv30_dependencies.json")
    requested = locked["installed_distributions"]
    tags = {tag: i for i, tag in enumerate(sys_tags())}
    # Reuse cached bytes only when their SHA256 matches the chosen PyPI file.
    cache = {}
    for path in (Path.home() / ".cache/pip").rglob("*.body"):
        if path.stat().st_size < 1000:
            continue
        with path.open("rb") as stream:
            if stream.read(4) != b"PK\x03\x04":
                continue
        try:
            with zipfile.ZipFile(path) as archive:
                metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
                if len(metadata) == 1:
                    message = email.message_from_bytes(archive.read(metadata[0]))
                    cache.setdefault((normalized(message["Name"]), message["Version"]), []).append(path)
        except (OSError, zipfile.BadZipFile, RuntimeError):
            continue

    def one(name, version):
        started = time.monotonic()
        url = f"https://pypi.org/pypi/{quote(name)}/{quote(version)}/json"
        def fetch_metadata():
            result = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                "--connect-timeout", "10", "--max-time", "30", "--retry", "1", "--retry-delay", "1", url],
                capture_output=True, text=True, timeout=65)
            if result.returncode:
                raise OSError(result.stderr.strip())
            return json.loads(result.stdout)
        try:
            metadata = retry(fetch_metadata)
            if normalized(metadata["info"]["name"]) != normalized(name) or Version(metadata["info"]["version"]) != Version(version):
                raise ValueError("PyPI release identity differs from lock")
            choices = []
            for entry in metadata["urls"]:
                if entry["packagetype"] != "bdist_wheel" or entry.get("yanked"):
                    continue
                wheel_name, wheel_version, build, wheel_tags = parse_wheel_filename(entry["filename"])
                matches = [tags[tag] for tag in wheel_tags if tag in tags]
                if normalized(wheel_name) == normalized(name) and wheel_version == Version(version) and matches:
                    choices.append((min(matches), entry["filename"], entry))
            if not choices:
                raise ValueError("No compatible non-yanked wheel for this locked release")
            entry = sorted(choices, key=lambda v: (v[0], v[1]))[0][2]
            if Path(entry["filename"]).name != entry["filename"]:
                raise ValueError("Unsafe wheel filename")
            parsed = urlparse(entry["url"])
            if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
                raise ValueError("Wheel URL is outside the official PyPI file host")
            expected = entry["digests"]["sha256"]
            destination = wheelhouse / entry["filename"]
            source = "existing_verified_wheelhouse"
            if not destination.exists() or digest(destination) != expected:
                cached = next((p for p in cache.get((normalized(name), version), []) if digest(p) == expected), None)
                if cached is not None:
                    shutil.copyfile(cached, destination)
                    source = "pip_cache_bytes_verified_against_PyPI_SHA256"
                else:
                    def download():
                        fd, temporary_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".partial", dir=wheelhouse)
                        temporary = Path(temporary_name)
                        try:
                            os.close(fd)
                            transfer = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                "--connect-timeout", "10", "--max-time", "300", "--retry", "1", "--retry-delay", "1",
                                "--output", str(temporary), entry["url"]], capture_output=True, text=True, timeout=610)
                            if transfer.returncode:
                                raise OSError(transfer.stderr.strip())
                            if digest(temporary) != expected:
                                raise ValueError("Downloaded wheel SHA256 differs from PyPI")
                            temporary.replace(destination)
                        finally:
                            temporary.unlink(missing_ok=True)
                    retry(download)
                    source = "official_PyPI_download"
            if digest(destination) != expected:
                raise ValueError("Wheelhouse file fingerprint changed")
            return {"name": name, "version": version, "status": "complete", "filename": destination.name,
                "sha256": expected, "bytes": destination.stat().st_size, "source": source,
                "release_url": url, "file_url": entry["url"], "wall_seconds": time.monotonic() - started}
        except Exception as error:
            return {"name": name, "version": version, "status": "failed", "error_type": type(error).__name__,
                "error": str(error), "wall_seconds": time.monotonic() - started}

    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(one, name, version) for name, version in requested.items()]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(evidence / "progress.json", {"completed": len(results), "planned": len(requested), "packages": results})
            print(json.dumps({k: result.get(k) for k in ("name", "version", "status", "bytes", "source", "error")}), flush=True)
    passed = len(results) == len(requested) and all(r["status"] == "complete" for r in results)
    summary = {"passed": passed, "scope": "exact_locked_wheel_acquisition_not_installation_validation",
        "python": sys.version, "locked_dependencies_sha256": digest(ROOT / "delivery/rv30_dependencies.json"),
        "script_sha256": digest(__file__ if isinstance(__file__, Path) else Path(__file__)),
        "wheelhouse": str(wheelhouse), "packages": sorted(results, key=lambda r: normalized(r["name"])),
        "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(evidence / "wheelhouse.json", summary)
    if passed:
        (evidence / "requirements.hashed.txt").write_text("".join(
            f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in summary["packages"]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
