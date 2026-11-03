"""Build a verified offline wheelhouse from the locked installed environment.

Missing original wheels are reconstructed from installed RECORD payloads and
explicitly labelled as local snapshots, never as original publisher wheels.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import io
import json
import os
from pathlib import Path
import re
import stat
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", required=True)
    parser.add_argument("--evidence-directory", required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, [0, 1])
    if os.getpriority(os.PRIO_PROCESS, 0) < 10:
        os.nice(10 - os.getpriority(os.PRIO_PROCESS, 0))
    from finance.io import read_json, write_json
    from finance.provenance import verify_freeze
    from packaging.tags import parse_tag, sys_tags
    from packaging.utils import parse_wheel_filename
    from packaging.version import Version
    frozen = verify_freeze(read_json(ROOT / "configs/rv30.json"))
    lock = frozen["payload"]["dependencies"]
    wheelhouse = Path(args.wheelhouse).resolve()
    evidence = ROOT / args.evidence_directory
    wheelhouse.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    progress = {"packages": []}
    for candidate in (evidence / "wheelhouse.json", evidence / "curl_transport_progress.json"):
        if candidate.exists():
            progress = read_json(candidate)
            break
    originals = {(normalized(r["name"]), r["version"]): r for r in progress["packages"] if r["status"] == "complete"}
    tags = set(sys_tags())
    rows = []
    for name, version in lock["installed_distributions"].items():
        original = originals.get((normalized(name), version))
        if original is not None and sha(wheelhouse / original["filename"]) == original["sha256"]:
            rows.append(original)
            print(name + ': original wheel verified', flush=True)
            continue
        distribution = metadata.distribution(name)
        if distribution.version != version:
            raise ValueError("Installed distribution changed: " + name)
        site = Path(distribution.locate_file("")).resolve()
        files = distribution.files
        if not files:
            raise ValueError("Distribution has no installed file inventory: " + name)
        metadata_paths = [str(p) for p in files if len(Path(str(p)).parts) == 2 and str(p).endswith(".dist-info/METADATA")]
        if len(metadata_paths) != 1:
            raise ValueError("Ambiguous distribution metadata: " + name)
        info_directory = str(Path(metadata_paths[0]).parent)
        wheel = distribution.read_text("WHEEL")
        wheel_tags = [line.split(": ", 1)[1] for line in wheel.splitlines() if line.startswith("Tag: ")]
        compatible_tag = next((tag for tag in wheel_tags if parse_tag(tag) & tags), None)
        if compatible_tag is None:
            raise ValueError("Installed wheel has no compatible tag: " + name)
        filename = f"{normalized(name).replace('-', '_')}-{version}-{compatible_tag}.whl"
        parsed_name, parsed_version, _, parsed_tags = parse_wheel_filename(filename)
        if normalized(parsed_name) != normalized(name) or parsed_version != Version(version) or not (tags & parsed_tags):
            raise ValueError("Invalid reconstructed wheel identity: " + name)
        destination = wheelhouse / filename
        temporary = destination.with_suffix(".whl.building")
        payload, omitted_scripts, relocated_data = [], [], []
        verified = 0
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for relative in sorted(files, key=str):
                archive_name = str(relative)
                source = Path(distribution.locate_file(relative)).resolve()
                if source != site and site not in source.parents:
                    # Entry point scripts are generated from entry_points.txt by pip.
                    bin_roots = {Path(sys.prefix) / "bin", Path(sys.base_prefix) / "bin", site.parents[2] / "bin"}
                    if any(root in source.parents for root in bin_roots):
                        omitted_scripts.append(archive_name)
                        continue
                    prefixes = (Path(sys.prefix), Path(sys.base_prefix), site.parents[2])
                    prefix = next((root for root in prefixes if root in source.parents), None)
                    data_relative = source.relative_to(prefix) if prefix is not None else None
                    if data_relative is None or data_relative.parts[0] not in ("share", "include", "lib", "lib64"):
                        raise ValueError("Distribution payload leaves its environment prefix: " + archive_name)
                    archive_name = f"{normalized(name).replace('-', '_')}-{version}.data/data/{data_relative.as_posix()}"
                    relocated_data.append({"installed_record_path": str(relative), "archive_path": archive_name})
                if ("__pycache__" in Path(archive_name).parts or archive_name.endswith((".pyc", ".pyo")) or
                        archive_name in {info_directory + "/RECORD", info_directory + "/INSTALLER", info_directory + "/REQUESTED", info_directory + "/direct_url.json"}):
                    continue
                if not source.is_file() or Path(archive_name).is_absolute() or ".." in Path(archive_name).parts:
                    raise ValueError("Missing or unsafe installed payload: " + archive_name)
                digest = sha(source)
                encoded = base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode()
                if relative.hash is not None:
                    if relative.hash.mode != "sha256" or relative.hash.value != encoded:
                        raise ValueError("Installed RECORD hash mismatch: " + archive_name)
                    verified += 1
                size = source.stat().st_size
                info = zipfile.ZipInfo(archive_name)
                info.create_system = 3
                info.external_attr = (stat.S_IFREG | stat.S_IMODE(source.stat().st_mode)) << 16
                with archive.open(info, "w", force_zip64=True) as target, source.open("rb") as input_file:
                    for block in iter(lambda: input_file.read(1024 * 1024), b""):
                        target.write(block)
                payload.append([archive_name, "sha256=" + encoded, str(size)])
            record = io.StringIO(newline="")
            writer = csv.writer(record, lineterminator="\n")
            writer.writerows(payload)
            writer.writerow([info_directory + "/RECORD", "", ""])
            archive.writestr(info_directory + "/RECORD", record.getvalue())
        temporary.replace(destination)
        payload_path = evidence / (normalized(name) + "_installed_payload.json")
        write_json(payload_path, {"name": name, "version": version, "source_prefix": sys.prefix,
            "payload_files": payload, "verified_original_RECORD_hashes": verified,
            "relocated_prefix_data": relocated_data,
            "omitted_generated_bin_scripts": omitted_scripts,
            "note": "pip regenerates entry point scripts from packaged entry_points.txt; this is an installed-file backup, not the original publisher wheel."})
        row = {"name": name, "version": version, "status": "complete", "filename": filename,
            "sha256": sha(destination), "bytes": destination.stat().st_size,
            "source": "local_locked_installed_file_snapshot_not_original_publisher_wheel",
            "payload_inventory": str(payload_path.relative_to(ROOT)), "payload_inventory_sha256": sha(payload_path),
            "verified_original_RECORD_hashes": verified}
        rows.append(row)
        write_json(evidence / "offline_progress.json", {"completed": len(rows), "planned": len(lock["installed_distributions"]), "packages": rows})
        print(name + ': installed-file snapshot complete', flush=True)
    # A resolver can prefer another tag of the same version before checking its
    # hash. Expose exactly the selected file for each package to offline pip.
    selected_directory = wheelhouse / "locked"
    selected_directory.mkdir(exist_ok=True)
    selected_names = {row["filename"] for row in rows}
    if {p.name for p in selected_directory.iterdir()} - selected_names:
        raise ValueError("Locked wheel directory contains unrelated files")
    for row in rows:
        selected = selected_directory / row["filename"]
        if selected.exists():
            if sha(selected) != row["sha256"]:
                raise ValueError("Preserve a previous locked wheel before replacing it: " + selected.name)
        else:
            os.link(wheelhouse / row["filename"], selected)
    summary = {"passed": len(rows) == len(lock["installed_distributions"]), "base_signature": frozen["signature"],
        "scope": "exact_locked_offline_wheelhouse_mixed_publisher_and_installed_file_snapshots",
        "source_prefix": sys.prefix, "python": sys.version, "wheelhouse": str(selected_directory),
        "acquisition_directory": str(wheelhouse),
        "dependencies_sha256": sha(ROOT / "delivery/rv30_dependencies.json"), "packages": rows,
        "completed_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(Path(__file__))}
    write_json(evidence / "offline_wheelhouse.json", summary)
    (evidence / "requirements.hashed.txt").write_text("".join(
        f"{r['name']}=={r['version']} --hash=sha256:{r['sha256']}\n" for r in rows))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
