"""Package accepted documentation; no models, market arrays or old ZIP extraction."""
from pathlib import Path
import hashlib
import json
import shutil
from zipfile import ZipFile, ZIP_DEFLATED
from verify_documentation import verify_previous_edition

BASE = Path(__file__).resolve().parent
PACKAGE = "AAPL_READOUT_DOCUMENTATION_20261110"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    acceptance = json.loads((BASE / "review/DOCUMENT_ACCEPTANCE.json").read_text())
    assert acceptance["status"] == "passed_document_delivery_research_goal_unachieved"
    verify_previous_edition(BASE)
    build = json.loads((BASE / "review/DOCUMENT_BUILD.json").read_text())
    previews = {str(Path(d["preview_directory"]).relative_to(BASE)) for d in build["documents"]}
    for doc in build["documents"]:
        assert sha(BASE / f'{doc["name"]}_EN.md') == doc["source_sha256"]
        assert sha(BASE / f'{doc["name"]}_EN.docx') == doc["docx_sha256"]
        assert sha(BASE / f'{doc["name"]}_EN.pdf') == doc["pdf_sha256"]
    accepted = []
    forbidden = {".npy", ".npz", ".pt", ".pth", ".csv", ".parquet", ".arrow", ".pkl", ".pickle"}
    for path in sorted(BASE.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(BASE)
        if rel.parts[0] == "release" or rel == Path("DOCUMENTATION_MANIFEST.json"):
            continue
        if "__pycache__" in rel.parts or "initial_render" in rel.parts:
            continue
        if any(part.startswith("lo_") for part in rel.parts):
            continue
        if rel.parts[0] == "page_previews" and not any(str(rel).startswith(p + "/") for p in previews):
            continue
        assert path.suffix.lower() not in forbidden, f"Unexpected market/model file: {rel}"
        accepted.append(path)
    manifest = {str(p.relative_to(BASE)): sha(p) for p in accepted}
    manifest_path = BASE / "DOCUMENTATION_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    destination = BASE / "release" / PACKAGE
    destination.mkdir(parents=True, exist_ok=False)
    for source in accepted + [manifest_path]:
        target = destination / source.relative_to(BASE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    archive = destination.with_suffix(".zip")
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as z:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                z.write(path, str(Path(PACKAGE) / path.relative_to(destination)))
    report = {"status": "packaged_pending_detached_documentation_check",
              "archive": str(archive), "archive_bytes": archive.stat().st_size,
              "archive_sha256": sha(archive), "manifest_sha256": sha(manifest_path),
              "manifest_file_count": len(manifest), "package_directory": str(destination),
              "self_contained_historical_retraining": False,
              "new_numerical_reproduction": False, "independent_market_confirmation": False,
              "research_goal_achieved": False}
    (BASE / "release/PACKAGE_BUILD.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
