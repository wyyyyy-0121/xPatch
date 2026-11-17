"""Export and call the recorded historical multiscale study by task.

All eight models are retained. The latest completed chronological fold is
chosen without consulting scores. Partial exports require an explicit flag.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from .bars import Bars, file_digest
from .historical_bundle_review import REVIEW_STATUS, read_record, require
from .inference import (_checked_file, _json_hash, _read_bars, export_development_bundle,
                        load_development_bundle)
from .spec import MODELS, OUT, PROFILES, TARGETS, tasks


TASK_KEYS = {(t["profile"], t["horizon"], t["target"]) for t in tasks()}


def write_new(path, value):
    with Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def original_protocol(folder, expected):
    folder = Path(folder).resolve()
    require(folder.is_relative_to(OUT.resolve()), "Fold is outside the study output tree")
    for parent in (folder, *folder.parents):
        if not parent.is_relative_to(OUT.resolve()):
            break
        candidate = parent / "PRIVATE_PROTOCOL.json"
        if candidate.is_file() and file_digest(candidate) == expected:
            return candidate
    raise ValueError("Original protocol matching the saved fold was not found")


def export_study_package(*, coverage_path, permission_path, source_manifest_path,
                         admission_dir, output_directory, allow_partial=False):
    records = {}
    values = {}
    for name, path in (("coverage", coverage_path), ("permission", permission_path),
                       ("source_manifest", source_manifest_path)):
        value, digest = read_record(path)
        records[name] = {"path": str(Path(path).resolve()), "sha256": digest}
        values[name] = value
    coverage = values["coverage"]
    rows = coverage.get("folds", [])
    keys = [(r["profile"], r["horizon"], r["target"], r["fold"]) for r in rows]
    require(coverage.get("status") == "private_verified_coverage_snapshot" and
            len(keys) == len(set(keys)) and
            set(keys) == {(*task, fold) for task in TASK_KEYS for fold in range(1, 6)},
            "Expected all 100 unique historical task-fold records")
    complete = defaultdict(list)
    for row in rows:
        if row["status"] == "completed":
            complete[(row["profile"], row["horizon"], row["target"])].append(row)
    full = all(len(complete[key]) == 5 for key in TASK_KEYS)
    require(full or allow_partial, "The 100-fold study is incomplete; partial export must be explicit")
    require(any(complete.values()), "No completed study folds are available")
    out = Path(output_directory).resolve()
    require(out.is_relative_to(OUT.resolve()), "Study package must be inside the output tree")
    out.mkdir(parents=True, exist_ok=False, mode=0o700)
    packaging = {"schema_version": 1, "status": REVIEW_STATUS,
                 "created_at_utc": datetime.now(timezone.utc).isoformat(),
                 "independent_test": False, "Q3_values_read": False, **records}
    review_path = out / "PACKAGING_RECORD.json"
    write_new(review_path, packaging)
    entries = []
    for key in sorted(TASK_KEYS):
        available = sorted(complete[key], key=lambda r: r["fold"])
        entry = {"profile": key[0], "horizon": key[1], "target": key[2],
                 "completed_folds": [r["fold"] for r in available],
                 "missing_folds": [f for f in range(1, 6) if f not in {r["fold"] for r in available}]}
        if not available:
            entries.append({**entry, "status": "unavailable", "selected_fold": None})
            continue
        chosen = available[-1]
        evidence = chosen["evidence"]
        require("fold_directory" in evidence, "Selected fold lacks a unified saved-model receipt")
        folder = Path(evidence["fold_directory"])
        receipt, receipt_hash = read_record(folder / "fold_receipt.json", evidence["receipt_sha256"])
        protocol = original_protocol(folder, receipt["config"]["protocol_identity"])
        relative = "bundles/" + f"{key[0]}_h{key[1]}_{key[2]}"
        exported = export_development_bundle(folder, expected_receipt_sha256=receipt_hash,
            output_directory=out / relative, admission_dir=Path(admission_dir),
            review_path=review_path, protocol_path=protocol)
        require(exported["status"] == "historical_exploration_inference_bundle",
                "Study package cannot substitute synthetic or differently admitted models")
        entries.append({**entry, "status": "available", "selected_fold": chosen["fold"],
                        "bundle": relative, "manifest_sha256": exported["manifest_sha256"],
                        "fold_receipt_sha256": receipt_hash})
    registry = {"schema_version": 1,
                "status": "historical_development_study_package" if full else
                          "partial_historical_development_study_package",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "independent_test": False, "Q3_values_read": False,
                "selection": "latest completed chronological fold per task; all eight models",
                "core_comparison_complete": full, "tasks": entries,
                "planned_tasks": 20, "available_tasks": sum(e["status"] == "available" for e in entries),
                "five_fold_tasks": sum(len(e["completed_folds"]) == 5 for e in entries),
                "packaging_record_sha256": file_digest(review_path),
                "package_source_sha256": file_digest(Path(__file__)),
                "limits": ["All forecasts remain exposed historical development.",
                           "The selected fold is an evaluation model, not a full-history production refit.",
                           "This package does not train models or reset any compute budget."]}
    registry_path = out / "REGISTRY.json"
    write_new(registry_path, registry)
    digest = file_digest(registry_path)
    load_study_package(out, expected_registry_sha256=digest)
    return {"status": registry["status"], "package": str(out), "registry_sha256": digest,
            "available_tasks": registry["available_tasks"], "five_fold_tasks": registry["five_fold_tasks"]}


def load_study_package(directory, *, expected_registry_sha256, device="cpu"):
    directory = Path(directory).resolve()
    path = _checked_file(directory, "REGISTRY.json", expected_registry_sha256)
    registry = json.loads(path.read_text(encoding="utf8"))
    entries = registry.get("tasks", [])
    keys = [(e["profile"], e["horizon"], e["target"]) for e in entries]
    require(registry.get("schema_version") == 1 and len(keys) == len(set(keys)) and
            set(keys) == TASK_KEYS and registry.get("planned_tasks") == 20 and
            registry.get("independent_test") is False and registry.get("Q3_values_read") is False and
            registry.get("package_source_sha256") == file_digest(Path(__file__)),
            "Study registry identity, scope or task inventory differs")
    _checked_file(directory, "PACKAGING_RECORD.json", registry.get("packaging_record_sha256"))
    for entry in entries:
        done = entry.get("completed_folds")
        require(isinstance(done, list) and done == sorted(set(done)) and
                all(type(f) is int and 1 <= f <= 5 for f in done) and
                entry.get("missing_folds") == [f for f in range(1, 6) if f not in done],
                "Registry fold coverage differs")
        if entry.get("status") == "unavailable":
            require(not done and entry.get("selected_fold") is None and "bundle" not in entry,
                    "An unavailable task cannot contain a model bundle")
        else:
            require(entry.get("status") == "available" and done and
                    entry.get("selected_fold") == done[-1], "Registry fold selection differs")
            manifest = _checked_file(directory, entry["bundle"] + "/manifest.json", entry["manifest_sha256"])
            saved = json.loads(manifest.read_text(encoding="utf8"))
            require(saved.get("data_kind") == "historical_exploration" and
                    all(saved.get(k) == entry[k] for k in ("profile", "horizon", "target")) and
                    saved.get("fold_receipt_sha256") == entry.get("fold_receipt_sha256"),
                    "Registry points to a different task or saved fold")
    available = sum(e["status"] == "available" for e in entries)
    full_count = sum(len(e["completed_folds"]) == 5 for e in entries)
    full = full_count == 20
    require(registry.get("available_tasks") == available and registry.get("five_fold_tasks") == full_count and
            registry.get("core_comparison_complete") is full and
            registry.get("status") == ("historical_development_study_package" if full else
                                      "partial_historical_development_study_package"),
            "Registry completion claim differs from its evidence")
    return StudyPredictor(directory, registry, expected_registry_sha256, device)


class StudyPredictor:
    def __init__(self, directory, registry, digest, device):
        self.directory, self.registry, self.digest, self.device = directory, registry, digest, device
        self._content_digest = _json_hash(registry)
        self._predictors = {}

    def predict(self, bars: Bars, *, profile, horizon, target, model, as_of_ns, history_identity):
        _checked_file(self.directory, "REGISTRY.json", self.digest)
        require(_json_hash(self.registry) == self._content_digest and
                self.registry["package_source_sha256"] == file_digest(Path(__file__)),
                "Loaded study package identity changed")
        key = (profile, horizon, target)
        require(type(horizon) is int and key in TASK_KEYS, "Unsupported study task")
        entry = next(e for e in self.registry["tasks"] if (e["profile"], e["horizon"], e["target"]) == key)
        require(entry["status"] == "available", "Requested task has no completed saved fold")
        if key not in self._predictors:
            self._predictors[key] = load_development_bundle(self.directory / entry["bundle"],
                expected_manifest_sha256=entry["manifest_sha256"], device=self.device)
        result = self._predictors[key].predict(bars, profile=profile, horizon=horizon,
            target=target, model=model, as_of_ns=as_of_ns, history_identity=history_identity)
        return {**result, "study_registry_sha256": self.digest, "selected_fold": entry["selected_fold"],
                "task_completed_folds": entry["completed_folds"],
                "core_comparison_complete": self.registry["core_comparison_complete"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("coverage", "permission", "source-manifest", "admission", "output"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    forecast = commands.add_parser("forecast")
    forecast.add_argument("--package", type=Path, required=True)
    forecast.add_argument("--registry-sha256", required=True)
    forecast.add_argument("--bars", type=Path, required=True)
    forecast.add_argument("--profile", choices=[p.name for p in PROFILES], required=True)
    forecast.add_argument("--horizon", type=int, required=True)
    forecast.add_argument("--target", choices=TARGETS, required=True)
    forecast.add_argument("--model", choices=MODELS, required=True)
    forecast.add_argument("--as-of-ns", type=int, required=True)
    forecast.add_argument("--history-identity", required=True)
    forecast.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.command == "export":
        result = export_study_package(coverage_path=args.coverage, permission_path=args.permission,
            source_manifest_path=args.source_manifest, admission_dir=args.admission,
            output_directory=args.output, allow_partial=args.allow_partial)
    else:
        result = load_study_package(args.package, expected_registry_sha256=args.registry_sha256,
                                   device=args.device).predict(_read_bars(args.bars), profile=args.profile,
            horizon=args.horizon, target=args.target, model=args.model, as_of_ns=args.as_of_ns,
            history_identity=args.history_identity)
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
