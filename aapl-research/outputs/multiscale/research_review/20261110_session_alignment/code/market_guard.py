"""Process-level file/network allowlist for the frozen diagnostic sources.

The hook supplements inspected source; it is not an OS sandbox or a ZIP-member
decoder monitor. Market adapter source restricts NPZ members to the stated ones.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path("/home/ayaka/wslcode/aapl-research").resolve()
BASE = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/multiscale/admission/alpaca_20261107T052228573242Z"
FORBIDDEN_OUTPUT_MARKERS = ("bars", "close", "weight", "sealed", "target", "prediction")


def install_guard(protocol, protocol_path):
    output = Path(protocol["output_directory"]).resolve()
    if not output.is_relative_to(BASE) or output == BASE:
        raise ValueError("Diagnostic output must be an explicit subdirectory of this stage")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.sched_getaffinity(0) != {26}:
        raise RuntimeError("Frozen diagnostic requires GPU hidden and CPU26 affinity")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        if os.environ.get(key) != "1":
            raise RuntimeError("Numerical thread limit changed")
    allowed = {Path(protocol_path).resolve(), SOURCE / "admission.json"}
    allowed.update(Path(p).resolve() for p in protocol["code_sha256"])
    allowed.update(Path(p).resolve() for p in protocol.get("metadata_sha256", {}))
    for rec in protocol.get("bars", {}).values():
        allowed.add(Path(rec["path"]).resolve())
    for folds in protocol.get("fits", {}).values():
        allowed.update(Path(rec["path"]).resolve() for rec in folds.values())
    if any(p.is_relative_to(ROOT / "outputs/fusion_next") for p in allowed):
        raise ValueError("This diagnostic does not admit foundation-project inputs")
    state = {"guard_installed": True, "GPU_hidden": True, "CPU_affinity": [26],
             "source_open_counts": {}, "rejected_attempts": [],
             "scope": "Python open/network hook plus inspected source, not OS sandbox or NPZ-member monitor"}

    def reject(event, path):
        state["rejected_attempts"].append({"event": event, "path": str(path)})
        raise PermissionError("Frozen session-alignment access rejected: " + str(path))

    def hook(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
            reject(event, "network/subprocess forbidden")
        if event != "open" or isinstance(args[0], int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any("sealed" in part.lower() or part.lower() in ("q3", "2026q3") for part in path.parts):
            reject(event, path)
        mode = args[1]
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = (isinstance(mode, str) and any(k in mode for k in "wax+")) or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if path.is_relative_to(output):
            if path.suffix.lower() not in ("", ".json", ".npz"):
                reject(event, path)
            if path.suffix.lower() == ".npz" and any(k in path.name.lower() for k in FORBIDDEN_OUTPUT_MARKERS):
                reject(event, path)
            return
        if writing:
            reject(event, path)
        if path in allowed:
            key = str(path)
            state["source_open_counts"][key] = state["source_open_counts"].get(key, 0) + 1
            return
        # Standard runtime libraries and civil-time zone tables are permitted.
        # This branch never permits numerical market archives or personal files.
        if path.suffix.lower() in (".npz", ".npy", ".pt", ".pth", ".safetensors", ".csv", ".parquet", ".pkl"):
            reject(event, path)
        if path.is_relative_to(ROOT / ".venv") or any(path.is_relative_to(Path(p)) for p in ("/usr/lib", "/usr/local/lib", "/usr/share/zoneinfo", "/lib")):
            return
        reject(event, path)

    sys.addaudithook(hook)
    return state


def synthetic_guard_acceptance():
    """Run in a separate CPU26 process; never open the approved market paths."""
    import socket
    output = BASE / "guard_acceptance"
    output.mkdir(parents=True, exist_ok=True)
    source = output / "APPROVED_SYNTHETIC_SOURCE.json"
    source.write_text('{"synthetic":true}\n')
    protocol = {"output_directory": str(output), "code_sha256": {str(Path(__file__).resolve()): "synthetic"},
                "metadata_sha256": {}, "bars": {}, "fits": {}}
    state = install_guard(protocol, Path(__file__).resolve())
    assert json.loads(source.read_text())["synthetic"]
    rejected = []
    cases = [ROOT / "outputs/fusion_next/data/close.npy",
             ROOT / "outputs/fusion_next/data/observed.npy",
             ROOT / "outputs/fusion_next/data/timestamps_ns.npy",
             SOURCE / "bars_minute.npz", SOURCE / "bars_hour.npz",
             ROOT / "data/2026Q3/sealed.csv", ROOT / "data/disguised_market.json",
             output / "close.npz", output / "sealed.npz", output / "arbitrary.npy",
             output / "holdout_sealed/fit.npz",
             ROOT / "outputs/multiscale/old_model.pt"]
    for path in cases:
        try:
            open(path, "rb")
        except PermissionError:
            rejected.append(str(path))
        else:
            raise AssertionError("Forbidden path accepted: " + str(path))
    for action in (lambda: socket.getaddrinfo("example.com", 443),
                   lambda: socket.socket().connect(("127.0.0.1", 9))):
        try:
            action()
        except PermissionError:
            pass
        else:
            raise AssertionError("Network action accepted")
    receipt = {"status": "passed_actual_guard_rejections", "forbidden_file_cases": rejected,
               "network_cases_rejected": 2, "market_files_opened": 0, "state": state}
    (output / "GUARD_ACCEPTANCE.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"status": receipt["status"], "forbidden_file_cases": len(rejected)}))


if __name__ == "__main__":
    synthetic_guard_acceptance()
