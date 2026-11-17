"""Standard-library-only protocol, resource, and atomic-commit helpers."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import sys
import sysconfig
import tempfile
import time

BASE = Path(__file__).resolve().parent
MEMORY_BYTES = 512 * 1024 * 1024
MAIN_FILES = ("analytic_core.py", "runtime_common.py", "main_worker.py", "freeze_protocol.py", "supervise.py",
              "independent_replay.py")
SPEC_FILES = ("DESIGN_ZH.md", "DESIGN_SPEC.json", "PREPARATION_IDENTITIES.json")


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def canonical_digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        require(not path.exists(), "Refuse to replace original receipt: " + str(path))
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, path)
            os.unlink(temporary)
        else:
            os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def set_environment(cpu=24):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[key] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    require(cpu in (24, 26), "Only the two frozen analysis CPUs are allowed")
    os.sched_setaffinity(0, {cpu})
    require(os.sched_getaffinity(0) == {cpu}, "Analysis affinity differs")


def memory_limit():
    # Linux RLIMIT_RSS is not enforced. A 512 MiB address-space hard ceiling is
    # deliberately stricter than the requested resident-memory cap. No fallback
    # raises it if an installed numerical runtime cannot import within the cap.
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = MEMORY_BYTES if hard == resource.RLIM_INFINITY else min(MEMORY_BYTES, hard)
    resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
    require(resource.getrlimit(resource.RLIMIT_AS)[1] <= MEMORY_BYTES, "Address-space hard cap failed")


def peak_rss_bytes():
    # Linux ru_maxrss is KiB; this stage is explicitly a Linux/WSL experiment.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def runtime_identity():
    distribution = importlib.metadata.distribution("numpy")
    numpy_init = Path(distribution.locate_file("numpy/__init__.py")).resolve()
    executable = Path(sys.executable).resolve()
    return {"python": sys.version, "python_executable": str(executable), "python_executable_sha256": sha(executable),
            "numpy_version": distribution.version, "numpy_init": str(numpy_init), "numpy_init_sha256": sha(numpy_init),
            "platform": sys.platform, "byteorder": sys.byteorder}


def load_protocol(path):
    path = Path(path).resolve()
    require(path == BASE / "EXECUTION_PROTOCOL.json", "Only the one frozen execution protocol is accepted")
    protocol = read_json(path)
    require(protocol.get("schema") == "origin_noise_execution_v1" and protocol.get("execution_admitted") is True,
            "Matching reviewed execution protocol required")
    require(protocol["maximum_total_wall_seconds"] == 180 and protocol["final_reserve_seconds"] == 60 and
            protocol["main_cpu"] == 24 and protocol["numeric_threads"] == 1 and
            protocol["memory_hard_limit_bytes"] == MEMORY_BYTES and protocol["GPU"] is False and
            protocol["market_payloads_admitted"] is False, "Frozen resource/scope limits differ")
    for name in MAIN_FILES + SPEC_FILES:
        require(str(BASE / name) in protocol["inputs"], "Required source/design missing from identity freeze: " + name)
    started, deadline = protocol["started_epoch"], protocol["deadline_epoch"]
    require(isinstance(started, (int, float)) and isinstance(deadline, (int, float)) and
            math.isfinite(started) and math.isfinite(deadline) and deadline - started == 180,
            "Absolute 180-second deadline malformed")
    for raw, expected in protocol["inputs"].items():
        candidate = Path(raw)
        require(candidate.is_absolute() and str(candidate.resolve()) == raw, "Input path not canonical")
        require(candidate.is_relative_to(BASE), "Execution inputs must be preparation-local source/metadata snapshots")
        require(candidate.suffix in (".py", ".json", ".md"), "Array/model input is forbidden")
        require(sha(candidate) == expected, "Frozen source identity changed: " + raw)
    require(runtime_identity() == protocol["runtime"], "Frozen Python/NumPy runtime identity differs")
    require(Path(protocol["output_root"]) == BASE / "run", "Output root differs from frozen stage")
    require(sha(BASE / "DESIGN_SPEC.json") == protocol["design_spec_sha256"], "Design specification differs")
    return protocol, sha(path)


def install_file_guard(protocol, protocol_path):
    allowed = {Path(path).resolve() for path in protocol["inputs"]} | {Path(protocol_path).resolve()}
    output = Path(protocol["output_root"]).resolve()
    roots = []
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        raw = sysconfig.get_paths().get(key)
        if raw:
            roots.append(Path(raw).resolve())
    roots.append(Path(protocol["runtime"]["numpy_init"]).resolve().parent)
    extra = (Path("/dev/null"), Path("/proc/self/statm"), Path("/proc/self/status"))

    def audit(event, arguments):
        if event != "open" or not arguments or isinstance(arguments[0], int):
            return
        raw = arguments[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(raw)).resolve()
        mode = arguments[1] if len(arguments) > 1 else "r"
        flags = arguments[2] if len(arguments) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
            isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))
        if writing:
            require(path.is_relative_to(output) or path == Path("/dev/null"), "Write outside frozen output tree")
            return
        require(path.suffix.lower() not in (".npz", ".npy", ".pt", ".pth", ".safetensors"),
                "Market/model/array payload loading is forbidden")
        require(path in allowed or path in extra or path.is_relative_to(output) or
                any(path.is_relative_to(root) for root in roots), "Read outside frozen text/runtime/output scope: " + str(path))

    sys.addaudithook(audit)


def check_resources(protocol, *, stopped=False, respect_reserve=True, cpu=24):
    require(peak_rss_bytes() <= MEMORY_BYTES, "Peak resident memory exceeds hard limit")
    require(cpu in (24, 26) and os.sched_getaffinity(0) == {cpu}, "CPU affinity changed")
    cutoff = protocol["deadline_epoch"] - (60 if respect_reserve else 0)
    if stopped or time.time() >= cutoff:
        raise InterruptedError("Signal or unchanged absolute cutoff; committed units retained")
