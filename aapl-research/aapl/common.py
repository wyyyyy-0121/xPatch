from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def identity(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"All new output must be under {ROOT}: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def runtime(seed=2026):
    import numpy as np
    import torch
    if len(os.sched_getaffinity(0)) > 18:
        raise RuntimeError("Launch with scripts/python.sh: new research affinity must be <=18 CPUs")
    torch.set_num_threads(int(os.environ.get('AAPL_CPU_THREADS', '4')))
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    return "cuda" if torch.cuda.is_available() else "cpu"


def precision_policy(config):
    import torch
    value = config.get('precision', 'fp32')
    if value not in ('fp32', 'tf32', 'bf16'):
        raise ValueError('Unadmitted numeric precision')
    torch.backends.cuda.matmul.fp32_precision = 'tf32' if value == 'tf32' else 'ieee'
    torch.backends.cudnn.conv.fp32_precision = 'tf32' if value == 'tf32' else 'ieee'
    return value


def sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()
