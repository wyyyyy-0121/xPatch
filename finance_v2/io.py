"""Small deterministic artifact helpers."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(body)
    temporary.replace(path)


def source_hashes():
    paths = list((ROOT / "finance_v2").rglob("*.py"))
    paths += [ROOT / p for p in ("models/xPatch.py", "layers/ema.py", "layers/network.py",
                                "layers/decomp.py", "layers/revin.py")]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
