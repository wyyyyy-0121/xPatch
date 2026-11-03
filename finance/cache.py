"""Read-only, fingerprinted mmap inputs shared by persistent numerical workers."""
from pathlib import Path

import numpy as np
import pandas as pd

from .data import RVData
from .io import ROOT, read_json, sha, stable_hash, write_json


ARRAYS = ("dates", "days", "day_index", "returns", "volume", "log_range")


def save_data(data, directory, identity):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name in ARRAYS:
        values = getattr(data, name)
        if name in ("dates", "days"):
            values = np.asarray(values, dtype="U10")
        np.save(directory / f"{name}.npy", values, allow_pickle=False)
    np.save(directory / "timestamps.npy", data.timestamps.asi8, allow_pickle=False)
    metadata = {"identity": identity, "close_by_date": {day: str(stamp) for day, stamp in data.close_by_date.items()},
        "regular_audits": data.regular_audits}
    metadata["files"] = {p.name: sha(p) for p in sorted(directory.glob("*.npy"))}
    write_json(directory / "data.json", metadata)
    return metadata


def load_data(directory, identity=None, verify=True):
    directory = Path(directory)
    metadata = read_json(directory / "data.json")
    if identity is not None and identity != metadata["identity"]:
        raise ValueError("Cached data belongs to another manifest")
    if verify:
        for name, digest in metadata["files"].items():
            if sha(directory / name) != digest:
                raise ValueError(f"Data mmap changed: {name}")
    arrays = {name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in ARRAYS}
    timestamps = pd.DatetimeIndex(np.load(directory / "timestamps.npy", mmap_mode="r"), tz="UTC")
    return RVData(timestamps=timestamps, **arrays, regular_audits=metadata["regular_audits"],
        close_by_date={day: pd.Timestamp(stamp) for day, stamp in metadata["close_by_date"].items()})


def prepare_data(config):
    manifest = ROOT / config["data_manifest"]
    audited = read_json(manifest)
    for asset, entry in audited["assets"].items():
        if sha(ROOT / entry["csv"]) != entry["csv_sha256"]:
            raise ValueError(f"Audited source changed: {asset}")
    identity = stable_hash(audited)
    directory = ROOT / "artifacts/rv30/data" / identity
    if (directory / "data.json").exists():
        load_data(directory, identity)
    else:
        data = RVData.from_manifest(config["data_manifest"])
        save_data(data, directory, identity)
    return directory, identity
