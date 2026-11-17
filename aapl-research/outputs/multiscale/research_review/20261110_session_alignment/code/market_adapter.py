"""Explicitly admitted original-cache adapter for one fit-internal diagnostic.

Importing this module opens no market file. load_admitted requires a separately
frozen execution protocol; synthetic engineering acceptance does not admit it.
Only original pre-Q3 minute/hour bars and fit.npy inside four original split
archives are decoded. No Data/Bars constructors, old scalers, targets, predictions
or weights are imported or loaded. The worker owns the process audit guard,
protocol identity, atomic checkpoints, fixed block order and unchanged deadline.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
from zoneinfo import ZoneInfo

import numpy as np

from alignment_core import features


ROOT = Path("/home/ayaka/wslcode/aapl-research")
SOURCE = ROOT / "outputs/multiscale/admission/alpaca_20261107T052228573242Z"
CACHE = ROOT / "outputs/multiscale/controlled_followup/20261108_v2/cache"
ADMISSION_SHA = "f6fc738205532cf5016d74b2baa4691692cb03674c3db9b6a8ccf8ce8126ae86"
BAR_KEYS = {"start_ns", "end_ns", "close", "observed", "session_date"}
NY = ZoneInfo("America/New_York")
SPLITS = [{"date": "2020-08-31", "ratio": 4.0}]
SPEC = {
    "minute": {"lookback": 480, "horizon": 60, "native_ns": 60_000_000_000,
               "task": "minute_h60_future_log_return", "bar_file": "bars_minute.npz",
               "bars_sha256": "c81d57cd1e190456e1fdbf59d7493c915704551fa2e6c5870b5bd66f4cd94332",
               "folds": {"1": (64261, "ae69a8e1a37676403b71d91f0f949de5bdb196e2caf31191d448b3ab7ccff326"),
                         "2": (128761, "93c5608c9b1ff3b3d1c724589dfd7d049e040405f7a716c59f57218a7ba2808b")}},
    "hourly": {"lookback": 96, "horizon": 8, "native_ns": 3_600_000_000_000,
               "task": "hourly_h8_future_log_return", "bar_file": "bars_hour.npz",
               "bars_sha256": "c41851d735d179c3243dd3356204d17845b55289c56c59f9e170c0b5f432881c",
               "folds": {"1": (1139, "38079b182cd2a01409cb3649596e1838c68fc509652b2d201efc975cc2aeb42a"),
                         "2": (2294, "ad5db4224a6fae358341f6e6576deb0d4dddb0489089b76991b1e82758a83bb9")}},
}


def _check(protocol):
    deadline = float(protocol["deadline_epoch"])
    if not np.isfinite(deadline) or time.time() >= deadline:
        raise TimeoutError("Unchanged session-alignment execution deadline reached")


def _sha(path, protocol):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            _check(protocol)
            result.update(block)
    return result.hexdigest()


def _registered_path(record, expected):
    path = Path(record["path"])
    if not path.is_absolute() or path.resolve() != expected.resolve():
        raise ValueError("Protocol path differs from the original admitted cache")
    return path


def _require_sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Missing or malformed frozen file SHA256")
    return value


def _gate(profile, protocol):
    if profile not in SPEC:
        raise ValueError("Only original minute H60 and hourly H8 are admitted")
    if (protocol.get("status") != "frozen_before_execution"
            or protocol.get("market_arrays_admitted") is not True
            or protocol.get("fit_labels_admitted") is not True
            or protocol.get("Q3_allowed", False) or protocol.get("network_allowed", False)):
        raise RuntimeError("Separate frozen market/fit-label admission is required")
    _check(protocol)
    if Path(protocol["source_root"]).resolve() != SOURCE.resolve():
        raise ValueError("Wrong original bar admission directory")
    own_path = str(Path(__file__).resolve())
    expected_self = protocol.get("market_adapter_sha256", protocol.get("code_sha256", {}).get(own_path))
    if _sha(own_path, protocol) != _require_sha(expected_self):
        raise ValueError("Adapter source is not the frozen source")
    core_path = str(Path(__file__).resolve().with_name("alignment_core.py"))
    expected_core = protocol.get("code_sha256", {}).get(core_path)
    if _sha(core_path, protocol) != _require_sha(expected_core):
        raise ValueError("Core source is not the frozen source")
    if protocol.get("split_actions") != SPLITS:
        raise ValueError("Origin-known split actions changed")


def _read_fit(record, profile, fold, protocol):
    spec = SPEC[profile]
    expected_path = CACHE / spec["task"] / ("fold_" + fold + ".npz")
    path = _registered_path(record, expected_path)
    file_sha = _require_sha(record["file_sha256"])
    if _sha(path, protocol) != file_sha:
        raise ValueError("Original split archive byte identity changed")
    expected_count, expected_fit_sha = spec["folds"][fold]
    if record["count"] != expected_count or record["fit_sha256"] != expected_fit_sha:
        raise ValueError("Protocol does not register the original fit cohort")
    # Reading the ZIP directory is not decoding train/stop/validation. Only this
    # literal member is ever opened or passed to np.load.
    with zipfile.ZipFile(path, "r") as archive:
        if archive.namelist().count("fit.npy") != 1:
            raise ValueError("Fit member is absent or duplicated")
        with archive.open("fit.npy", "r") as member:
            value = np.load(io.BytesIO(member.read()), allow_pickle=False)
    _check(protocol)
    if (value.ndim != 1 or value.dtype != np.dtype("int64") or len(value) != expected_count
            or not np.all(np.diff(value) > 0)):
        raise ValueError("Original fit indices have wrong type/order/count")
    canonical = hashlib.sha256(np.asarray(value, dtype="<i8").tobytes()).hexdigest()
    if canonical != expected_fit_sha:
        raise ValueError("Original fit canonical index hash changed")
    return value.copy(), {"path": str(path), "file_sha256": file_sha,
                          "decoded_member": "fit.npy", "fit_sha256": canonical,
                          "count": expected_count}


def _validate_bars(bars, spec):
    if set(bars) != BAR_KEYS:
        raise ValueError("Only the five original bar fields may be decoded")
    n = len(bars["end_ns"])
    if not n or any(a.ndim != 1 or len(a) != n for a in bars.values()):
        raise ValueError("Original bar fields must be matching nonempty vectors")
    if (bars["start_ns"].dtype != np.dtype("int64") or bars["end_ns"].dtype != np.dtype("int64")
            or bars["observed"].dtype != np.dtype("bool") or bars["close"].dtype != np.dtype("float64")
            or bars["session_date"].dtype.kind != "U"):
        raise ValueError("Unexpected original bar field dtypes")
    if (not np.all(np.diff(bars["start_ns"]) > 0) or not np.all(np.diff(bars["end_ns"]) > 0)
            or not np.all(bars["end_ns"] - bars["start_ns"] == spec["native_ns"])):
        raise ValueError("Original complete-bar ordering/duration changed")
    dates = bars["session_date"]
    if (np.any(dates > "2026-06-30") or np.any(dates < "2021-01-04")
            or np.any(dates[1:] < dates[:-1])):
        raise ValueError("Nonchronological or non-pre-Q3 session dates")
    for date in np.unique(dates):
        if datetime.strptime(str(date), "%Y-%m-%d").date().isoformat() != date:
            raise ValueError("Invalid session date encoding")
    observed_close = bars["close"][bars["observed"]]
    if not np.isfinite(observed_close).all() or np.any(observed_close <= 0):
        raise ValueError("Observed original closes must be finite and positive")


def _clock(end_ns, session_dates):
    """Original v3b eight-column clock from completed-time New York civil time."""
    rows = []
    for stamp, session in zip(end_ns, session_dates):
        seconds, remainder = divmod(int(stamp), 1_000_000_000)
        if remainder:
            raise ValueError("Original completed bars must have whole-second timestamps")
        local = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone(NY)
        if local.date().isoformat() != str(session) or local.weekday() > 4:
            raise ValueError("New York completed-time date disagrees with original session")
        u = (local.hour * 60 + local.minute + local.second / 60.0 - 570.0) / 390.0
        if not 0 < u <= 1:
            raise ValueError("Completed-time clock outside original regular-session interval")
        rows.append([np.sin(2 * np.pi * u), np.cos(2 * np.pi * u),
                     np.sin(4 * np.pi * u), np.cos(4 * np.pi * u)]
                    + [float(local.weekday() == day) for day in range(4)])
    return np.asarray(rows, dtype=np.float64).reshape(len(end_ns), 8)


class AdmittedData:
    """Immutable admitted caches; accessors reject every non-fit origin."""

    def __init__(self, profile, bars, fits, protocol, receipt):
        self.profile, self.protocol, self.receipt = profile, protocol, receipt
        spec = SPEC[profile]
        self.lookback, self.horizon, self.native_ns = spec["lookback"], spec["horizon"], spec["native_ns"]
        self._bars, self._fits = bars, fits
        self.end_ns = bars["end_ns"]
        self._allowed = np.zeros(len(self.end_ns), dtype=bool)
        union = np.unique(np.concatenate(list(fits.values())))
        if np.any(union < self.lookback - 1) or np.any(union + self.horizon >= len(self.end_ns)):
            raise ValueError("An original fit row lacks its full frozen history or target")
        # Prefix-sum verification preserves EVERY original row. No observed-mask
        # or feature availability filter is allowed to select an easier cohort.
        bad = np.r_[0, np.cumsum(~bars["observed"], dtype=np.int64)]
        if (np.any(bad[union + 1] != bad[union - self.lookback + 1])
                or np.any(bad[union + self.horizon + 1] != bad[union + 1])):
            raise ValueError("Original fit history/target has unobserved bars; refuse, do not drop")
        if profile == "minute" and np.any(bars["session_date"][union] != bars["session_date"][union + self.horizon]):
            raise ValueError("Original minute H60 target crosses a session; refuse, do not drop")
        self._allowed[union] = True
        self._log_close = np.full(len(self.end_ns), np.nan, dtype=np.float64)
        self._log_close[bars["observed"]] = np.log(bars["close"][bars["observed"]])
        self._split = np.zeros(len(self.end_ns), dtype=np.float64)
        for action in SPLITS:
            self._split[bars["session_date"] >= action["date"]] += np.log(action["ratio"])
        self.receipt.update({"unique_original_fit_origins": len(union),
                             "all_original_fit_rows_validated_without_dropping": True,
                             "history_and_future_observed_verified": True,
                             "minute_same_session_verified": profile == "minute",
                             "stop_outer_Q3_targets_read": False})
        for value in list(bars.values()) + list(fits.values()) + [self._allowed, self._log_close, self._split]:
            value.flags.writeable = False

    def fit(self, fold):
        _check(self.protocol)
        return self._fits[str(fold)].copy()

    def _indices(self, indices):
        _check(self.protocol)
        value = np.asarray(indices)
        if (value.ndim != 1 or value.dtype.kind not in "iu" or len(value) == 0
                or np.any(value < 0) or np.any(value >= len(self.end_ns))):
            raise ValueError("Expected a nonempty original-fit origin vector")
        value = np.asarray(value, dtype=np.int64)
        if not np.all(self._allowed[value]):
            raise ValueError("Only registered original fit origins may access features/targets")
        return value

    def x_and_calendar(self, indices):
        origins = self._indices(indices)
        positions = origins[:, None] + np.arange(1 - self.lookback, 1)
        x = (self._log_close[positions] + self._split[positions]
             - self._split[origins, None] - self._log_close[origins, None])
        clock = _clock(self.end_ns[origins], self._bars["session_date"][origins])
        if not np.isfinite(x).all():
            raise ValueError("Nonfinite original origin-known relative-close history")
        return x, self.end_ns[positions], self._bars["session_date"][positions], clock

    def batch_features(self, indices):
        x, ends, dates, clock = self.x_and_calendar(indices)
        return features(x, ends, dates, clock, self.native_ns)

    def targets(self, indices):
        origins = self._indices(indices)
        future = origins[:, None] + np.arange(1, self.horizon + 1)
        # Exact original target semantics: nominal future closes / nominal
        # origin close. No future split adjustment and no archived target file.
        target = np.log(self._bars["close"][future] / self._bars["close"][origins, None])
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite original complete cumulative-return path")
        return target


def load_admitted(profile, protocol):
    """Load one profile only after exact source/cache/protocol admission checks."""
    _gate(profile, protocol)
    admission_path = SOURCE / "admission.json"
    if _sha(admission_path, protocol) != ADMISSION_SHA:
        raise ValueError("Original admission metadata changed")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("split_actions") != SPLITS:
        raise ValueError("Original known split action metadata changed")
    spec = SPEC[profile]
    record = protocol["bars"][profile]
    path = _registered_path(record, SOURCE / spec["bar_file"])
    if record["sha256"] != spec["bars_sha256"] or admission["files"][spec["bar_file"]] != spec["bars_sha256"]:
        raise ValueError("Protocol/admission does not register the original bar byte identity")
    if _sha(path, protocol) != spec["bars_sha256"]:
        raise ValueError("Original admitted bar cache changed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != BAR_KEYS or len(archive.files) != len(BAR_KEYS):
            raise ValueError("Unexpected fields in the original bar archive")
        bars = {key: archive[key].copy() for key in sorted(BAR_KEYS)}
    _check(protocol)
    _validate_bars(bars, spec)
    fit_records = protocol["fits"][profile]
    if set(fit_records) != {"1", "2"}:
        raise ValueError("Exactly original folds 1 and 2 must be registered")
    fits, fit_receipts = {}, []
    for fold in ("1", "2"):
        fits[fold], receipt = _read_fit(fit_records[fold], profile, fold, protocol)
        fit_receipts.append(receipt)
    receipt = {"profile": profile, "lookback": spec["lookback"], "horizon": spec["horizon"],
               "bars": {"path": str(path), "sha256": spec["bars_sha256"], "decoded_keys": sorted(BAR_KEYS)},
               "fits": fit_receipts, "admission_sha256": ADMISSION_SHA,
               "deadline_epoch": float(protocol["deadline_epoch"]), "stop_outer_split_members_decoded": False,
               "old_weights_scalers_targets_predictions_loaded": False, "Q3_read": False,
               "live_receipt_timeliness_established": False}
    data = AdmittedData(profile, bars, fits, protocol, receipt)
    _check(protocol)
    return data
