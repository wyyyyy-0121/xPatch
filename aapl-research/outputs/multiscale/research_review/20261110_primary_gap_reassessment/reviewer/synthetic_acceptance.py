"""Price-free acceptance for the separately frozen calendar audit."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import time
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    protocol = json.loads(Path(args.protocol).read_text())
    assert protocol["status"] == "frozen_before_execution"
    assert protocol["mode"] == "price_free_synthetic_calendar"
    assert time.time() < protocol["deadline_epoch"]
    assert set(os.sched_getaffinity(0)) == {26}
    for name, expected in protocol["code_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    spec = importlib.util.spec_from_file_location("timestamp_audit", BASE / "audit_timestamps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def calendar(short=(), omit=()):
        values, labels = [], []
        for day in pd.bdate_range("2021-03-01", "2021-04-02"):
            date = str(day.date())
            if date in omit:
                continue
            count = 210 if date in short else 390
            start = pd.Timestamp(date + " 09:30", tz="America/New_York")
            grid = pd.date_range(start, periods=count, freq="min").as_unit("ns").asi8
            values.extend(grid + module.MINUTE)
            labels.extend([date] * count)
        return np.asarray(values, dtype=np.int64), np.asarray(labels)

    end, dates = calendar()
    info = module.timeline(end, dates, "minute")
    o = int(np.flatnonzero((dates == "2021-03-23") & (info["clocks"] == "12:30:00"))[0])
    row = module.origin_record(o, info, 480, 60, "minute")
    assert row["previous1"]["calendar_usable"]
    assert row["previous1"]["fixed_nominal_previous_session_shift_matches"]

    first = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1]])
    last = np.r_[first[1:] - 1, len(end) - 1]
    selected = np.concatenate([a + 59 + 60 * np.arange((b - a + 1) // 60) for a, b in zip(first, last)])
    hourly = module.timeline(end[selected], dates[selected], "hourly")
    o = len(selected) - 3
    row = module.origin_record(o, hourly, 96, 8, "hourly")
    assert row["previous1"]["slot_exists"] and not row["previous1"]["mature_le"]
    assert row["previous2"]["calendar_usable"] and row["nearest_matured_visible_session_lag"] == 2
    equal = module.origin_record(o, hourly, 96, 6, "hourly")
    assert equal["previous1"]["mature_le"] and not equal["previous1"]["mature_lt"]

    short_end, short_dates = calendar(short=("2021-03-22",))
    short_info = module.timeline(short_end, short_dates, "minute")
    def short_origin(clock):
        return int(np.flatnonzero((short_dates == "2021-03-23") & (short_info["clocks"] == clock))[0])
    missing = module.origin_record(short_origin("14:00:00"), short_info, 480, 60, "minute")
    assert not missing["previous1"]["slot_exists"]
    illegal = module.origin_record(short_origin("12:30:00"), short_info, 480, 60, "minute")
    assert illegal["previous1"]["mature_le"] and not illegal["previous1"]["time_horizon_legal"]
    assert not illegal["previous1"]["calendar_usable"]

    varied_end, varied_dates = calendar(omit=("2021-03-08",))
    varied = module.timeline(varied_end, varied_dates, "minute")
    origins = np.flatnonzero((varied["weekday"] == 1) & (varied["clocks"] == "12:30:00"))
    records = [module.origin_record(int(i), varied, 480, 60, "minute") for i in origins if i >= 479]
    assert len({r["clock"] for r in records}) == 1
    assert len({r["gap_signature"] for r in records}) > 1
    assert any(module.origin_record(int(i), info, 480, 60, "minute")["historical_DST_offset_change"]
               for i in np.flatnonzero((dates == "2021-03-15") & (info["clocks"] == "12:30:00")))

    module.install_guard([], [], BASE / "timestamp_audit")
    blocked = []
    for name in ("outputs/fusion_next/data/AAPL/close.npy", "outputs/fusion_next/data/AAPL/observed.npy",
                 "outputs/forbidden/bars_hour.npz", "outputs/forbidden/targets.npz",
                 "outputs/forbidden/predictions.npz", "outputs/forbidden/model.pt",
                 "holdout_sealed/private.npy",
                 str((BASE / "timestamp_audit/close.npy").relative_to(module.ROOT)),
                 str((BASE / "timestamp_audit/observed.npy").relative_to(module.ROOT)),
                 str((BASE / "timestamp_audit/holdout_sealed/private.npy").relative_to(module.ROOT))):
        try:
            with (module.ROOT / name).open("rb"):
                pass
        except RuntimeError:
            blocked.append(name)
        else:
            raise AssertionError("Forbidden data was opened")
    assert len(blocked) == 10
    result = {"status": "passed_price_free_synthetic_calendar_acceptance",
              "regular_minute_previous_session_mature_and_in_window": True,
              "regular_hour_previous_session_immature_previous2_mature": True,
              "completion_equality_distinguished": True, "halfday_missing_slot_retained": True,
              "mature_but_cross_session_minute_path_rejected": True,
              "clock_equivalence_does_not_imply_historical_gap_uniqueness": True,
              "historical_DST_offset_change_detected": True, "forbidden_paths_rejected": blocked,
              "real_timestamp_or_split_arrays_read": False, "price_labels_weights_read": False,
              "Q3_read": False, "protocol_sha256": hashlib.sha256(Path(args.protocol).read_bytes()).hexdigest()}
    assert time.time() < protocol["deadline_epoch"]
    module.atomic(BASE / "timestamp_audit/SYNTHETIC_ACCEPTANCE.json", result)
    print(json.dumps({"status": result["status"], "forbidden_open_attempts_rejected": len(blocked)}))


if __name__ == "__main__":
    main()
