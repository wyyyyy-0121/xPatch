"""Exercise the actual market worker functions with synthetic in-memory data.

No market loading function is invoked. The small synthetic dimensions test checkpoint control
flow; original L480/L96 numerical formulas and maximum shape have separate
accepted fixtures. This is not an end-to-end market admission test.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import alignment_core as core
import engineering as eng
import run_market as worker
import market_adapter as adapter

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "engineering/native_pipeline"


class SyntheticData:
    profile = "synthetic_only"
    lookback = 12
    horizon = 3

    def __init__(self):
        rng = np.random.default_rng(570)
        self.X = rng.normal(size=(3072, core.feature_dimensions(12)["total"]))
        self.X[:, 11] = 0
        self.X[:, -2] = self.X[:, 0] * self.X[:, 13]
        self.X[:, -1] = self.X[:, 1] * self.X[:, 14]
        self.Y = (.01*self.X[:, -2, None] * np.arange(1, 4)[None]
                  + rng.normal(0, .003, (3072, 3)))
        self.arm_phase = None; self.counter = 0; self.fired = False; self.pause = 0
        self.target_calls = 0

    def trigger(self, phase):
        self.arm_phase = phase; self.counter = 0; self.fired = False

    def batch_features(self, indices):
        phase = "prefix" if int(indices[0]) < 1536 else "tail"
        if phase == self.arm_phase:
            self.counter += 1
            if self.counter == 2 and not self.fired:
                self.fired = True
                os.kill(os.getpid(), signal.SIGUSR1)
        if self.pause:
            time.sleep(self.pause)
        return self.X[indices].copy()

    def targets(self, indices):
        self.target_calls += 1
        return self.Y[indices].copy()


def synthetic_adapter_cases():
    results = []
    for profile, L, H, minutes in (("minute", 480, 60, 1), ("hourly", 96, 8, 60)):
        ends, dates = eng.synthetic_calendar(minutes)
        native = minutes*60*10**9
        close = np.exp(5+np.sin(np.arange(len(ends))*.013)*.01)
        possible = np.arange(L-1, len(ends)-H, dtype=np.int64)
        if profile == "minute":
            possible = possible[dates[possible] == dates[possible+H]]
        chosen = possible[np.linspace(0, len(possible)-1, 32, dtype=np.int64)]
        bars = {"end_ns": ends, "start_ns": ends-native, "session_date": dates,
                "close": close, "observed": np.ones(len(ends), dtype=bool)}
        data = adapter.AdmittedData(profile, bars, {"1": chosen}, {"deadline_epoch": eng.deadline()}, {})
        x, ew, dw, clock = data.x_and_calendar(chosen)
        positions = chosen[:, None]+np.arange(1-L, 1)
        logc = np.log(close)
        expected = (logc[positions]+np.log(4.))-np.log(4.)-logc[chosen, None]
        assert np.array_equal(x, expected)
        np.testing.assert_allclose(clock, eng.clock_values(ends[chosen]), rtol=0, atol=1e-14)
        expected_y = np.log(close[chosen[:, None]+np.arange(1, H+1)]/close[chosen, None])
        assert np.array_equal(data.targets(chosen), expected_y)
        assert np.array_equal(data.batch_features(chosen), core.features(expected, ew, dw, clock, native))
        results.append({"profile": profile, "x_nominal_split_formula_bitwise": True,
                        "target_raw_nominal_formula_bitwise": True,
                        "clock_scalar_vector_max_abs": float(np.max(np.abs(clock-eng.clock_values(ends[chosen])))),
                        "synthetic_AdmittedData_only_no_load_admitted_called": True})
    return results


def moments_equal(a, b):
    for key in worker.MOMENTS:
        assert np.array_equal(a[key], b[key]), key


def output_equal(a, b):
    for phase in ("fit", "tail"):
        with np.load(a / phase / "CHECKPOINT.npz", allow_pickle=False) as za, np.load(b / phase / "CHECKPOINT.npz", allow_pickle=False) as zb:
            for k in (("count", "sx", "sy", "xx", "xy", "cursor") if phase == "fit" else ("count", "cursor", "sse")):
                assert np.array_equal(za[k], zb[k]), (phase, k)
    for arm in core.ARMS:
        with np.load(a / "fit" / ("MODEL_"+arm+".npz"), allow_pickle=False) as za, np.load(b / "fit" / ("MODEL_"+arm+".npz"), allow_pickle=False) as zb:
            for k in ("coef", "x_mean", "x_std", "y_mean", "columns"):
                assert np.array_equal(za[k], zb[k]), (arm, k)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eng.guard()
    protocol_file = BASE / "NATIVE_PIPELINE_PROTOCOL.json"
    protocol = eng.read(protocol_file)
    assert protocol["original_engineering_deadline_epoch"] == eng.deadline()
    assert all(eng.sha(Path(p)) == h for p, h in protocol["source_sha256"].items())
    assert os.sched_getaffinity(0) == {26} and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    eng.check_time()
    p = dict(worker.REQUIRED, deadline_epoch=eng.deadline())
    identity = hashlib.sha256(b"synthetic-native-pipeline").hexdigest()
    protocol_sha = eng.sha(protocol_file)
    data = SyntheticData(); prefix = np.arange(1536, dtype=np.int64); tail = np.arange(1536, 3072, dtype=np.int64)
    signal.signal(signal.SIGUSR1, worker.receive_signal)
    result = {"status": "running", "market_arrays_or_load_admitted_used": False,
              "synthetic_only_L": 12, "synthetic_only_H": 3, "protocol_sha256": protocol_sha,
              "original_deadline_epoch": eng.deadline(), "deadline_reset": False}
    try:
        result["synthetic_adapter_cases"] = synthetic_adapter_cases()
        baseline = worker.accumulate(data, prefix, OUT / "base_features", "features", p, protocol_sha, identity)
        feature_gate = worker.summary_gate(baseline, 12, p)
        assert feature_gate["passed"]
        data.trigger("prefix"); worker.STOP = False
        try:
            worker.accumulate(data, prefix, OUT / "resumed_features", "features", p, protocol_sha, identity)
        except worker.Paused:
            pass
        else:
            raise AssertionError("Actual SIGUSR1 did not pause actual features function")
        assert data.fired
        with np.load(OUT / "resumed_features/CHECKPOINT.npz", allow_pickle=False) as z:
            assert int(z["cursor"]) == 512
        data.trigger(None); worker.STOP = False
        recovered = worker.accumulate(data, prefix, OUT / "resumed_features", "features", p, protocol_sha, identity)
        moments_equal(baseline, recovered)
        assert data.target_calls == 0
        state = {"data": data, "prefix": prefix, "tail": tail, "identity": identity, "fold": 1}
        worker.fit_and_score(dict(state, name="baseline"), baseline, p, protocol_sha, OUT)
        for phase, name in (("prefix", "resumed_fit"), ("tail", "resumed_tail")):
            data.trigger(phase); worker.STOP = False
            try:
                worker.fit_and_score(dict(state, name=name), baseline, p, protocol_sha, OUT)
            except worker.Paused:
                pass
            else:
                raise AssertionError("Actual SIGUSR1 did not pause actual "+name)
            assert data.fired
            checkpoint_phase = "fit" if phase == "prefix" else "tail"
            with np.load(OUT / name / checkpoint_phase / "CHECKPOINT.npz", allow_pickle=False) as z:
                assert int(z["cursor"]) == 512
            data.trigger(None); worker.STOP = False
            worker.fit_and_score(dict(state, name=name), baseline, p, protocol_sha, OUT)
            output_equal(OUT / "baseline", OUT / name)
        # Expire during actual tail accumulation; resume the same frozen budget.
        models = {}
        for arm in core.ARMS:
            with np.load(OUT / "baseline/fit" / ("MODEL_"+arm+".npz"), allow_pickle=False) as z:
                models[arm] = {k: z[k].copy() for k in ("coef", "x_mean", "x_std", "y_mean", "columns")}
        timeout_p = dict(p, deadline_epoch=time.time()+.18)
        eng.atomic_json(OUT / "TIMEOUT_CASE.json", timeout_p)
        data.pause = .1; worker.STOP = False
        try:
            worker.score_tail(data, tail, models, OUT / "timeout_tail", timeout_p, protocol_sha, identity)
        except worker.Paused:
            pass
        else:
            raise AssertionError("Actual tail deadline did not pause")
        with np.load(OUT / "timeout_tail/CHECKPOINT.npz", allow_pickle=False) as z:
            before = {k: z[k].copy() for k in z.files}
        assert 0 < int(before["cursor"]) < len(tail)
        data.pause = 0
        try:
            worker.score_tail(data, tail, models, OUT / "timeout_tail", timeout_p, protocol_sha, identity)
        except worker.Paused:
            pass
        else:
            raise AssertionError("Expired actual tail protocol advanced")
        with np.load(OUT / "timeout_tail/CHECKPOINT.npz", allow_pickle=False) as z:
            assert all(np.array_equal(z[k], v) for k, v in before.items())
        eng.check_time()
        result.update(status="passed_actual_pipeline_functions_synthetic_lifecycle",
            feature_gate=feature_gate, actual_SIGUSR1_feature_fit_tail=True,
            resumed_moments_models_and_tail_sse_bitwise=True,
            actual_tail_timeout_checkpoint_cursor=int(before["cursor"]),
            expired_actual_tail_protocol_no_advance=True, all_256row_blocks_preserved=True,
            completed_epoch=time.time())
        eng.atomic_json(OUT / "ACCEPTANCE.json", result)
        print(json.dumps({"status": result["status"], "timeout_cursor": int(before["cursor"])}))
    except BaseException as exc:
        result.update(status="failed_native_synthetic_lifecycle", error_type=type(exc).__name__, error=str(exc), failed_epoch=time.time())
        eng.atomic_json(OUT / "FAILURE.json", result)
        raise


if __name__ == "__main__":
    main()
