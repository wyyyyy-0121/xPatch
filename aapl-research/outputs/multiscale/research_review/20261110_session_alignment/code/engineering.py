"""Price-free acceptance, actual checkpoint lifecycle and maximum-shape cost.

This program never admits market files. All array fixtures are generated locally
inside engineering/; subprocess cases bind an immutable fixture and deadline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

import numpy as np
import alignment_core as core

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "engineering"
RULES = BASE / "ENGINEERING_RULES.json"
STOP = False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def guard():
    """Reject any numerical inputs outside this synthetic-output directory."""
    def hook(event, args):
        if event.startswith("socket.") and event in ("socket.connect", "socket.getaddrinfo"):
            raise PermissionError("No network in synthetic stage")
        if event != "open" or isinstance(args[0], int):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.suffix.lower() in (".npz", ".npy", ".pt", ".pth", ".safetensors", ".pkl"):
            if not p.is_relative_to(OUT.resolve()):
                raise PermissionError("Price-free stage refuses external numerical file")
    sys.addaudithook(hook)


def deadline():
    return float(read(RULES)["deadline_epoch"])


def check_time():
    if time.time() >= deadline():
        raise TimeoutError("Original engineering deadline reached")


def receive_signal(signum, frame):
    global STOP
    STOP = True


def validate_moments(stats):
    n, _, _ = core.validate_stats(stats)
    if n:
        moment = np.diag(stats["xx"]) / n
        squared_mean = (stats["sx"] / n) ** 2
        variance = moment - squared_mean
        limit = 1e-10 * np.maximum(1., np.maximum(moment, squared_mean))
        if np.any(variance < -limit):
            raise ValueError("Raw-moment cancellation exceeds frozen caller gate")


def worker(protocol_path, folder, stop_after=None):
    p = read(protocol_path)
    if p["code_sha256"] != {str(x): sha(x) for x in (Path(__file__), BASE / "code/alignment_core.py")}:
        raise ValueError("Changed synthetic worker source")
    if sha(p["fixture"]) != p["fixture_sha256"]:
        raise ValueError("Changed synthetic fixture")
    if p["engineering_deadline_epoch"] != deadline():
        raise ValueError("Changed absolute engineering deadline")
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "CHECKPOINT.npz"
    identity = sha(protocol_path)
    with np.load(p["fixture"], allow_pickle=False) as z:
        X, Y = z["X"], z["Y"]
    stats = core.empty_stats(X.shape[1], Y.shape[1]); cursor = 0
    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as z:
            if str(z["protocol_sha256"]) != identity:
                raise ValueError("Checkpoint source/deadline identity changed")
            stats = {k: z[k].copy() for k in ("count", "sx", "sy", "xx", "xy")}
            stats["count"] = int(stats["count"]); cursor = int(z["cursor"])
        validate_moments(stats)
        if stats["count"] != cursor or cursor % p["block"] != 0 and cursor != len(X):
            raise ValueError("Invalid complete-block checkpoint cursor")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, receive_signal)
    atomic_json(folder / "STARTED.json", {"pid": os.getpid(), "cursor": cursor})
    accepted = 0
    while cursor < len(X):
        reason = "signal" if STOP else "deadline" if time.time() >= min(deadline(), p["deadline_epoch"]) else None
        if reason:
            if not checkpoint.exists():
                atomic_npz(checkpoint, **stats, cursor=np.int64(cursor), protocol_sha256=np.asarray(identity))
            atomic_json(folder / "STATUS.json", {"status": "stopped", "reason": reason, "cursor": cursor,
                        "deadline_epoch": p["deadline_epoch"], "protocol_sha256": identity})
            return
        end = min(cursor + p["block"], len(X))
        core.update_stats(stats, X[cursor:end], Y[cursor:end]); validate_moments(stats)
        cursor = end; accepted += 1
        atomic_npz(checkpoint, **stats, cursor=np.int64(cursor), protocol_sha256=np.asarray(identity))
        atomic_json(folder / "CURSOR.json", {"cursor": cursor})
        if stop_after is not None and accepted >= stop_after:
            atomic_json(folder / "STATUS.json", {"status": "stopped", "reason": "planned_checkpoint", "cursor": cursor,
                        "deadline_epoch": p["deadline_epoch"], "protocol_sha256": identity})
            return
        if p["pause_per_block"]:
            time.sleep(p["pause_per_block"])
    predictions = {}
    for arm, cols in core.columns(p["lookback"]).items():
        model = core.solve(stats, cols)
        if model["diagnostics"]["normal_equation_relative_inf"] > 1e-10:
            raise ValueError("Synthetic worker stationarity failed")
        predictions[arm] = core.predict(model, X[:16])
    atomic_npz(folder / "PREDICTIONS.npz", **predictions)
    atomic_json(folder / "STATUS.json", {"status": "completed", "cursor": cursor,
                "deadline_epoch": p["deadline_epoch"], "protocol_sha256": identity})


def clock_values(ends):
    ny = ZoneInfo("America/New_York")
    dates = [dt.datetime.fromtimestamp(int(t) / 1e9, dt.timezone.utc).astimezone(ny) for t in ends]
    u = np.array([(d.hour * 60 + d.minute + d.second / 60 - 570) / 390 for d in dates])
    return np.column_stack((np.sin(2*np.pi*u), np.cos(2*np.pi*u), np.sin(4*np.pi*u),
                            np.cos(4*np.pi*u), *[np.array([d.weekday() == k for d in dates], float) for k in range(4)]))


def synthetic_calendar(native_minutes):
    """Synthetic half days around a real DST boundary, not an exchange calendar."""
    ny = ZoneInfo("America/New_York"); stamps = []; sessions = []
    first = dt.date(2021, 3, 1)
    for k in range(42):
        day = first + dt.timedelta(days=k)
        if day.weekday() > 4:
            continue
        length = 210 if k in (11, 25) else 390
        start = dt.datetime.combine(day, dt.time(9, 30), ny)
        for minute in range(native_minutes, length + 1, native_minutes):
            stamps.append(int((start + dt.timedelta(minutes=minute)).timestamp() * 1e9))
            sessions.append(day.isoformat())
    return np.array(stamps, np.int64), np.array(sessions, dtype="U10")


def feature_and_solver_cases():
    results = []; rng = np.random.default_rng(2026)
    for L, H, minutes in ((480, 60, 1), (96, 8, 60)):
        check_time(); ends, dates = synthetic_calendar(minutes)
        log_close = np.cumsum(rng.normal(0, .002, len(ends)))
        origins = np.arange(L-1, len(ends), dtype=np.int64)
        positions = origins[:, None] + np.arange(1-L, 1)
        x = log_close[positions] - log_close[origins, None]
        clock = clock_values(ends[origins])
        full = core.features(x, ends[positions], dates[positions], clock, minutes*60*10**9)
        common = core.feature_dimensions(L)["common"]
        direct = np.column_stack((x,
            (dates[positions[:, 1:]] != dates[positions[:, :-1]]).astype(float),
            np.log(np.diff(ends[positions], axis=1)/(minutes*60*1e9)), clock))
        np.testing.assert_allclose(full[:, :common], direct, rtol=0, atol=1e-12)
        for k, o in enumerate(origins[:32]):
            diffs = np.diff(x[k]); g = (dates[positions[k, 1:]] != dates[positions[k, :-1]])
            np.testing.assert_allclose(full[k, common:], [np.dot(g, diffs), np.dot(g[::-1], diffs)], rtol=0, atol=1e-12)
        # Four-corner mixed differences prove a nonlinear interaction on the
        # varied, feasible historical-calendar design domain, without labels.
        pairs = None
        for i in range(len(origins)):
            for j in range(i+1, min(len(origins), i+2*L)):
                g1 = full[i, L:L+L-1]; g2 = full[j, L:L+L-1]
                if np.any(g1 != g2) and np.any(g1[::-1] != g2[::-1]):
                    pairs = (i, j); break
            if pairs:
                break
        if pairs is None:
            raise AssertionError("Synthetic calendar fails to distinguish history masks")
        i, j = pairs
        x1 = rng.normal(0, .02, L); x1 -= x1[-1]; x0 = np.zeros(L)
        four_x = np.stack((x1, x1, x0, x0))
        four_p = positions[[i, j, i, j]]
        four = core.features(four_x, ends[four_p], dates[four_p], clock[[i,j,i,j]], minutes*60*10**9)
        mixed = four[0] - four[1] - four[2] + four[3]
        np.testing.assert_allclose(mixed[:common], 0, rtol=0, atol=1e-12)
        assert np.all(np.abs(mixed[common:]) > 1e-8)
        assert np.std(full[:, common]) > 1e-12 and np.std(full[:, common+1]) > 1e-12
        assert not np.allclose(full[:, common], full[:, common+1], rtol=0, atol=1e-12)
        # Modify all future prices/timestamps and reconstruct one original input.
        o = int(origins[len(origins)//2]); ix = np.arange(o-L+1, o+1)
        lc2 = log_close.copy(); e2 = ends.copy(); da2 = dates.copy()
        lc2[o+1:] += 9; e2[o+1:] += 10**12; da2[o+1:] = "2099-01-01"
        before = core.features((log_close[ix]-log_close[o])[None], ends[ix][None], dates[ix][None], clock_values(ends[[o]]), minutes*60*10**9)
        after = core.features((lc2[ix]-lc2[o])[None], e2[ix][None], da2[ix][None], clock_values(e2[[o]]), minutes*60*10**9)
        assert np.array_equal(before, after)
        # Direct centered raw-X reconstruction is independent of moment algebra.
        chosen = np.linspace(0, len(full)-1, min(128, len(full)), dtype=np.int64)
        X = full[chosen]; Y = rng.normal(0, .01, (len(X), H))
        stats = core.empty_stats(X.shape[1], H)
        for k in range(0, len(X), 31):
            core.update_stats(stats, X[k:k+31], Y[k:k+31]); validate_moments(stats)
        arm_results = {}
        for arm, cols in core.columns(L).items():
            model = core.solve(stats, cols)
            mx = X[:, cols].mean(0); sd = X[:, cols].std(0)
            sd = np.where(sd > 1e-12, sd, 1.)
            z = (X[:, cols]-mx)/sd; ym = Y.mean(0)
            coef = np.linalg.solve(z.T@z/len(X)+.1*np.eye(len(cols)), z.T@(Y-ym)/len(X))
            expected = ym + z@coef; actual = core.predict(model, X)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-10)
            assert model["diagnostics"]["normal_equation_relative_inf"] <= 1e-10
            arm_results[arm] = {"max_abs_direct_prediction_difference": float(np.max(np.abs(actual-expected))), **model["diagnostics"]}
        fit = np.arange(100, 400, dtype=np.int64); linear_ends = np.arange(600, dtype=np.int64)
        prefix, tail, boundary = core.partition_fit(fit, linear_ends, H)
        assert boundary == 4*len(fit)//5 and np.array_equal(tail, fit[boundary:])
        assert np.all(linear_ends[prefix+H] < linear_ends[tail[0]])
        results.append({"L": L, "H": H, "rows": len(full), "mixed_summary_differences": mixed[common:].tolist(),
                        "direct_features": True, "future_mutation_bitwise_identity": True, "strict_purge": True, "arms": arm_results})
    return results


def run_child(protocol, folder, stop_after=None):
    cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--protocol", str(protocol), "--folder", str(folder)]
    if stop_after is not None:
        cmd += ["--stop-after", str(stop_after)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish_child(proc):
    while proc.poll() is None:
        if time.time() >= deadline():
            proc.terminate()
            raise TimeoutError("Original synthetic stage expired")
        time.sleep(.02)
    stdout, stderr = proc.communicate()
    if proc.returncode:
        raise RuntimeError(f"Child failed {proc.returncode}: {stdout}\n{stderr}")


def compare_checkpoints(a, b):
    with np.load(a / "CHECKPOINT.npz", allow_pickle=False) as za, np.load(b / "CHECKPOINT.npz", allow_pickle=False) as zb:
        for k in ("count", "sx", "sy", "xx", "xy", "cursor"):
            assert np.array_equal(za[k], zb[k]), k
    with np.load(a / "PREDICTIONS.npz", allow_pickle=False) as za, np.load(b / "PREDICTIONS.npz", allow_pickle=False) as zb:
        for arm in core.ARMS:
            assert np.array_equal(za[arm], zb[arm]), arm


def lifecycle_cases():
    rng = np.random.default_rng(1406); L = 96; p = core.feature_dimensions(L)["total"]
    fixture = OUT / "synthetic_fixture.npz"
    atomic_npz(fixture, X=rng.normal(size=(4096, p)), Y=rng.normal(0, .01, size=(4096, 8)))
    proto = {"fixture": str(fixture), "fixture_sha256": sha(fixture), "lookback": L,
             "block": 256, "deadline_epoch": min(deadline(), time.time()+180),
             "engineering_deadline_epoch": deadline(), "pause_per_block": .03,
             "code_sha256": {str(x): sha(x) for x in (Path(__file__), BASE / "code/alignment_core.py")}}
    protocol = OUT / "LIFECYCLE_PROTOCOL.json"; atomic_json(protocol, proto)
    baseline = OUT / "uninterrupted"; resumed = OUT / "resumed"; signaled = OUT / "signaled"
    finish_child(run_child(protocol, baseline))
    finish_child(run_child(protocol, resumed, 2)); assert read(resumed / "STATUS.json")["reason"] == "planned_checkpoint"
    finish_child(run_child(protocol, resumed)); compare_checkpoints(baseline, resumed)
    proc = run_child(protocol, signaled)
    until = time.time()+15
    while not (signaled / "CURSOR.json").exists():
        if proc.poll() is not None or time.time() > until:
            raise RuntimeError("Signal case never reached an actual checkpoint")
        time.sleep(.005)
    proc.send_signal(signal.SIGTERM); finish_child(proc)
    status = read(signaled / "STATUS.json"); assert status["reason"] == "signal" and status["cursor"] >= 256
    finish_child(run_child(protocol, signaled)); compare_checkpoints(baseline, signaled)
    timeout_proto = dict(proto, deadline_epoch=min(deadline(), time.time()+1.5), pause_per_block=.2)
    timeout_path = OUT / "TIMEOUT_PROTOCOL.json"; atomic_json(timeout_path, timeout_proto)
    timed = OUT / "timed_out"; finish_child(run_child(timeout_path, timed))
    status = read(timed / "STATUS.json")
    assert status["reason"] == "deadline" and 0 < status["cursor"] < 4096
    before = sha(timed / "CHECKPOINT.npz"); finish_child(run_child(timeout_path, timed))
    assert sha(timed / "CHECKPOINT.npz") == before
    return {"planned_resume_statistics_and_predictions_bitwise": True,
            "actual_SIGTERM_resume_statistics_and_predictions_bitwise": True,
            "actual_timeout_mid_progress_checkpoint": status,
            "expired_protocol_did_not_advance_or_extend": True}


def maximum_shape_cost():
    check_time(); rng = np.random.default_rng(2026); p = 1448; H = 60
    X = rng.normal(size=(256, p)); Y = rng.normal(0, .01, size=(256, H))
    stats = core.empty_stats(p, H); start = time.monotonic()
    for _ in range(4):
        core.update_stats(stats, X, Y); validate_moments(stats)
    update_seconds = time.monotonic()-start
    start = time.monotonic(); diags = {}
    for arm, cols in core.columns(480).items():
        model = core.solve(stats, cols); diags[arm] = model["diagnostics"]
        assert diags[arm]["normal_equation_relative_inf"] <= 1e-10
    solve_seconds = time.monotonic()-start
    return {"rows": 1024, "full_p": p, "H": H, "update_seconds": update_seconds,
            "three_solve_seconds": solve_seconds,
            "conservative_seconds_for_four_states": 4*(update_seconds/1024*196455 + 4*solve_seconds)+180,
            "synthetic_dense_max_shape_not_market_runtime_guarantee": True, "diagnostics": diags}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("mode", choices=("accept", "worker"))
    parser.add_argument("--protocol"); parser.add_argument("--folder"); parser.add_argument("--stop-after", type=int)
    args = parser.parse_args(); OUT.mkdir(parents=True, exist_ok=True); guard()
    if args.mode == "worker":
        worker(Path(args.protocol), Path(args.folder), args.stop_after); return
    check_time()
    protocol = read(args.protocol)
    assert protocol["engineering_rules_sha256"] == sha(RULES)
    assert all(sha(Path(p)) == h for p, h in protocol["code_sha256"].items())
    assert os.sched_getaffinity(0) == {26}
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    receipt = {"status": "running", "protocol_sha256": sha(args.protocol), "deadline_epoch": deadline(),
               "market_data_or_weights_read": False, "Q3_read": False, "GPU": False}
    try:
        forbidden = []
        for p in (BASE.parents[3] / "data/close.npy", BASE.parents[3] / "data/bars_minute.npz"):
            try:
                open(p, "rb")
            except PermissionError:
                forbidden.append(str(p))
            else:
                raise AssertionError("External numerical path was not rejected")
        receipt["actual_forbidden_file_rejections"] = forbidden
        receipt["feature_and_solver_cases"] = feature_and_solver_cases()
        atomic_json(OUT / "PROGRESS.json", receipt)
        receipt["lifecycle"] = lifecycle_cases()
        atomic_json(OUT / "PROGRESS.json", receipt)
        receipt["maximum_shape_cost"] = maximum_shape_cost()
        check_time(); receipt.update(status="passed_price_free_engineering_and_cost", completed_epoch=time.time())
        atomic_json(OUT / "ENGINEERING_ACCEPTANCE.json", receipt)
        print(json.dumps({"status": receipt["status"], "cost": receipt["maximum_shape_cost"]["conservative_seconds_for_four_states"]}))
    except BaseException as e:
        receipt.update(status="failed_price_free_engineering", error_type=type(e).__name__, error=str(e), failed_epoch=time.time())
        atomic_json(OUT / "ENGINEERING_FAILURE.json", receipt)
        raise


if __name__ == "__main__":
    main()
