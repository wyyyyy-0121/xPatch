"""One frozen synthetic FP32 input qualification. No training or data-file input."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

ROOTS = ["/home/ayaka/.venvs/xpatch-ls/lib/python3.13/site-packages",
         "/home/ayaka/.venvs/xpatch-v2/lib/python3.13/site-packages",
         "/home/ayaka/wslcode/aapl-research/.venv/lib/python3.13/site-packages"]
STOP = False
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stopped(_number, _frame):
    global STOP
    STOP = True
def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--protocol", required=True)
    cli.add_argument("--output", required=True)
    args = cli.parse_args()
    stage = Path(__file__).resolve().parent.parent
    protocol_path, output = Path(args.protocol).resolve(), Path(args.output).resolve()
    assert protocol_path.parent == output.parent == stage and output.suffix == ".json"
    assert protocol_path != output and not output.exists(), "No overwrite or rerun"
    temporary = output.with_name("." + output.name + "." + str(os.getpid()) + ".tmp")
    protocol = json.loads(protocol_path.read_text())
    began = time.monotonic()
    report = {"status": "terminal_noadmit", "checks": [], "training": False,
              "inference": False, "market_data_read": False, "Q3_read": False,
              "protocol_sha256": sha(protocol_path), "source_sha256": sha(__file__),
              "new_market_candidate_admitted": False, "full_research_goal_achieved": False}

    def check(name, condition):
        if not bool(condition):
            raise AssertionError(name)
        report["checks"].append(name)

    def deadline():
        if STOP or time.time() >= protocol["deadline_epoch"] - 3:
            raise TimeoutError("signal_or_original_qualification_deadline")

    signal.signal(signal.SIGUSR1, stopped)
    signal.signal(signal.SIGTERM, stopped)
    try:
        check("CPU24_CUDA_disabled", set(os.sched_getaffinity(0)) == {24}
              and os.environ.get("CUDA_VISIBLE_DEVICES") == "-1")
        check("fixed_recipe", protocol["counts"] == [2048, 512, 4096]
              and protocol["shapes"] == [[480, 60], [96, 8]]
              and protocol["data_seed"] == 20261110 and protocol["alphas"] == [.25, 0.0]
              and protocol["relative_reconstruction_limit"] == 1e-6)
        check("finite_maximum_300s", math.isfinite(protocol["deadline_epoch"])
              and math.isfinite(protocol["started_epoch"])
              and protocol["started_epoch"] <= time.time()
              and 0 < protocol["deadline_epoch"] - protocol["started_epoch"] <= 300)
        check("qualification_only", all(protocol[k] is False for k in
              ["training_allowed", "market_data_allowed", "Q3_allowed", "GPU", "paid_resources"]))
        deadline()
        sources = {Path(k).resolve(): v for k, v in protocol["source_sha256"].items()}
        check("self_identity_frozen", sources.get(Path(__file__).resolve()) == sha(__file__))
        for path, wanted in sources.items():
            check("source_identity:" + path.name, path.suffix in {".py", ".json", ".md"}
                  and path.stat().st_size < 2_000_000 and sha(path) == wanted)
        extras = [Path(p).resolve() for p in protocol["additional_runtime_roots"]]
        check("installed_runtime_roots", set(map(str, extras)) == set(ROOTS)
              and all(p.is_dir() for p in extras))
        check("runtime_entries_bound", len(protocol["runtime_file_sha256"]) == 3
              and str(Path(ROOTS[0]) / "numpy/__init__.py") in protocol["runtime_file_sha256"])
        for name, wanted in protocol["runtime_file_sha256"].items():
            path = Path(name).resolve()
            check("runtime_identity:" + name, any(path.is_relative_to(p) for p in extras)
                  and sha(path) == wanted)
        runtime = extras + [Path(p).resolve() for p in
                           ["/usr", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev",
                            sys.prefix, sys.base_prefix]]
        allowed = set(sources) | {protocol_path, Path(__file__).resolve()}

        def audit(event, values):
            if event.startswith("socket."):
                raise PermissionError("No network input")
            if event != "open" or not isinstance(values[0], (str, bytes, os.PathLike)):
                return
            path = Path(os.fsdecode(values[0])).resolve()
            mode, flags = values[1], values[2]
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and path not in {output, temporary}:
                raise PermissionError("Unadmitted qualification write: " + str(path))
            if not writing and path not in allowed and not any(path.is_relative_to(p) for p in runtime):
                raise PermissionError("Unadmitted qualification read: " + str(path))

        sys.dont_write_bytecode = True
        sys.addaudithook(audit)
        for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            os.environ[key] = "1"
        import numpy as np
        report["numpy"] = np.__version__
        check("actual_numpy_entry", protocol["runtime_file_sha256"].get(str(Path(np.__file__).resolve())) == sha(np.__file__))
        report["shapes"] = []
        nfit, nstop, ntest = protocol["counts"]
        n = nfit + nstop + ntest
        for lookback, horizon in protocol["shapes"]:
            deadline()
            rng = np.random.default_rng(protocol["data_seed"] + lookback)
            increments = rng.standard_normal((n, lookback - 1))
            raw = np.zeros((n, lookback), dtype=np.float64)
            raw[:, :-1] = -np.cumsum(increments[:, ::-1], axis=1)[:, ::-1] / math.sqrt(lookback)
            fit = raw[:nfit]
            xm = float(fit.sum(dtype=np.float64) / fit.size)
            xs = math.sqrt(max(float(np.square(fit).sum(dtype=np.float64) / fit.size - xm * xm), 0.0))
            xs = xs if xs > 1e-12 else 1.0
            x = ((raw - xm) / xs).astype(np.float32)
            x64, c = x.astype(np.float64), float(x[0, -1])
            z = math.sqrt(lookback) * xs * (x[:, -1].astype(np.float64) - x[:, -2].astype(np.float64))
            noise = rng.standard_normal((n, horizon)).cumsum(axis=1)
            factor = np.arange(1, horizon + 1, dtype=np.float64) / math.sqrt(horizon)
            d = np.zeros_like(x64)
            d[:, 1:] = np.diff(x64, axis=1)
            dm = float(d[:nfit].sum(dtype=np.float64) / d[:nfit].size)
            ds = math.sqrt(max(float(np.square(d[:nfit]).mean(dtype=np.float64) - dm * dm), 0.0))
            check("finite_positive_scale:" + str(lookback), math.isfinite(ds) and ds > 0)
            e = ((d - dm) / ds).astype(np.float32)
            dhat = e.astype(np.float64) * ds + dm
            xhat = np.empty_like(x64)
            xhat[:, -1] = c
            xhat[:, :-1] = c - np.cumsum(dhat[:, :0:-1], axis=1)[:, ::-1]
            xhat32 = xhat.astype(np.float32)
            penult = (c - dhat[:, -1]).astype(np.float32)
            zhat = math.sqrt(lookback) * xs * (c - penult.astype(np.float64))
            record = {"shape": [lookback, horizon], "x_mean": xm, "x_std": xs,
                      "d_mean": dm, "d_std": ds, "anchor": c, "splits": []}
            report["shapes"].append(record)
            check("finite_arrays:" + str(lookback), all(np.isfinite(v).all() for v in
                  [x, z, e, dhat, xhat, zhat, noise]) and np.all(x64[:, -1] == c))
            for name, sl in [("fit", slice(0, nfit)), ("stop", slice(nfit, nfit + nstop)),
                             ("test", slice(nfit + nstop, n))]:
                deadline()
                reference = float(np.sqrt(np.square(x64[sl]).mean()))
                rmse = float(np.sqrt(np.square(xhat[sl] - x64[sl]).mean()))
                exact_z = zhat[sl].tobytes() == z[sl].tobytes()
                rec = {"split": name, "count": len(x[sl]), "wholeX_FP32_exact":
                       xhat32[sl].tobytes() == x[sl].tobytes(), "RMSE": rmse,
                       "reference_RMS": reference, "relative_RMSE": rmse / reference if reference > 0 else None,
                       "z_bitwise_exact": exact_z, "base_X_sha256": hashlib.sha256(x[sl].tobytes()).hexdigest(),
                       "encoded_X_sha256": hashlib.sha256(e[sl].tobytes()).hexdigest(),
                       "base_z_sha256": hashlib.sha256(z[sl].tobytes()).hexdigest()}
                record["splits"].append(rec)
                check("reconstruction_bound:" + str(lookback) + ":" + name,
                      reference > 0 and rec["relative_RMSE"] <= 1e-6)
                check("exact_signal:" + str(lookback) + ":" + name, exact_z)
            seen, duplicates, conflicts = {}, 0, 0
            for i, row in enumerate(e):
                if i % 128 == 0:
                    deadline()
                key, value = row.tobytes(), z[i:i + 1].tobytes()
                if key in seen:
                    duplicates += 1
                    conflicts += seen[key] != value
                else:
                    seen[key] = value
            record.update(encoded_row_duplicates=duplicates, conflicting_signal_collisions=conflicts,
                          label_sha256={str(a): hashlib.sha256((noise + a * z[:, None] * factor[None, :]).tobytes()).hexdigest()
                                        for a in protocol["alphas"]})
            check("no_conflicting_collisions:" + str(lookback), conflicts == 0)
        deadline()
        report["status"] = "passed_fixed_design_signal_preserving_input_qualification"
        report["wholeX_exact"] = all(s["wholeX_FP32_exact"] for r in report["shapes"] for s in r["splits"])
    except BaseException as error:
        report["failure"] = {"type": type(error).__name__, "message": str(error)}
    report.update(checks_passed=len(report["checks"]), elapsed_seconds=time.monotonic() - began,
                  finished_epoch=time.time(), limits="Fixed support only; not full-information/population equivalence or a pure optimization cause. Python audit guard is not an OS sandbox.")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, output)
    print(json.dumps({"status": report["status"], "checks_passed": report["checks_passed"]}))
    return 0 if report["status"].startswith("passed_") else 2


if __name__ == "__main__":
    sys.exit(main())
