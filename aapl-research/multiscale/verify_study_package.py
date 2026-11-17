"""Exercise saved study predictors in fresh processes on past-only inputs.

One saved validation origin per available task is checked against every model's
original forecast. This is a package and API check; the full-array restoration
audit and independent error recomputation remain separate evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .bars import file_digest
from .historical_bundle_review import read_record, require
from .spec import MODELS, OUT, PROFILES, ROOT
from .study_package import load_study_package, write_new


NEURAL = {"mlp", "gru", "lstm", "xpatch"}


def before(bars, last):
    from .bars import Bars
    part = slice(None, last + 1)
    return Bars(bars.frequency, bars.start_ns[part], bars.end_ns[part], bars.close[part],
                bars.observed[part], bars.session_date[part])


def expected_rejection(call, label):
    try:
        call()
    except ValueError:
        return label
    raise ValueError("Invalid forecast input was accepted: " + label)


def check_task(package, digest, key, admission_dir):
    import numpy as np
    import torch
    from .bars import Bars
    from .development import bars_identity
    from .inference import _cutoff_ns
    from .matrix_executor import read_bars

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and
            1 <= len(os.sched_getaffinity(0)) <= 2, "Audit requires hidden CUDA and at most two CPUs")
    torch.set_num_threads(min(2, len(os.sched_getaffinity(0))))
    torch.set_num_interop_threads(1)
    predictor = load_study_package(package, expected_registry_sha256=digest, device="cpu")
    entry = next(e for e in predictor.registry["tasks"] if
                 (e["profile"], e["horizon"], e["target"]) == key)
    require(entry["status"] == "available", "Selected package task is unavailable")
    review = json.loads((Path(package) / "PACKAGING_RECORD.json").read_text(encoding="utf8"))
    coverage, _ = read_record(review["coverage"]["path"], review["coverage"]["sha256"])
    row = next(r for r in coverage["folds"] if (r["profile"], r["horizon"], r["target"], r["fold"]) ==
               (*key, entry["selected_fold"]))
    folder = Path(row["evidence"]["fold_directory"])
    receipt, rh = read_record(folder / "fold_receipt.json", entry["fold_receipt_sha256"])
    admission, _ = read_record(Path(admission_dir) / "admission.json")
    profile = next(p for p in PROFILES if p.name == key[0])
    bars = read_bars(Path(admission_dir), admission, profile.frequency)
    require(bars_identity(bars) == receipt["identity_fields"]["bars_sha256"], "Audit source bars differ")
    with np.load(folder / "prediction_last_close.npz", allow_pickle=False) as saved:
        origin = int(saved["origins"][0])
    history = before(bars, origin)
    require(int(history.end_ns[-1]) < _cutoff_ns(), "Audit input crosses the protected cutoff")
    kwargs = {"profile": key[0], "horizon": key[1], "target": key[2],
              "as_of_ns": int(history.end_ns[-1]), "history_identity": "recorded-admission:" + bars_identity(history)}
    checks, example = [], None
    for model in MODELS:
        path = folder / ("prediction_" + model + ".npz")
        require(file_digest(path) == receipt["models"][model]["prediction_sha256"], "Saved forecast changed")
        with np.load(path, allow_pickle=False) as saved:
            require(int(saved["origins"][0]) == origin, "Saved models use different first origins")
            expected = np.asarray(saved["prediction"][0], dtype=np.float64)
        rng = torch.get_rng_state().clone()
        result = predictor.predict(history, model=model, **kwargs)
        require(torch.equal(rng, torch.get_rng_state()), "Forecast changed the caller's Torch RNG")
        actual = np.asarray(result["prediction"], dtype=np.float64)
        rtol, atol = (1e-5, 1e-6) if model in NEURAL else (1e-10, 1e-12)
        require(actual.shape == expected.shape == (key[1],) and np.isfinite(actual).all() and
                np.allclose(actual, expected, rtol=rtol, atol=atol), "Packaged forecast differs: " + model)
        require(result["status"] == "historical_exploration_forecast" and
                result["selected_fold"] == entry["selected_fold"] and
                result["core_comparison_complete"] == predictor.registry["core_comparison_complete"] and
                result["independent_test"] is False and result["decision_ns"] == kwargs["as_of_ns"],
                "Forecast scope, fold selection or decision differs")
        future = result["future_targets"]
        require(len(future) == key[1] and all(t["observed"] is None and
                t["target_completion_ns"] is None and kwargs["as_of_ns"] <
                t["scheduled_session_or_bar_end_ns"] < _cutoff_ns() for t in future),
                "Forecast schedule is not a protected, unobserved future path")
        checks.append({"model": model, "rtol": rtol, "atol": atol,
                       "max_absolute_difference": float(np.max(np.abs(actual - expected))),
                       "exact_array_equal": bool(np.array_equal(actual, expected))})
        if model == "xpatch":
            example = result
    guards = []
    guards.append(expected_rejection(lambda: predictor.predict(history, model="unknown", **kwargs), "unknown_model"))
    changed = {**kwargs, "horizon": 999}
    guards.append(expected_rejection(lambda: predictor.predict(history, model="last_close", **changed), "wrong_horizon"))
    changed = {**kwargs, "as_of_ns": kwargs["as_of_ns"] - 1}
    guards.append(expected_rejection(lambda: predictor.predict(history, model="last_close", **changed), "wrong_decision"))
    observed = history.observed.copy()
    observed[-1] = False
    missing = Bars(history.frequency, history.start_ns, history.end_ns, history.close,
                   observed, history.session_date)
    guards.append(expected_rejection(lambda: predictor.predict(missing, model="last_close", **kwargs), "unobserved_input"))
    keep = np.ones(len(history.close), dtype=bool)
    keep[-3] = False
    missing = Bars(history.frequency, history.start_ns[keep], history.end_ns[keep], history.close[keep],
                   history.observed[keep], history.session_date[keep])
    guards.append(expected_rejection(lambda: predictor.predict(missing, model="last_close", **kwargs), "omitted_bar"))
    return {"status": "historical_study_task_package_verified", "profile": key[0], "horizon": key[1],
            "target": key[2], "selected_fold": entry["selected_fold"], "models": checks,
            "input_guards_checked": guards, "past_only_input_last_decision_ns": kwargs["as_of_ns"],
            "saved_target_arrays_opened": False, "training_started": False, "Q3_values_read": False,
            "example": example}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", help="Internal child task: PROFILE:HORIZON:TARGET")
    args = parser.parse_args()
    if args.task:
        profile, horizon, target = args.task.split(":")
        write_new(args.output, check_task(args.package, args.registry_sha256,
                                        (profile, int(horizon), target), args.admission))
        return
    predictor = load_study_package(args.package, expected_registry_sha256=args.registry_sha256)
    out = args.output.resolve()
    require(out.is_relative_to(OUT.resolve()), "Package audit output must remain in the output tree")
    out.mkdir(parents=True, exist_ok=False)
    (out / "task_checks").mkdir()
    started = time.monotonic()
    results = []
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="2",
               OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", PYTHONPATH=str(ROOT))
    for entry in predictor.registry["tasks"]:
        if entry["status"] != "available":
            continue
        key = f"{entry['profile']}:{entry['horizon']}:{entry['target']}"
        path = out / "task_checks" / (key.replace(":", "_") + ".json")
        command = [sys.executable, "-m", "multiscale.verify_study_package", "--task", key,
                   "--package", str(args.package.resolve()), "--registry-sha256", args.registry_sha256,
                   "--admission", str(args.admission.resolve()), "--output", str(path)]
        run = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=120)
        require(run.returncode == 0, "Fresh-process task package check failed: " + key + "\n" + run.stderr[-3500:])
        value = json.loads(path.read_text(encoding="utf8"))
        results.append({"task": key, "check": str(path), "check_sha256": file_digest(path), "result": value})
        print(json.dumps({"stage": "verified_task_package", "tasks_checked": len(results), "task": key}), flush=True)
    payload = {"status": "available_historical_task_packages_verified", "created_at_utc": datetime.now(timezone.utc).isoformat(),
               "registry_sha256": args.registry_sha256, "script_sha256": file_digest(Path(__file__)),
               "planned_tasks": 20, "tasks_verified": len(results), "fresh_processes": len(results),
               "models_verified": len(results) * len(MODELS), "results": results,
               "core_comparison_complete": predictor.registry["core_comparison_complete"],
               "training_started": False, "Q3_values_read": False, "saved_target_arrays_opened": False,
               "wall_seconds": time.monotonic() - started,
               "limitations": ["Only one saved validation origin per available task is checked by this package audit.",
                               "Neural forecasts use the existing single-origin float32 API tolerance; differences are reported.",
                               "This is separate from full-forecast restoration and independent metric recomputation."]}
    write_new(out / "PACKAGE_AUDIT.json", payload)
    print(json.dumps({k: payload[k] for k in ("status", "tasks_verified", "models_verified", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
