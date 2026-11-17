"""Independently verify annual coverage and metrics on fictional saved paths."""
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


root = Path("/home/ayaka/wslcode/aapl-research/outputs/multiscale")
source = root / "admission/synthetic_test_20261107T122518466783Z"
run = root / "synthetic_matrix_validation/runs/c904cb2f49f94c63363768cf"
report_dir = root / "synthetic_matrix_validation/reports/20261107T122631678893Z"
report_file = report_dir / "SYNTHETIC_ONLY_report.json"
csv_file = report_dir / "SYNTHETIC_ONLY_model_rows.csv"
admission = json.loads((source / "admission.json").read_text(encoding="utf8"))
state = json.loads((run / "matrix_state.json").read_text(encoding="utf8"))
report = json.loads(report_file.read_text(encoding="utf8"))
if ("SYNTHETIC TEST ONLY" not in (source / "SYNTHETIC_ONLY.txt").read_text(encoding="utf8") or
        admission.get("data_kind") != "synthetic_test" or state.get("data_kind") != "synthetic_test" or
        report.get("status") != "synthetic_test_report_only" or report.get("independent_test") is not False):
    raise ValueError("Only isolated fictional data are allowed in this verification")
with np.load(source / "bars_month.npz", allow_pickle=False) as stored:
    labels = stored["session_date"].copy()
task = next(t for t in report["tasks"] if t["profile"] == "annual" and t["target"] == "future_close")
if task["completed_folds"] != [1, 2] or task["status"] != "partial_descriptive":
    raise ValueError("The fixture must remain a partial two-fold synthetic report")
models = tuple(task["summary"]["models"])
outcomes, predicted, chosen, previous = [], {name: [] for name in models}, [], None
for fold in task["completed_folds"]:
    record = state["jobs"][f"annual/h12/future_close/fold{fold:02d}"]
    folder = (run / record["directory"]).resolve()
    if not folder.is_relative_to(run.resolve()):
        raise ValueError("Synthetic fold path escapes its run")
    with np.load(folder / "targets.npz", allow_pickle=False) as saved:
        origins, actual = saved["origins"].copy(), saved["actual"].copy()
    positions = []
    for index, origin in enumerate(origins):
        origin = int(origin)
        if previous is None or origin >= previous + 12:
            positions.append(index)
            chosen.append(origin)
            previous = origin
    outcomes.extend(actual[positions, -1].tolist())
    for model in models:
        with np.load(folder / f"prediction_{model}.npz", allow_pickle=False) as saved:
            if not np.array_equal(saved["origins"], origins):
                raise ValueError("Synthetic model origins are not aligned")
            predicted[model].extend(saved["prediction"][positions, -1].tolist())
expected = {"origins": len(chosen), "first_origin_session": str(labels[chosen[0]]),
            "last_origin_session": str(labels[chosen[-1]]), "first_target_session": str(labels[chosen[0] + 1]),
            "last_target_session": str(labels[chosen[-1] + 12])}
outcomes = np.asarray(outcomes, dtype=float)
base_mse = float(np.mean(np.square(np.asarray(predicted["last_close"]) - outcomes)))
with csv_file.open(encoding="utf8", newline="") as stream:
    rows = {row["model"]: row for row in csv.DictReader(stream)
            if row["profile"] == "annual" and row["target"] == "future_close"}
if set(rows) != set(models):
    raise ValueError("Synthetic report model coverage differs")
for name in models:
    values = task["summary"]["models"][name]["annual_nonoverlapping_origins"]
    residual = np.asarray(predicted[name], dtype=float) - outcomes
    mse, mae = float(np.mean(np.square(residual))), float(np.mean(np.abs(residual)))
    relative = 1 - mse / base_mse if base_mse > 0 else None
    if any(values.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Synthetic annual date coverage differs for {name}")
    if (not np.isclose(values["endpoint_mse"], mse, rtol=1e-10, atol=1e-15) or
            not np.isclose(values["endpoint_mae"], mae, rtol=1e-10, atol=1e-15) or
            (relative is not None and not np.isclose(values["relative_improvement_vs_persistence"], relative,
                                                      rtol=1e-10, atol=1e-15))):
        raise ValueError(f"Synthetic annual error recomputation differs for {name}")
    row = rows[name]
    if row["annual_nonoverlapping_origins"] != str(expected["origins"]) or any(
            row["annual_" + key] != expected[key] for key in expected if key != "origins"):
        raise ValueError(f"Synthetic annual CSV coverage differs for {name}")
receipt = {"status": "passed_synthetic_test_only", "created_at_utc": datetime.now(timezone.utc).isoformat(),
           "market_data_read": False, "formal_experiment": False, "independent_test": False,
           "completed_synthetic_folds": task["completed_folds"], "models_verified": len(models),
           "synthetic_nonoverlap_coverage": expected,
           "report_sha256": hashlib.sha256(report_file.read_bytes()).hexdigest(),
           "csv_sha256": hashlib.sha256(csv_file.read_bytes()).hexdigest(),
           "limitations": ["Synthetic partial-fold verification only; no market results or publication rights.",
                           "The dates and errors refer to fictional generated series."]}
path = Path(__file__).with_name("SYNTHETIC_ONLY_receipt.json")
with path.open("x", encoding="utf8") as output:
    json.dump(receipt, output, indent=2, allow_nan=False)
print(json.dumps({"status": receipt["status"], "receipt": str(path), "models_verified": len(models)}))
