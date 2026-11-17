"""Bounded CPU Ridge shape check on inner-development SIP minute data."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from .benchmark_local import inputs
from .features import fit_scaler
from .ridge import fit
from .spec import OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--fit-count", type=int, default=32768)
    args = parser.parse_args()
    if not 4096 <= args.fit_count <= 65536 or args.fit_count % 128:
        raise ValueError("fit-count must be a multiple of 128 between 4096 and 65536")
    windows, fit_origins, stop_origins, data_id = inputs(args.admission, args.fit_count)
    start = time.monotonic()
    scaler = fit_scaler(windows, fit_origins, 60, "future_log_return")
    scaled_at = time.monotonic()
    model = fit(windows, scaler, fit_origins, 60, "future_log_return", alpha=1.0)
    fitted_at = time.monotonic()
    pred = model.predict(windows, scaler, stop_origins)
    if pred.shape != (128, 60) or not np.isfinite(pred).all():
        raise ValueError("Invalid Ridge stop-partition prediction shape")
    end = time.monotonic()
    directory = OUT / "local_ridge_benchmark" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "ridge_coefficients.npz", coefficients=model.coefficients)
    result = {"status": "passed", "purpose": "bounded engineering shape/timing only; no forecast score or selection",
              "source": "prior SIP minute data, new H60 origin gate provisional", "data_identity": data_id,
              "lookback": 480, "horizon": 60, "fit_origins": len(fit_origins), "stop_origins": len(stop_origins),
              "alpha": model.alpha, "scaler_seconds": scaled_at - start,
              "fit_seconds": fitted_at - scaled_at, "predict_seconds": end - fitted_at,
              "total_seconds": end - start, "cpu_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "coefficient_file_bytes": (directory / "ridge_coefficients.npz").stat().st_size,
              "Q3_values_read": False,
              "limitations": ["Single bounded subset, not a full fold or hyperparameter search.",
                              "No external validation, scoring, or independent test."]}
    (directory / "benchmark.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"directory": str(directory), "status": "passed", "fit_origins": len(fit_origins),
                      "scaler_seconds": result["scaler_seconds"], "fit_seconds": result["fit_seconds"],
                      "predict_seconds": result["predict_seconds"], "total_seconds": result["total_seconds"]}))


if __name__ == "__main__":
    main()
