"""Exercise resumable halving/refit on tiny pre-2023 data; not a research comparison."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from aapl.common import runtime, utc_now, write_json
from aapl.data import Market
from aapl.search_space import generate
import aapl.runner as runner


if __name__ == "__main__":
    started = time.perf_counter()
    runtime()
    market = Market.load()
    days = sorted(set(market.dates))[:20]
    # CPU avoids competing with the sole GPU pressure-test job.
    old_runtime = runner.runtime
    def cpu_runtime(seed=2026):
        old_runtime(seed)
        return "cpu"
    runner.runtime = cpu_runtime
    split = {"fold": "implementation_smoke_only", "inner": [
        {"fit_dates": days[:2 + i * 3], "validation_dates": days[3 + i * 3:4 + i * 3],
         "purge_dates": days[2 + i * 3:3 + i * 3]} for i in range(3)]}
    records = []
    for name in ("xpatch", "patchtst", "ridge", "lightgbm"):
        cfg = generate()[name][0]
        if name in ("xpatch", "patchtst"):
            cfg.update(lookback=48, patch=8, stride=8, batch_size=128, revin=False)
            if name == "patchtst":
                cfg.update(d_model=32, layers=1, dropout=0.1)
        if name == "lightgbm":
            cfg.update(n_estimators=8, num_leaves=7, min_child_samples=20)
        directory = ROOT / "outputs/executor_smoke/current" / name
        selected = runner.run_inner(name, [cfg], split, market, directory)
        resumed = runner.run_inner(name, [cfg], split, market, directory)
        assert selected == resumed
        if name in ("xpatch", "patchtst"):
            summary, predict, retained = runner.neural_job(cfg, 2026, days[:3], days[4:5], 2,
                directory / "refit", market, refit=True)
            assert predict(market.select(days[5:6])).shape == (138,)
            del retained, predict
        records.append({"model": name, "three_inner_blocks": True, "completed_selection": True,
                        "completed_search_resume_identical": True, "refit_checked": name in ("xpatch", "patchtst")})
    write_json(ROOT / "outputs/executor_smoke.json", {"passed": True, "records": records,
               "scope": "CPU implementation smoke, tiny 2021 data; not HPO performance or research results",
               "completed_at_utc": utc_now(), "wall_seconds": time.perf_counter() - started})
    print("Executor smoke passed", time.perf_counter() - started, flush=True)
