"""Exercise fixed-budget prediction, calibration receipts and reloads on 2021 data."""
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import aapl.jobpool as workers
import aapl.runner as runner
from aapl.common import read_json, runtime, utc_now, write_json
from aapl.data import Market
from aapl.search_space import generate
from aapl.resources import ResourceMonitor


if __name__ == '__main__':
    started = time.perf_counter()
    runtime()
    market = Market.load()
    workers.MARKET = market
    days = sorted(set(market.dates))[:30]
    records = []
    for name in ('ridge', 'lightgbm', 'xpatch', 'patchtst', 'chronos2_frozen', 'chronos2_lora'):
        config = generate()[name][0]
        if name in ('xpatch', 'patchtst'):
            config.update(lookback=48, patch=8, stride=8, batch_size=128, revin=False)
            if name == 'patchtst':
                config.update(d_model=32, layers=1, dropout=0.1)
        if name == 'lightgbm':
            config.update(num_leaves=7, min_child_samples=20)
        if name in ('xpatch', 'patchtst', 'chronos2_lora'):
            config.update(read_json(ROOT / 'outputs/optimization_admission.json')['adopted'].get(name, {}))
        spec = {'config': config, 'seed': 2026, 'budget': 2 if name != 'lightgbm' else 8,
                'stage': 'implementation_smoke', 'protocol_sha256': '2021_smoke_only_not_research',
                'fit_dates': days[:3], 'calibration_dates': days[4:6], 'evaluation_dates': days[7:10],
                'directory': str(ROOT / 'outputs/pipeline_smoke' / name)}
        receipt = workers.predict_work(spec)
        repeated = workers.predict_work(spec)
        assert receipt == repeated
        with np.load(Path(spec['directory']) / 'predictions.npz') as saved:
            np.testing.assert_array_equal(saved['origins'], market.select(spec['evaluation_dates']))
            assert np.isfinite(saved['prediction']).all()
            assert len(saved['prediction']) == 414
        assert receipt['evaluation_target_read_for_scoring'] is False
        assert receipt['resources']['passed']
        records.append({'model': name, 'fixed_budget_prediction': True, 'calibration_receipt': True,
                        'completed_job_cache_identical': True, 'evaluation_labels_not_scored': True})
    write_json(ROOT / 'outputs/pipeline_smoke.json', {'passed': True, 'records': records,
               'completed_at_utc': utc_now(), 'wall_seconds': time.perf_counter() - started,
               'scope': 'All core adapters, fixed-budget final-prediction path; tiny 2021 data, no development or sealed outcomes'})
    print('Pipeline smoke passed', time.perf_counter() - started, flush=True)
