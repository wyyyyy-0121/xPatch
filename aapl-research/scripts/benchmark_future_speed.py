"""Measure exact-input reuse on admitted development windows, without training."""
import argparse
from datetime import datetime, timezone
import json
import statistics
import time

import numpy as np
import torch

from fusion.next.common import OUT, digest, source_identity, write_json
from fusion.next.panel import Panel
from scripts import future_speed_runtime as fast


def run(repeats=16):
    panel = Panel.load()
    original = Panel.windows
    rng = np.random.default_rng(1701)
    comparisons = 0
    timings = []
    for fit_end in ('2024-12-30', '2025-06-27', '2025-12-30'):
        scalers = panel.scaler(fit_end)
        eligible = {sid: panel.origins[name][panel.dates[panel.origins[name]] <= fit_end]
                    for sid, name in enumerate(panel.stocks)}
        def sample():
            stocks = rng.integers(len(panel.stocks), size=256)
            rows = np.empty(256, dtype=np.int64)
            for sid, candidates in eligible.items():
                selected = stocks == sid
                rows[selected] = rng.choice(candidates, size=int(selected.sum()))
            return stocks, rows
        for information in ('ohlcv', 'close'):
            for lookback in (48, 96, 240):
                config = {'lookback': lookback, 'information': information}
                stocks, rows = sample()
                experts = rng.standard_normal((256, 4)).astype(np.float32)
                for labels in (True, False):
                    a = original(panel, stocks, rows, config, scalers, experts, include_labels=labels)
                    b = fast.windows(original, panel, stocks, rows, config, scalers, experts, include_labels=labels)
                    if set(a) != set(b):
                        raise AssertionError('Batch fields differ')
                    for field in a:
                        torch.testing.assert_close(a[field], b[field], rtol=0, atol=0, equal_nan=True)
                    comparisons += 1
        # Alternate order on identical rows, so neither path always gets warm pages.
        config = {'lookback': 240, 'information': 'ohlcv'}
        fast.clear()
        start = time.perf_counter()
        fast.windows(original, panel, stocks, rows, config, scalers, experts)
        cold = time.perf_counter()-start
        measured = {'original': [], 'optimized': []}
        for i in range(repeats):
            stocks, rows = sample()
            for name in (('original', 'optimized') if i % 2 else ('optimized', 'original')):
                start = time.perf_counter()
                if name == 'original':
                    original(panel, stocks, rows, config, scalers, experts)
                else:
                    fast.windows(original, panel, stocks, rows, config, scalers, experts)
                measured[name].append(time.perf_counter()-start)
        baseline = statistics.median(measured['original'])
        optimized = statistics.median(measured['optimized'])
        timings.append({'fit_end': fit_end, 'batch_size': 256, 'lookback': 240,
                        'original_median_seconds': baseline, 'optimized_median_seconds': optimized,
                        'warm_speedup': baseline/optimized, 'cold_seconds': cold,
                        'amortization_batches': max(0, cold-optimized)/(baseline-optimized)
                            if baseline > optimized else None})
    report = {'at_utc': datetime.now(timezone.utc).isoformat(),
              'base_engine_source_identity': source_identity(),
              'runtime_sha256': digest(OUT.parents[1]/'scripts/future_speed_runtime.py'),
              'exact_input_parity_passed': True, 'parity_batches': comparisons,
              'lookbacks': [48, 96, 240], 'information': ['ohlcv', 'close'],
              'training_and_inference_inputs_checked': True, 'Q3_accessed': False,
              'training_performed': False, 'cache_memory_cap_bytes': fast.MAX_BYTES,
              'warm_speedup': min(row['warm_speedup'] for row in timings),
              'timing_origins': 'new random admitted origins each repeat, identical across paired paths',
              'timings': timings, 'scope': 'CPU input preparation only; no end-to-end GPU speed claim'}
    write_json(OUT/'performance/future_speed_benchmark.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=16)
    run(parser.parse_args().repeats)
