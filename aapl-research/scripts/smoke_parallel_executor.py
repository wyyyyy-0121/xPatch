"""A real pooled halving/promotion/pruning/resume rehearsal on tiny 2021 data."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aapl.common import read_json, runtime, utc_now, write_json
from aapl.data import Market
from aapl.jobpool import initialize, search_work
from aapl.runner import run_inner
from aapl.search_space import generate


if __name__ == '__main__':
    started = time.perf_counter()
    runtime()
    market = Market.load()
    days = sorted(set(market.dates))[:20]
    split = {'fold': 'pooled_implementation_smoke', 'inner': [
        {'fit_dates': days[:2 + i], 'purge_dates': days[2 + i:3 + i],
         'validation_dates': days[3 + i:4 + i]} for i in range(3)]}
    configs = generate()['xpatch'][:5]
    for cfg in configs:
        cfg.update(lookback=48, patch=8, stride=8, batch_size=128, revin=False)
    options = read_json(ROOT / 'outputs/optimization_admission.json')['adopted']['xpatch']
    root = ROOT / 'outputs/executor_smoke/pooled'
    context = mp.get_context('spawn')
    slots = context.Queue()
    for slot in range(3):slots.put(slot)
    with ProcessPoolExecutor(max_workers=3, mp_context=context, initializer=initialize, initargs=(slots,)) as pool:
        selected = run_inner('xpatch', configs, split, market, root,
                             dispatch=lambda specs: list(pool.map(search_work, specs)), execution=options)
    assert selected == run_inner('xpatch', configs, split, market, root, execution=options)
    assert len(read_json(root / 'ranking_rung0.json')) == 5
    assert len(read_json(root / 'ranking_rung2.json')) == 4
    tombstones = list(root.rglob('checkpoints_pruned.json'))
    assert tombstones
    assert all(not (p.parent / row['file']).exists() for p in tombstones for row in read_json(p)['files'])
    write_json(ROOT / 'outputs/parallel_executor_smoke.json', {'passed': True,
               'three_worker_real_search': True, 'three_inner_blocks': True,
               'promotion_and_successful_loser_pruning': True, 'completed_search_resume_identical': True,
               'tombstones': len(tombstones), 'scope': 'tiny 2021 implementation rehearsal only',
               'completed_at_utc': utc_now(), 'wall_seconds': time.perf_counter() - started})
    print('Pooled executor smoke passed', time.perf_counter() - started, flush=True)
