from pathlib import Path

from aapl.common import ROOT, digest, identity, read_json

OUT = ROOT / 'outputs/fusion'
RESEARCH_ID = 'aapl-foundation-fusion-20261105'
EXPERTS = ('ridge', 'lightgbm', 'xpatch', 'patchtst', 'chronos2_frozen')
SEEDS = (2026, 2027, 2028)


def source_hash():
    return identity({str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT / 'fusion').glob('*.py'))})


def selected_experts():
    return {name: read_json(ROOT / f'outputs/formal/search/{name}/selection.json')[0]
            for name in EXPERTS}


def candidate_grid():
    rows = []
    for basis in ('foundation', 'without_foundation', 'static'):
        for alpha in (1., 10., 100., 1000.):
            rows.append(dict(kind='ridge', basis=basis, alpha=alpha))
    for basis in ('foundation', 'without_foundation'):
        for hidden in (8, 16):
            for decay in (0.001, 0.01):
                for residual in (0.0, 0.1):
                    rows.append(dict(kind='gate', basis=basis, hidden=hidden, decay=decay,
                                     residual_cap=residual, lr=0.001, epochs=24))
    return [{**r, 'candidate_id': identity(r)[:16]} for r in rows]
