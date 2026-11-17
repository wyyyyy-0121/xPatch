"""References for an isolated, post-observation long-scale replication."""
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
V3 = BASE.parents[1] / '20261109_v3'
sys.path.insert(0, str(V3 / 'code'))
import study as original

ROOT = original.ROOT
PRIOR = original.PRIOR
atomic, read, sha, now, npz = original.atomic, original.read, original.sha, original.now, original.npz
make_job, task_key, source_hashes = original.make_job, original.task_key, original.source_hashes


def implementation_hashes():
    return {p.name: sha(p) for p in sorted((BASE/'code').glob('*.py'))
            if p.name in ('common.py', 'controller.py', 'worker.py')}


def jobs():
    result = []
    for profile, horizon in (('annual',12), ('quarterly',3), ('monthly',21)):
        for target in ('future_close','future_log_return'):
            prefix = 'price' if target == 'future_close' else 'anchor'
            for head in ('random','baseline'):
                for fold in range(1,6):
                    for seed in (2026,2027,2028):
                        result.append(make_job(profile,horizon,target,'xpatch',prefix+'_'+head,fold,seed))
    assert len(result)==180 and len({j['id'] for j in result})==180
    return result


def folder(cfg, manifest=None):
    manifest = read(BASE/'protocol/JOBS.json') if manifest is None else manifest
    return Path(manifest['locations'][cfg['id']])
