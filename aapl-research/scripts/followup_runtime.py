"""Prepare the November 8 experiment and isolate future process writes.

This module never launches a GPU job. A draft is not a launch authorization or
an admission receipt. Runtime routing is confined to its calling process.
"""
import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import torch

from fusion.next.common import ROOT, OUT as BASE, PLAN, digest, identity, read_json, source_identity, write_json
from scripts.review_fusion_development import FOLLOW, QUARTERS, allowed


EARLIEST = '2026-11-08T09:00:00+08:00'
DEADLINE = '2026-11-09T09:34:22.509412+00:00'
RESERVE_HOURS = 8.


def candidates(reference):
    """Three fixed cells; the fourth cell reuses the exact incumbent recipe."""
    if (reference.get('family') != 'fusion' or not reference.get('foundation')
            or reference.get('foundation_mode') != 'frozen' or reference.get('loss') != 'scaled'):
        raise ValueError('The fixed matrix requires a frozen-foundation, scaled-loss reference')
    if reference.get('sampler_policy') != 'balanced_complete_coverage_v1':
        raise ValueError('Retain the admitted coverage sampler')
    result = []
    for name, foundation, loss in [('A_foundation_raw', True, 'raw'),
                                    ('B_no_foundation_scaled', False, 'scaled'),
                                    ('C_no_foundation_raw', False, 'raw')]:
        config = copy.deepcopy({k: v for k, v in reference.items() if k != 'candidate_id'})
        config.update(foundation=foundation, loss=loss)
        if not foundation:
            config['representation_cache'] = False
        config['candidate_id'] = 'followup_' + identity(config)[:12]
        result.append({'name': name, 'reference_candidate_id': reference['candidate_id'],
                       'config': config, 'old_optimizer_or_checkpoint_reused': False})
    return result


def launch_window(limit, projected_hours, now=None):
    """Validate time only. A successful return does not admit a training job."""
    now = now or datetime.now(timezone.utc)
    if limit.get('total_wall_hours') != 100 or limit.get('deadline_utc') != DEADLINE:
        raise ValueError('The original 100-hour deadline must not change')
    if now < datetime.fromisoformat(EARLIEST):
        raise RuntimeError('Next large computation is scheduled for November 8, 09:00 Shanghai')
    left = (datetime.fromisoformat(DEADLINE) - now).total_seconds()/3600
    if left <= 0:
        raise TimeoutError('Original absolute deadline elapsed')
    if not np.isfinite(projected_hours) or projected_hours <= 0:
        raise ValueError('Use an actual positive stage runtime projection')
    if 1.5*projected_hours + RESERVE_HOURS > left:
        raise RuntimeError('Stage plus 50% margin and delivery reserve exceeds remaining time')
    return {'earliest_utc': datetime.fromisoformat(EARLIEST).astimezone(timezone.utc).isoformat(),
            'deadline_utc': DEADLINE, 'remaining_hours': left,
            'projected_hours': projected_hours, 'margin_multiplier': 1.5,
            'delivery_reserve_hours': RESERVE_HOURS, 'training_admitted': False}


def prepare(snapshot):
    """Create a reviewable draft while the original run is still active."""
    snapshot = Path(snapshot).resolve()
    if not snapshot.is_relative_to(FOLLOW/'snapshots'):
        raise ValueError('Use an isolated completed-development snapshot')
    manifest = read_json(snapshot/'manifest.json')
    if (manifest['identity'] != identity({k:v for k,v in manifest.items() if k != 'identity'})
            or manifest['source_identity'] != source_identity() or manifest['plan_sha256'] != digest(PLAN)
            or manifest.get('Q2_Q3_predictions_included') is not False or manifest.get('training_launched') is not False):
        raise ValueError('Development snapshot identity differs')
    for name, row in manifest['files'].items():
        if not allowed(name) or digest(snapshot/name) != row['sha256']:
            raise ValueError('Completed-development snapshot was modified')
    scope = read_json(snapshot/'performance/scope_100h.json')
    limit = manifest['snapshot']['limit']
    if (scope['training_budget_policy']['total_coverage_equivalent_epochs'] != 2
            or scope['validation'] != list(QUARTERS) or scope['seeds'] != [2026, 2027, 2028]
            or limit.get('total_wall_hours') != 100 or limit.get('deadline_utc') != DEADLINE):
        raise ValueError('Original scientific scope or time limit differs')
    selected = read_json(snapshot/'selections/fusion.json')['selected']
    proposed = candidates(selected)
    value = {'status': 'draft_not_runnable', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
             'source_identity': source_identity(), 'source_snapshot_identity': manifest['identity'],
             'scope_identity': scope['identity'], 'base_plan_sha256': digest(PLAN),
             'followup_plan_sha256': digest(ROOT/'FUSION_FOLLOWUP_PLAN.md'),
             'runtime_sha256': digest(Path(__file__)),
             'guard_sha256': digest(ROOT/'scripts/followup_guard.py'), 'limit': limit,
             'earliest_large_run': EARLIEST, 'reference': selected, 'candidates': proposed,
             'reference_status': 'provisional until original F5, F6 and actual F8 finish',
             'validation': list(QUARTERS), 'pilot_seeds': [2026], 'expanded_seeds': [2026,2027,2028],
             'total_coverage_equivalent_epochs': 2, 'pilot_full_fits': len(proposed)*len(QUARTERS),
             'pilot_GPU_hours_ceiling': 6, 'conditional_extension_GPU_hours_ceiling': 6,
             'delivery_reserve_hours': RESERVE_HOURS, 'new_GPU_benchmark_performed': False,
             'training_launched': False, 'Q2_Q3_values_read': False,
             'release_requirements': ['Original run and local delivery have actually completed.',
                 'Reference and matrix are finalized and hashed before the first new formal score.',
                 'Actual new-shape timing, numerical and checkpoint checks pass.',
                 'One execution lock and an independent whole-process-group absolute time guard are active.',
                 'Start date, remaining-time margin and delivery reserve pass.',
                 'Every new output/checkpoint is isolated; the original OOF and caches stay read only.']}
    value['identity'] = identity(value)
    path = FOLLOW/'preparation'/('draft_'+value['identity'][:16]+'.json')
    write_json(path, value)
    write_json(FOLLOW/'preparation/latest.json', {'draft':str(path.relative_to(ROOT)),
               'identity':value['identity'], 'status':value['status'], 'training_launched':False})
    return path, value


class ReadOnlyWarmCache:
    """Reuse exact immutable latents; compute misses only in the new run cache."""
    def __init__(self, local, base_cache):
        from fusion.next import cache
        self.local = local
        self.base_cache = Path(base_cache)
        self.write_root = cache.OUT
        self.local_data_root = cache.OUT/'data'
        self.originals = {}
        self.reused_rows = 0
        self.local_rows = 0

    def original(self, sid):
        if sid not in self.originals:
            name = self.local.panel.stocks[sid]
            # The local cache's data link has the original manifest bytes.
            spec = {**self.local.base, 'stock':name,
                    'data_manifest_sha256':digest(self.local_data_root/name/'manifest.json')}
            directory = self.base_cache/identity(spec)
            members = [directory/name for name in ('identity.json','latent.npy','filled.npy')]
            if not all(p.exists() for p in members):
                self.originals[sid] = None
            else:
                if read_json(members[0]) != spec:
                    raise ValueError('Warm-cache identity differs')
                values = np.load(members[1], mmap_mode='r')
                filled = np.load(members[2], mmap_mode='r')
                if (values.shape != (len(self.local.panel.dates),self.local.width)
                        or values.dtype != self.local.dtype or filled.shape != (len(values),)
                        or filled.dtype != np.dtype(bool)):
                    raise ValueError('Warm-cache shape or precision differs')
                self.originals[sid] = (values, filled)
        return self.originals[sid]

    @torch.no_grad()
    def get(self, batch):
        from fusion.next import cache
        if cache.OUT != self.write_root:
            raise ValueError('Warm cache cannot outlive its isolated runtime binding')
        stocks = batch['stock'].detach().cpu().numpy()
        rows = batch['row'].detach().cpu().numpy()
        if (stocks.shape != rows.shape or rows.ndim != 1 or not len(rows)
                or rows.min() < 0 or rows.max() >= len(self.local.panel.dates)
                or stocks.min() < 0 or stocks.max() >= len(self.local.panel.stocks)):
            raise ValueError('Invalid latent-cache origins')
        if np.any(self.local.panel.dates[rows] > self.local.allowed_end):
            raise ValueError('Warm-cache request crosses its declared time boundary')
        result = np.empty((len(rows),self.local.width), dtype=np.float32)
        reused = np.zeros(len(rows),dtype=bool)
        for sid in np.unique(stocks):
            original = self.original(int(sid))
            if original is None:
                continue
            values, filled = original
            select = np.flatnonzero(stocks == sid)
            hit = select[filled[rows[select]]]
            if len(hit):
                result[hit] = values[rows[hit]]
                reused[hit] = True
        missing = np.flatnonzero(~reused)
        if len(missing):
            sub = {name:batch[name][missing] for name in ('stock','row','foundation_context','local')}
            result[missing] = self.local.get(sub).float().cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite reused foundation latent')
        self.reused_rows += int(reused.sum())
        self.local_rows += len(missing)
        return torch.as_tensor(result,device=batch['local'].device)


@contextmanager
def isolated_outputs(destination, read_root=BASE):
    """Route training writes in this process, fence JSON writes, restore on exit.

    Install the existing FP32/speed adapters after entering in a dedicated
    worker process. Data, weights and OOF are loaded from the original root;
    construct ExpertStore before this context, since it shares Trainer's module.
    """
    from fusion.next import common,accounting,cache,checkpoint,experiment,trainer,hardware,foundation
    destination, read_root = Path(destination).resolve(), Path(read_root).resolve()
    if (not destination.is_relative_to(FOLLOW/'runs') or destination == FOLLOW/'runs'
            or destination.is_relative_to(read_root) or read_root.is_relative_to(destination)):
        raise ValueError('Follow-up writes must use a separate run directory')
    if not (read_root/'data').is_dir():
        raise ValueError('Original read-only data is missing')
    destination.mkdir(parents=True,exist_ok=True)
    lock = (destination/'execution.lock').open('a')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError('A follow-up process already owns this output directory')
    saved, caches = [], []
    original_writer = common.write_json
    def fenced(path, value):
        if not Path(path).resolve().is_relative_to(destination):
            raise ValueError('Follow-up write leaves its isolated output directory')
        return original_writer(path,value)
    def set_value(module,name,value):
        saved.append((module,name,getattr(module,name)))
        setattr(module,name,value)
    try:
        link = destination/'data'
        if link.is_symlink():
            if link.resolve() != (read_root/'data').resolve():
                raise ValueError('Follow-up input data link changed')
        elif link.exists():
            raise ValueError('Do not replace an existing follow-up data directory')
        else:
            link.symlink_to(read_root/'data',target_is_directory=True)
        for name in ('portable_inference_policy.json','future_speed_policy.json'):
            source = read_root/'performance'/name
            if source.exists():
                target = destination/'performance'/name
                target.parent.mkdir(exist_ok=True)
                if target.exists() and digest(target) != digest(source):
                    raise ValueError('Follow-up numerical/speed policy changed')
                if not target.exists():
                    shutil.copyfile(source,target)
        for module in (common,accounting,cache,experiment,trainer,hardware):
            set_value(module,'OUT',destination)
        for module in (common,accounting,cache,checkpoint,experiment,trainer,hardware,foundation):
            set_value(module,'write_json',fenced)
        original_cache = trainer.RepresentationCache
        def warm(*args,**kwargs):
            value = ReadOnlyWarmCache(original_cache(*args,**kwargs), read_root/'representation_cache')
            caches.append(value)
            return value
        set_value(trainer,'RepresentationCache',warm)
        binding = {'base_source_identity':source_identity(), 'read_root':str(read_root),
                   'write_root':str(destination), 'runtime_sha256':digest(Path(__file__)),
                   'original_inputs_read_only':True, 'original_checkpoints_reused':False}
        receipt = destination/'runtime_binding.json'
        if receipt.exists() and read_json(receipt) != binding:
            raise ValueError('Follow-up runtime binding changed')
        fenced(receipt,binding)
        yield SimpleNamespace(write_root=destination,read_root=read_root,caches=caches,binding=binding)
    finally:
        for module,name,value in reversed(saved):
            setattr(module,name,value)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare only; never launch a training job')
    parser.add_argument('--snapshot',type=Path,required=True)
    args = parser.parse_args()
    path,value = prepare(args.snapshot)
    print({'draft':str(path),'status':value['status'],'candidate_count':len(value['candidates']),
           'training_launched':False})
