"""Bounded sustained CUDA timing for the user's shortened nine-hour window.

This is a training-only diagnostic, not a scored trial or a reset of old probes.
The old manifest and all three previous short-probe reservations are preserved.
"""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from fusion.next.common import ROOT, OUT as BASE, digest, identity, read_json, utc_now, write_json
from scripts import followup_guard as guard
from scripts.followup_admission import check_frozen
from scripts.followup_runtime import FOLLOW, isolated_outputs
from scripts.run_fusion_followup import (original_idle, execution_locks, runtime_signature,
    bind_guarded_worker, state_hash, clone_prefix, attempts, spent_seconds)

ENTRY = Path(__file__).resolve()
DEADLINE = '2026-11-07T11:11:20+00:00'
BLOCK = 256


def worker(manifest_path, directory):
    request = bind_guarded_worker(directory, ENTRY, 'exploration')
    manifest = check_frozen(manifest_path)
    original_idle(manifest, query_gpu=False)
    from scripts import portable_inference_runtime as numerical, future_speed_runtime as speed
    from fusion.next.panel import Panel
    from fusion.next.trainer import Trainer, ExpertStore
    from fusion.next.experiment import model_for
    from fusion.next.recipe import for_fit
    from fusion.next.hardware import runtime, gpu_telemetry
    from fusion.next.accounting import Usage
    if numerical._installed or speed._installed:
        raise RuntimeError('Use a clean diagnostic interpreter')
    panel = Panel.load(); store = ExpertStore(panel)
    if store.sources != manifest['expert_sources'] or store.key != manifest['expert_scope_key']:
        raise ValueError('Frozen expert sources changed')
    candidate = next(c for c in manifest['candidates'] if c['name'] == 'A_foundation_raw')
    fold = manifest['folds']['2026Q1']
    config = {**for_fit(candidate['config'], fold['fit_end']), 'seed':2026}
    signature = runtime_signature()
    with isolated_outputs(directory/'work'):
        numerical.install(); speed.install()
        def make(namespace):
            if runtime(2026) != 'cuda': raise RuntimeError('Real CUDA is required')
            return Trainer(panel, model_for(panel,config,'cuda'), config, fold['fit_end'], namespace,2026,store,'cuda')
        with Usage('performance','nine-hour sustained diagnostic',gpu=True) as usage:
            started=time.perf_counter(); trainer=make('sustained')
            torch.cuda.synchronize(); initialization=time.perf_counter()-started
            blocks=[]; copy_seconds=None; copy_bytes=None
            for endpoint in (BLOCK,2*BLOCK,3*BLOCK,4*BLOCK):
                started=time.perf_counter(); trainer.run_until(endpoint,usage)
                torch.cuda.synchronize(); blocks.append(time.perf_counter()-started)
                write_json(directory/'diagnostic_progress.json',{'step':endpoint,'block_seconds':blocks,'updated_at_utc':utc_now()})
                if endpoint==3*BLOCK:
                    from fusion.next.checkpoint import resolve_checkpoint
                    committed,_=resolve_checkpoint(trainer.path);copy_bytes=committed.stat().st_size
                    started=time.perf_counter()
                    clone_prefix(trainer.path,trainer.path.parents[1]/'resumed/seed_2026.pt')
                    copy_seconds=time.perf_counter()-started
            expected=state_hash(trainer)
            del trainer;gc.collect();torch.cuda.empty_cache()
            trainer=make('resumed');trainer.resume();trainer.run_until(4*BLOCK,usage)
            if state_hash(trainer)!=expected: raise ValueError('Sustained exact CUDA resume failed')
            sid=panel.stocks.index('AAPL');origins=panel.origins['AAPL']
            rows=origins[(panel.dates[origins]<=fold['fit_end']) & store.available[sid][origins]][:512]
            stocks=np.full(len(rows),sid); expert=store.predict(stocks,rows)
            values=[];timings=[]
            for _ in range(2):
                started=time.perf_counter();values.append(trainer.predict(stocks,rows,expert))
                torch.cuda.synchronize();timings.append(time.perf_counter()-started)
            difference=float(np.max(np.abs(values[0]-values[1])))
            if difference!=0 or not np.isfinite(values).all(): raise ValueError('FP32 repeat differs')
            result={'schema':1,'artifact_type':'sustained_training_only_diagnostic',
                'parent_manifest_identity':manifest['identity'],'candidate':candidate,
                'entry_sha256':digest(ENTRY),'guard_request_identity':request['identity'],
                'runtime_signature':signature,'worker':guard.process_identity(),
                'previous_probe_reservations':3,'diagnostic_reservation_number':4,
                'updates_per_block':BLOCK,'block_seconds':blocks,
                'seconds_per_update_upper':max(blocks)/BLOCK,
                'initialization_seconds':initialization,'checkpoint_copy_bytes':copy_bytes,
                'checkpoint_copy_seconds':copy_seconds,'exact_resume_passed':True,
                'FP32_repeat_max_difference':difference,
                'prediction_seconds_per_origin_upper':max(timings)/len(rows),
                'training_only':True,'Q2_Q3_scored':False,'predictive_success_demonstrated':False,
                'telemetry':gpu_telemetry(),'completed_at_utc':utc_now()}
            result['identity']=identity(result);write_json(directory/'diagnostic_result.json',result)
    return result


def launch(manifest_path):
    from datetime import datetime
    manifest=check_frozen(manifest_path)
    run=FOLLOW/'runs'/manifest['identity']
    previous=attempts(run,'probe')
    if len(previous)!=3:raise ValueError('Expected exactly three preserved prior reservations')
    for p in previous: spent_seconds(p,600)
    directory=run/'attempts/probe_04_sustained_9h'
    if directory.exists():raise RuntimeError('Diagnostic reserved already; inspect exact existing processes')
    if time.time()+600>=datetime.fromisoformat(DEADLINE).timestamp():raise TimeoutError('Nine-hour GPU deadline too near')
    with execution_locks():
        original_idle(manifest)
        directory.mkdir(parents=True)
        write_json(directory/'diagnostic_admission.json',{'authorization':'User requested immediate plan change with nine GPU hours left',
            'parent_manifest_sha256':digest(manifest_path),'entry_sha256':digest(ENTRY),
            'GPU_deadline_utc':DEADLINE,'candidate':'A_foundation_raw','training_only':True,'stage_seconds':600})
        process,receipt=guard.start(directory,[sys.executable,str(ENTRY),'--manifest',str(manifest_path),
            '--worker','--directory',str(directory)],manifest['limit'],600)
        write_json(directory/'scheduler_dispatch.json',{'scheduler':guard.process_identity(),'guard':receipt,'created_at_utc':utc_now()})
        code=process.wait()
        if code:raise RuntimeError('Bounded diagnostic failed; preserve evidence')
        print(read_json(directory/'diagnostic_result.json'),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--directory',type=Path)
    args=parser.parse_args()
    worker(args.manifest,args.directory) if args.worker else launch(args.manifest.resolve())
