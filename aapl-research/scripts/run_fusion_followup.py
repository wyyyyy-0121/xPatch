"""Guarded training-only probes, fixed pilot and conditional seed extension.

No entry changes the original engine, fits on final data or opens a holdout.
The pilot stops for review; extending one recipe to three seeds requires the
recomputed prespecified decision and a separate cumulative remaining-budget gate.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import gc
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import psutil
import torch

from fusion.next.common import ROOT, OUT as BASE, digest, identity, read_json, utc_now, write_json
from scripts import followup_guard as guard
from scripts.followup_admission import check_frozen, identified
from scripts.followup_runtime import FOLLOW, DEADLINE, isolated_outputs, launch_window
from scripts.review_fusion_development import QUARTERS, validate_rows


PROBE_SECONDS = 600
MAX_PROBES = 5
PROBE_UPDATES = 20
PILOT_SECONDS = 6*3600
EXTENSION_SECONDS = 6*3600
ENTRY = Path(__file__).resolve()


def runtime_signature():
    """Query the GPU without creating a CUDA context in the scheduler."""
    result = subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.total,driver_version',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10,check=True)
    devices = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(devices) != 1:
        raise RuntimeError('The admitted follow-up uses exactly one visible GPU')
    return {'gpu':devices[0],'torch':str(torch.__version__),'CUDA_runtime':torch.version.cuda,
            'python':sys.version.split()[0],'affinity':sorted(os.sched_getaffinity(0)),
            'threads':int(os.environ.get('AAPL_CPU_THREADS',os.environ.get('OMP_NUM_THREADS','4'))),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def original_idle(manifest,query_gpu=True):
    if os.environ.get('FUSION_EXECUTION_PROFILE') != 'autodl':
        raise RuntimeError('New GPU probes and formal training run only on the admitted cloud host')
    if read_json(BASE/'cloud/hard_time_limit.json') != manifest['limit']:
        raise ValueError('Cloud original absolute limit differs')
    state = read_json(BASE/'cloud/bounded_run_state.json')
    dispatch = read_json(BASE/'cloud/bounded_run_dispatch.json')
    limiter = read_json(BASE/'cloud/hard_time_limit_state.json')
    if (state['stage'] != 'F6_complete_return_ready' or state['scope_identity'] != manifest['scope_identity']
            or state['source_identity'] != manifest['source_identity'] or guard.exact_alive(dispatch)):
        raise RuntimeError('Original cloud task has not actually completed and exited')
    if not guard.exact_alive(limiter):
        raise RuntimeError('Original independent absolute time limiter is not alive')
    for path in (BASE/'usage_events').glob('*.json'):
        row = read_json(path)
        if (row['status'] == 'running' and row.get('gpu') and row.get('boot_id') == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                and guard.exact_alive({'pid':row['pid'],'created_unix':row['process_created_unix']})):
            raise RuntimeError('Original accounted GPU work is still alive')
    if query_gpu:
        result = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                                capture_output=True,text=True,timeout=10,check=True)
        if result.stdout.strip():
            raise RuntimeError('GPU already has compute processes; do not start competing work')


@contextmanager
def execution_locks():
    """Hold original queue and follow-up scheduler locks for the entire guard."""
    FOLLOW.mkdir(exist_ok=True)
    paths = [BASE/'cloud/bounded_run.lock',FOLLOW/'scheduler.lock']
    locks = []
    try:
        for path in paths:
            stream = path.open('a');locks.append(stream)
            try:
                fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('An original or follow-up executor still holds its lock')
        yield
    finally:
        for stream in reversed(locks):
            fcntl.flock(stream,fcntl.LOCK_UN);stream.close()


def spent_seconds(request,stage):
    """Known exact exit receipts account actual time; unknown exits cost the cap."""
    state_path = request.parent/'guard_state.json'
    dispatch_path = request.parent/'guard_dispatch.json'
    row = identified(request)
    state = read_json(state_path) if state_path.exists() else None
    if state and state['request_identity'] != row['identity']:
        raise ValueError('Previous guard state belongs to another request')
    if dispatch_path.exists() and guard.exact_alive(read_json(dispatch_path)['guard']):
        raise RuntimeError('A previous follow-up guard is still alive')
    if state and state.get('worker') and guard.group_members(state['worker']):
        raise RuntimeError('A previous follow-up worker group is still alive')
    if state and state['stage'] in ('completed','stopped','failed','refused'):
        end = datetime.fromisoformat(state['updated_at_utc']).timestamp()
        return max(0,end-row['requested_at_unix'])
    return stage  # Do not erase unobserved work or restart on an observation timeout.


def attempts(run,mode):
    prefix = {'probe':'probe_','pilot':'pilot_attempt_','extend':'extension_attempt_','fit':'fit_attempt_'}[mode]
    return [p/'guard_request.json' for p in sorted((run/'attempts').glob(prefix+'*')) if p.is_dir()]


def stage_remaining(run,mode='pilot'):
    ceiling = {'pilot':PILOT_SECONDS,'extend':EXTENSION_SECONDS,'fit':guard.MAX_STAGE_SECONDS}[mode]
    return max(0,ceiling-sum(spent_seconds(p,ceiling) for p in attempts(run,mode) if p.exists()))


def probe_table(run,manifest,signature):
    """Only real guarded probes of each exact recipe can project the pilot."""
    found = {}
    for request_path in attempts(run,'probe'):
        if not request_path.exists():
            continue
        directory = request_path.parent
        state_path,report_path = directory/'guard_state.json',directory/'probe_result.json'
        if not state_path.exists() or not report_path.exists():
            continue
        state,report = read_json(state_path),identified(report_path)
        request = identified(request_path)
        name = report['candidate_name']
        expected = next((r for r in manifest['candidates'] if r['name'] == name),None)
        if (state['stage'] != 'completed' or state['worker_returncode'] != 0
                or state['request_identity'] != request['identity'] or report['guard_request_identity'] != request['identity']
                or report['worker'] != state['worker'] or report['manifest_identity'] != manifest['identity']
                or report['runtime_signature'] != signature or report['entry_sha256'] != digest(ENTRY)
                or report['actual_CUDA_probe'] is not True or report['training_data_only'] is not True
                or expected is None or report['candidate_config_identity'] != identity(expected['config'])
                or report['fit_end'] != manifest['folds']['2026Q1']['fit_end']
                or report['lookback'] != expected['config']['lookback']
                or report['batch_size'] != expected['config']['batch_size']
                or report['exact_checkpoint_resume_passed'] is not True or report['prediction_parity_max_difference'] != 0
                or report['updates_measured'] != PROBE_UPDATES):
            raise ValueError('GPU probe identity, runtime, checkpoint or precision differs')
        for field in ('seconds_per_update_upper','seconds_per_prediction_origin_upper','initialization_seconds_upper'):
            if not np.isfinite(report[field]) or report[field] <= 0:
                raise ValueError('GPU timing must be measured, finite and positive')
        # A repeated probe cannot replace a slower measurement with a favorable
        # speed; keep the conservative maximum across successful attempts.
        old = found.get(name)
        if old:
            report = {**report,**{field:max(old[field],report[field]) for field in
                ('seconds_per_update_upper','seconds_per_prediction_origin_upper','initialization_seconds_upper')}}
            fields = ('checkpoint_copy_bytes','checkpoint_copy_seconds_upper')
            if all(field in old and field in report for field in fields):
                if any(not np.isfinite(r[field]) or r[field]<=0 for r in (old,report) for field in fields):
                    raise ValueError('Checkpoint copy measurements must be finite and positive')
                if old[fields[1]]/old[fields[0]]>report[fields[1]]/report[fields[0]]:
                    report.update({field:old[field] for field in fields})
        found[name] = report
    if set(found) != {r['name'] for r in manifest['candidates']}:
        raise RuntimeError('Need completed numerical/recovery/timing probes for all three fixed recipes')
    return found


def projection(manifest,table,names=None,seeds=(2026,)):
    rows = []
    for candidate in manifest['candidates']:
        if names is not None and candidate['name'] not in names:continue
        measured = table[candidate['name']]
        for seed in seeds:
            for quarter in QUARTERS:
                f = manifest['folds'][quarter]
                seconds = (measured['initialization_seconds_upper']
                           + f['total_updates']*measured['seconds_per_update_upper']
                           + f['score_origins']*measured['seconds_per_prediction_origin_upper'])
                rows.append({'candidate':candidate['name'],'quarter':quarter,'seed':seed,
                             'updates':f['total_updates'],'score_origins':f['score_origins'],'seconds':seconds})
    return {'fits':rows,'seconds':sum(row['seconds'] for row in rows),
            'margin_multiplier':1.5,'cold_first_prefix_and_checkpoint_costs_included':True,
            'whole_training_speedup_claimed':False}


def check_budget(mode,manifest,run,signature,now=None,manifest_path=None):
    """Date and budget gates are pure checks and never issue a training command."""
    now = now or datetime.now(timezone.utc)
    left = (datetime.fromisoformat(DEADLINE)-now).total_seconds()
    if mode == 'probe':
        if len(attempts(run,'probe')) >= MAX_PROBES:
            raise RuntimeError('All five preparation probes have been reserved/spent')
        for path in attempts(run,'probe'):
            if path.exists():spent_seconds(path,PROBE_SECONDS)
        if left < PROBE_SECONDS+guard.RESERVE_SECONDS:
            raise RuntimeError('No time for a bounded preparation probe and delivery reserve')
        return {'mode':mode,'stage_seconds':PROBE_SECONDS,'maximum_total_probes':MAX_PROBES,
                'remaining_seconds':left,'large_training_admitted':False}
    if mode not in ('pilot','extend'):
        raise ValueError('Only fixed pilot, conditional extension or preparation probe can launch')
    result_path = 'pilot/pilot_result.json' if mode == 'pilot' else 'extension/extension_result.json'
    if (run/result_path).exists():
        raise RuntimeError('The fixed '+mode+' already completed; review it instead of launching a duplicate')
    # Reject early dates before even accepting any purported benchmark.
    launch_window(manifest['limit'],1e-9,now)
    extra = {}
    if mode == 'extend':
        from scripts.review_fusion_pilot import checked_review
        if manifest_path is None:raise ValueError('Extension needs its exact frozen manifest path')
        saved,plan = checked_review(manifest_path,manifest)
        extra = {'pilot_review':saved,'extension_plan':plan}
    table = probe_table(run,manifest,signature)
    measured = (projection(manifest,table,plan['train_candidates'],plan['new_seeds'])
                if mode == 'extend' else projection(manifest,table))
    window = launch_window(manifest['limit'],measured['seconds']/3600,now)
    remaining = stage_remaining(run,mode)
    if 1.5*measured['seconds']+60 > remaining:
        raise RuntimeError('Measured '+mode+' plus margin and stop grace exceeds its remaining 6-hour cap')
    return {'mode':mode,'stage_seconds':remaining,'window':window,'projection':measured,
            'large_training_admitted':True,'resumed_attempts_receive_no_new_six_hour_budget':True,**extra}


def bind_guarded_worker(attempt,entry,purpose):
    """A worker must be this exact live guard's direct session leader."""
    attempt = guard.checked_directory(attempt)
    request = identified(attempt/'guard_request.json')
    state = read_json(attempt/'guard_state.json')
    current = guard.process_identity()
    if (request['purpose'] != purpose
            or request['delivery_reserve_seconds'] != guard.PURPOSE_RESERVES[purpose]):
        raise RuntimeError('Research worker cannot use the reserved delivery-fit time' if purpose=='exploration'
                           else 'Delivery worker requires its exact delivery-fit guard purpose')
    if (state.get('stage') != 'running' or state['request_identity'] != request['identity']
            or state['worker'] != current or state['guard']['pid'] != os.getppid()
            or not guard.exact_alive(state['guard']) or not guard.exact_alive(request['owner'])
            or os.getpgid(0) != os.getpid() or os.getsid(0) != os.getpid()
            or request['guard_sha256'] != digest(ROOT/'scripts/followup_guard.py')
            or time.time() >= request['stop_by_unix']):
        raise RuntimeError('Worker is not bound to a live independent guard and caller')
    if psutil.Process(os.getppid()).cmdline()[1:] != [str(ROOT/'scripts/followup_guard.py'),'--request',str(attempt/'guard_request.json')]:
        raise ValueError('Worker parent command is not the exact group guard')
    if request['command'] != [sys.executable,str(entry),*sys.argv[1:]]:
        raise ValueError('Worker arguments differ from the guarded command')
    return request


def bound_worker(attempt):
    return bind_guarded_worker(attempt,ENTRY,'exploration')


def state_hash(value):
    """Small changing model state, RNG and optimizer payload; frozen weights are pinned separately."""
    from fusion.next import checkpoint
    import hashlib
    result = hashlib.sha256()
    def add(item):
        if isinstance(item,torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            result.update(str((array.shape,array.dtype.str)).encode());result.update(array.tobytes())
        elif isinstance(item,np.ndarray):
            result.update(str((item.shape,item.dtype.str)).encode());result.update(item.tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=str):
                result.update(repr(key).encode());add(item[key])
        elif isinstance(item,(list,tuple)):
            result.update(type(item).__name__.encode())
            for part in item:add(part)
        else:result.update(repr(item).encode())
    add({'model':checkpoint.changing_state(value.model),'optimizer':value.optimizer.state_dict(),
         'samplers':{k:v.state_dict() for k,v in value.samplers.items()},
         'rng':checkpoint.capture_rng(),'step':value.step,'last_loss':value.last_loss})
    return result.hexdigest()


def clone_prefix(source,destination):
    """Copy a new run's committed probe prefix into its paired recovery trial."""
    from fusion.next import checkpoint
    from fusion.next.common import OUT
    source,destination = Path(source),Path(destination)
    if (not source.resolve().is_relative_to(OUT/'training')
            or not destination.resolve().is_relative_to(OUT/'training')):
        raise ValueError('Probe checkpoint copies must stay inside the new training namespace')
    if destination.exists() or destination.with_suffix('.pt.json').exists():
        raise ValueError('Preserve the existing paired recovery prefix')
    committed,receipt = checkpoint.resolve_checkpoint(source)
    generation = destination.parent/('.'+destination.stem+'.checkpoints')/'probe_prefix.pt'
    if generation.exists():raise ValueError('Paired checkpoint generation already exists')
    generation.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(committed,generation)
    if digest(generation) != receipt['sha256']:
        raise ValueError('Paired checkpoint copy differs')
    write_json(destination.with_suffix('.pt.json'),{**receipt,'generation':str(generation.relative_to(destination.parent))})
    destination.symlink_to(generation.relative_to(destination.parent))


def run_probe(panel,store,manifest,candidate,attempt,request,signature):
    from fusion.next.accounting import Usage
    from fusion.next.experiment import model_for
    from fusion.next.hardware import runtime,gpu_telemetry
    from fusion.next.recipe import for_fit
    from fusion.next.trainer import Trainer
    f = manifest['folds']['2026Q1']
    cfg = {**for_fit(candidate['config'],f['fit_end']),'seed':2026}
    def make(namespace):
        runtime(2026)
        return Trainer(panel,model_for(panel,cfg,'cuda'),cfg,f['fit_end'],namespace,2026,store,'cuda')
    with Usage('performance','followup probe '+candidate['name']):
        started = time.perf_counter();continuous = make('probe_continuous')
        torch.cuda.synchronize();build1 = time.perf_counter()-started
        started = time.perf_counter();continuous.run_until(PROBE_UPDATES//2)
        torch.cuda.synchronize();first = time.perf_counter()-started
        copy_path = continuous.path.parents[1]/'probe_resumed/seed_2026.pt'
        from fusion.next.checkpoint import resolve_checkpoint
        committed,_ = resolve_checkpoint(continuous.path)
        copy_bytes = committed.stat().st_size
        started = time.perf_counter();clone_prefix(continuous.path,copy_path)
        copy_seconds = time.perf_counter()-started
        started = time.perf_counter();continuous.run_until(PROBE_UPDATES)
        torch.cuda.synchronize();second = time.perf_counter()-started
        expected = state_hash(continuous);resources = continuous.guard.check()
        del continuous;gc.collect();torch.cuda.empty_cache()
        started = time.perf_counter();resumed = make('probe_resumed');resumed.resume()
        torch.cuda.synchronize();build2 = time.perf_counter()-started
        resumed.run_until(PROBE_UPDATES)
        if state_hash(resumed) != expected:
            raise ValueError('Actual CUDA checkpoint/resume differs from continuous training')
        sid = panel.stocks.index('AAPL');origins = panel.origins['AAPL']
        origins = origins[(panel.dates[origins]<=f['fit_end']) & store.available[sid][origins]]
        rows = origins[:2*cfg['batch_size']]
        if len(rows) != 2*cfg['batch_size']:
            raise ValueError('Not enough strictly earlier training origins for prediction timing')
        expert = store.predict(np.full(len(rows),sid),rows)
        timings,values = [],[]
        for _ in range(2):
            started = time.perf_counter()
            values.append(resumed.predict(np.full(len(rows),sid),rows,expert))
            torch.cuda.synchronize();timings.append(time.perf_counter()-started)
        difference = float(np.max(np.abs(values[0]-values[1])))
        if difference != 0 or not np.isfinite(values).all():
            raise ValueError('FP32 prediction repeats are not finite and exactly consistent')
        latest_resources = resumed.guard.check()
        resources = {name:(min(resources[name],value) if name == 'minimum_available_ram_bytes'
                           else max(resources[name],value)) for name,value in latest_resources.items()}
        value = {'schema':1,'manifest_identity':manifest['identity'],'candidate_name':candidate['name'],
                 'candidate_config_identity':identity(candidate['config']),'entry_sha256':digest(ENTRY),
                 'guard_request_identity':request['identity'],'worker':guard.process_identity(),
                 'runtime_signature':signature,'actual_CUDA_probe':True,'training_data_only':True,
                 'fit_end':f['fit_end'],'lookback':cfg['lookback'],'batch_size':cfg['batch_size'],
                 'updates_measured':PROBE_UPDATES,'exact_checkpoint_resume_passed':True,
                 'prediction_parity_max_difference':difference,
                 'initialization_seconds_upper':max(build1,build2),
                 'seconds_per_update_upper':max(first,second)/(PROBE_UPDATES//2),
                 'seconds_per_prediction_origin_upper':max(timings)/len(rows),
                 'checkpoint_copy_bytes':copy_bytes,'checkpoint_copy_seconds_upper':copy_seconds,
                 'first_prefix_seconds':first,'second_prefix_seconds':second,
                 'checkpoint_io_in_update_timings':True,'adaptation_phase_timing_separately_measured':False,
                 'resources':resources,'telemetry_sample':gpu_telemetry(),
                 'full_training_or_predictive_improvement_demonstrated':False,'completed_at_utc':utc_now()}
        value['identity'] = identity(value);write_json(attempt/'probe_result.json',value)
        del resumed;gc.collect();torch.cuda.empty_cache()
    return value


def run_pilot(panel,store,manifest,destination):
    from fusion.next.experiment import neural_trial,score_rows
    from fusion.next.evaluation import point_scores
    completed = []
    for candidate in manifest['candidates']:
        for quarter in QUARTERS:
            result,prediction = neural_trial(panel,candidate['config'],quarter,2026,1.,'pilot',store,'fusion_search')
            _,rows = score_rows(panel,quarter)
            validate_rows(rows,prediction,panel.dates,quarter)
            score = point_scores(panel.targets['AAPL'][rows],prediction)
            if score != result['scores']:
                raise ValueError('Pilot score differs from recomputed common-origin predictions')
            completed.append({'candidate':candidate['name'],'quarter':quarter,'seed':2026,
                              'scores':score,'prediction_sha256':result['prediction_sha256']})
            write_json(destination/'pilot_progress.json',{'manifest_identity':manifest['identity'],
                       'completed_fits':completed,'expected_fits':9,'updated_at_utc':utc_now()})
    report = {'manifest_identity':manifest['identity'],'stage':'pilot_complete_review_required',
              'completed_fits':completed,'expected_fits':9,'total_coverage_equivalent_epochs':2,
              'Q2_Q3_prediction_arrays_loaded':False,'independent_success_claimed':False,
              'extension_or_final_fit_started':False,'completed_at_utc':utc_now()}
    write_json(destination/'pilot_result.json',report)
    return report


def run_extension(panel,store,manifest,destination,plan):
    from fusion.next.experiment import neural_trial,score_rows
    from fusion.next.evaluation import point_scores
    completed = []
    recipes = {r['name']:r['config'] for r in manifest['candidates']}
    for name in plan['train_candidates']:
        for seed in plan['new_seeds']:
            for quarter in QUARTERS:
                result,prediction = neural_trial(panel,recipes[name],quarter,seed,1.,'extension',store,'fusion_search')
                _,rows = score_rows(panel,quarter)
                validate_rows(rows,prediction,panel.dates,quarter)
                score = point_scores(panel.targets['AAPL'][rows],prediction)
                if score != result['scores']:
                    raise ValueError('Extension score differs from recomputed common-origin predictions')
                completed.append({'candidate':name,'quarter':quarter,'seed':seed,
                                  'scores':score,'prediction_sha256':result['prediction_sha256']})
                write_json(destination/'extension_progress.json',{'manifest_identity':manifest['identity'],
                           'extension_plan':plan,'completed_fits':completed,'expected_fits':plan['new_full_fits'],
                           'updated_at_utc':utc_now()})
    report = {'manifest_identity':manifest['identity'],'stage':'extension_complete_review_required',
              'extension_plan':plan,'completed_fits':completed,'expected_fits':plan['new_full_fits'],
              'total_coverage_equivalent_epochs':2,'Q2_Q3_prediction_arrays_loaded':False,
              'independent_success_claimed':False,'final_fit_started':False,'completed_at_utc':utc_now()}
    write_json(destination/'extension_result.json',report)
    return report


def worker(manifest_path,attempt,destination,mode,candidate_name):
    request = bound_worker(attempt)
    manifest = check_frozen(manifest_path)
    if mode not in ('probe','pilot','extend') or request['limit'] != manifest['limit']:
        raise ValueError('Guarded worker mode or absolute limit differs')
    if os.environ.get('FUSION_EXECUTION_PROFILE') != 'autodl':
        raise RuntimeError('Formal/probe worker requires the admitted cloud CUDA runtime')
    # Parent holds original lock and must remain alive; do not take a competing
    # lock or rerun the idle-GPU query after this worker starts its own CUDA job.
    original_idle(manifest,query_gpu=False)
    if mode != 'probe':
        launch_window(manifest['limit'],1e-9)
    from scripts import portable_inference_runtime as numerical,future_speed_runtime as speed
    if numerical._installed or speed._installed:
        raise RuntimeError('Use a clean worker without a preinstalled original-output bootstrap')
    from fusion.next.panel import Panel
    from fusion.next.trainer import ExpertStore
    panel = Panel.load();store = ExpertStore(panel)
    if store.sources != manifest['expert_sources'] or store.key != manifest['expert_scope_key']:
        raise ValueError('Original OOF experts changed after matrix freeze')
    signature = runtime_signature()
    admission = identified(Path(attempt)/'launch_admission.json')
    if (admission['manifest_identity'] != manifest['identity'] or admission['runtime_signature'] != signature
            or admission['mode'] != mode or admission['entry_sha256'] != digest(ENTRY)
            or admission['destination'] != str(Path(destination).resolve())
            or admission['budget']['mode'] != mode
            or request['stage_seconds'] != admission['budget']['stage_seconds']):
        raise ValueError('Worker differs from the scheduler admission')
    plan = None
    if mode == 'extend':
        from scripts.review_fusion_pilot import checked_review
        saved,plan = checked_review(manifest_path,manifest)
        if admission['budget']['pilot_review'] != saved or admission['budget']['extension_plan'] != plan:
            raise ValueError('Extension decision changed after scheduling')
    with isolated_outputs(destination):
        numerical.install();speed.install()
        if mode == 'probe':
            candidate = next((r for r in manifest['candidates'] if r['name'] == candidate_name),None)
            if candidate is None or admission['candidate_name'] != candidate_name:
                raise ValueError('Probe candidate is outside the fixed matrix')
            return run_probe(panel,store,manifest,candidate,Path(attempt),request,signature)
        # Recheck the same benchmark and budget before any formal model loads.
        table = probe_table(FOLLOW/'runs'/manifest['identity'],manifest,signature)
        measured = (projection(manifest,table,plan['train_candidates'],plan['new_seeds'])
                    if mode == 'extend' else projection(manifest,table))
        if admission['budget']['projection'] != measured or admission['budget']['large_training_admitted'] is not True:
            raise ValueError('Formal benchmark/budget admission changed')
        launch_window(manifest['limit'],measured['seconds']/3600)
        return (run_extension(panel,store,manifest,Path(destination),plan) if mode == 'extend'
                else run_pilot(panel,store,manifest,Path(destination)))


def stop_guard(process,attempt):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=7)
        except subprocess.TimeoutExpired:
            state_path = attempt/'guard_state.json'
            if state_path.exists():
                state = read_json(state_path)
                if state.get('worker') and guard.group_members(state['worker']):
                    try:os.killpg(state['worker']['pid'],signal.SIGKILL)
                    except ProcessLookupError:pass
            process.kill();process.wait()
    # A killed guard cannot execute its own finally block. The scheduler still
    # owns the exact receipt and must clear that worker group before unlocking.
    state_path = attempt/'guard_state.json'
    if state_path.exists():
        state = read_json(state_path)
        if state.get('worker') and guard.group_members(state['worker']):
            try:os.killpg(state['worker']['pid'],signal.SIGKILL)
            except ProcessLookupError:pass


def launch(manifest_path,mode,candidate_name=None):
    manifest_path = Path(manifest_path).resolve()
    manifest = check_frozen(manifest_path)
    run = FOLLOW/'runs'/manifest['identity']
    with execution_locks():
        from scripts import portable_inference_runtime as numerical,future_speed_runtime as speed
        if numerical._installed or speed._installed:
            raise RuntimeError('Use a clean launcher without the old original-output bootstrap')
        original_idle(manifest)
        signature = runtime_signature()
        budget = check_budget(mode,manifest,run,signature,manifest_path=manifest_path)
        if mode == 'probe' and candidate_name not in {r['name'] for r in manifest['candidates']}:
            raise ValueError('Choose one of the three fixed probe recipes')
        if mode != 'probe' and candidate_name is not None:
            raise ValueError('Formal stages use their complete fixed recipe and seed list')
        previous = attempts(run,mode)
        tag = {'probe':'probe_','pilot':'pilot_attempt_','extend':'extension_attempt_'}[mode]+str(len(previous)+1).zfill(2)
        attempt = run/'attempts'/tag
        if attempt.exists():
            raise RuntimeError('Preserve the existing attempt; do not overwrite missing/partial startup evidence')
        destination = attempt if mode == 'probe' else run/('pilot' if mode == 'pilot' else 'extension')
        admission = {'manifest_identity':manifest['identity'],'mode':mode,'candidate_name':candidate_name,
                     'destination':str(destination),'runtime_signature':signature,'budget':budget,
                     'entry_sha256':digest(ENTRY),'issued_at_utc':utc_now()}
        admission['identity'] = identity(admission);write_json(attempt/'launch_admission.json',admission)
        command = [sys.executable,str(ENTRY),'--worker','--manifest',str(manifest_path),
                   '--attempt',str(attempt),'--destination',str(destination),'--mode',mode]
        if candidate_name is not None:command += ['--candidate',candidate_name]
        os.environ['PYTHONPATH'] = str(ROOT)
        process,dispatch = guard.start(attempt,command,manifest['limit'],budget['stage_seconds'],purpose='exploration')
        try:
            timeout = max(1,dispatch['stop_by_unix']-time.time()+10)
            code = process.wait(timeout=timeout)
        except BaseException:
            stop_guard(process,attempt);raise
        finally:
            stop_guard(process,attempt)
        state = read_json(attempt/'guard_state.json')
        if code != 0 or state['stage'] != 'completed':
            raise RuntimeError('Follow-up guard stopped or failed; inspect preserved attempt logs')
        expected = (attempt/'probe_result.json' if mode == 'probe' else
                    destination/('pilot_result.json' if mode == 'pilot' else 'extension_result.json'))
        result = read_json(expected)
        if result['manifest_identity'] != manifest['identity']:
            raise ValueError('Completed worker result belongs to another matrix')
        print({'mode':mode,'attempt':str(attempt),'completed':True,
               'large_training':mode != 'probe','extension_started':mode == 'extend'})
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--mode',choices=['probe','pilot','extend'],required=True)
    parser.add_argument('--candidate')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--destination',type=Path,help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.attempt or not args.destination:
            parser.error('Internal workers need exact guard attempt and output paths')
        worker(args.manifest,args.attempt,args.destination,args.mode,args.candidate)
    else:
        if args.attempt or args.destination:
            parser.error('Output/attempt paths are determined by the frozen matrix')
        launch(args.manifest,args.mode,args.candidate)
