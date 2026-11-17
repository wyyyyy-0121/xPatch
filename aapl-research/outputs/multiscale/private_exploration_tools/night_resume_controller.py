"""Resume tonight's frozen AAPL queues without resetting their budgets.

At the authorized time, live frozen workers retain their exact in-memory state;
if the WSL session disappeared, verified periodic checkpoints are used instead.
Independent systemd timers freeze each queue when its remaining budget expires.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


PROJECT=Path('/home/ayaka/wslcode/aapl-research')
PYTHON=PROJECT/'.venv/bin/python'
TOOLS=PROJECT/'outputs/multiscale/private_exploration_tools'
WORKER=TOOLS/'night_resume_worker.py'
BUDGET_STOP=TOOLS/'night_budget_stop.py'
MATRIX_WORKER=TOOLS/'minute_matrix_queue.py'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno());tmp=f.name
    os.replace(tmp,path)


def proc(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();rest=raw[raw.rfind(')')+2:].split()
        argv=[x.decode('utf8','replace') for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
        return {'pid':pid,'state':rest[0],'start_ticks':int(rest[19]),'argv':argv}
    except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):return None


def match(pid,start_ticks):
    item=proc(pid)
    return item if item and item['start_ticks']==start_ticks else None


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise ValueError('Cannot read available memory')


def user_bus():
    path=f'/run/user/{os.getuid()}'
    os.environ.setdefault('XDG_RUNTIME_DIR',path)
    os.environ.setdefault('DBUS_SESSION_BUS_ADDRESS','unix:path='+path+'/bus')
    if not Path(path+'/bus').exists():raise ValueError('The systemd user bus is unavailable')


def load_context(snapshot_path,plan_path):
    snapshot_path=Path(snapshot_path).resolve();plan_path=Path(plan_path).resolve()
    snapshot=json.loads(snapshot_path.read_text());plan=json.loads(plan_path.read_text())
    validation_path=snapshot_path.parent/'CHECKPOINT_RESUME_VALIDATION.json'
    validation=json.loads(validation_path.read_text());manifest=json.loads(Path(snapshot['pause_manifest']).read_text())
    affinity_path=snapshot_path.parent/'AFFINITY_BEFORE.json'
    affinity_validation_path=snapshot_path.parent/'AFFINITY_LIMIT_VALIDATION.json'
    affinity=json.loads(affinity_path.read_text());affinity_validation=json.loads(affinity_validation_path.read_text())
    if (snapshot['status']!='snapshot_complete' or validation['status']!='all_checkpoint_resume_identities_verified' or
            validation['training_started'] is not False or digest(snapshot_path)!=plan['snapshot_sha256'] or
            digest(validation_path)!=plan['validation_sha256'] or digest(WORKER)!=plan['resume_worker_sha256'] or
            validation['snapshot_sha256']!=digest(snapshot_path) or validation['resume_worker_sha256']!=digest(WORKER) or
            digest(affinity_path)!=plan['affinity_snapshot_sha256'] or
            digest(affinity_validation_path)!=plan['affinity_validation_sha256'] or
            affinity_validation.get('status')!='all_threads_limited' or affinity['temporary_affinity']!=[24,26] or
            digest(BUDGET_STOP)!=plan['budget_stop_sha256'] or digest(Path(__file__))!=plan['controller_sha256'] or
            digest(Path(snapshot['pause_manifest']))!=snapshot['pause_manifest_sha256'] or
            plan['resume_at_local']!=snapshot['resume_at_local']):
        raise ValueError('Recovery plan, validation or source identity changed')
    if len(snapshot['workers'])!=6 or len(snapshot['queues'])!=4:
        raise ValueError('Unexpected paused queue inventory')
    for group in snapshot['queues'].values():
        if digest(group['plan'])!=group['plan_sha256'] or group['remaining_active_seconds']<0:
            raise ValueError('A queue plan or remaining budget changed')
        if group['script']=='minute_matrix_queue.py' and json.loads(Path(group['plan']).read_text())['script_sha256']!=digest(MATRIX_WORKER):
            raise ValueError('The original minute matrix worker source changed')
    original_boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()==snapshot['boot_id']
    saved_processes={x['pid']:x for x in manifest['processes']}
    parents={};live=[];fallback=[]
    for parent in manifest['parents']:
        record=saved_processes[parent]
        current=match(parent,record['start_ticks']) if original_boot else None
        if current and current['state'] not in ('T','t'):
            raise ValueError('An original queue parent is no longer frozen')
        parents[str(parent)]={'record':record,'present':bool(current)}
    for item in snapshot['workers']:
        current=match(item['pid'],item['process_start_ticks']) if original_boot else None
        if current:
            if current['state'] not in ('T','t'):
                raise ValueError('An original training worker is no longer frozen')
            if not any(item['script'] in token for token in current['argv']):
                raise ValueError('A paused PID no longer belongs to the expected worker')
            record=next(x for x in affinity['processes'] if x['pid']==item['pid'])
            for thread in record['threads']:
                if not os.sched_getaffinity(thread['tid']).issubset({24,26}):
                    raise ValueError('A paused worker thread exceeds the temporary CPU limit')
            live.append(item['pid'])
        else:
            if not (Path(item['fold_dir'])/'fold_state.json').is_file():
                raise ValueError('A missing process has no resumable fold state')
            fallback.append(item['pid'])
    return snapshot,plan,manifest,parents,live,fallback,affinity


def restore_thread_affinities(item,affinity):
    record=next(x for x in affinity['processes'] if x['pid']==item['pid'])
    if record['start_ticks']!=item['process_start_ticks']:raise ValueError('Affinity snapshot process identity differs')
    for thread in record['threads']:
        if not Path(f"/proc/{item['pid']}/task/{thread['tid']}").exists():
            raise ValueError('A frozen worker thread disappeared')
        os.sched_setaffinity(thread['tid'],set(thread['affinity_before']))


def verification_for(item):
    plan=Path(item['plan']);key=item['job'];script=item['script']
    if script=='minute_parallel_folds_45.py':return plan.parent/f'fold_{int(key):02d}_verification.json'
    if script=='minute_matrix_queue.py':return plan.parent/'verifications'/(key+'.json')
    return plan.parent/(key+'_verification.json')


def verified_completion(item):
    folder=Path(item['fold_dir']);receipt=folder/'fold_receipt.json'
    if not receipt.is_file():return None
    record=json.loads(receipt.read_text())
    if record.get('status')!='completed' or record.get('Q3_values_read') is not False:return None
    standard=verification_for(item)
    alternatives=[standard,standard.with_name(standard.stem+'_night_resumed.json')]
    for path in alternatives:
        if not path.is_file():continue
        verification=json.loads(path.read_text())
        matches=verification.get('current_source_matches_recorded',{})
        if (verification.get('fold_receipt_sha256')==digest(receipt) and
                matches and all(matches.values())):
            return {'receipt':str(receipt),'receipt_sha256':digest(receipt),
                    'verification':str(path),'verification_sha256':digest(path)}
    return None


def schedule_watchdogs(snapshot_path,snapshot):
    user_bus();units={}
    try:
        for parent,group in snapshot['queues'].items():
            remaining=int(group['remaining_active_seconds'])
            if remaining<=0:continue
            unit=f'aapl-night-budget-153438-{parent}';seconds=max(1,remaining-3)
            args=['systemd-run','--user','--unit='+unit,'--on-active='+str(seconds)+'s',
                  '--timer-property=AccuracySec=1s','/usr/bin/python3',str(BUDGET_STOP),str(snapshot_path),parent]
            start=time.monotonic();p=subprocess.run(args,capture_output=True,text=True,timeout=20)
            if p.returncode!=0:raise RuntimeError('Cannot register independent budget timer: '+p.stderr[-300:])
            units[parent]={'unit':unit,'started_monotonic':start,'fire_after_seconds':seconds,
                           'deadline_monotonic':start+seconds}
            status=subprocess.run(['systemctl','--user','is-active',unit+'.timer'],capture_output=True,text=True,timeout=10)
            if status.returncode!=0 or status.stdout.strip()!='active':raise RuntimeError('Budget timer is not active: '+unit)
        return units
    except Exception:
        for value in units.values():
            subprocess.run(['systemctl','--user','stop',value['unit']+'.timer'],capture_output=True,text=True,timeout=10)
        raise


def stop_watchdog(value):
    subprocess.run(['systemctl','--user','stop',value['unit']+'.timer'],capture_output=True,text=True,timeout=10)


def freeze_group(snapshot_path,parent,reason):
    proc_=subprocess.run(['/usr/bin/python3',str(BUDGET_STOP),str(snapshot_path),str(parent),reason],
                         capture_output=True,text=True,timeout=20)
    if proc_.returncode!=0:raise RuntimeError('Could not freeze a queue: '+proc_.stderr[-300:])


def mark_queue(snapshot,parent,changes):
    group=snapshot['queues'][str(parent)];path=Path(group['state']);state=json.loads(path.read_text())
    state.update(changes);atomic_json(path,state)
    return state


def update_entry(snapshot,item,status,completed=None,pid=None):
    parent=str(item['parent_pid']);group=snapshot['queues'][parent];path=Path(group['state']);state=json.loads(path.read_text())
    entry=(state['jobs'][item['job']] if item['script']=='minute_matrix_queue.py' else state['workers'][item['job']])
    entry['status']=status
    if pid is not None:entry['pid']=pid
    if status=='running' and 'slot' in item:entry['cores']=item['slot']
    if completed:
        entry.update(exit_code=0,receipt_sha256=completed['receipt_sha256'],
                     verification_sha256=completed['verification_sha256'],verification=completed['verification'])
    if item['script']=='minute_matrix_queue.py':
        state['completed']=sum(v['status']=='completed' for v in state['jobs'].values())
    atomic_json(path,state)


def kill_original_parent(parent,parents,snapshot):
    if not parents[parent]['present']:return
    rec=parents[parent]['record'];current=match(rec['pid'],rec['start_ticks'])
    if not current:return
    if current['state'] not in ('T','t'):raise ValueError('Original queue parent was unexpectedly resumed')
    # Only after all of its original workers have exited; this releases its stale clock and lock.
    for w in snapshot['workers']:
        if str(w['parent_pid'])==parent:
            child=match(w['pid'],w['process_start_ticks'])
            if child and child['state'] not in ('Z','X','x'):return
    os.kill(rec['pid'],signal.SIGKILL);parents[parent]['present']=False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',required=True);parser.add_argument('--plan',required=True)
    parser.add_argument('--dry-run',action='store_true');args=parser.parse_args()
    snapshot_path=Path(args.snapshot).resolve();plan_path=Path(args.plan).resolve()
    initial_snapshot=json.loads(snapshot_path.read_text());initial_plan=json.loads(plan_path.read_text())
    if digest(snapshot_path)!=initial_plan['snapshot_sha256']:
        raise ValueError('Recovery snapshot identity changed')
    target=datetime.fromisoformat(initial_snapshot['resume_at_local']).astimezone(timezone.utc)
    if args.dry_run:
        snapshot,plan,manifest,parents,live,fallback,affinity=load_context(snapshot_path,plan_path)
        print(json.dumps({'status':'recovery_preflight_only','resume_at_local':snapshot['resume_at_local'],
                          'live_frozen_workers':len(live),'checkpoint_fallback_workers':len(fallback),
                          'queues':len(snapshot['queues']),'remaining_seconds':{k:v['remaining_active_seconds'] for k,v in snapshot['queues'].items()},
                          'training_started':False},ensure_ascii=False),flush=True)
        return
    if datetime.now(timezone.utc)<target:raise ValueError('The authorized resume time has not arrived')
    lock_path=snapshot_path.parent/'NIGHT_RESUME.lock';state_path=snapshot_path.parent/'NIGHT_RESUME_STATE.json'
    with lock_path.open('a+') as lock:
        try:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'resume_controller_already_running'}),flush=True);return
        if state_path.exists():
            old=json.loads(state_path.read_text())
            if old.get('status') in ('completed','finished_with_paused_jobs'):
                print(json.dumps({'status':'resume_already_handled','previous':old['status']}),flush=True);return
            # A crashed controller cannot safely restart clocks. Freeze everything for review.
            for parent in initial_snapshot['queues']:freeze_group(snapshot_path,parent,'controller_recovery_required')
            old['status']='failed_paused';old['reason']='previous_controller_did_not_finish';atomic_json(state_path,old)
            return
        snapshot,plan,manifest,parents,live,fallback,affinity=load_context(snapshot_path,plan_path)
        watchdogs=schedule_watchdogs(snapshot_path,snapshot)
        resume_started=datetime.now(timezone.utc).isoformat()
        state={'status':'starting','resume_started_at_utc':resume_started,'snapshot_sha256':digest(snapshot_path),
               'recovery_plan_sha256':digest(plan_path),'watchdogs':watchdogs,'workers':{},'groups':{},
               'note':'Original budgets exclude the requested paused interval; budget timers are independent.'}
        for parent,group in snapshot['queues'].items():
            state['groups'][parent]={'status':'running' if group['remaining_active_seconds']>0 else 'paused_budget_exhausted',
                                     'remaining_active_seconds_at_resume':group['remaining_active_seconds']}
        atomic_json(state_path,state)
        active={};streams={};cancelled=False
        matrix_parent=next(p for p,g in snapshot['queues'].items() if g['script']=='minute_matrix_queue.py')
        matrix_plan_path=Path(snapshot['queues'][matrix_parent]['plan']);matrix_plan=json.loads(matrix_plan_path.read_text())
        slots=[tuple(x) for x in matrix_plan['allowed_cpu_slots']]
        def stop(signum,frame):
            nonlocal cancelled
            cancelled=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            for parent,group in snapshot['queues'].items():
                mark_queue(snapshot,parent,{'status':'running_night_resume',
                    'night_pause':{'paused_at_utc':manifest['paused_at_utc'],'resumed_at_utc':resume_started,
                                   'active_seconds_before_pause':group['active_seconds_before_pause'],
                                   'remaining_active_seconds_at_resume':group['remaining_active_seconds']}})
            logdir=snapshot_path.parent/'night_resume_logs';logdir.mkdir(exist_ok=True)
            for item in snapshot['workers']:
                parent=str(item['parent_pid']);slot=tuple(item['original_affinity'])
                if slot not in slots:raise ValueError('A paused worker has an unexpected CPU slot')
                if snapshot['queues'][parent]['remaining_active_seconds']<=0:
                    update_entry(snapshot,item,'paused_budget_exhausted');continue
                old=match(item['pid'],item['process_start_ticks']) if item['pid'] in live else None
                if old:
                    restore_thread_affinities(item,affinity);os.kill(item['pid'],signal.SIGCONT)
                    actual_pid=item['pid'];start_ticks=item['process_start_ticks'];mode='live_exact'
                else:
                    log=(logdir/f"restore_{item['pid']}.log").open('a',encoding='utf8');streams[f"restore_{item['pid']}"]=log
                    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(PROJECT))
                    child=subprocess.Popen([str(PYTHON),str(WORKER),'--snapshot',str(snapshot_path),'--pid',str(item['pid']),
                                            '--cores',','.join(map(str,slot))],stdout=log,stderr=subprocess.STDOUT,
                                           start_new_session=True,env=env,cwd=str(PROJECT))
                    actual_pid=child.pid;current=proc(actual_pid)
                    if not current:raise ValueError('Checkpoint worker did not start')
                    start_ticks=current['start_ticks'];mode='periodic_checkpoint_fallback'
                running={**item,'actual_pid':actual_pid,'actual_start_ticks':start_ticks,'slot':list(slot),'mode':mode}
                active[(parent,item['job'])]=running
                state['workers'][f"{parent}:{item['job']}"]={'status':'running','pid':actual_pid,'mode':mode,'slot':list(slot)}
                update_entry(snapshot,item,'running',pid=actual_pid)
                atomic_json(state_path,state)
            state['status']='running';atomic_json(state_path,state)
            while True:
                now=time.monotonic()
                for parent,group in snapshot['queues'].items():
                    if state['groups'][parent]['status']!='running':continue
                    watchdog=watchdogs.get(parent);marker=snapshot_path.parent/f'BUDGET_STOP_{parent}.json'
                    if marker.exists() or (watchdog and now>=watchdog['deadline_monotonic']):
                        freeze_group(snapshot_path,parent,'original_compute_budget_exhausted')
                        state['groups'][parent]['status']='paused_budget_exhausted'
                        mark_queue(snapshot,parent,{'status':'paused_budget_exhausted','reason':'original_compute_budget_exhausted',
                                                    'elapsed_seconds':group['maximum_wall_seconds']})
                        for identity,item in list(active.items()):
                            if identity[0]==parent:
                                update_entry(snapshot,item,'paused_budget_exhausted');state['workers'][f'{parent}:{item["job"]}']['status']='paused_budget_exhausted'
                                del active[identity]
                for identity,item in list(active.items()):
                    current=match(item['actual_pid'],item['actual_start_ticks'])
                    if current and current['state'] not in ('Z','X','x'):continue
                    parent,key=identity;completion=verified_completion(item)
                    if completion:
                        update_entry(snapshot,item,'completed',completed=completion)
                        state['workers'][f'{parent}:{key}']['status']='completed';del active[identity]
                    else:
                        update_entry(snapshot,item,'failed');state['workers'][f'{parent}:{key}']['status']='failed';del active[identity]
                        freeze_group(snapshot_path,parent,'worker_failed')
                        state['groups'][parent]['status']='failed_paused'
                        mark_queue(snapshot,parent,{'status':'failed_paused','reason':'worker_failed'})
                        for other,other_item in list(active.items()):
                            if other[0]==parent:
                                update_entry(snapshot,other_item,'paused_after_peer_failure')
                                state['workers'][f'{parent}:{other_item["job"]}']['status']='paused_after_peer_failure';del active[other]
                if cancelled or available_memory()<2*1024**3:
                    reason='controller_cancelled' if cancelled else 'low_available_memory'
                    for parent in snapshot['queues']:
                        if state['groups'][parent]['status']=='running':
                            freeze_group(snapshot_path,parent,reason);state['groups'][parent]['status']='paused_'+reason
                            mark_queue(snapshot,parent,{'status':'paused_'+reason,'reason':reason})
                    active.clear();break
                for parent,group in snapshot['queues'].items():
                    if parent==matrix_parent or state['groups'][parent]['status']!='running':continue
                    if not any(identity[0]==parent for identity in active):
                        qstate=json.loads(Path(group['state']).read_text())
                        if all(v['status']=='completed' for v in qstate['workers'].values()):
                            state['groups'][parent]['status']='completed'
                            elapsed=min(group['maximum_wall_seconds'],group['active_seconds_before_pause']+
                                        max(0,time.monotonic()-watchdogs[parent]['started_monotonic']))
                            mark_queue(snapshot,parent,{'status':'completed_private_development_only','elapsed_seconds':elapsed})
                            stop_watchdog(watchdogs[parent]);kill_original_parent(parent,parents,snapshot)
                if state['groups'][matrix_parent]['status']=='running':
                    matrix_state=json.loads(Path(snapshot['queues'][matrix_parent]['state']).read_text())
                    busy={tuple(item['slot']) for item in active.values()}
                    free=[slot for slot in slots if slot not in busy]
                    pending=[j for j in matrix_plan['jobs'] if matrix_state['jobs'][j['key']]['status']=='pending']
                    while free and pending and available_memory()>=4*1024**3 and time.monotonic()+30<watchdogs[matrix_parent]['deadline_monotonic']:
                        slot=free.pop(0);job=pending.pop(0);key=job['key']
                        log=(logdir/f'matrix_{key}.log').open('a',encoding='utf8');streams['matrix_'+key]=log
                        env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=str(PROJECT))
                        child=subprocess.Popen([str(PYTHON),str(MATRIX_WORKER),'--worker','--plan',str(matrix_plan_path),
                                                '--job',key,'--cores',','.join(map(str,slot))],stdout=log,stderr=subprocess.STDOUT,
                                               start_new_session=True,env=env,cwd=str(PROJECT))
                        current=proc(child.pid)
                        if not current:raise ValueError('A pending matrix worker did not start')
                        item={'parent_pid':int(matrix_parent),'script':'minute_matrix_queue.py','job':key,
                              'plan':str(matrix_plan_path),'fold_dir':str(matrix_plan_path.parent/'jobs'/key),
                              'actual_pid':child.pid,'actual_start_ticks':current['start_ticks'],'slot':list(slot),'mode':'scheduled'}
                        active[(matrix_parent,key)]=item
                        state['workers'][f'{matrix_parent}:{key}']={'status':'running','pid':child.pid,'mode':'scheduled','slot':list(slot)}
                        update_entry(snapshot,item,'running',pid=child.pid);atomic_json(state_path,state)
                    matrix_state=json.loads(Path(snapshot['queues'][matrix_parent]['state']).read_text())
                    if (all(v['status']=='completed' for v in matrix_state['jobs'].values()) and
                            not any(i[0]==matrix_parent for i in active)):
                        state['groups'][matrix_parent]['status']='completed'
                        group=snapshot['queues'][matrix_parent]
                        elapsed=min(group['maximum_wall_seconds'],group['active_seconds_before_pause']+
                                    max(0,time.monotonic()-watchdogs[matrix_parent]['started_monotonic']))
                        mark_queue(snapshot,matrix_parent,{'status':'completed_scheduled_jobs_only','elapsed_seconds':elapsed})
                        stop_watchdog(watchdogs[matrix_parent]);kill_original_parent(matrix_parent,parents,snapshot)
                for parent in snapshot['queues']:
                    if state['groups'][parent]['status']=='running':kill_original_parent(parent,parents,snapshot)
                atomic_json(state_path,state)
                if all(v['status']!='running' for v in state['groups'].values()):break
                time.sleep(3)
            state['status']='completed' if all(v['status']=='completed' for v in state['groups'].values()) else 'finished_with_paused_jobs'
            state['finished_at_utc']=datetime.now(timezone.utc).isoformat();atomic_json(state_path,state)
            print(json.dumps({'status':state['status'],'groups':state['groups'],'state':str(state_path)},ensure_ascii=False),flush=True)
        except Exception as exc:
            for parent in snapshot['queues']:
                try:freeze_group(snapshot_path,parent,'controller_error')
                except Exception:pass
            state['status']='failed_paused';state['reason']=f'{type(exc).__name__}: {exc}';atomic_json(state_path,state)
            raise
        finally:
            for stream in streams.values():stream.close()


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'night_resume_controller_error','error':f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False),file=sys.stderr,flush=True)
        raise
