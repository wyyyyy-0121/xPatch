"""Enforce the user's total cloud wall-time cap without changing frozen model code."""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import os
from pathlib import Path
import signal
import time

import psutil

from fusion.next.common import ROOT,OUT,digest,read_json,utc_now,write_json


def project_process(process):
    try:
        args=process.cmdline()
        module=args[args.index('-m')+1] if '-m' in args else ''
        owned=(module.startswith('fusion.next.') or
               any(Path(arg).name in ('continue_fusion_cloud.py','run_fusion_100h.py') for arg in args[1:]))
        return owned and Path(process.cwd()).resolve()==ROOT.resolve()
    except (psutil.Error,OSError,ValueError,IndexError):return False


def targets():
    rows=[]
    for process in psutil.process_iter():
        try:
            if process.pid!=os.getpid() and project_process(process):
                rows.append({'pid':process.pid,'created_unix':process.create_time(),'command':process.cmdline()})
        except psutil.NoSuchProcess:continue
    return rows


def send_exact(row,number):
    try:
        process=psutil.Process(row['pid'])
        if abs(process.create_time()-row['created_unix'])>=.01 or not project_process(process):return False
        process.send_signal(number);return True
    except psutil.NoSuchProcess:return False


def setup():
    path=OUT/'cloud/hard_time_limit.json'
    if path.exists():
        receipt=read_json(path)
        if receipt['total_wall_hours']!=100:raise ValueError('Existing limit differs from the user-approved 100 hours')
        return receipt
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    jobs=[read_json(p) for p in (OUT/'usage_events').glob('*.json')]
    jobs=[row for row in jobs if row.get('boot_id')==boot]
    if not jobs:raise ValueError('Need an actual first cloud job; do not invent a rental start time')
    anchor=min(datetime.fromisoformat(row['started_at_utc']) for row in jobs)
    deadline=anchor+timedelta(hours=100)
    receipt={'authorization':'User explicitly chose a 100-hour hard limit',
        'total_wall_hours':100,'anchor_utc':anchor.isoformat(),'deadline_utc':deadline.isoformat(),
        'anchor_policy':'first recorded research job on this cloud boot; all later idle/CPU/GPU time counts',
        'provider_rental_start_not_inferred':True,'boot_id':boot,'created_at_utc':utc_now(),
        'checkpoint_grace_seconds':600,'termination_grace_seconds':60,
        'script_sha256':digest(Path(__file__)),'Q3_accessed':False}
    write_json(path,receipt);return receipt


def run(inspect_only=False):
    if os.environ.get('FUSION_EXECUTION_PROFILE')!='autodl':raise RuntimeError('Cloud-only time limiter')
    receipt=setup()
    if inspect_only:print({'limit':receipt,'current_targets':targets()});return
    lock=(OUT/'cloud/hard_time_limit.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A cloud time limiter is already running')
    deadline=datetime.fromisoformat(receipt['deadline_utc']).timestamp();sent={}
    while True:
        remaining=deadline-time.time();rows=targets()
        number=(signal.SIGKILL if remaining<=0 else
                signal.SIGTERM if remaining<=receipt['termination_grace_seconds'] else
                signal.SIGINT if remaining<=receipt['checkpoint_grace_seconds'] else None)
        events=[]
        for row in rows:
            key=(row['pid'],row['created_unix'],number)
            if number is not None and key not in sent:
                sent[key]=send_exact(row,number);events.append({**row,'signal':int(number),'sent':sent[key]})
        state={'limit':receipt,'updated_at_utc':utc_now(),'remaining_seconds':max(0,remaining),
            'pid':os.getpid(),'created_unix':psutil.Process().create_time(),
            'stage':'enforcing_deadline' if number is not None else 'armed',
            'current_targets':rows,'signal_events':events,'Q3_accessed':False}
        write_json(OUT/'cloud/hard_time_limit_state.json',state)
        if events:
            stamp=utc_now().replace(':','').replace('.','_')
            write_json(OUT/'cloud'/('time_limit_signals_'+stamp+'.json'),state)
        if remaining<=0 and not rows:return
        time.sleep(2 if number is not None else min(30,max(1,remaining-600)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inspect-only',action='store_true')
    run(parser.parse_args().inspect_only)
