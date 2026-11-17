"""Keep cloud F2 -> admission -> F3 running, preserving prior evidence first."""
import io
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

from .accounting import recover_interrupted,totals
from .common import ROOT,OUT,digest,read_json,source_files,source_identity,utc_now,write_json


STATE=OUT/'cloud/driver_state.json'


def state(stage,**details):
    write_json(STATE,{'stage':stage,'source_identity':source_identity(),
        'updated_at_utc':utc_now(),'Q3_accessed':False,**details})


def archive_source():
    code=source_identity();files=source_files();directory=OUT/'performance';directory.mkdir(exist_ok=True)
    path=directory/('source_'+code+'.tar.gz')
    if not path.exists():
        with tarfile.open(path,'w:gz') as archive:
            for name,file in files.items():archive.add(file,arcname=name,recursive=False)
            data=json.dumps({'source_identity':code,'files':{name:digest(file) for name,file in files.items()}},sort_keys=True).encode()
            info=tarfile.TarInfo('source_manifest.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
    return {'path':str(path.relative_to(ROOT)),'sha256':digest(path)}


def launch(module,args,cpus,label,threads=2):
    path=OUT/'cloud/logs'/(label+'.log');path.parent.mkdir(parents=True,exist_ok=True)
    command=['taskset','-c',cpus,sys.executable,'-m',module,*args]
    env={**os.environ,'AAPL_CPU_THREADS':str(threads)}
    with path.open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(OUT/'cloud'/(label+'_process.json'),{'pid':process.pid,'command':command,
        'started_at_utc':utc_now(),'source_identity':source_identity(),'log':str(path.relative_to(ROOT))})
    return process


def run():
    if os.environ.get('FUSION_EXECUTION_PROFILE')!='autodl':raise RuntimeError('Cloud driver requires the cloud profile')
    if not (OUT/'search_manifest.json').exists():
        # A heartbeat timeout is not evidence of death. Accounting recovers only
        # missing exact PID/creation identities before a new GPU job can start.
        while True:
            recover_interrupted();active=totals()['active_jobs']
            if not active:break
            state('waiting_for_existing_measurements',active_jobs=active);time.sleep(5)
        old=read_json(OUT/'performance_admission.json')
        stamp=utc_now().replace(':','').replace('.','_')
        backup=OUT/'cloud'/('prior_F2_'+stamp);backup.mkdir()
        for name in ('performance_admission.json','performance_cpu.json','performance_stress_progress.json','usage_summary.json'):
            path=OUT/name
            if path.exists():shutil.copy2(path,backup/name)
        write_json(backup/'receipt.json',{'source_identity':old['source_identity'],
            'files':{p.name:digest(p) for p in backup.iterdir() if p.is_file()},'saved_at_utc':utc_now()})
        if not old.get('numerical_and_resource_checks_passed'):
            raise RuntimeError('Prior maximum-shape numerical checks failed; inspect before repeating the same policy')
        receipt=archive_source();state('F2_final_source_measurement',source_archive=receipt)
        choices=read_json(OUT/'native_scaling.json')['choices']
        native_args=['--xpatch-batch-size',str(choices['xpatch']['batch_size']),
                     '--patchtst-batch-size',str(choices['patchtst']['batch_size'])]
        gpu=launch('fusion.next.performance',['--batch-size','256','--precision','bf16','--stress-seconds','900',*native_args],
            '8-24','performance_gpu_final')
        cpu=launch('fusion.next.performance',['--cpu-only'],'0-7','performance_cpu_final',threads=8)
        while gpu.poll() is None or cpu.poll() is None:
            state('F2_final_source_measurement',gpu_pid=gpu.pid,cpu_pid=cpu.pid,
                gpu_returncode=gpu.poll(),cpu_returncode=cpu.poll(),source_archive=receipt)
            if (gpu.poll() is not None and gpu.returncode) or (cpu.poll() is not None and cpu.returncode):
                # The independent sibling measurement can finish. Do not kill
                # or discard valid work, and do not advance a failed stage.
                state('F2_measurement_failure',gpu_pid=gpu.pid,cpu_pid=cpu.pid,
                    gpu_returncode=gpu.poll(),cpu_returncode=cpu.poll())
                raise RuntimeError('F2 measurement subprocess failed; inspect retained logs')
            time.sleep(5)
        if not read_json(OUT/'performance_admission.json').get('numerical_and_resource_checks_passed'):
            raise RuntimeError('Selected batch/precision did not pass final-source F2 checks')
        state('F2_cost_projection_and_search_freeze')
        result=launch('fusion.next.budget',[],'0-24','budget_final')
        if result.wait():raise RuntimeError('Full-scope cost projection or immutable search freeze failed')
    from .experiment import admitted
    spec=admitted()
    if spec.get('training_budget_policy',{}).get('total_coverage_equivalent_epochs')!=2:
        raise RuntimeError('User replaced the old budget with exactly two total coverage-equivalent epochs')
    if not (OUT/'F3_complete.json').exists():
        process=launch('fusion.next.experiment',['F3'],'0-24','F3_full_coverage')
        while process.poll() is None:
            state('F3_running',pid=process.pid,search_identity=spec['identity'],
                projected_hours=spec['projection']['formal_wall_hours_projection_with_margin'])
            time.sleep(30)
        if process.returncode:raise RuntimeError('F3 stopped; inspect saved checkpoints and logs before recovery')
    state('F3_complete_F4_ready',search_identity=spec['identity'])


if __name__=='__main__':
    (OUT/'cloud').mkdir(parents=True,exist_ok=True)
    lock=(OUT/'cloud/driver.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('A cloud driver is already running')
    try:run()
    except BaseException as error:
        state('stopped_for_inspection',error_type=type(error).__name__,error=str(error));raise
