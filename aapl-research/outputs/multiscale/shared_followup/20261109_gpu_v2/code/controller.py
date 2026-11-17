"""Measured local-GPU queue, authorized deadline, checkpoint on stop."""
from experiment import *
import experiment as study
import fcntl
import subprocess

SLOTS = [(0,2),(4,6),(8,10),(12,14),(16,18),(20,22),(1,3),(5,7)]


def main():
    assert set(os.sched_getaffinity(0)).issubset({24, 26})
    rules = freeze(); assert rules['admitted'] and rules['fixed_epochs'] == 20 and rules['GPU'] is True
    slots = SLOTS[:rules['maximum_concurrent_gpu_workers']]
    assert 1 <= len(slots) <= 8
    slot_arms = rules.get('slot_arms')
    assert slot_arms == ['tied','untied'] and len(slots) == 2
    active = {}; pending = [j for j in rules['jobs'] if not (BASE / 'runs' / j['id'] / 'DONE.json').exists()]
    (BASE / 'logs').mkdir(exist_ok=True)
    error = None
    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    with (BASE / 'audit/QUEUE_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while pending or active:
            stopping = study.STOP or time.time() >= DEADLINE or error is not None
            for slot, value in list(active.items()):
                if stopping and not value.get('signaled') and value['p'].poll() is None:
                    value['p'].send_signal(signal.SIGUSR1); value['signaled'] = True
                if value['p'].poll() is not None:
                    value['log'].close()
                    if not (BASE / 'runs' / value['job']['id'] / 'DONE.json').exists() and not stopping:
                        error = {'job': value['job'], 'exit_code': value['p'].returncode}
                    del active[slot]
            stopping = study.STOP or time.time() >= DEADLINE or error is not None
            for slot in slots:
                if slot in active or not pending or stopping or error is not None:
                    continue
                arm = slot_arms[slots.index(slot)]
                index = next((i for i,j in enumerate(pending) if j['arm'] == arm), None)
                if index is None:
                    continue
                job = pending.pop(index); log = (BASE / 'logs' / (job['id']+'.log')).open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='2',
                    MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8')
                process = subprocess.Popen(['taskset','-c',','.join(map(str,slot)),sys.executable,
                    str(BASE / 'code/experiment.py'),'--stage','worker','--job',job['id']],
                    env=env, stdout=log, stderr=subprocess.STDOUT)
                active[slot] = {'job': job, 'p': process, 'log': log}
            completed = sum((BASE / 'runs' / j['id'] / 'DONE.json').exists() for j in rules['jobs'])
            atomic(BASE / 'audit/QUEUE_STATE.json', {'status': 'stopping' if stopping else 'running',
                'pid': os.getpid(), 'completed': completed, 'planned': len(rules['jobs']),
                'pending': len(pending), 'active': [{'pid': v['p'].pid,'cpus':list(k),'job':v['job']['id']}
                    for k,v in active.items()], 'error': error, 'deadline_epoch': DEADLINE, 'at_utc': now()})
            if stopping and not active:
                atomic(BASE / 'audit/PAUSED.json', {'status': 'paused', 'completed': completed,
                    'planned': len(rules['jobs']), 'error': error, 'deadline_epoch': DEADLINE,
                    'checkpoints_preserved': True, 'at_utc': now()})
                return
            time.sleep(1)
    atomic(BASE / 'audit/COMPLETE.json', {'status':'completed','bundles':len(rules['jobs']),
        'deadline_epoch':DEADLINE,'Q3_values_read':False,'at_utc':now()})


    if study.STOP or time.time() >= DEADLINE:
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    with (BASE/'logs/evaluation.log').open('a') as log:
        process = subprocess.Popen(['taskset','-c','24,26',sys.executable,
            str(BASE/'code/evaluate_shared.py'),'--stage','run'], env=env,
            stdout=log, stderr=subprocess.STDOUT)
        atomic(BASE/'audit/ANALYSIS_PROCESS.json', {'pid':process.pid,'parent_pid':os.getpid(),
            'deadline_epoch':DEADLINE,'at_utc':now()})
        signaled = False
        while process.poll() is None:
            if (study.STOP or time.time() >= DEADLINE) and not signaled:
                process.send_signal(signal.SIGUSR1); signaled = True
            time.sleep(1)
    complete = BASE/'analysis/ANALYSIS_COMPLETE.json'
    atomic(BASE/'audit/PIPELINE_STATE.json', {'status':'completed' if complete.exists() and process.returncode == 0 else 'incomplete',
        'training_bundles':len(rules['jobs']),'analysis_complete':complete.exists(),
        'analysis_exit_code':process.returncode,'deadline_epoch':DEADLINE,'at_utc':now()})


if __name__ == '__main__':
    main()
