"""Bounded 20-versus-60-epoch price diagnostic after the main matched queue."""
from study import *
import signal, subprocess, fcntl, time

SLOTS = [(0, 2), (4, 6), (8, 10), (12, 14), (16, 18), (20, 22), (1, 3), (5, 7)]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def prepare():
    path = BASE/'protocol/CAP_JOBS.json'
    if path.exists():
        saved = read(path)
        assert saved['worker_sha256'] == sha(BASE/'code/cap_worker.py')
        assert saved['controller_sha256'] == sha(__file__)
        assert saved['budget_sha256'] == sha(BASE/'protocol/BUDGET.json')
        return saved
    assert read(BASE/'audit/REPLICATION_COMPLETE.json')['jobs'] == 240
    source_jobs = [j for j in read(BASE/'protocol/REPLICATION_JOBS.json')['jobs']
                   if j['profile'] == 'hourly' and j['horizon'] == 1 and j['variant'] == 'price_random']
    assert len(source_jobs) == 30
    jobs = []; costs = []
    for cfg in source_jobs:
        folder = BASE/'runs'/cfg['id']; result = read(folder/'DONE.json')
        longer = dict(cfg, id=cfg['id']+'_cap60', max_epochs=60)
        jobs.append({'cfg': longer, 'source_job': cfg['id'], 'source_latest_sha256': sha(folder/'latest.pt')})
        extra = 40*result['training_wall_seconds']/result['epochs'] if result['epochs'] == 20 else 0.
        costs.append(extra+15.)
    remaining = read(BASE/'protocol/BUDGET.json')['deadline_epoch']-time.time()
    slots = [0.]*8
    for cost in sorted(costs, reverse=True):
        index = int(np.argmin(slots)); slots[index] += cost
    projected = 1.3*max(slots)+180.
    receipt = {'jobs': jobs, 'admitted': projected < remaining,
               'remaining_seconds': remaining, 'projected_seconds_including_reserve': projected,
               'worker_sha256': sha(BASE/'code/cap_worker.py'), 'controller_sha256': sha(__file__),
               'training_hashes': source_hashes(), 'budget_sha256': sha(BASE/'protocol/BUDGET.json'),
               'deadline_epoch': read(BASE/'protocol/BUDGET.json')['deadline_epoch'],
               'selection': 'all 30 original hourly H1 price cells, no score selection', 'at_utc': now()}
    atomic(path, receipt)
    print(json.dumps({'cap_probe_admitted': receipt['admitted'], 'cells': len(jobs),
                      'projected_seconds': projected, 'remaining_seconds': remaining}), flush=True)
    return receipt


def main():
    manifest = prepare()
    if not manifest['admitted']:
        return
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    deadline = manifest['deadline_epoch']; jobs = manifest['jobs']
    pending = [j for j in jobs if not (BASE/'runs'/j['cfg']['id']/'DONE.json').exists()]
    active = {}; error = None; last = 0.
    with (BASE/'audit/QUEUE_LOCK').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while pending or active:
            stopping = STOP or time.time() >= deadline or error is not None
            for slot, r in list(active.items()):
                if stopping and not r.get('signaled') and r['process'].poll() is None:
                    r['process'].send_signal(signal.SIGUSR1); r['signaled'] = True
                if r['process'].poll() is not None:
                    r['log'].close()
                    if not (BASE/'runs'/r['job']['cfg']['id']/'DONE.json').exists() and not stopping:
                        error = {'job': r['job']['cfg']['id'], 'exitcode': r['process'].returncode}
                        atomic(BASE/'audit/CAP_ERROR.json', error)
                    del active[slot]
            for slot in [s for s in SLOTS if s not in active]:
                if stopping or error or not pending:
                    break
                item = pending.pop(0); cfg = item['cfg']
                log = (BASE/'logs'/(cfg['id']+'.log')).open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
                process = subprocess.Popen(['taskset', '-c', ','.join(map(str, slot)), sys.executable,
                          str(BASE/'code/cap_worker.py'), '--job', cfg['id']], env=env, cwd=ROOT,
                          stdout=log, stderr=subprocess.STDOUT)
                active[slot] = {'process': process, 'job': item, 'log': log}
            if time.monotonic()-last > 5 or stopping:
                last = time.monotonic()
                done = sum((BASE/'runs'/j['cfg']['id']/'DONE.json').exists() for j in jobs)
                atomic(BASE/'audit/CAP_STATE.json', {'status': 'stopping' if stopping else 'running',
                       'completed': done, 'planned': len(jobs), 'pid': os.getpid(),
                       'active': [{'pid': r['process'].pid, 'job': r['job']['cfg']['id'], 'cpus': list(s)}
                                  for s, r in active.items()], 'deadline_epoch': deadline, 'at_utc': now()})
            if stopping and not active:
                atomic(BASE/'audit/CAP_PAUSED.json', {'completed': done, 'planned': len(jobs),
                       'error': error, 'deadline_epoch': deadline, 'at_utc': now()})
                if error:
                    raise RuntimeError(str(error))
                return
            time.sleep(1)
    atomic(BASE/'audit/CAP_COMPLETE.json', {'status': 'completed', 'jobs': len(jobs),
           'at_utc': now(), 'deadline_epoch': deadline, 'Q3_values_read': False})
    print('All cap comparisons complete.', flush=True)


if __name__ == '__main__':
    main()
