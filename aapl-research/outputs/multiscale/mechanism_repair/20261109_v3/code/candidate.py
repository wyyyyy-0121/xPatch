"""Apply the registered mechanism condition, freeze once, admit the whole matrix."""
from study import *
import argparse, fcntl, signal, subprocess, time

SLOTS = [(0, 2), (4, 6), (8, 10), (12, 14), (16, 18), (20, 22), (1, 3), (5, 7)]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def prepare():
    path = BASE/'protocol/CANDIDATE_JOBS.json'
    if path.exists():
        saved = read(path)
        assert saved['worker_sha256'] == sha(BASE/'code/candidate_worker.py')
        assert saved['controller_sha256'] == sha(__file__)
        assert saved['rules_sha256'] == sha(BASE/'protocol/CANDIDATE_RULES.json')
        assert saved['budget_sha256'] == sha(BASE/'protocol/BUDGET.json')
        return saved
    result = read(BASE/'analysis/REPLICATION_RESULTS.json')
    rules = read(BASE/'protocol/CANDIDATE_RULES.json'); required = rules['required_contrast']
    selected = None; condition_checks = []
    for model in rules['architecture_precedence']:
        rows = [r for r in result['contrasts'] if r['model'] == model and
                all(r[key] == required[key] for key in ('task', 'primary', 'reference', 'mode'))]
        assert len(rows) == 1
        eligible = rows[0]['replicated_development_mechanism_supported']
        condition_checks.append({'model': model, 'supported': eligible, 'contrast': rows[0]})
        if selected is None and eligible:
            selected = model
    common = {'architecture': selected, 'condition_checks': condition_checks,
              'rules_sha256': sha(BASE/'protocol/CANDIDATE_RULES.json'),
              'worker_sha256': sha(BASE/'code/candidate_worker.py'), 'controller_sha256': sha(__file__),
              'training_hashes': source_hashes(), 'budget_sha256': sha(BASE/'protocol/BUDGET.json'),
              'replication_analysis_sha256': sha(BASE/'analysis/REPLICATION_RESULTS.json'),
              'deadline_epoch': read(BASE/'protocol/BUDGET.json')['deadline_epoch'], 'at_utc': now(),
              'Q3_values_read': False, 'independent_confirmation': False}
    if selected is None:
        common.update(admitted=False, reason='registered mechanism condition not supported', jobs=[])
        atomic(path, common); return common
    primary = set(rules['primary_return_tasks']); jobs = []
    for profile, (_, _, horizons) in PROFILES.items():
        for horizon in horizons:
            for target in ('future_close', 'future_log_return'):
                task = task_key(profile, horizon, target)
                seeds = (2026, 2027, 2028) if task in primary or task == 'hourly_h1_future_close' else (2026,)
                variant = 'anchor_baseline' if target == 'future_log_return' else 'price_baseline'
                for fold in range(1, 6):
                    for seed in seeds:
                        jobs.append(make_job(profile, horizon, target, selected, variant, fold, seed))
    assert len(jobs) == 130 and len({j['id'] for j in jobs}) == 130
    # Estimate before launching any new candidate output, using all observed recipes.
    screened = read(BASE/'protocol/JOBS.json')['jobs']
    replicated = read(BASE/'protocol/REPLICATION_JOBS.json')['jobs']
    observed = {j['id']: j for j in screened+replicated if j['model'] == selected}
    profile_rates = {}; rates = []
    for cfg in observed.values():
        r = read(BASE/'runs'/cfg['id']/'DONE.json')
        rec = read(PRIOR/'cache'/task_key(cfg['profile'], cfg['horizon'], cfg['target'])/'TASK.json')['folds'][str(cfg['fold'])]
        c = rec['counts']; batches = (c['fit']+127)//128+(c['stop']+127)//128
        rate = r['training_wall_seconds']/(r['epochs']*batches)
        profile_rates.setdefault(cfg['profile'], []).append(rate); rates.append(rate)
    costs = []; reused = 0
    for cfg in jobs:
        if (BASE/'runs'/cfg['id']/'DONE.json').exists():
            assert read(BASE/'runs'/cfg['id']/'DONE.json')['job'] == cfg
            reused += 1; continue
        c = read(PRIOR/'cache'/task_key(cfg['profile'], cfg['horizon'], cfg['target'])/'TASK.json')['folds'][str(cfg['fold'])]['counts']
        rate = max(profile_rates.get(cfg['profile'], rates))
        costs.append(20*((c['fit']+127)//128+(c['stop']+127)//128)*rate+30.)
    slots = [0.]*8
    for cost in sorted(costs, reverse=True):
        i = int(np.argmin(slots)); slots[i] += cost
    projected = 1.3*max(slots)+300.
    remaining = common['deadline_epoch']-time.time()
    common.update(jobs=jobs, reused_cells=reused, new_cells=len(costs),
                  projected_wall_seconds_including_reserve=projected, remaining_seconds=remaining,
                  admitted=projected < remaining,
                  reason='whole matrix fits unchanged deadline' if projected < remaining else
                         'whole matrix does not fit unchanged deadline; no partial matrix admitted')
    atomic(path, common)
    print(json.dumps({key: common[key] for key in ('architecture', 'admitted', 'reason', 'reused_cells',
                     'new_cells', 'projected_wall_seconds_including_reserve', 'remaining_seconds')}), flush=True)
    return common


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); manifest = prepare()
    if args.prepare_only or not manifest['admitted']:
        return
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    jobs = manifest['jobs']; deadline = manifest['deadline_epoch']
    pending = [j for j in jobs if not (BASE/'runs'/j['id']/'DONE.json').exists()]
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
                    if not (BASE/'runs'/r['job']['id']/'DONE.json').exists() and not stopping:
                        error = {'job': r['job']['id'], 'exitcode': r['process'].returncode}
                        atomic(BASE/'audit/CANDIDATE_ERROR.json', error)
                    del active[slot]
            for slot in [s for s in SLOTS if s not in active]:
                if stopping or error or not pending:
                    break
                cfg = pending.pop(0); log = (BASE/'logs'/(cfg['id']+'.log')).open('a')
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
                p = subprocess.Popen(['taskset', '-c', ','.join(map(str, slot)), sys.executable,
                       str(BASE/'code/candidate_worker.py'), '--job', cfg['id']], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT)
                active[slot] = {'process': p, 'job': cfg, 'log': log}
            if time.monotonic()-last > 5 or stopping:
                last = time.monotonic()
                completed = sum((BASE/'runs'/j['id']/'DONE.json').exists() for j in jobs)
                atomic(BASE/'audit/CANDIDATE_STATE.json', {'status': 'stopping' if stopping else 'running',
                       'completed': completed, 'planned': len(jobs), 'pid': os.getpid(), 'pending': len(pending),
                       'active': [{'pid': r['process'].pid, 'job': r['job']['id'], 'cpus': list(s)}
                                  for s, r in active.items()], 'deadline_epoch': deadline, 'at_utc': now()})
            if stopping and not active:
                atomic(BASE/'audit/CANDIDATE_PAUSED.json', {'completed': completed, 'planned': len(jobs),
                       'error': error, 'deadline_epoch': deadline, 'at_utc': now()})
                if error:
                    raise RuntimeError(str(error))
                return
            time.sleep(1)
    atomic(BASE/'audit/CANDIDATE_COMPLETE.json', {'status': 'all_candidate_cells_complete', 'jobs': len(jobs),
           'at_utc': now(), 'deadline_epoch': deadline, 'Q3_values_read': False})
    print('All 130 candidate cells complete.', flush=True)


if __name__ == '__main__':
    main()
