"""Admit whole registered factorial groups, then run inside the original deadline."""
from study import *
import argparse, fcntl, signal, subprocess, time

SLOTS = [(0, 2), (4, 6), (8, 10), (12, 14), (16, 18), (20, 22), (1, 3), (5, 7)]
STOP = False


def stop(*_):
    global STOP
    STOP = True


def schedule_seconds(costs):
    slots = [0.]*8
    for cost in sorted(costs, reverse=True):
        i = int(np.argmin(slots)); slots[i] += cost
    return max(slots, default=0.)


def prepare():
    manifest_path = BASE/'protocol/REPLICATION_JOBS.json'
    if manifest_path.exists():
        saved = read(manifest_path)
        assert saved['training_hashes'] == source_hashes()
        assert saved['rules_sha256'] == sha(BASE/'protocol/CONTINUATION_RULES.json')
        assert saved['worker_sha256'] == sha(BASE/'code/replication_worker.py')
        return saved
    results = read(BASE/'analysis/SCREENING_RESULTS.json')
    assert results['matched_interventions_verified'], 'Do not advance unmatched interventions'
    rules = read(BASE/'protocol/CONTINUATION_RULES.json')
    budget = read(BASE/'protocol/BUDGET.json')
    remaining = budget['deadline_epoch']-time.time()
    screening = read(BASE/'protocol/JOBS.json')['jobs']
    jobs = []; costs = []; decisions = []
    groups = {(r['task'], r['model']): r for r in results['replication_groups']}
    for task, model in rules['priority_independent_of_score_size']:
        group = groups[(task, model)]
        decision = {'task': task, 'model': model, 'screening_reasons': group['reasons'],
                    'eligible': group['mechanism_replicate'], 'admitted': False}
        if not decision['eligible']:
            decision['reason'] = 'registered direction gate not met'
            decisions.append(decision); continue
        samples = [j for j in screening if task_key(j['profile'], j['horizon'], j['target']) == task
                   and j['model'] == model]
        rates = []
        for cfg in samples:
            r = read(BASE/'runs'/cfg['id']/'DONE.json')
            rec = read(PRIOR/'cache'/task/'TASK.json')['folds'][str(cfg['fold'])]
            c = rec['counts']; batches = (c['fit']+127)//128+(c['stop']+127)//128
            rates.append(r['training_wall_seconds']/(r['epochs']*batches))
        # Use the slowest observed configuration, including startup/checkpoint costs.
        rate = max(rates)
        variants = list(dict.fromkeys(j['variant'] for j in samples))
        expected_variants = 1 if model == 'tiny' else 4
        assert len(variants) == expected_variants
        proposed = []; proposed_costs = []
        for variant in variants:
            for fold in rules['folds']:
                for seed in rules['seeds']:
                    first = samples[0]
                    cfg = make_job(first['profile'], first['horizon'], first['target'], model, variant, fold, seed)
                    proposed.append(cfg)
                    if (BASE/'runs'/cfg['id']/'DONE.json').exists():
                        assert read(BASE/'runs'/cfg['id']/'DONE.json')['job'] == cfg
                        continue
                    c = read(PRIOR/'cache'/task/'TASK.json')['folds'][str(fold)]['counts']
                    batches = (c['fit']+127)//128+(c['stop']+127)//128
                    proposed_costs.append(cfg['max_epochs']*batches*rate+30.)
        projected = 1.3*schedule_seconds(costs+proposed_costs)+300.
        decision.update(total_cells=len(proposed), reused_screening_cells=len(proposed)-len(proposed_costs),
                        worst_observed_seconds_per_fit_stop_batch=rate,
                        cumulative_projected_wall_seconds=projected, remaining_wall_seconds=remaining)
        if projected < remaining:
            jobs.extend(proposed); costs.extend(proposed_costs)
            decision.update(admitted=True, reason='complete group fits unchanged remaining deadline')
        else:
            decision['reason'] = 'complete group does not fit unchanged remaining deadline'
        decisions.append(decision)
    manifest = {'stage': 'C_mechanism_replication', 'jobs': jobs, 'groups': decisions,
                'training_hashes': source_hashes(),
                'worker_sha256': sha(BASE/'code/replication_worker.py'),
                'controller_sha256': sha(__file__),
                'rules_sha256': sha(BASE/'protocol/CONTINUATION_RULES.json'),
                'screening_analysis_sha256': sha(BASE/'analysis/SCREENING_RESULTS.json'),
                'unchanged_budget_sha256': sha(BASE/'protocol/BUDGET.json'),
                'deadline_epoch': budget['deadline_epoch'], 'at_utc': now(),
                'remaining_wall_seconds_at_admission': remaining,
                'projected_wall_seconds_including_reserve': 1.3*schedule_seconds(costs)+300.,
                'Q3_values_read': False, 'method_success_claimed': False}
    atomic(manifest_path, manifest)
    atomic(BASE/'audit/REPLICATION_ADMISSION.json', {k: v for k, v in manifest.items() if k != 'jobs'})
    print(json.dumps({'admitted_total_cells': len(jobs),
                      'new_fits': len(costs), 'group_decisions': decisions}), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); manifest = prepare()
    if args.prepare_only:
        return
    assert sha(__file__) == manifest['controller_sha256']
    assert read(BASE/'audit/SCREENING_COMPLETE.json')['jobs'] == 52
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, stop)
    deadline = manifest['deadline_epoch']; jobs = manifest['jobs']
    assert sha(BASE/'protocol/BUDGET.json') == manifest['unchanged_budget_sha256']
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
                        error = {'job': r['job'], 'exitcode': r['process'].returncode}
                        atomic(BASE/'audit/REPLICATION_ERROR.json', error)
                    del active[slot]
            for slot in [s for s in SLOTS if s not in active]:
                if stopping or error is not None or not pending:
                    break
                memory = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                                  if x.startswith('MemAvailable:')))*1024
                if memory < 2*1024**3:
                    break
                job = pending.pop(0); folder = BASE/'runs'/job['id']; folder.mkdir(exist_ok=True)
                assert not (folder/'ERROR.json').exists(), 'Unreviewed worker error'
                env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', OMP_NUM_THREADS='2',
                           MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
                log = (BASE/'logs'/(job['id']+'.log')).open('a')
                p = subprocess.Popen(['taskset', '-c', ','.join(map(str, slot)), sys.executable,
                       str(BASE/'code/replication_worker.py'), '--job', job['id']], cwd=ROOT,
                       env=env, stdout=log, stderr=subprocess.STDOUT)
                active[slot] = {'process': p, 'job': job, 'log': log}
            if time.monotonic()-last > 5 or stopping:
                last = time.monotonic()
                done = sum((BASE/'runs'/j['id']/'DONE.json').exists() for j in jobs)
                atomic(BASE/'audit/REPLICATION_STATE.json', {'status': 'stopping' if stopping else 'running',
                       'pid': os.getpid(), 'planned': len(jobs), 'completed': done, 'pending': len(pending),
                       'active': [{'pid': r['process'].pid, 'cpus': list(s), 'job': r['job']['id']}
                                  for s, r in active.items()], 'deadline_epoch': deadline, 'at_utc': now()})
            if stopping and not active:
                atomic(BASE/'audit/REPLICATION_PAUSED.json', {'completed': done, 'planned': len(jobs),
                       'error': error, 'deadline_epoch': deadline, 'at_utc': now()})
                if error:
                    raise RuntimeError(str(error))
                return
            time.sleep(1)
    complete = {'status': 'all_admitted_replication_cells_complete', 'jobs': len(jobs),
                'admitted_groups': sum(r['admitted'] for r in manifest['groups']),
                'unadmitted_eligible_groups': [r for r in manifest['groups'] if r['eligible'] and not r['admitted']],
                'deadline_epoch': deadline, 'at_utc': now(), 'Q3_values_read': False}
    atomic(BASE/'audit/REPLICATION_COMPLETE.json', complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    main()
