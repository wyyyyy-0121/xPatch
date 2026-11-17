"""Freeze the paired matrix before measuring new forecast scores."""
from study import *
import time

if __name__=='__main__':
    assert read(BASE/'analysis/COMPARISON_AUDIT.json')['status']=='comparison_differences_verified'
    jobs=screening_jobs()
    if (BASE/'protocol/JOBS.json').exists():
        assert read(BASE/'protocol/JOBS.json')['jobs']==jobs
    else:atomic(BASE/'protocol/JOBS.json',{'stage':'B_screening','jobs':jobs,'market_scores_seen':False,'at_utc':now()})
    frozen={'hashes':source_hashes(),'plan_sha256':sha(BASE/'PLAN_ZH.md'),
            'jobs_sha256':sha(BASE/'protocol/JOBS.json'),'data_sha256':sha(PRIOR/'protocol/DATA.json'),'at_utc':now()}
    if (BASE/'protocol/CODE_FREEZE.json').exists():assert read(BASE/'protocol/CODE_FREEZE.json')['hashes']==frozen['hashes']
    else:atomic(BASE/'protocol/CODE_FREEZE.json',frozen)
    if not (BASE/'protocol/BUDGET.json').exists():
        start=time.time()
        atomic(BASE/'protocol/BUDGET.json',{'start_epoch':start,'deadline_epoch':start+4*3600,
            'maximum_wall_seconds':4*3600,'maximum_workers':8,'cpus_per_worker':2,'analysis_cpus_total':2,
            'gpu':False,'stage_B_jobs':52,'source':'User requested new plan, new goal and immediate research on 2026-11-09.',
            'previous_budget_unchanged':True,'includes_short_benchmark_and_recovery':True})
    print(json.dumps({'jobs':len(jobs),'budget':read(BASE/'protocol/BUDGET.json'),'frozen':True}),flush=True)
