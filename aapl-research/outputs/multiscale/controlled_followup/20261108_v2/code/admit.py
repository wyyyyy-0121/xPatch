from common import *
import math,time
from datetime import datetime,timezone

def main():
    bench=read(BASE/'audit/SYNTHETIC_THROUGHPUT.json')
    rates={(r['family'],r['model'],r['field'],r['lookback']):r for r in bench['models']}
    all_jobs=read(BASE/'protocol/ALL_JOBS.json')['jobs'];projections=[]
    for compressed in (False,True):
        jobs=[j for j in all_jobs if not (compressed and j['phase']=='cross_scale' and j['seed']!=2026)]
        costs=[]
        for j in jobs:
            n=read(BASE/'cache'/key(j['profile'],j['horizon'],j['target'])/'TASK.json')['folds'][str(j['fold'])]['counts']
            r=rates[(j['family'],j['model'],j['field'],j['lookback'])]
            train=math.ceil(n['fit']/128)*r['train_seconds_per_128']
            val=math.ceil(n['stop']/128)*r['forward_seconds_per_128']
            # Four seconds per job accounts for loading; six seconds for checkpoints and repeated array conversion.
            costs.append(20*(train+val)+math.ceil(n['validation']/128)*r['forward_seconds_per_128']+10.)
        # Eight simultaneous two-CPU workers, 60% measured-single-job capacity assumed,
        # plus 30% contingency and 20 minutes reserved for recovery, launch and final inference.
        projection=max(sum(costs)/(8*.60),max(costs))*1.30+1200
        projections.append({'variant':'compressed' if compressed else 'standard','new_neural_fits':len(jobs),
            'sum_job_seconds':sum(costs),'projected_wall_seconds':projection,'cpu_parallel_efficiency_assumption':.60,
            'contingency_fraction':.30,'extra_launch_recovery_inference_seconds':1200})
    remaining=read(BASE/'protocol/BUDGET.json')['deadline_epoch']-time.time()
    selected=next((x for x in projections if x['projected_wall_seconds']<=remaining),None)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if available<9*1024**3:selected=None
    result={'status':'admitted' if selected else 'not_admitted','admitted':selected is not None,
        'projections':projections,'selected':selected,'remaining_seconds_at_admission':remaining,
        'memory_available_bytes':available,'all_new_market_scores_hidden_before_variant_choice':True,
        'benchmark_sha256':sha(BASE/'audit/SYNTHETIC_THROUGHPUT.json'),'at_utc':datetime.now(timezone.utc).isoformat()}
    atomic(BASE/'audit/COMPUTE_ADMISSION.json',result)
    if selected:
        compressed=selected['variant']=='compressed'
        jobs=[j for j in all_jobs if not (compressed and j['phase']=='cross_scale' and j['seed']!=2026)]
        # Complete mechanistic pairs before other tasks; expensive later folds first within each stage.
        phase_order={'loss_probe':0,'mechanism_initial':1,'mechanism_formal':2,'cross_scale':3}
        jobs.sort(key=lambda j:(phase_order[j['phase']],-j['fold'],j['id']))
        atomic(BASE/'protocol/SELECTED_JOBS.json',{'variant':selected['variant'],'selection_basis':'throughput only',
            'admission_sha256':sha(BASE/'audit/COMPUTE_ADMISSION.json'),'jobs':jobs})
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
