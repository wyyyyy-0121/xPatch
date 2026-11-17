from common import *
import argparse,subprocess,torch

def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['continuous','split']);ap.add_argument('--steps',type=int)
    args=ap.parse_args()
    cfg=next(j for j in read(BASE/'protocol/ALL_JOBS.json')['jobs'] if j['profile']=='hourly' and
        j['horizon']==8 and j['target']=='future_log_return' and j['family']=='tiny' and j['field']=='long' and
        j['loss']=='aligned' and j['fold']==1 and j['seed']==2026)
    if args.part:
        from engine import run
        run(cfg,stop_after_steps=args.steps,output_override=BASE/'audit/recovery'/args.part)
        return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for part,steps in [('continuous',8),('split',3),('split',5)]:
        subprocess.run(['taskset','-c','30,31',sys.executable,__file__,'--part',part,'--steps',str(steps)],env=env,check=True)
    a=torch.load(BASE/'audit/recovery/continuous/latest.pt',weights_only=True)
    b=torch.load(BASE/'audit/recovery/split/latest.pt',weights_only=True)
    checks={k:same(a[k],b[k]) for k in ('identity','model','optimizer','numpy_rng','torch_rng','scaler')}
    checks.update({k:same(a['state'][k],b['state'][k]) for k in ('epoch','cursor','permutation','steps','loss_sum','seen','history')})
    assert all(checks.values()),checks
    atomic(BASE/'audit/RECOVERY_CHECK.json',{'status':'passed','checks':checks,
        'actual_process_exits_and_resume':True,'continuous_steps':8,'resumed_parts':[3,5],
        'outer_predictions_or_scores_produced':False,'deadline_preserved':read(BASE/'protocol/BUDGET.json')['deadline_epoch']})

if __name__=='__main__':main()
