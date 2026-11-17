"""Synthetic throughput, baseline-function checks and real exit/resume before queue launch."""
from study import *
import subprocess, time, torch
from networks import create, objective

def same(a,b):
    if isinstance(a,torch.Tensor):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert len(os.sched_getaffinity(0))==2 and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    assert source_hashes()==read(BASE/'protocol/CODE_FREEZE.json')['hashes']
    jobs=read(BASE/'protocol/JOBS.json')['jobs'];measurements=[];start=time.monotonic()
    for profile,h,target in [('minute',60,'future_log_return'),('hourly',8,'future_log_return'),('hourly',1,'future_close')]:
        models=('lstm','xpatch','tiny') if target=='future_log_return' else ('lstm','xpatch')
        for model in models:
            cfg=next(j for j in jobs if j['profile']==profile and j['horizon']==h and j['model']==model and
                j['initialization']=='baseline' and j['fold']==1)
            task=task_key(profile,h,target);s=read(PRIOR/'cache'/task/'scaler_1.json')
            scaler=s['tiny' if model=='tiny' else 'original']
            torch.manual_seed(2026);net=create(cfg,scaler)
            x=torch.randn(128,cfg['lookback']);a=torch.randn(128,1);y=torch.randn(128,h)
            args=(x,) if model=='tiny' else (x[:,:,None],a)
            net.eval()
            with torch.inference_mode():
                p=net(*args).numpy()*np.asarray(scaler['y_std'])+np.asarray(scaler['y_mean'])
            expected=np.zeros_like(p) if target=='future_log_return' else \
                np.broadcast_to((a.numpy()*scaler['anchor_std']+scaler['anchor_mean']),p.shape)
            assert np.allclose(p,expected,rtol=1e-6,atol=1e-6),(model,float(abs(p-expected).max()))
            opt=torch.optim.Adam(net.parameters(),lr=.001)
            for _ in range(3):
                net.train();opt.zero_grad(set_to_none=True);loss=objective(net(*args),y,cfg['loss']);loss.backward();opt.step()
            begun=time.perf_counter()
            for _ in range(8):
                net.train();opt.zero_grad(set_to_none=True);loss=objective(net(*args),y,cfg['loss']);loss.backward();opt.step()
            train=(time.perf_counter()-begun)/8
            net.eval();begun=time.perf_counter()
            with torch.inference_mode():
                for _ in range(8):net(*args)
            forward=(time.perf_counter()-begun)/8
            measurements.append({'profile':profile,'horizon':h,'model':model,'train_seconds_per_128':train,
                'forward_seconds_per_128':forward,'initial_function_check':True,'parameters':sum(p.numel() for p in net.parameters())})
            assert time.monotonic()-start<600,'Ten-minute admission cap exceeded'
    # Counterfactual initialization changes the registered heads; all other parameters stay paired.
    for model in ('lstm','xpatch'):
        cfg=next(j for j in jobs if j['model']==model and j['profile']=='minute' and j['fold']==1 and j['variant']=='anchor_random')
        scaler=read(PRIOR/'cache/minute_h60_future_log_return/scaler_1.json')['original']
        torch.manual_seed(2026);a=create(cfg,scaler)
        torch.manual_seed(2026);b=create({**cfg,'initialization':'baseline'},scaler)
        ignored=('head.',) if model=='lstm' else ('backbone.fc8.','anchor_head.')
        assert all(torch.equal(a.state_dict()[n],v) for n,v in b.state_dict().items() if not n.startswith(ignored))
    cfg=next(j for j in jobs if j['model']=='tiny' and j['profile']=='hourly' and j['fold']==1)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    processes=[]
    for part,steps in [('continuous',8),('split',3),('split',5)]:
        out=BASE/'audit/recovery'/part
        command=[sys.executable,str(BASE/'code/worker.py'),'--job',cfg['id'],'--steps',str(steps),'--folder',str(out)]
        run=subprocess.run(command,env=env,capture_output=True,text=True,timeout=180)
        assert run.returncode==0,run.stderr
        processes.append({'part':part,'steps':steps,'exitcode':run.returncode,'stdout':run.stdout.strip()})
    a=torch.load(BASE/'audit/recovery/continuous/latest.pt',map_location='cpu',weights_only=True)
    b=torch.load(BASE/'audit/recovery/split/latest.pt',map_location='cpu',weights_only=True)
    checks={k:same(a[k],b[k]) for k in ('identity','model','optimizer','scaler','numpy_rng','torch_rng')}
    checks.update({k:same(a['state'][k],b['state'][k]) for k in ('epoch','cursor','permutation','steps','history','loss_sum','seen','gradients_first_two')})
    assert all(checks.values()),checks
    atomic(BASE/'audit/RECOVERY_CHECK.json',{'status':'passed','checks':checks,'actual_processes':processes,
        'deadline_preserved':read(BASE/'protocol/BUDGET.json')['deadline_epoch'],'outer_scores_produced':False})
    cost=0.
    for cfg in jobs:
        m=next(x for x in measurements if x['profile']==cfg['profile'] and x['horizon']==cfg['horizon'] and x['model']==cfg['model'])
        counts=read(PRIOR/'cache'/task_key(cfg['profile'],cfg['horizon'],cfg['target'])/'TASK.json')['folds'][str(cfg['fold'])]['counts']
        batches=lambda n:(n+127)//128
        # Include twenty full epochs, stopping, up to two prediction replays and startup/checkpoint allowance.
        cost+=20*(batches(counts['fit'])*m['train_seconds_per_128']+batches(counts['stop'])*m['forward_seconds_per_128'])
        cost+=2*batches(counts['validation'])*m['forward_seconds_per_128']+15
    projection=1.3*cost/(8*.6)+120
    budget=read(BASE/'protocol/BUDGET.json');remaining=budget['deadline_epoch']-time.time()
    receipt={'status':'admitted' if projection<remaining else 'not_admitted','admitted':projection<remaining,
        'measurements':measurements,'projected_52_jobs_wall_seconds':projection,'remaining_seconds':remaining,
        'parallel_efficiency_assumption':.6,'contingency_fraction':.3,'all_twenty_epochs_included':True,
        'new_outer_scores_seen':False,'at_utc':now(),'recovery_passed':True,'source_hashes_unchanged':True}
    atomic(BASE/'audit/COMPUTE_ADMISSION.json',receipt)
    print(json.dumps(receipt),flush=True)
    if not receipt['admitted']:raise SystemExit(2)

if __name__=='__main__':main()
