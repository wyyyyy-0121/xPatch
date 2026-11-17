"""Bounded synthetic GRU/LSTM serial versus two-worker CPU comparison."""
import argparse,json,os,resource,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path


def worker(kind,cores):
    os.sched_setaffinity(0,set(cores))
    import torch
    from torch import nn
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(2026)
    model={'gru':nn.GRU,'lstm':nn.LSTM}[kind](input_size=1,hidden_size=32,batch_first=True)
    head=nn.Linear(33,60);opt=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=0.001)
    x=torch.randn(128,480,1);anchor=torch.randn(128,1);y=torch.randn(128,60)
    def update():
        opt.zero_grad(set_to_none=True);features=model(x)[0][:,-1];pred=head(torch.cat((features,anchor),dim=1));loss=(pred-y).square().mean()
        if not torch.isfinite(loss):raise ValueError('nonfinite synthetic loss')
        loss.backward();opt.step();return float(loss.detach())
    for _ in range(5):update()
    start=time.monotonic()
    for _ in range(30):loss=update()
    print(json.dumps({'status':'synthetic_worker_completed','kind':kind,'cores':cores,'steps':30,
                      'elapsed_seconds':time.monotonic()-start,'final_synthetic_loss':loss,
                      'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}),flush=True)


def run():
    from multiscale.spec import OUT
    from multiscale.bars import file_digest
    this=Path(__file__).resolve()
    jobs=[('gru',[0,2]),('lstm',[4,6])]
    allowed=os.sched_getaffinity(0)
    if any(not set(cores).issubset(allowed) for _,cores in jobs):raise ValueError('required CPU cores unavailable')
    def args(kind,cores):return [sys.executable,str(this),'--worker',kind,'--cores',','.join(map(str,cores))]
    serial=[];started=time.monotonic()
    for kind,cores in jobs:
        p=subprocess.run(args(kind,cores),capture_output=True,text=True,timeout=90,check=True)
        serial.append(json.loads(p.stdout.strip().splitlines()[-1]))
    serial_wall=time.monotonic()-started
    started=time.monotonic();processes=[subprocess.Popen(args(k,c),stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for k,c in jobs]
    parallel=[]
    try:
        for p in processes:
            stdout,stderr=p.communicate(timeout=90)
            if p.returncode!=0:raise ValueError('synthetic worker failed: '+stderr[-500:])
            parallel.append(json.loads(stdout.strip().splitlines()[-1]))
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:p.kill();p.wait()
    parallel_wall=time.monotonic()-started
    for a,b in zip(serial,parallel,strict=True):
        if a['kind']!=b['kind'] or abs(a['final_synthetic_loss']-b['final_synthetic_loss'])>1e-6:
            raise ValueError('serial and parallel synthetic worker outcomes differ')
    out={'status':'synthetic_concurrency_probe_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'purpose':'short CPU parallelism indication, not a full-run forecast',
         'workload':'synthetic GRU and LSTM; batch 128, sequence 480, output 60; 5 warmup and 30 timed updates each',
         'device':'cpu','gpu_used':False,'market_data_read':False,'threads_per_worker':2,
         'worker_cpu_affinity':{kind:cores for kind,cores in jobs},'serial_wall_seconds':serial_wall,
         'parallel_wall_seconds':parallel_wall,'observed_wall_speedup':serial_wall/parallel_wall,
         'serial_workers':serial,'parallel_workers':parallel,'script_sha256':file_digest(this),
         'limits':['One short serial-versus-parallel pair; results vary with machine load and thermal conditions.',
                   'Does not benchmark full data I/O, validation, checkpoints, all model types or the largest fold.',
                   'Does not authorize long runs, GPU use or cloud spending.']}
    directory=OUT/'private_compute_projections';directory.mkdir(parents=True,exist_ok=True)
    path=directory/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'_cpu_concurrency.json')
    with path.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'status':out['status'],'serial_seconds':round(serial_wall,2),'parallel_seconds':round(parallel_wall,2),
                      'observed_wall_speedup':round(out['observed_wall_speedup'],2),'receipt':str(path)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',choices=('gru','lstm'));p.add_argument('--cores')
    args=p.parse_args()
    if args.worker:worker(args.worker,[int(x) for x in args.cores.split(',')])
    else:run()
