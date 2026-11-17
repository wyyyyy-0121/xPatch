from common import *
import time,torch
from models import create,objective
from multiscale.training import atomic_torch_save

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert len(os.sched_getaffinity(0))==2 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    started=time.monotonic();out=[]
    for lookback,h in ((480,60),(96,8),(252,21),(36,3),(12,12)):
        families=[('tiny','tiny',f) for f in ('short','long')]
        if lookback in (480,96):families += [('original',m,'long') for m in ('mlp','gru','lstm','xpatch')]
        for family,model_name,field in families:
            cfg={'family':family,'model':model_name,'field':field,'lookback':lookback,'horizon':h}
            torch.manual_seed(2026);model=create(cfg,np.zeros(h));opt=torch.optim.Adam(model.parameters(),lr=.001)
            x=torch.randn(128,lookback);a=torch.randn(128,1);args=(x,) if family=='tiny' else (x[:,:,None],a)
            y=torch.randn(128,h)
            for _ in range(3):
                opt.zero_grad(set_to_none=True);loss=objective(model(*args),y,'aligned');loss.backward();opt.step()
            begin=time.perf_counter();n=0
            for _ in range(20):
                opt.zero_grad(set_to_none=True);loss=objective(model(*args),y,'aligned');loss.backward();opt.step();n+=1
                if time.monotonic()-started>820:break
            train=(time.perf_counter()-begin)/n;model.eval();begin=time.perf_counter()
            with torch.inference_mode():
                for _ in range(10):model(*args)
            forward=(time.perf_counter()-begin)/10
            begin=time.perf_counter();p=BASE/'audit/benchmark_latest.pt'
            atomic_torch_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()},p)
            saved=torch.load(p,weights_only=True);model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
            out.append({'family':family,'model':model_name,'field':field,'lookback':lookback,'horizon':h,
                'train_seconds_per_128':train,'forward_seconds_per_128':forward,
                'checkpoint_roundtrip_seconds':time.perf_counter()-begin,'synthetic_only':True,
                'steps':n,'parameters':sum(p.numel() for p in model.parameters())})
            if time.monotonic()-started>850:raise RuntimeError('Benchmark 15-minute cap reached before required shapes')
    # Causal feature prefix must not change if the later supplied history is changed.
    model=create({'family':'tiny','field':'long','horizon':60},np.zeros(60));model.eval()
    x=torch.randn(2,480);xx=x.clone();xx[:,240:]+=100
    with torch.inference_mode():
        assert torch.equal(model.representation(x)[:,:,:240],model.representation(xx)[:,:,:240])
        assert torch.count_nonzero(model(x))==0
    atomic(BASE/'audit/SYNTHETIC_THROUGHPUT.json',{'status':'measured_without_market_scores','models':out,
        'elapsed_seconds':time.monotonic()-started,'market_fits':0,'causal_prefix_and_zero_initialization':True,
        'affinity':sorted(os.sched_getaffinity(0))})
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
