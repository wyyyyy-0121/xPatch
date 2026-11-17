from common import *
import torch,time
from models import create
from multiscale.training import atomic_torch_save

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.manual_seed(2026)
    assert len(os.sched_getaffinity(0))<=2 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    results={}
    for kind in ('full','lstm'):
        model=create(kind,np.zeros(60));opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        if kind=='lstm':args=(torch.randn(128,480,1),torch.randn(128,1))
        else:args=(torch.randn(128,4),torch.randn(128,480),torch.randn(128,4))
        y=torch.randn(128,60);begin=time.monotonic()
        for _ in range(3):
            opt.zero_grad();loss=(model(*args)-y).square().mean();loss.backward();opt.step()
        start=time.monotonic();steps=0
        for _ in range(60):
            opt.zero_grad();loss=(model(*args)-y).square().mean();loss.backward();opt.step();steps+=1
            if time.monotonic()-begin>850:break
        train=(time.monotonic()-start)/steps;model.eval();start=time.monotonic()
        with torch.inference_mode():
            for _ in range(30):model(*args)
        forward=(time.monotonic()-start)/30
        start=time.monotonic();path=BASE/'audit'/f'{kind}_benchmark_checkpoint.pt'
        atomic_torch_save({'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':torch.get_rng_state()},path)
        ck=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer'])
        results[kind]={'train_seconds_per_128':train,'forward_seconds_per_128':forward,
            'checkpoint_roundtrip_seconds':time.monotonic()-start,'steps':steps,'synthetic_only':True,
            'cpu_mask':sorted(os.sched_getaffinity(0))}
    atomic(BASE/'audit/SYNTHETIC_THROUGHPUT.json',{'status':'measured','models':results,'market_fits':0})
    print(json.dumps(results))

if __name__=='__main__':main()
