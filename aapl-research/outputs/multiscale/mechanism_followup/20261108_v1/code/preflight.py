from common import *
import torch,time,copy
from models import create,Small,PROBES
from multiscale.features import Scaler
from multiscale.models import make_model

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert len(os.sched_getaffinity(0))<=2 and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    data=read(BASE/'protocol/DATA.json');bars,_,folds,admission=bars_and_folds()
    from multiscale.features import CloseWindows
    windows=CloseWindows(bars,480,admission['split_actions'])
    checks=[]
    for f in folds:
        rec=data['folds'][str(f['fold'])]['models']['lstm']
        saved=torch.load(rec['checkpoint'],map_location='cpu',weights_only=True)
        assert saved['identity']==rec['identity'] and saved['config']['seed']==2026
        model=make_model('lstm',480,60);model.load_state_dict(saved['model']);model.eval()
        idx=f['validation'][:512];x,a=windows.arrays(idx);scaler=Scaler(**saved['scaler']);x,a=scaler.transform_x(x,a)
        with torch.inference_mode():p=scaler.invert_y(model(torch.from_numpy(x[:,:,None]),torch.from_numpy(a)).numpy())
        with np.load(rec['prediction'],allow_pickle=False) as old:expected=old['prediction'][:512]
        assert np.array_equal(p,expected)
        checks.append({'fold':f['fold'],'lstm_exact_restored_origins':len(idx),'checkpoint_sha256':sha(rec['checkpoint'])})
    x,_=windows.arrays(folds[0]['validation'][:16]);t,r,g,vol=small_features(x)
    assert np.isfinite(t).all() and np.isfinite(r).all() and np.isfinite(g).all()
    configs=['trend','short','fixed_mix','mix_noshrink','full','short_plain','short_const','short_state']
    counts={}
    for name in configs:
        m=create(name,np.zeros(60));p=m(torch.from_numpy(t),torch.from_numpy(r),torch.from_numpy(g))
        assert p.shape==(16,60) and torch.equal(p,torch.zeros_like(p))
        (p-torch.ones_like(p)*.01).square().mean().backward()
        assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters() if q.requires_grad)
        counts[name]={'trainable':sum(q.numel() for q in m.parameters() if q.requires_grad),'total':sum(q.numel() for q in m.parameters())}
    # Dropout/RNG/Adam replay exercises a stochastic original-model probe.
    torch.manual_seed(2026);m=create('mlp_noanchor');opt=torch.optim.Adam(m.parameters(),lr=.001)
    xx=torch.randn(16,480,1);aa=torch.randn(16,1);yy=torch.randn(16,60)
    def update(model,optimizer):
        model.train();optimizer.zero_grad();loss=(model(xx,aa)-yy).square().mean();loss.backward();optimizer.step()
    update(m,opt);ck={'model':copy.deepcopy(m.state_dict()),'optimizer':copy.deepcopy(opt.state_dict()),'rng':torch.get_rng_state().clone()}
    update(m,opt);expected=copy.deepcopy(m.state_dict())
    n=create('mlp_noanchor');no=torch.optim.Adam(n.parameters(),lr=.001);n.load_state_dict(ck['model']);no.load_state_dict(ck['optimizer']);torch.set_rng_state(ck['rng']);update(n,no)
    assert all(torch.equal(expected[k],n.state_dict()[k]) for k in expected)
    atomic(BASE/'audit/PREFLIGHT.json',{'status':'passed','lstm_reuse':checks,'small_parameters':counts,
        'zero_initialization':True,'finite_gradients':True,'dropout_rng_adam_replay_exact':True,
        'data_split_hashes_matched_original':True,'Q3_values_read':False,'new_market_fits':0})
    print(json.dumps({'status':'passed','small_parameters':counts,'lstm_folds_verified':len(checks)}))

if __name__=='__main__':main()
