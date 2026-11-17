from common import *
from datetime import datetime,timezone
import time,argparse

def linear_controls():
    """Ridge gets exactly all 4+480+4 past features used by the candidate."""
    cache=cache_arrays();folder=BASE/'controls';folder.mkdir(exist_ok=True)
    def xrows(rr):return np.column_stack([np.asarray(cache[k][rr],dtype='float64') for k in ('trend','residual','gate')])
    for fold in range(1,6):
        out=folder/f'linear_f{fold}';out.mkdir(exist_ok=True)
        if (out/'DONE.json').exists():continue
        start=time.time();cpu=time.process_time();parts=np.load(BASE/'cache'/f'fold_{fold}.npz')
        fit=rows_for(parts['fit']);stop=rows_for(parts['stop']);val=rows_for(parts['validation']);n=len(fit)
        xx=np.zeros((488,488));xy=np.zeros((488,60));sx=np.zeros(488);sy=np.zeros(60)
        for i in range(0,n,4096):
            rr=fit[i:i+4096];x=xrows(rr);y=cache['actual'][rr]
            xx+=x.T@x;xy+=x.T@y;sx+=x.sum(0);sy+=y.sum(0)
        mx=sx/n;my=sy/n;std=np.sqrt(np.maximum(np.diag(xx)/n-mx**2,0));std=np.where(std>1e-12,std,1)
        gram=(xx-n*np.outer(mx,mx))/std[:,None]/std[None,:];rhs=(xy-n*np.outer(mx,my))/std[:,None]
        best=None;candidates=[]
        for alpha in (.1,1.,10.):
            coef=np.linalg.solve(gram+alpha*np.eye(488),rhs);total=0.
            for i in range(0,len(stop),4096):
                rr=stop[i:i+4096];p=((xrows(rr)-mx)/std)@coef+my
                total+=float(np.square(p[:,-1]-cache['actual'][rr,-1]).sum())
            score=total/len(stop);candidates.append({'alpha':alpha,'inner_stop_mse':score})
            if best is None or score<best[0]:best=(score,alpha,coef)
        _,alpha,coef=best;p=np.concatenate([((xrows(val[i:i+4096])-mx)/std)@coef+my for i in range(0,len(val),4096)])
        np.savez_compressed(out/'prediction.npz',prediction=p,origins=parts['validation'])
        np.savez_compressed(out/'model.npz',coef=coef,x_mean=mx,x_std=std,y_mean=my)
        y=cache['actual'][val];mse=float(np.mean((p[:,-1]-y[:,-1])**2));zero=float(np.mean(y[:,-1]**2))
        atomic(out/'DONE.json',{'status':'completed','fold':fold,'selected_alpha':alpha,'inner_candidates':candidates,
            'features':488,'information':'same trend, residual history and gate inputs as candidate',
            'endpoint_mse':mse,'zero_mse':zero,'improvement_pct':100*(1-mse/zero),
            'wall_seconds':time.time()-start,'cpu_seconds':time.process_time()-cpu,'Q3_values_read':False,
            'prediction_sha256':sha(out/'prediction.npz'),'model_sha256':sha(out/'model.npz')})
        print(json.dumps({'control':'linear','fold':fold,'improvement_pct':100*(1-mse/zero)}),flush=True)

def shrink_controls(available_only=False):
    import torch
    from multiscale.features import CloseWindows,Scaler
    from multiscale.models import make_model
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    bars,_,folds,admission=bars_and_folds();windows=CloseWindows(bars,480,admission['split_actions']);cache=cache_arrays()
    for f in folds:
        fold=f['fold'];stop=f['inner']['stop'];val=f['validation']
        for seed in (2026,2027,2028):
            out=BASE/'controls'/f'shrink_lstm_f{fold}_s{seed}';out.mkdir(parents=True,exist_ok=True)
            if (out/'DONE.json').exists():continue
            root=BASE/'runs'/f'formal_lstm_f{fold}_s{seed}'
            if available_only and not (root/'DONE.json').exists():continue
            assert (root/'DONE.json').exists()
            start=time.time();saved=torch.load(root/'best.pt',map_location='cpu',weights_only=True)
            scaler=Scaler(**saved['scaler']);model=make_model('lstm',480,60);model.load_state_dict(saved['model']);model.eval();pv=[]
            with torch.inference_mode():
                for i in range(0,len(stop),512):
                    x,a=windows.arrays(stop[i:i+512]);x,a=scaler.transform_x(x,a)
                    pv.append(scaler.invert_y(model(torch.from_numpy(x[:,:,None]),torch.from_numpy(a)).numpy())[:,-1])
            ps=np.concatenate(pv);ys=cache['actual'][rows_for(stop),-1];den=float(np.sum(ps**2))
            alpha=float(np.clip(np.sum(ps*ys)/den,0,1)) if den>0 else 0.
            with np.load(root/'prediction.npz') as a:
                assert np.array_equal(a['origins'],val);p=a['prediction']*alpha
            np.savez_compressed(out/'prediction.npz',prediction=p,origins=val)
            atomic(out/'DONE.json',{'status':'completed','fold':fold,'seed':seed,'alpha':alpha,
                'calibration':'closed-form [0,1] scalar minimizing H60 MSE only on inner stop labels',
                'outer_labels_used_to_fit_alpha':False,'checkpoint_sha256':sha(root/'best.pt'),
                'prediction_sha256':sha(out/'prediction.npz'),'wall_seconds':time.time()-start,'Q3_values_read':False})
            print(json.dumps({'control':'lstm_shrink','fold':fold,'seed':seed,'alpha':alpha}),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['linear','shrink']);ap.add_argument('--available-only',action='store_true');args=ap.parse_args()
    linear_controls() if args.mode=='linear' else shrink_controls(args.available_only)
