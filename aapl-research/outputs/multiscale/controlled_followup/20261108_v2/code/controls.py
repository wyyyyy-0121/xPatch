from common import *
import torch
from multiscale.features import CloseWindows,Scaler
from multiscale.models import make_model
from multiscale.bars import targets

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert len(os.sched_getaffinity(0))<=2
    for profile,h,target in MECHANISM:
        task=key(profile,h,target);meta=read(BASE/'cache'/task/'TASK.json');bars=admitted_bars(meta['frequency'])
        windows=CloseWindows(bars,meta['lookback'],read(SOURCE/'admission.json')['split_actions'])
        allidx=np.load(BASE/'cache'/profile/'origins.npy',mmap_mode='r');cache=np.load(BASE/'cache'/profile/'x.npy',mmap_mode='r')
        for fn,fold in meta['folds'].items():
            out=BASE/'controls'/task/f'fold_{fn}';out.mkdir(parents=True,exist_ok=True)
            if (out/'DONE.json').exists():continue
            parts=dict(np.load(fold['split_file']));fit,stop,val=[parts[k] for k in ('fit','stop','validation')]
            D=meta['lookback'];xx=np.zeros((D,D));xy=np.zeros((D,h));sx=np.zeros(D);sy=np.zeros(h)
            for i in range(0,len(fit),4096):
                idx=fit[i:i+4096];x=np.asarray(cache[rows_for(idx,allidx)],dtype='float64');y=targets(bars,idx,h,target)
                xx+=x.T@x;xy+=x.T@y;sx+=x.sum(0);sy+=y.sum(0)
            mx=sx/len(fit);my=sy/len(fit);std=np.sqrt(np.maximum(np.diag(xx)/len(fit)-mx**2,0));std=np.where(std>1e-12,std,1.)
            gram=(xx-len(fit)*np.outer(mx,mx))/std[:,None]/std[None,:];rhs=(xy-len(fit)*np.outer(mx,my))/std[:,None]
            best=None;scores=[]
            for alpha in (.1,1.,10.):
                coef=np.linalg.solve(gram+alpha*np.eye(D),rhs);total=0.
                for i in range(0,len(stop),4096):
                    idx=stop[i:i+4096];p=((cache[rows_for(idx,allidx)]-mx)/std)@coef+my
                    y=targets(bars,idx,h,target);total+=float(((p[:,-1]-y[:,-1])**2).sum())
                score=total/len(stop);scores.append({'alpha':alpha,'stop_mse':score})
                if best is None or score<best[0]:best=(score,alpha,coef)
            _,alpha,coef=best
            p=np.concatenate([((cache[rows_for(val[i:i+4096],allidx)]-mx)/std)@coef+my for i in range(0,len(val),4096)])
            save_npz(out/'ridge_prediction.npz',prediction=p,origins=val)
            save_npz(out/'ridge_model.npz',coef=coef,x_mean=mx,x_std=std,y_mean=my)
            rec=fold['models']['lstm'];ck=torch.load(rec['checkpoint'],weights_only=True)
            assert ck['identity']==rec['identity'];net=make_model('lstm',meta['lookback'],h);net.load_state_dict(ck['model']);net.eval()
            sc=Scaler(**ck['scaler']);den=num=0.
            with torch.inference_mode():
                for i in range(0,len(stop),128):
                    idx=stop[i:i+128];x,a=windows.arrays(idx);x,a=sc.transform_x(x,a)
                    ps=sc.invert_y(net(torch.from_numpy(x[:,:,None]),torch.from_numpy(a)).numpy())[:,-1]
                    ys=targets(bars,idx,h,target)[:,-1];num+=float((ps*ys).sum());den+=float((ps*ps).sum())
            shrink=float(np.clip(num/den,0,1)) if den else 0.
            with np.load(rec['prediction']) as a:
                assert np.array_equal(a['origins'],val);ps=a['prediction']*shrink
            save_npz(out/'shrink_prediction.npz',prediction=ps,origins=val)
            atomic(out/'DONE.json',{'status':'completed','task':task,'fold':int(fn),'ridge_alpha':alpha,
                'ridge_inner_candidates':scores,'ridge_input':'same demeaned historical log closes as tiny model; per-column fit scaling',
                'lstm_shrink_alpha':shrink,'expert_seed':2026,'fit_on_outer_labels':False,
                'ridge_prediction_sha256':sha(out/'ridge_prediction.npz'),
                'shrink_prediction_sha256':sha(out/'shrink_prediction.npz'),'Q3_values_read':False})
            print(json.dumps({'controls_complete':task,'fold':fn}),flush=True)

if __name__=='__main__':main()
