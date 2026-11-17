"""Fixed-expert expanding-past calibration; no scored-fold labels enter fitting."""
from common import *
import copy,torch,pandas as pd
from scipy.optimize import minimize
from multiscale.features import CloseWindows
from multiscale.training import atomic_torch_save

def features(bars,origins,lookback,short,long):
    windows=CloseWindows(bars,lookback,read(SOURCE/'admission.json')['split_actions'])
    import exchange_calendars as xc
    calendar=xc.get_calendar('XNYS',start=str(bars.session_date[0]),end=str(bars.session_date[-1]))
    bounds={}
    for date in np.unique(bars.session_date[origins]):
        op=calendar.session_open(str(date)).as_unit('ns').value;cl=calendar.session_close(str(date)).as_unit('ns').value
        bounds[str(date)]=(op,cl)
    output=[]
    for i in range(0,len(origins),4096):
        idx=origins[i:i+4096];x,_=windows.arrays(idx);r=np.diff(x,axis=1)
        vs=np.sqrt(np.mean(r[:,-short:]**2,axis=1));vl=np.sqrt(np.mean(r[:,-long:]**2,axis=1))
        elapsed=np.array([(bars.end_ns[j]-bounds[str(bars.session_date[j])][0])/
            (bounds[str(bars.session_date[j])][1]-bounds[str(bars.session_date[j])][0]) for j in idx])
        a=np.column_stack((vs/np.maximum(vl,1e-8),abs(r[:,-short:].sum(1))/np.maximum(np.sqrt(short)*vl,1e-8),
            r[:,-long:].sum(1)/np.maximum(np.sqrt(long)*vl,1e-8),elapsed))
        assert np.isfinite(a).all() and ((elapsed>=0)&(elapsed<=1)).all()
        output.append(a)
    return np.concatenate(output)

def load_fold(meta,fn):
    rec=meta['folds'][str(fn)]
    with np.load(rec['targets']) as a:
        origins=a['origins'];actual=a['actual'];decision=a['decision_ns'];complete=a['target_completion_ns']
    preds=[]
    for model in ('mlp','gru','lstm','xpatch'):
        mr=rec['models'][model];assert sha(mr['prediction'])==mr['prediction_sha256']
        with np.load(mr['prediction']) as a:
            assert np.array_equal(a['origins'],origins);preds.append(a['prediction'])
    p=np.stack(preds+[np.zeros_like(actual)],axis=1)
    return origins,actual,decision,complete,p

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert len(os.sched_getaffinity(0))<=2
    config=read(BASE/'protocol/PREDECLARATION.json')['fusion'];records=[]
    for profile,h,target in MECHANISM:
        task=key(profile,h,target);meta=read(BASE/'cache'/task/'TASK.json');bars=admitted_bars(meta['frequency'])
        folds={i:load_fold(meta,i) for i in range(1,6)}
        allorig=np.concatenate([folds[i][0] for i in range(1,6)])
        feat=features(bars,allorig,meta['lookback'],*config['short_long_windows'][profile])
        fi={};offset=0
        for i in range(1,6):fi[i]=feat[offset:offset+len(folds[i][0])];offset+=len(folds[i][0])
        for fn in (3,4,5):
            out=BASE/'fusion'/task/f'fold_{fn}';out.mkdir(parents=True,exist_ok=True)
            if (out/'DONE.json').exists():records.append(read(out/'DONE.json'));continue
            vo,vy,vd,vc,vp=folds[fn]
            origins=np.concatenate([folds[i][0] for i in range(1,fn)])
            y=np.concatenate([folds[i][1][:,-1] for i in range(1,fn)])
            decision=np.concatenate([folds[i][2] for i in range(1,fn)])
            complete=np.concatenate([folds[i][3] for i in range(1,fn)])
            p=np.concatenate([folds[i][4][:,:,-1] for i in range(1,fn)])
            x=np.concatenate([fi[i] for i in range(1,fn)])
            eligible=complete<vd[0];origins,y,decision,complete,p,x=[v[eligible] for v in (origins,y,decision,complete,p,x)]
            assert len(y) and complete.max()<vd[0] and decision.max()<vd[0]
            sigma=float(y.std());sigma=sigma if sigma>1e-12 else 1.
            pp=p/sigma;yy=y/sigma
            gram=pp.T@pp/len(y);rhs=pp.T@yy/len(y)
            res=minimize(lambda w:float(w@gram@w-2*w@rhs),np.full(5,.2),
                jac=lambda w:2*(gram@w-rhs),bounds=[(0,1)]*5,
                constraints={'type':'eq','fun':lambda w:w.sum()-1,'jac':lambda w:np.ones(5)},
                method='SLSQP',options={'ftol':1e-12,'maxiter':1000})
            assert res.success and np.all(res.x>=-1e-9) and abs(res.x.sum()-1)<1e-9
            weights=np.maximum(res.x,0);weights/=weights.sum()
            save_npz(out/'constant_prediction.npz',prediction=np.einsum('neh,e->nh',vp,weights),origins=vo)
            save_npz(out/'equal_prediction.npz',prediction=vp[:,:4].mean(1),origins=vo)
            boundary=max(1,min(len(y)-1,int(len(y)*.8)));st=np.arange(boundary,len(y))
            fit=np.flatnonzero(complete[:boundary]<decision[boundary])
            gate_records=[]
            if len(fit)>=100 and len(st)>=30 and y[fit].std()>1e-12:
                xm=x[fit].mean(0);xs=x[fit].std(0);xs=np.where(xs>1e-12,xs,1.)
                scale=float(y[fit].std());xf=torch.from_numpy((x[fit]-xm)/xs);xt=torch.from_numpy((x[st]-xm)/xs)
                pf=torch.from_numpy(p[fit]/scale);pt=torch.from_numpy(p[st]/scale)
                yf=torch.from_numpy(y[fit]/scale);yt=torch.from_numpy(y[st]/scale)
                xv=torch.from_numpy((fi[fn]-xm)/xs)
                for seed in (2026,2027,2028):
                    torch.manual_seed(seed);gate=torch.nn.Linear(4,5,dtype=torch.float64)
                    torch.nn.init.normal_(gate.weight,0,.01);torch.nn.init.zeros_(gate.bias)
                    opt=torch.optim.Adam(gate.parameters(),lr=.01);best=None;best_loss=float('inf');stale=0;history=[]
                    for step in range(301):
                        with torch.inference_mode():
                            stop_loss=float(((torch.softmax(gate(xt),1)*pt).sum(1)-yt).square().mean())
                        history.append({'step':step,'stop_standardized_endpoint_mse':stop_loss})
                        if stop_loss<best_loss:
                            best_loss=stop_loss;best=copy.deepcopy(gate.state_dict());best_step=step;stale=0
                        else:stale+=1
                        if step==300 or stale>=20:break
                        opt.zero_grad(set_to_none=True);w=torch.softmax(gate(xf),1)
                        loss=((w*pf).sum(1)-yf).square().mean()+.001*(w-.2).square().sum(1).mean()
                        loss.backward();assert all(z.grad is not None and torch.isfinite(z.grad).all() for z in gate.parameters())
                        opt.step()
                    gate.load_state_dict(best)
                    with torch.inference_mode():vw=torch.softmax(gate(xv),1).numpy()
                    pred=np.einsum('neh,ne->nh',vp,vw)
                    save_npz(out/f'gate_s{seed}_prediction.npz',prediction=pred,origins=vo)
                    atomic_torch_save({'weights':gate.state_dict(),'x_mean':xm.tolist(),'x_std':xs.tolist(),
                        'seed':seed,'best_step':best_step,'experts':['mlp','gru','lstm','xpatch','zero']},out/f'gate_s{seed}.pt')
                    gate_records.append({'seed':seed,'best_step':best_step,'steps':len(history)-1,
                        'mean_weights':vw.mean(0).tolist(),'history':history,
                        'prediction_sha256':sha(out/f'gate_s{seed}_prediction.npz')})
            result={'status':'completed','task':task,'evaluation_fold':fn,'calibration_origins':len(y),
                'calibration_max_label_completion_ns':int(complete.max()),'evaluation_first_decision_ns':int(vd[0]),
                'inner_fit_origins':len(fit),'inner_stop_origins':len(st),'constant_weights':weights.tolist(),
                'constant_fit_success':bool(res.success),'gates':gate_records,'expert_seed':2026,
                'outer_labels_used_in_fitting':False,'scope':'three-fold expanding historical development replay',
                'prediction_files':{p.name:sha(p) for p in out.glob('*_prediction.npz')},'Q3_values_read':False}
            atomic(out/'DONE.json',result);records.append(result)
            print(json.dumps({'fusion_complete':task,'fold':fn}),flush=True)
    atomic(BASE/'fusion/COMPLETE.json',{'status':'completed','records':records,
        'new_expert_fits':0,'constant_fits':len(records),'gate_fits':sum(len(r['gates']) for r in records),
        'scope':'Only expert seed 2026; three gate seeds do not establish full-pipeline robustness'})

if __name__=='__main__':main()
