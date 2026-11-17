from common import *
from dataclasses import asdict
from datetime import datetime,timezone
from multiscale.features import CloseWindows,fit_scaler
from multiscale.bars import targets,eligible_origins
from multiscale.splits import rolling_folds,warmup_folds
from multiscale.development import validate_split,bars_identity
from multiscale.training import digest_indices

def prepare():
    admission=read(SOURCE/'admission.json');records=reference_records();tasks=[]
    old_data=read(PREVIOUS/'protocol/DATA.json')
    for profile,(freq,lookback,horizons) in PROFILES.items():
        bars=admitted_bars(freq);windows=CloseWindows(bars,lookback,admission['split_actions'])
        folder=BASE/'cache'/profile;folder.mkdir(exist_ok=True)
        origins=eligible_origins(bars,lookback,min(horizons),same_session=freq=='minute')
        op=folder/'origins.npy';xp=folder/'x.npy'
        if not (folder/'CACHE.json').exists():
            np.save(op,origins)
            x=np.lib.format.open_memmap(xp,mode='w+',dtype='float32',shape=(len(origins),lookback))
            for i in range(0,len(origins),4096):
                a,_=windows.arrays(origins[i:i+4096]);a-=a.mean(1,keepdims=True)
                x[i:i+len(a)]=a
            x.flush();del x
            atomic(folder/'CACHE.json',{'origins_sha256':sha(op),'x_sha256':sha(xp),
                'rows':len(origins),'lookback':lookback,'information':'demeaned split-aware past log closes',
                'bars_sha256':bars_identity(bars),'Q3_values_read':False})
        else:
            c=read(folder/'CACHE.json');assert c['origins_sha256']==sha(op) and c['x_sha256']==sha(xp)
        allorig=np.load(op,mmap_mode='r');x=np.load(xp,mmap_mode='r')
        for h in horizons:
            eligible=eligible_origins(bars,lookback,h,same_session=freq=='minute')
            folds=warmup_folds(bars,eligible,h) if profile in ('annual','quarterly') else rolling_folds(bars,eligible,h)
            assert len(folds)==5
            for target in ('future_close','future_log_return'):
                task=key(profile,h,target);td=BASE/'cache'/task;td.mkdir(exist_ok=True)
                info={'task':task,'profile':profile,'frequency':freq,'lookback':lookback,'horizon':h,'target':target,'folds':{}}
                for fold in folds:
                    fn=fold['fold'];train,fit,stop,val=validate_split(bars,fold,h)
                    rr=next(r for r in records if r['profile']==profile and int(r['horizon'])==h and r['target']==target and int(r['fold'])==fn and r['model']=='mlp')
                    corefolder=Path(rr['training_summary']).parent.parent
                    if (corefolder/'fold_receipt.json').exists():
                        original=read(corefolder/'fold_receipt.json')['identity_fields']
                        expected={n:original[n+'_indices_sha256'] for n in ('train','fit','stop','validation')}
                        target_path=corefolder/'targets.npz'
                    else:
                        expected=read(corefolder/'PRIVATE_PROTOCOL.json')['split_sha256']
                        assert profile=='minute' and h==60 and target=='future_log_return' and fn==1
                        target_path=Path(old_data['folds']['1']['targets'])
                    hashes={n:digest_indices(z) for n,z in zip(('train','fit','stop','validation'),(train,fit,stop,val))}
                    assert hashes==expected,(task,fn)
                    with np.load(target_path) as z:
                        assert np.array_equal(z['origins'],val)
                        assert np.array_equal(z['actual'],targets(bars,val,h,target))
                    sp=td/f'fold_{fn}.npz';save_npz(sp,train=train,fit=fit,stop=stop,validation=val)
                    scalep=td/f'scaler_{fn}.json'
                    if not scalep.exists():
                        sums=np.zeros(h);squares=np.zeros(h);sx=ssx=0.;count=0
                        for i in range(0,len(fit),4096):
                            indices=fit[i:i+4096];a=np.asarray(x[rows_for(indices,allorig)],dtype='float64')
                            y=targets(bars,indices,h,target)
                            if target=='future_close':y-=bars.close[indices,None]
                            sx+=float(a.sum());ssx+=float((a*a).sum());count+=a.size
                            sums+=y.sum(0);squares+=(y*y).sum(0)
                        mean=sums/len(fit);std=np.sqrt(np.maximum(squares/len(fit)-mean**2,0));std=np.where(std>1e-12,std,1.)
                        xm=sx/count;xs=max(np.sqrt(max(ssx/count-xm*xm,0)),1e-12)
                        atomic(scalep,{'tiny':{'x_mean':xm,'x_std':xs,'y_mean':mean.tolist(),'y_std':std.tolist()},
                           'original':asdict(fit_scaler(windows,fit,h,target)),
                           'fitted_origin_count':len(fit),'split_sha256':hashes})
                    models={}
                    for model in ('mlp','gru','lstm','xpatch'):
                        r=next(r for r in records if r['profile']==profile and int(r['horizon'])==h and r['target']==target and int(r['fold'])==fn and r['model']==model)
                        summary=Path(r['training_summary']);assert sha(summary)==r['training_summary_sha256']
                        p=summary.parent.parent/f'prediction_{model}.npz';ck=summary.parent/'best.pt'
                        models[model]={'summary':str(summary),'summary_sha256':sha(summary),'checkpoint':str(ck),
                            'checkpoint_sha256':sha(ck),'prediction':str(p),'prediction_sha256':sha(p),
                            'identity':r['training_identity']}
                    info['folds'][str(fn)]={'split_file':str(sp),'split_sha256':hashes,
                        'counts':dict(zip(('train','fit','stop','validation'),map(len,(train,fit,stop,val)))),
                        'targets':str(target_path),'targets_sha256':sha(target_path),'models':models,
                        'scaler_file':str(scalep),'scaler_sha256':sha(scalep)}
                atomic(td/'TASK.json',info);tasks.append(info)
        print(json.dumps({'profile':profile,'cache_ready':True,'rows':len(origins)}),flush=True)
    atomic(BASE/'protocol/DATA.json',{'status':'matched_original_twenty_tasks','admission':str(SOURCE),
        'admission_sha256':sha(SOURCE/'admission.json'),'task_files':{t['task']:sha(BASE/'cache'/t['task']/'TASK.json') for t in tasks},
        'Q3_values_read':False,'independent_confirmation':False,'at_utc':datetime.now(timezone.utc).isoformat()})

if __name__=='__main__':prepare()
