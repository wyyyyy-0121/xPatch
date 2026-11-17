from common import *
import torch,time,shutil
from predictor import Predictor
from multiscale.features import CloseWindows

def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    assert len(os.sched_getaffinity(0))<=2
    jobs=read(BASE/'protocol/SELECTED_JOBS.json')['jobs'];records=[]
    old=read(BASE/'audit/PREDICTOR_REPLAY.json') if (BASE/'audit/PREDICTOR_REPLAY.json').exists() else {}
    already={r['job']:r for r in old.get('records',[]) if r.get('passed')}
    for job in jobs:
        folder=BASE/'runs'/job['id'];done=read(folder/'DONE.json')
        assert sha(folder/'best.pt')==done['checkpoint_sha256']
        assert sha(folder/'prediction.npz')==done['prediction_sha256']
        if job['id'] in already and already[job['id']]['checkpoint_sha256']==sha(folder/'best.pt'):
            records.append(already[job['id']]);continue
        task=key(job['profile'],job['horizon'],job['target']);meta=read(BASE/'cache'/task/'TASK.json');fold=meta['folds'][str(job['fold'])]
        origins=np.load(fold['split_file'])['validation']
        # Replay reads origins and saved predictions only; no target-array file is opened here.
        with np.load(folder/'prediction.npz') as a:
            assert np.array_equal(origins,a['origins']);expected=a['prediction']
        model=Predictor(folder/'best.pt');bars=admitted_bars(job['frequency'])
        allidx=np.load(BASE/'cache'/job['profile']/'origins.npy',mmap_mode='r')
        cache=np.load(BASE/'cache'/job['profile']/'x.npy',mmap_mode='r')
        windows=CloseWindows(bars,job['lookback'],read(SOURCE/'admission.json')['split_actions'])
        maximum=0.;exact=True
        for i in range(0,len(origins),128):
            idx=origins[i:i+128];anchor=bars.close[idx]
            x=np.array(cache[rows_for(idx,allidx)],copy=True) if job['family']=='tiny' else windows.arrays(idx)[0]
            p=model.predict_prepared(x,anchor);q=expected[i:i+len(idx)]
            exact &= np.array_equal(p,q);maximum=max(maximum,float(np.max(abs(p-q))))
        assert exact,(job['id'],maximum)
        records.append({'job':job['id'],'passed':True,'origins':len(origins),'steps':job['horizon'],
            'exact_same_batch_replay':exact,'maximum_absolute_error':maximum,
            'checkpoint_sha256':sha(folder/'best.pt'),'prediction_sha256':sha(folder/'prediction.npz')})
        if len(records)%10==0:
            atomic(BASE/'audit/PREDICTOR_REPLAY.json',{'status':'running','records':records,'targets_opened':False})
            print(json.dumps({'replayed':len(records),'total':len(jobs)}),flush=True)
    # Fix demonstrations by task and final fold/seed, with no score-based choice.
    examples=BASE/'delivery';(examples/'models').mkdir(parents=True,exist_ok=True)
    for profile,(_,_,hs) in PROFILES.items():
        h=max(hs);job=next(j for j in jobs if j['family']=='tiny' and j['profile']==profile and j['horizon']==h
             and j['target']=='future_log_return' and j['field']=='long' and j['loss']=='aligned' and j['fold']==5 and j['seed']==2026)
        bars=admitted_bars(job['frequency']);task=key(profile,h,job['target']);f=read(BASE/'cache'/task/'TASK.json')['folds']['5']
        origins=np.load(f['split_file'])['validation'][:2];windows=CloseWindows(bars,job['lookback'],read(SOURCE/'admission.json')['split_actions'])
        x,a=windows.arrays(origins);closes=np.exp(x)*a
        shutil.copy2(BASE/'runs'/job['id']/'best.pt',examples/'models'/f'{profile}.pt')
        p=Predictor(examples/'models'/f'{profile}.pt').predict(closes)
        save_npz(examples/f'{profile}_example.npz',closes=closes,expected=p)
    atomic(BASE/'audit/PREDICTOR_REPLAY.json',{'status':'passed','records':records,'predictors':len(records),
        'origin_predictor_pairs':sum(r['origins'] for r in records),'targets_opened':False,
        'all_exact_same_batch':True,'portable_predictor':True,'Q3_values_read':False})

if __name__=='__main__':main()
