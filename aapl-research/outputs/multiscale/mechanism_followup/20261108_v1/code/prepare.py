from common import *
from datetime import datetime,timezone
import csv,shutil
from multiscale.features import CloseWindows
from multiscale.bars import targets
from multiscale.training import digest_indices
from multiscale.development import validate_split,bars_identity

def main():
    bars,origins,folds,admission=bars_and_folds()
    windows=CloseWindows(bars,480,admission['split_actions'])
    previous=ROOT/'outputs/multiscale/completed_core_evidence/20261108_parallel8_continuation'
    with (previous/'results/TRAINING_MODELS.csv').open() as f:records=list(csv.DictReader(f))
    data={'at_utc':datetime.now(timezone.utc).isoformat(),'bars_sha256':bars_identity(bars),
          'admission_sha256':sha(SOURCE/'admission.json'),'Q3_values_read':False,'independent_confirmation':False,
          'folds':{},'source_files':admission['files']}
    for f in folds:
        number=f['fold'];parts=validate_split(bars,f,60)
        d={'split_sha256':{k:digest_indices(v) for k,v in zip(('train','fit','stop','validation'),parts)},
           'counts':{k:len(v) for k,v in zip(('train','fit','stop','validation'),parts)},'models':{}}
        for m in ('mlp','gru','lstm','xpatch'):
            r=next(r for r in records if r['profile']=='minute' and r['horizon']=='60' and
                   r['target']=='future_log_return' and r['fold']==str(number) and r['model']==m)
            summary=Path(r['training_summary']);assert sha(summary)==r['training_summary_sha256']
            folder=summary.parent.parent;pred=folder/f'prediction_{m}.npz';checkpoint=summary.parent/'best.pt'
            d['models'][m]={'summary':str(summary),'summary_sha256':sha(summary),'prediction':str(pred),
                'prediction_sha256':sha(pred),'checkpoint':str(checkpoint),'checkpoint_sha256':sha(checkpoint),
                'identity':r['training_identity']}
            with np.load(pred,allow_pickle=False) as a:assert np.array_equal(a['origins'],parts[3])
        if number==1:
            old=read(Path(d['models']['mlp']['summary']).parent.parent/'PRIVATE_PROTOCOL.json')
            assert d['split_sha256']==old['split_sha256']
            d['targets']=str(ROOT/'outputs/multiscale/private_minute_full_controls/20261107T140007359046Z/targets.npz')
        else:
            folder=Path(d['models']['mlp']['summary']).parent.parent
            rec=read(folder/'fold_receipt.json')['identity_fields']
            for k,v in d['split_sha256'].items():assert v==rec[k+'_indices_sha256']
            d['targets']=str(folder/'targets.npz')
        with np.load(d['targets'],allow_pickle=False) as a:
            assert np.array_equal(a['origins'],parts[3])
            assert np.array_equal(a['actual'],targets(bars,parts[3],60,'future_log_return'))
        data['folds'][str(number)]=d
        np.savez(BASE/'cache'/f'fold_{number}.npz',**dict(zip(('train','fit','stop','validation'),parts)))
    atomic(BASE/'protocol/DATA.json',data)
    n=len(origins);np.save(BASE/'cache/origins.npy',origins)
    arrays={k:np.lib.format.open_memmap(BASE/'cache'/f'{k}.npy',mode='w+',dtype=d,shape=shape)
        for k,d,shape in [('trend','float32',(n,4)),('residual','float32',(n,480)),('gate','float32',(n,4)),
                          ('volatility','float64',(n,)),('actual','float64',(n,60))]}
    for start in range(0,n,4096):
        idx=origins[start:start+4096];x,_=windows.arrays(idx)
        tr,re,ga,vo=small_features(x)
        for k,v in zip(('trend','residual','gate','volatility'),(tr,re,ga,vo)):arrays[k][start:start+len(idx)]=v
        arrays['actual'][start:start+len(idx)]=targets(bars,idx,60,'future_log_return')
    for a in arrays.values():a.flush()
    atomic(BASE/'protocol/CACHE.json',{'status':'complete','rows':n,'lookback':480,'horizon':60,
        'files':{p.name:sha(p) for p in (BASE/'cache').glob('*')},'Q3_values_read':False,
        'inputs':'only supplied past 480 closes; EMA initialized at first supplied close',
        'y_access':'training labels and already-exposed development scores only; no post-cutoff file opened'})
    print(json.dumps({'status':'cache_and_folds_ready','origins':n,'fold_counts':{k:v['counts'] for k,v in data['folds'].items()}}))

if __name__=='__main__':main()
