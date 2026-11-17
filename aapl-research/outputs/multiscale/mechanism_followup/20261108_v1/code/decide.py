from common import *
from datetime import datetime,timezone

def decide():
    cache=cache_arrays();data=read(BASE/'protocol/DATA.json');allmetrics=[];orientations=[]
    for fold in (1,2):
        parts=np.load(BASE/'cache'/f'fold_{fold}.npz');v=parts['validation'];rr=rows_for(v)
        threshold=float(np.median(cache['volatility'][rows_for(parts['fit'])]));high=cache['volatility'][rr]>threshold
        y=cache['actual'][rr,-1];m={}
        for kind in ('xpatch_noresidual','xpatch_notrend'):
            job=f'probe_{kind}_f{fold}_s2026';done=read(BASE/'runs'/job/'DONE.json');assert done['status']=='completed'
            with np.load(BASE/'runs'/job/'prediction.npz') as a:
                assert np.array_equal(a['origins'],v);p=a['prediction'][:,-1]
            m[kind]={label:float(np.mean((p[mask]-y[mask])**2)) for label,mask in [('low',~high),('high',high)]}
        dlow=m['xpatch_noresidual']['low']-m['xpatch_notrend']['low']
        dhigh=m['xpatch_noresidual']['high']-m['xpatch_notrend']['high']
        orient=1 if dlow<0<dhigh else -1 if dlow>0>dhigh else 0
        orientations.append(orient);allmetrics.append({'fold':fold,'vol_threshold':threshold,'branch_losses':m,
            'trend_minus_residual_low':dlow,'trend_minus_residual_high':dhigh,'reversal_orientation':orient})
    passed=orientations[0]!=0 and orientations[0]==orientations[1]
    pre=read(BASE/'protocol/PREDECLARATION.json');chosen=pre['if_gate_passes' if passed else 'if_gate_fails']
    for kind in ('mlp_noanchor','gru_mean','lstm_mean','xpatch_noanchor','xpatch_notrend','xpatch_noresidual'):
        base_model={'mlp_noanchor':'mlp','gru_mean':'gru','lstm_mean':'lstm'}.get(kind,'xpatch');improvements=[]
        for fold in (1,2):
            result=read(BASE/'runs'/f'probe_{kind}_f{fold}_s2026'/'DONE.json')
            with np.load(data['folds'][str(fold)]['models'][base_model]['prediction']) as a:p=a['prediction'][:,-1]
            v=np.load(BASE/'cache'/f'fold_{fold}.npz')['validation'];y=cache['actual'][rows_for(v),-1]
            original=float(np.mean((p-y)**2));new=result['metrics']['endpoint_mse']
            improvements.append(100*(1-new/original))
        allmetrics.append({'probe':kind,'improvement_vs_original_pct':improvements,'screen_supported':all(x>0 for x in improvements)})
    result={'status':'decision_frozen','at_utc':datetime.now(timezone.utc).isoformat(),'two_stream_gate_passed':passed,
        'primary':chosen['primary'],'formal_kinds':chosen['formal_kinds'],'evidence':allmetrics,
        'reason':'Prespecified complementarity screen passed' if passed else 'Prespecified complementarity screen failed; use predeclared local-plus-shrinkage revision',
        'Q3_values_read':False,'independent_confirmation':False}
    atomic(BASE/'protocol/STAGE_C_DECISION.json',result)
    jobs=read(BASE/'protocol/JOBS.json')['jobs']
    jobs += [{'id':f'formal_{kind}_f{fold}_s{seed}','phase':'formal','kind':kind,'fold':fold,'seed':seed}
             for kind in chosen['formal_kinds'] for fold in range(1,6) for seed in (2026,2027,2028)]
    atomic(BASE/'protocol/JOBS.json',{'jobs':jobs})
    print(json.dumps({'two_stream_gate_passed':passed,'primary':result['primary'],'formal_jobs':len(jobs)-12}),flush=True)
    return result

if __name__=='__main__':decide()
