"""Recompute fit means, inner calibration and saved full-path errors independently."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np

BASE=Path(__file__).resolve().parents[1]
V3=BASE.parent.with_name('20261109_v3')
sys.path.insert(0,str(V3/'code'))
from study import PRIOR,SOURCE,read,sha,atomic,now
from multiscale.matrix_executor import read_bars
from multiscale.bars import targets


def main():
    assert set(os.sched_getaffinity(0)).issubset({24,26}) and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    rules=read(BASE/'RULES.json');result=read(BASE/'audit/RESULTS.json')
    assert sha(BASE/'code/run_mean_controls.py')==rules['script_sha256']
    assert all(sha(path)==digest for path,digest in rules['input_hashes'].items())
    admission=read(SOURCE/'admission.json');cache={};checks=[];aggregates={}
    for task in rules['rules']['tasks']:
        meta=read(PRIOR/'cache'/task/'TASK.json');h=meta['horizon']
        if meta['frequency'] not in cache:
            cache[meta['frequency']]=read_bars(SOURCE,admission,meta['frequency'])
        bars=cache[meta['frequency']]
        for fold in range(1,6):
            rec=meta['folds'][str(fold)];done=read(BASE/'fits'/task/f'fold_{fold}'/'DONE.json')
            with np.load(rec['split_file'],allow_pickle=False) as a:
                fit,tail,val=(a[k].copy() for k in ('fit','stop','validation'))
            mean=np.zeros(h)
            # Deliberately differ from execution's 4096-row summation order.
            for start in range(0,len(fit),7919):
                idx=fit[start:start+7919];y=targets(bars,idx,h,meta['target'])
                if meta['target']=='future_close':y=y-bars.close[idx,None]
                mean+=y.sum(axis=0)/len(fit)
            assert np.allclose(mean,done['mean_path'],rtol=1e-12,atol=1e-12)
            btail=np.zeros((len(tail),h)) if meta['target']=='future_log_return' else np.broadcast_to(bars.close[tail,None],(len(tail),h))
            ytail=targets(bars,tail,h,meta['target']);q=(btail+np.asarray(done['mean_path'])[None,:]-btail)[:,-1];r=(ytail-btail)[:,-1]
            den=np.dot(q,q);alpha=float(np.clip(np.dot(q,r)/den,0,1)) if den else 0.
            assert np.isclose(alpha,done['alpha'],rtol=1e-11,atol=1e-11)
            b=np.zeros((len(val),h)) if meta['target']=='future_log_return' else np.broadcast_to(bars.close[val,None],(len(val),h))
            y=targets(bars,val,h,meta['target'])
            path=BASE/'predictions'/task/f'fold_{fold}.npz'
            assert sha(path)==done['prediction_sha256']
            with np.load(path,allow_pickle=False) as a:
                assert np.array_equal(a['origins'],val)
                raw=b+np.asarray(done['mean_path'])[None,:];shrink=b+done['alpha']*(raw-b)
                assert np.array_equal(raw,a['raw']) and np.array_equal(shrink,a['shrink'])
                for name,pred in [('baseline',b),('raw',raw),('shrink',shrink)]:
                    e=pred-y;squared=e*e;absolute=abs(e)
                    assert np.array_equal(squared[:,-1],a['loss_'+name])
                    assert np.array_equal(absolute[:,-1],a['absolute_'+name])
                    assert np.array_equal(squared.mean(1),a['path_loss_'+name])
                    mse=float(squared[:,-1].mean());mae=float(absolute[:,-1].mean());pmse=float(squared.mean())
                    for key,value in [('endpoint_mse',mse),('endpoint_mae',mae),('path_mse',pmse)]:
                        assert np.isclose(value,done['metrics'][name][key],rtol=1e-13,atol=1e-14)
                    accu=aggregates.setdefault((task,name),{'n':0,'squared':0.,'absolute':0.,'path_squared':0.})
                    accu['n']+=len(val);accu['squared']+=float(squared[:,-1].sum());accu['absolute']+=float(absolute[:,-1].sum());accu['path_squared']+=float(squared.sum())/h
            checks.append({'task':task,'fold':fold,'fit_mean_different_summation_order_verified':True,
                           'inner_alpha_recomputed':True,'full_paths_exact':True,'all_saved_losses_exact':True})
    for row in result['pooled']:
        if row['method'] not in ('baseline','raw','shrink'):continue
        x=aggregates[(row['task'],row['method'])]
        for key,value in [('endpoint_mse',x['squared']/x['n']),('endpoint_mae',x['absolute']/x['n']),('path_mse',x['path_squared']/x['n'])]:
            assert np.isclose(row[key],value,rtol=1e-13,atol=1e-14)
    report={'status':'passed','task_count':20,'fold_count':100,'checks':checks,
            'source_hashes_unchanged':True,'independent_fit_sum_batch_size':7919,
            'saved_prediction_and_loss_arrays_exact':True,'pooled_endpoint_and_path_metrics_recomputed':True,
            'rules_sha256':sha(BASE/'RULES.json'),'results_sha256':sha(BASE/'audit/RESULTS.json'),
            'verifier_sha256':sha(__file__),'Q3_values_read':False,'at_utc':now()}
    atomic(BASE/'audit/REPLAY.json',report)
    print(json.dumps({'status':'passed','tasks':20,'folds':100,'full_paths_exact':True}),flush=True)


if __name__=='__main__':main()
