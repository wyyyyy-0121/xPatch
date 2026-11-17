"""Reproduce every full-task BN control, seed and block statistic from saved losses."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from portable_stats import paired_intervals

def close(a,b):
    if a is None or b is None:assert a is b
    else:assert np.allclose(a,b,rtol=1e-11,atol=1e-14),(a,b)

def main():
    expected=json.loads((ROOT/'CONTROL_AUDIT.json').read_text());out=[]
    for task,result in expected['tasks'].items():
        groups=[];totals={};seed_totals={str(s):{'adapted':0.,'adapted_shrink':0.} for s in (2026,2027,2028)};n=0
        for fold in range(1,6):
            with np.load(ROOT/'statistics'/task/f'fold_{fold}.npz',allow_pickle=False) as a:
                names=a['names'].tolist();groups.append({'names':names,'counts':a['counts'],'sums':a['sums']})
                origins=a['origins'];losses={name:a['loss_'+name] for name in names}
                assert sum(a['counts'])==len(origins)
                for name,value in losses.items():totals[name]=totals.get(name,0.)+float(value.sum())
                for seed in (2026,2027,2028):
                    for mode in ('adapted','adapted_shrink'):
                        seed_totals[str(seed)][mode]+=float(a[f'seed_{seed}_{mode}'].sum())
                    for mode in ('adapted','adapted_shrink'):
                        close(losses[mode],np.mean([a[f'seed_{s}_{mode}'] for s in (2026,2027,2028)],axis=0))
                n+=len(origins)
                expected_fold=result['folds'][fold-1]
                for mode in ('adapted','adapted_shrink'):
                    close(100*(1-losses[mode].mean()/losses['zero'].mean()),expected_fold[mode+'_gain_pct'])
        for name,value in totals.items():
            close(value/n,result['pooled_mse'][name]);close(100*(1-value/totals['zero']),result['improvement_over_zero_pct'][name])
        for seed,modes in seed_totals.items():
            for mode,value in modes.items():close(100*(1-value/totals['zero']),result['per_seed_improvement_over_zero_pct'][seed][mode])
        intervals=[]
        for block in ((1,5) if task.startswith('minute') else (2,5)):
            intervals+=paired_intervals(groups,[(mode,control) for mode in ('adapted','adapted_shrink') for control in result['controls']],block,task)
        assert len(intervals)==len(result['paired_intervals'])
        for row,reference in zip(intervals,result['paired_intervals']):
            for key in ('task','primary','reference','block','status','supports_improvement'):assert row[key]==reference[key]
            for key in ('mse_reference_minus_primary','difference_ci95','relative_improvement_pct','relative_ci95_pct'):close(row.get(key),reference.get(key))
        out.append({'task':task,'folds':5,'seeds':3,'pooled_controls':len(totals),'paired_intervals':len(intervals),'origins':n})
    print(json.dumps({'status':'passed','tasks':out,'independent_confirmation':False,'source_workspace_required':False}),flush=True)

if __name__=='__main__':main()
