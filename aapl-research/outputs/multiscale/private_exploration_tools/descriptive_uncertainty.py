"""Dependence-aware descriptive intervals for already-exposed private folds."""
from datetime import datetime,timezone
import json,time
from pathlib import Path

import numpy as np

from multiscale.aggregate_development import session_labels
from multiscale.bars import file_digest
from multiscale.evaluation import paired_block_bootstrap
from multiscale.spec import OUT,MODELS


def write_new(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')


def main():
    root=OUT/'private_three_scale_development/20261107T134128870773Z';p=json.loads((root/'PRIVATE_PROTOCOL.json').read_text());s=json.loads((root/'PRIVATE_QUEUE_STATE.json').read_text())
    minute_controls=OUT/'private_minute_full_controls/20261107T140007359046Z';minute_neural=OUT/'private_minute_full_neural/20261107T140317864464Z'
    if s.get('status')!='completed_private_development_only':raise ValueError('The three-scale development queue is incomplete')
    if json.loads((minute_neural/'PRIVATE_INDEPENDENT_CHECK.json').read_text())['status']!='private_full_minute_neural_independently_recomputed':raise ValueError('Minute first fold is not verified')
    source=OUT/'admission/alpaca_20261107T052228573242Z';ap=source/'admission.json';a=json.loads(ap.read_text())
    if file_digest(ap)!=p['admission_sha256'] or a.get('Q3_values_read') is not False:raise ValueError('Candidate identity changed')
    lengths={'hourly':[2,5],'monthly':[21,42],'quarterly':[3,6],'minute':[1,5]}
    tasks=[tuple(x) for x in p['task_order']]+[('minute',60,'future_log_return')]
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');parent=OUT/'private_uncertainty_analysis';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False)
    plan={'status':'private_posthoc_descriptive_intervals_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'tasks':tasks,'block_lengths_by_profile':lengths,'resamples':5000,'seed':2026,'confidence':0.95,
          'method':'non-circular moving blocks over original session grid; paired endpoint loss difference weighted by origins',
          'p_values_reported':False,'independent_test':False,'Q3_values_read':False,
          'source_protocol_sha256':file_digest(root/'PRIVATE_PROTOCOL.json'),'source_admission_sha256':file_digest(ap),
          'source_code_sha256':file_digest(Path(__file__)),'baseline':'last close for price, zero return for return',
          'limitations':['Development results were already observed; intervals are descriptive and do not establish independent significance.',
                         'Dependence, nonstationarity, split effects and multiple comparisons limit interpretation.',
                         'Minute task contains only one full fold.','Publication rights and source accuracy remain unconfirmed.']}
    write_new(directory/'PRIVATE_PROTOCOL.json',plan);started=time.monotonic();index=[]
    for profile,horizon,target in tasks:
        if profile=='minute':
            fold_entries=[{'target_dir':minute_controls,'control_dir':minute_controls,'neural_dir':minute_neural}];frequency='minute'
        else:
            jobs=[j for j in p['jobs'] if (j['profile'],j['horizon'],j['target'])==(profile,horizon,target)]
            if len(jobs)!=5:raise ValueError('A planned task lacks five folds')
            fold_entries=[{'target_dir':Path(s['jobs'][j['key']]['fold_directory']),
                           'control_dir':Path(s['jobs'][j['key']]['fold_directory']),
                           'neural_dir':Path(s['jobs'][j['key']]['fold_directory'])} for j in jobs]
            frequency=jobs[0]['frequency']
        labels=session_labels(source,a,frequency);groups=[];diffs={name:[] for name in MODELS};prior=None
        for entry in fold_entries:
            with np.load(entry['target_dir']/'targets.npz',allow_pickle=False) as z:
                truth=np.asarray(z['actual'][:,-1],dtype=np.float64);orig=z['origins'].copy();dec=z['decision_ns'].copy()
            if prior is not None and int(dec[0])<=prior:raise ValueError('Fold decision times overlap')
            prior=int(dec[-1]);groups.append(labels[orig]);
            with np.load(entry['control_dir']/'prediction_last_close.npz',allow_pickle=False) as z:baseline=np.asarray(z['prediction'][:,-1],dtype=np.float64)
            baseline_loss=np.square(baseline-truth)
            for name in MODELS:
                directory_for_model=entry['neural_dir'] if name in ('mlp','gru','lstm','xpatch') else entry['control_dir']
                with np.load(directory_for_model/f'prediction_{name}.npz',allow_pickle=False) as z:
                    if not np.array_equal(z['origins'],orig) or not np.array_equal(z['decision_ns'],dec):raise ValueError('Prediction origins differ')
                    pred=np.asarray(z['prediction'][:,-1],dtype=np.float64)
                diffs[name].append(np.square(pred-truth)-baseline_loss)
        groups=np.concatenate(groups);grid=np.unique(labels);grid=grid[(grid>=groups[0])&(grid<=groups[-1])]
        observed_groups=len(np.unique(groups));model_intervals={}
        for name in MODELS:
            model_intervals[name]=[];difference=np.concatenate(diffs[name])
            for length in lengths[profile]:
                if length>=len(grid) or observed_groups<2:
                    result={'block_length':length,'status':'insufficient_groups','groups':len(grid),'observed_groups':observed_groups,'confidence_interval':None}
                else:
                    result=paired_block_bootstrap(difference,groups,block_length=length,resamples=5000,seed=2026,confidence=.95,group_order=grid)
                    result.pop('centered_one_sided_p_improvement',None)
                    result['status']='descriptive_development_only' if result['bootstrap_status']=='ok' else 'degenerate_bootstrap'
                    result['interpretation']=('Descriptive paired loss difference only; negative favors the model. Not independent confirmation.'
                                              if result['bootstrap_status']=='ok' else
                                              'All resampled means are identical; no interval is reported.')
                model_intervals[name].append(result)
        result={'status':'private_descriptive_not_independent','profile':profile,'horizon':horizon,'target':target,
                'folds':len(fold_entries),'origins':len(groups),'models':model_intervals,'independent_test':False,
                'p_values_reported':False,'limits':plan['limitations']}
        path=directory/f'PRIVATE_{profile}_h{horizon}_{target}.json';write_new(path,result)
        index.append({'profile':profile,'horizon':horizon,'target':target,'path':str(path),'sha256':file_digest(path)})
        print(json.dumps({'stage':'descriptive_intervals_complete','profile':profile,'horizon':horizon,'target':target,
                          'completed_tasks':len(index),'total_tasks':len(tasks),'elapsed_seconds':round(time.monotonic()-started,1)},ensure_ascii=False),flush=True)
    result={'status':'private_descriptive_intervals_complete','tasks':index,'independent_test':False,
            'p_values_reported':False,'elapsed_seconds':time.monotonic()-started}
    write_new(directory/'PRIVATE_INDEX.json',result)
    print(json.dumps({'status':result['status'],'directory':str(directory),'tasks':len(index),
                      'elapsed_seconds':round(result['elapsed_seconds'],1)}),flush=True)


if __name__=='__main__':main()
