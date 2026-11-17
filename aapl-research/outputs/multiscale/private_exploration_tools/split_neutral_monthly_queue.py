"""Post-hoc monthly split-neutral return comparison on the same five folds."""
from datetime import datetime,timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from multiscale.bars import eligible_origins,file_digest
from multiscale.development import FoldConfig,atomic_json,bars_identity,run_development_fold,validate_split
from multiscale.features import CloseWindows
from multiscale.matrix_executor import read_bars
from multiscale.spec import MODELS,OUT,PROFILES
from multiscale.splits import rolling_folds
from multiscale.training import digest_indices
from multiscale.verify_development import check_fold
from split_neutral import independent_neutral_return,normalized_actions,share_adjusted_bars


def save_new(path,value):
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.write('\n')


def main():
    started=time.monotonic()
    source=OUT/'admission/alpaca_20261107T052228573242Z'
    ap=source/'admission.json'
    admission=json.loads(ap.read_text())
    terms=OUT/'source_docs/alpaca_terms.txt'
    quote='you agree to use the Services and Content solely for your own personal and non-commercial purposes'
    if (quote not in ' '.join(terms.read_text().split()) or admission.get('Q3_values_read') is not False
            or admission.get('experiment_scored') is not False):
        raise ValueError('Candidate or recorded personal-use term is invalid')
    prior=OUT/'private_three_scale_development/20261107T134128870773Z'
    prior_plan=json.loads((prior/'PRIVATE_PROTOCOL.json').read_text())
    prior_state=json.loads((prior/'PRIVATE_QUEUE_STATE.json').read_text())
    diagnostic=prior/'PRIVATE_SPLIT_NEUTRAL_LABEL_DIAGNOSTIC.json'
    diag=json.loads(diagnostic.read_text())
    if (prior_plan['admission_sha256']!=file_digest(ap) or diag['status']!='private_posthoc_label_diagnostic_only'
            or prior_state['status']!='completed_private_development_only'):
        raise ValueError('Prior task and label diagnostic are not verified')
    profile=next(p for p in PROFILES if p.name=='monthly')
    lookback=max(profile.lookbacks);horizon=21
    original=read_bars(source,admission,'day')
    actions=normalized_actions(admission['split_actions'])
    adjusted,factors=share_adjusted_bars(original,actions)
    op=source/'origins_monthly_h21.npy'
    if file_digest(op)!=admission['files'][op.name]:raise ValueError('Monthly origins changed')
    origins=np.load(op,allow_pickle=False)
    if (not np.array_equal(origins,eligible_origins(original,lookback,horizon,same_session=False)) or
            not np.array_equal(origins,eligible_origins(adjusted,lookback,horizon,same_session=False))):
        raise ValueError('Adjusted and original eligible origins differ')
    old_x,_=CloseWindows(original,lookback,actions).arrays(origins)
    adjusted_x,_=CloseWindows(adjusted,lookback,[]).arrays(origins)
    if not np.allclose(old_x,adjusted_x,rtol=1e-11,atol=1e-11):
        raise ValueError('Past log-relative feature history changed unexpectedly')
    folds=rolling_folds(adjusted,origins,horizon)
    jobs=[]
    for fold in folds:
        train,fit,stop,val=validate_split(adjusted,fold,horizon)
        previous=next(j for j in prior_plan['jobs'] if (j['profile'],j['horizon'],j['target'],j['fold'])==
                      ('monthly',horizon,'future_log_return',fold['fold']))
        hashes={name:digest_indices(values) for name,values in
                [('train',train),('fit',fit),('stop',stop),('validation',val)]}
        for name,value in hashes.items():
            if previous[name+'_indices_sha256']!=value:raise ValueError('Prior task split differs')
        jobs.append({'fold':fold['fold'],'fit_origins':len(fit),'stop_origins':len(stop),
                     'validation_origins':len(val),'split_sha256':hashes,
                     'prior_fold_directory':prior_state['jobs'][previous['key']]['fold_directory']})
    if len(jobs)!=5:raise ValueError('Expected five monthly folds')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    parent=OUT/'private_split_neutral_monthly';parent.mkdir(parents=True,exist_ok=True)
    directory=parent/stamp;directory.mkdir(exist_ok=False)
    exposure=OUT/'POST_CUTOFF_EXPOSURE.md';before=exposure.read_bytes();(directory/'EXPOSURE_BEFORE.md').write_bytes(before)
    plan={'status':'private_posthoc_split_neutral_development_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'user_request':'直接做实验；并行任务并行','scope':'personal non-commercial local development',
          'admission_sha256':file_digest(ap),'alpaca_terms_sha256':file_digest(terms),
          'original_bar_sha256':bars_identity(original),'adjusted_bar_sha256':bars_identity(adjusted),
          'transformation':'raw close multiplied by cumulative split ratios effective on or before each bar',
          'target':'future_log_return on share-adjusted price; excludes cash dividends',
          'feature_rule':'past log-relative history matches original split-aware windows; anchor is share-adjusted',
          'future_actions_as_features':False,'actions':actions,'origins_sha256':file_digest(op),
          'source_protocol_sha256':file_digest(prior/'PRIVATE_PROTOCOL.json'),
          'prior_label_diagnostic_sha256':file_digest(diagnostic),
          'profile':'monthly','horizon':horizon,'lookback':lookback,'seasonal_period':252,
          'recipe':{'max_epochs':20,'patience':3,'batch_size':128,'learning_rate':0.001,'seed':2026,
                    'ridge_alphas':[0.1,1.0,10.0]},'models':list(MODELS),'jobs':jobs,
          'utility_sha256':file_digest(Path(__file__).with_name('split_neutral.py')),
          'synthetic_test_sha256':file_digest(Path(__file__).with_name('test_split_neutral.py')),
          'script_sha256':file_digest(Path(__file__)),'exposure_before_sha256':hashlib.sha256(before).hexdigest(),
          'device':'cpu','cpu_threads':2,'maximum_wall_seconds':600,
          'data_cutoff':'2026-06-30','Q3_values_read':False,'independent_test':False,
          'publication_rights':'not confirmed; keep results local',
          'limits':['Target and anchor changed after original results were observed; post-hoc exploratory comparison.',
                    'Historical observations and overlapping validation windows are not independent confirmation.',
                    'This is a split-neutral price return, not a dividend-inclusive total return.',
                    'One seed and a 20-epoch limit do not establish convergence.']}
    pp=directory/'PRIVATE_PROTOCOL.json';save_new(pp,plan);protocol_hash=file_digest(pp)
    with exposure.open('a',encoding='utf8') as f:
        f.write('\n## 月度拆股中性收益：事后开发对照\n\n'
                f'{stamp}：看到名义收益和拆股敏感性后，登记同一月度 H21 五折的拆股中性价格收益对照。'
                '每个时点只乘以当时已经生效的拆股比例，输入历史对数相对路径与原方案一致，但锚点数值改变；'
                '未来拆股仅出现在实现后的标签中，不含现金分红。结果是已曝光的本机探索，不是独立确认。'
                f'方案见 {pp}。\n')
    state={'status':'running','protocol_sha256':protocol_hash,'completed':0,'total':5,'jobs':{}}
    state_path=directory/'PRIVATE_QUEUE_STATE.json';atomic_json(state_path,state)
    print(json.dumps({'status':'split_neutral_monthly_started','directory':str(directory),'folds':5,
                      'cpu_threads':2,'maximum_seconds':600},ensure_ascii=False),flush=True)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    verified_results={};label_checks=[]
    try:
        for job in jobs:
            if time.monotonic()-started>=600:raise TimeoutError('Monthly queue reached its wall limit')
            number=job['fold'];key=f'fold_{number:02d}';folder=directory/key
            state['jobs'][key]={'status':'running'};atomic_json(state_path,state)
            recipe=plan['recipe']
            cfg=FoldConfig('monthly',horizon,'future_log_return',lookback,252,tuple(recipe['ridge_alphas']),
                           recipe['max_epochs'],recipe['patience'],recipe['batch_size'],recipe['learning_rate'],
                           recipe['seed'],protocol_hash)
            result=run_development_fold(adjusted,folds[number-1],[],cfg,folder,
                                        'private-posthoc-split-neutral:'+file_digest(ap),device='cpu')
            checked=check_fold(folder)
            if not all(checked['current_source_matches_recorded'].values()):raise ValueError('Core source changed')
            with np.load(folder/'targets.npz',allow_pickle=False) as x:
                idx=x['origins'].copy();actual=x['actual'].copy()
            direct=independent_neutral_return(original,idx,horizon,actions)
            if not np.allclose(actual,direct,rtol=1e-12,atol=1e-12):
                raise ValueError('Independent split-neutral target formula mismatch')
            vp=directory/(key+'_verification.json');save_new(vp,checked)
            label_checks.append({'fold':number,'validation_origins':len(idx),'all_path_labels_match':True,
                                 'maximum_absolute_difference':float(np.max(np.abs(actual-direct)))})
            verified_results[number]=result
            state['jobs'][key]={'status':'completed','fold_directory':str(folder),
                                'receipt_sha256':file_digest(folder/'fold_receipt.json'),
                                'verification_sha256':file_digest(vp),'validation_origins':len(idx)}
            state['completed']+=1;state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
            print(json.dumps({'stage':'split_neutral_fold_verified','fold':number,'completed':state['completed'],
                              'total':5}),flush=True)
        count=sum(j['validation_origins'] for j in jobs)
        pooled={name:sum(j['validation_origins']*verified_results[j['fold']]['models'][name]['metrics']['endpoint_mse']
                         for j in jobs)/count for name in MODELS}
        direct={name:0.0 for name in MODELS}
        for job in jobs:
            folder=directory/f"fold_{job['fold']:02d}"
            with np.load(folder/'targets.npz',allow_pickle=False) as x:truth=x['actual'][:,-1]
            for name in MODELS:
                with np.load(folder/f'prediction_{name}.npz',allow_pickle=False) as x:pred=x['prediction'][:,-1]
                direct[name]+=float(np.square(pred-truth).sum())/count
        if any(not np.isclose(pooled[m],direct[m],rtol=1e-12,atol=1e-18) for m in MODELS):
            raise ValueError('Independent pooled error differs')
        if count!=diag['origins'] or not np.isclose(pooled['last_close'],diag['split_neutral_zero_baseline_mse'],
                                                     rtol=1e-12,atol=1e-18):
            raise ValueError('Zero baseline differs from the earlier label-only diagnostic')
        baseline=pooled['last_close']
        metrics={m:{'endpoint_mse':pooled[m],'relative_improvement':1-pooled[m]/baseline if baseline else None}
                 for m in MODELS}
        summary={'status':'five_posthoc_split_neutral_development_folds_only','validation_origins':count,
                 'models':metrics,'independent_test':False,'publication_rights':'not confirmed',
                 'limitations':plan['limits']}
        result_path=directory/'PRIVATE_RESULTS.json';save_new(result_path,summary)
        save_new(directory/'PRIVATE_LABEL_AND_AGGREGATE_CHECK.json',
                 {'status':'independent_label_and_saved_prediction_check','folds':label_checks,
                  'direct_endpoint_mse':direct,'same_zero_baseline_as_prior_diagnostic':True,
                  'past_feature_history_matches_original':True,'anchor_changed':True})
        with (directory/'PRIVATE_RESULTS.csv').open('x',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['model','endpoint_mse','relative_improvement'])
            writer.writeheader()
            for name in MODELS:writer.writerow({'model':name,**metrics[name]})
        state['status']='completed_posthoc_development_only';state['elapsed_seconds']=time.monotonic()-started
        state['results_sha256']=file_digest(result_path);atomic_json(state_path,state)
        with exposure.open('a',encoding='utf8') as f:
            f.write(f'\n{stamp}：月度拆股中性收益事后对照完成五折，结果见 {result_path}；仅为开发。\n')
        print(json.dumps({'status':state['status'],'directory':str(directory),'folds':5,
                          'validation_origins':count,'elapsed_seconds':state['elapsed_seconds']},
                         ensure_ascii=False),flush=True)
    except Exception as exc:
        state['status']='failed';state['error']=f'{type(exc).__name__}: {exc}'
        state['elapsed_seconds']=time.monotonic()-started;atomic_json(state_path,state)
        raise


if __name__=='__main__':
    try:main()
    except Exception as exc:
        print(json.dumps({'status':'split_neutral_monthly_failed','error':f'{type(exc).__name__}: {exc}'},
                         ensure_ascii=False),file=sys.stderr,flush=True)
        raise
