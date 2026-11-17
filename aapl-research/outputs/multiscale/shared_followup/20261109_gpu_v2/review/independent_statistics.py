"""Independent CPU arithmetic audit. Market entry requires every completion first.

No evaluator loss/statistics function is imported. --self-test generates all
arrays internally and does not inspect the running market study.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import time
import numpy as np

BASE=Path(__file__).resolve().parents[1]
DEADLINE=1794216709.0
SEEDS=(2026,2027,2028)
HORIZONS=(5,15,30,60)
MODES=('last','last_shrink','selected','selected_shrink')
METHODS=tuple(a+'_'+m for a in ('tied','untied') for m in MODES)
CANDIDATE='tied_selected_shrink'
META={'origins','session_date','day_grid'}
STOP=False
MAX_ERROR=0.
COMPARISONS=0


def guard():
    if STOP or time.time()>=DEADLINE:raise InterruptedError('Authorized deadline or signal')


def on_signal(*_):
    global STOP
    STOP=True


def read(path):
    guard()
    return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for value in iter(lambda:f.read(4*1024*1024),b''):
            guard();h.update(value)
    return h.hexdigest()


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno());name=f.name
    os.replace(name,path)


def close(a,b,label,rtol=2e-10,atol=1e-15):
    global MAX_ERROR,COMPARISONS
    x=np.asarray(a);y=np.asarray(b)
    assert x.shape==y.shape,(label,x.shape,y.shape)
    if x.size and x.dtype.kind in 'fiu':MAX_ERROR=max(MAX_ERROR,float(np.max(np.abs(x-y))))
    COMPARISONS+=1
    assert np.allclose(x,y,rtol=rtol,atol=atol,equal_nan=False),label


def required_completion_paths(base):
    jobs=[{'id':f'{a}_f{f}_s{s}','arm':a,'fold':f,'seed':s}
        for a in ('tied','untied') for f in range(1,6) for s in SEEDS]
    required=[base/'audit/COMPLETE.json',base/'analysis/ANALYSIS_COMPLETE.json']
    required += [base/'runs'/j['id']/'DONE.json' for j in jobs]
    missing=[str(p) for p in required if not p.is_file()]
    assert not missing,{'missing_completion_paths':missing}
    return jobs


def identity_gate(base):
    """Only JSON metadata and byte hashes until the entire gate has passed."""
    guard();jobs=required_completion_paths(base)
    rules=read(base/'protocol/RULES.json');auth=read(base/'protocol/AUTHORIZATION.json')
    complete=read(base/'audit/COMPLETE.json');analysis=read(base/'analysis/ANALYSIS_COMPLETE.json')
    evaluation=read(base/'protocol/EVALUATION_FREEZE.json')
    rulehash=sha(base/'protocol/RULES.json')
    assert rules['admitted'] and rules['fixed_epochs']==20 and rules['all_cells_required']
    assert auth['authorized'] and auth['local_gpu_authorized'] and not auth['paid_resources']
    assert rules['deadline_epoch']==auth['deadline_epoch']==evaluation['deadline_epoch']==DEADLINE
    assert rules['authorization_sha256']==sha(base/'protocol/AUTHORIZATION.json')
    assert rules['plan_sha256']==sha(base/'PLAN_ZH.md')
    assert rules['cost_sha256']==sha(base/rules['cost_file'])
    assert rules['candidate']==evaluation['candidate']==CANDIDATE
    assert rules['horizons']==evaluation['horizons']==list(HORIZONS)
    assert rules['seeds']==evaluation['seeds']==list(SEEDS) and rules['folds']==[1,2,3,4,5]
    assert {(j['arm'],j['fold'],j['seed']) for j in rules['jobs']}=={(j['arm'],j['fold'],j['seed']) for j in jobs}
    assert len(rules['jobs'])==30 and len({j['id'] for j in rules['jobs']})==30
    assert complete['status']=='completed' and complete['bundles']==30 and complete['deadline_epoch']==DEADLINE
    assert analysis['status']=='completed' and analysis['complete_bundles']==30 and analysis['market_training_complete']
    assert analysis['all_five_folds_three_seeds'] and not analysis['research_goal_achieved']
    assert analysis['rules_sha256']==evaluation['rules_sha256']==rulehash
    assert analysis['evaluation_freeze_sha256']==sha(base/'protocol/EVALUATION_FREEZE.json')
    assert evaluation['evaluator_sha256']==sha(base/'code/evaluate_shared.py')
    assert evaluation['blocks']==[1,5] and evaluation['repetitions']==5000 and evaluation['statistics_seed']==2026
    assert evaluation['legacy_h60_columns']==33
    for rec in (rules,evaluation,analysis):
        assert rec['Q3_values_read'] is False and rec['independent_confirmation'] is False
    for name,digest in rules['sources'].items():assert sha(base/'code'/name)==digest
    # All evaluation and analysis input identities are checked before arrays load.
    for name,digest in evaluation['input_hashes'].items():assert sha(name)==digest,('evaluation_input',name)
    for name,digest in analysis['files'].items():
        p=base/name;assert p.resolve().is_relative_to(base.resolve())
        assert sha(p)==digest,('analysis_file',name)
    assert analysis['statistics_sha256']==sha(base/'analysis/STATS.json')
    assert analysis['results_sha256']==sha(base/'analysis/RESULTS.json')
    receipts={};paired={}
    for j in jobs:
        folder=base/'runs'/j['id'];done=read(folder/'DONE.json')
        identity=hashlib.sha256(json.dumps({'job':j,'rule':rulehash},sort_keys=True).encode()).hexdigest()
        assert done['job']==j and done['status']=='completed' and done['identity']==identity
        assert done['rules_sha256']==rulehash and done['fixed_epochs']==20
        assert done['Q3_values_read'] is False and done['independent_confirmation'] is False
        assert done['prediction_sha256']==sha(folder/'predictions.npz')
        assert set(done['checkpoint_hashes'])=={'initial.pt','last.pt','best.pt'}
        for name,digest in done['checkpoint_hashes'].items():assert sha(folder/name)==digest
        assert set(done['alphas'])=={'last','selected'}
        for a in done['alphas'].values():assert set(a)=={'5','15','30','60'} and all(0.<=float(x)<=1. for x in a.values())
        assert [r['epoch'] for r in done['history']]==list(range(1,21))
        assert done['history'][-1]['steps']==done['steps']
        scores=[r['aggregate_standardized_inner_endpoint_mse'] for r in done['history']]
        assert done['best_epoch']==int(np.argmin(scores))+1
        paired.setdefault((j['fold'],j['seed']),[]).append(done['steps'])
        receipts[j['id']]=done
    assert all(len(v)==2 and v[0]==v[1] for v in paired.values())
    return rules,evaluation,analysis,receipts


def mean_alpha(mean_path,inner,h):
    """Independent least-squares scalar, using only this H inner endpoint."""
    path=np.asarray(mean_path,dtype='float64')[:h]
    predictor=np.full(len(inner),path[-1],dtype='float64')
    denominator=float(np.dot(predictor,predictor))
    alpha=float(np.clip(np.dot(predictor,np.asarray(inner)[:,h-1])/denominator,0.,1.)) if denominator else 0.
    return path,alpha


def independent_losses(predictions,actual,mean_path,alpha,legacy=None):
    """Construct every observation loss without using evaluator arithmetic."""
    y=np.asarray(actual,dtype='float64');n,h=y.shape
    losses={};paths={};absolute={}
    def add(name,p):
        error=np.asarray(p,dtype='float64')-y
        losses[name]=np.square(error[:,-1])
        paths[name]=np.sum(np.square(error),axis=1)/h
        absolute[name]=np.abs(error[:,-1])
    add('zero',np.zeros((n,h)))
    add('fit_mean_raw',np.broadcast_to(mean_path,(n,h)))
    add('fit_mean_shrink',np.broadcast_to(np.asarray(mean_path)*alpha,(n,h)))
    for method in METHODS:
        seed_names=[];summed=np.zeros((n,h),dtype='float64')
        for seed in SEEDS:
            p=np.asarray(predictions[(method,seed)],dtype='float64')
            assert p.shape==(n,h) and np.isfinite(p).all()
            name=f'{method}_seed_{seed}';add(name,p);seed_names.append(name);summed+=p
        for output in (losses,paths,absolute):
            output[method]=sum((output[k] for k in seed_names),np.zeros(n))/3.
        add(method+'_prediction_ensemble',summed/3.)
    if legacy is not None:
        assert len(legacy)==33 and 'zero' in legacy
        assert np.array_equal(legacy['zero'],losses['zero'])
        for name,value in legacy.items():
            a=np.asarray(value);assert a.shape==(n,) and np.isfinite(a).all() and (a>=0).all()
            losses['legacy_v3b_'+name]=a.copy()
    return losses,paths,absolute


def independent_bootstrap(folds,block,repetitions=5000):
    """Independent day multiplicities applied to per-day observation sums.

    Draw order intentionally matches the fixed protocol: fold order, batches of
    100 replicates, all block starts per replicate, seed2026 freshly per block.
    Matrix multiplication by sampled-day multiplicities differs from the
    evaluator's indexed sums implementation and preserves original row weights.
    """
    names=list(folds[0]['losses']);rng=np.random.default_rng(2026)
    if any(len(f['day_grid'])<=block for f in folds):return None
    sampled_sums=np.zeros((repetitions,len(names)));sampled_n=np.zeros(repetitions)
    total=np.zeros(len(names));n_total=0
    for f in folds:
        guard();assert list(f['losses'])==names
        dates=np.asarray(f['session_date']);grid=np.asarray(f['day_grid']);d=len(grid)
        assert len(dates) and np.all(grid[1:]>grid[:-1]) and np.all(dates[1:]>=dates[:-1])
        positions=np.searchsorted(grid,dates);assert np.array_equal(grid[positions],dates)
        counts=np.zeros(d,dtype='int64');np.add.at(counts,positions,1)
        sums=np.zeros((d,len(names)))
        for k,name in enumerate(names):np.add.at(sums[:,k],positions,f['losses'][name])
        total+=sums.sum(0);n_total+=len(dates)
        for offset in range(0,repetitions,100):
            guard();r=min(100,repetitions-offset)
            starts=rng.integers(0,d-block+1,size=(r,(d+block-1)//block))
            indices=np.reshape(starts[:,:,None]+np.arange(block),(r,-1))[:,:d]
            weights=np.zeros((r,d),dtype='float64')
            np.add.at(weights,(np.arange(r)[:,None],indices),1.)
            sampled_sums[offset:offset+r]+=weights@sums
            sampled_n[offset:offset+r]+=weights@counts
    assert np.all(sampled_n>0)
    return total/n_total,sampled_sums/sampled_n[:,None]


def independent_summary(folds,h,repetitions=5000):
    names=list(folds[0]['losses']);assert len(folds)==5
    n=sum(len(f['origins']) for f in folds)
    sums={k:sum(float(np.sum(f['losses'][k])) for f in folds) for k in names}
    fold_mse={k:[float(np.mean(f['losses'][k])) for f in folds] for k in names}
    summary={k:{'origins':n,'endpoint_mse':sums[k]/n,
        'improvement_pct_over_zero':100*(1-sums[k]/sums['zero']) if sums['zero'] else None,
        'fold_mse':fold_mse[k],
        'positive_folds_over_zero':sum(x<z for x,z in zip(fold_mse[k],fold_mse['zero']))} for k in names}
    controls=['zero','fit_mean_raw','fit_mean_shrink']+['untied_'+m for m in MODES]
    if h==60:
        controls.extend(k for k in names if k.startswith('legacy_v3b_'))
        assert len(controls)==40
    pairs=[('tied_last','untied_last')]+[(CANDIDATE,c) for c in controls]
    intervals=[]
    for block in (1,5):
        generated=independent_bootstrap(folds,block,repetitions)
        for primary,reference in pairs:
            row={'primary':primary,'reference':reference,'block':block,'repetitions':repetitions,
                'supports_improvement':False}
            if generated is None:intervals.append(dict(row,status='insufficient_groups'));continue
            point,samples=generated;i=names.index(primary);j=names.index(reference)
            delta=samples[:,j]-samples[:,i];varying=bool(np.any(delta!=delta[0]))
            ci=np.quantile(delta,[.025,.975]).tolist() if varying else None
            relative=100*(1-samples[:,i]/samples[:,j]) if point[j]>0 and np.all(samples[:,j]>0) else None
            row.update(status='descriptive' if varying else 'degenerate',
                mse_reference_minus_primary=float(point[j]-point[i]),difference_ci95=ci,
                relative_improvement_pct=float(100*(1-point[i]/point[j])) if point[j]>0 else None,
                relative_ci95_pct=np.quantile(relative,[.025,.975]).tolist() if varying and relative is not None else None,
                supports_improvement=bool(ci and ci[0]>0))
            intervals.append(row)
    def direction(primary,reference):
        seed_positive={}
        for seed in SEEDS:
            a=f'{primary}_seed_{seed}'
            b=f'{reference}_seed_{seed}' if reference in METHODS else reference
            seed_positive[str(seed)]=sums[a]<sums[b]
        return {'seed_positive':seed_positive,
            'positive_folds':sum(a<b for a,b in zip(fold_mse[primary],fold_mse[reference])),
            'both_block_intervals_support_improvement':all(r['supports_improvement'] for r in intervals if
                r['primary']==primary and r['reference']==reference)}
    mechanism=direction('tied_last','untied_last')
    mechanism['passed']=all(mechanism['seed_positive'].values()) and mechanism['positive_folds']>=4 and mechanism['both_block_intervals_support_improvement']
    directions={c:direction(CANDIDATE,c) for c in controls};zero=directions['zero']
    improvement=summary[CANDIDATE]['improvement_pct_over_zero']
    checks={'three_positive_seeds_over_zero':all(zero['seed_positive'].values()),
        'four_positive_folds_over_zero':zero['positive_folds']>=4,
        'minimum_one_percent_over_zero':improvement is not None and improvement>=1.,
        'both_block_intervals_against_every_control':all(v['both_block_intervals_support_improvement'] for v in directions.values())}
    return {'horizon':h,'summary':summary,'mechanism':mechanism,'candidate_control_directions':directions,
        'candidate_checks':checks,'candidate_performance_gate_passed':all(checks.values()),
        'intervals':intervals,'primary_for_study':h==60,'independent_confirmation':False}


def compare_summary(actual,expected,label):
    assert actual['horizon']==expected['horizon'] and expected['independent_confirmation'] is False
    assert set(actual['summary'])==set(expected['summary']),label
    for name,row in actual['summary'].items():
        for key,value in row.items():
            target=expected['summary'][name][key]
            if value is None:assert target is None,(label,name,key)
            else:close(value,target,(label,name,key))
    for k in ('mechanism','candidate_control_directions','candidate_checks','candidate_performance_gate_passed','primary_for_study'):
        assert actual[k]==expected[k],(label,k)
    assert len(actual['intervals'])==len(expected['intervals'])
    for a,b in zip(actual['intervals'],expected['intervals']):
        for k,value in a.items():
            if isinstance(value,(float,list)):close(value,b[k],(label,'interval',k),rtol=2e-9,atol=2e-14)
            else:assert value==b[k],(label,'interval',k,value,b[k])


def checkpoint_metadata(base,rules,receipts):
    import torch
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    rulehash=sha(base/'protocol/RULES.json')
    for jid,done in receipts.items():
        for name in ('initial.pt','last.pt','best.pt'):
            guard();saved=torch.load(base/'runs'/jid/name,map_location='cpu',weights_only=True)
            assert saved['rules_sha256']==rulehash and saved['identity']==done['identity'] and saved['job']==done['job']
            assert saved['scaler']==done['scaler']
            assert saved['epoch']==(0 if name=='initial.pt' else 20 if name=='last.pt' else done['best_epoch'])
            assert all(t.device.type=='cpu' for t in saved['model'].values() if isinstance(t,torch.Tensor))
            del saved


def verify_fold(base,train,fold,rules,receipts):
    """Only reached after the complete metadata/hash gate succeeds."""
    guard();data=train.data_for(fold);rec=data.rec;vi=data.splits['validation'];si=data.splits['stop']
    assert data.bars.session_date[-1]<='2026-06-30'
    assert data.bars.end_ns[data.splits['fit'][-1]+60]<data.bars.end_ns[si[0]]
    assert data.bars.end_ns[si[-1]+60]<data.bars.end_ns[vi[0]]
    def labels(origins):
        ix=origins[:,None]+np.arange(1,61)
        assert data.bars.observed[origins].all() and data.bars.observed[ix].all()
        assert np.all(data.bars.session_date[origins]==data.bars.session_date[origins+60])
        return np.log(data.bars.close[ix]/data.bars.close[origins,None])
    inner=labels(si);actual=labels(vi)
    with np.load(rec['targets'],allow_pickle=False) as cache:
        assert np.array_equal(cache['origins'],vi)
        close(actual,cache['actual'],('target_cache',fold),rtol=0.,atol=0.)
    legacy_path=base.parents[1]/'mechanism_repair/20261109_v3b/analysis/paired_losses/minute_h60_future_log_return'/f'fold_{fold}.npz'
    with np.load(legacy_path,allow_pickle=False) as old:
        assert np.array_equal(old['origins'],vi)
        dates=old['session_date'].copy();grid=old['day_grid'].copy()
        legacy={k:old[k].copy() for k in old.files if k not in META}
    assert np.array_equal(dates,data.bars.session_date[vi])
    assert np.array_equal(grid,np.unique(data.bars.session_date[(data.bars.session_date>=dates[0])&(data.bars.session_date<=dates[-1])]))
    pred={h:{} for h in HORIZONS}
    expected_keys={'origins'}|{m+'_h'+str(h) for m in MODES for h in HORIZONS}
    for arm in ('tied','untied'):
        for seed in SEEDS:
            guard();jid=f'{arm}_f{fold}_s{seed}';done=receipts[jid]
            assert done['scaler']==data.scaler
            with np.load(base/'runs'/jid/'predictions.npz',allow_pickle=False) as arrays:
                assert set(arrays.files)==expected_keys and np.array_equal(arrays['origins'],vi)
                for h in HORIZONS:
                    for mode in ('last','selected'):
                        raw=arrays[f'{mode}_h{h}'].copy();shrunk=arrays[f'{mode}_shrink_h{h}'].copy()
                        assert raw.shape==shrunk.shape==(len(vi),h)
                        assert np.array_equal(shrunk,raw*float(done['alphas'][mode][str(h)]))
                        pred[h][(arm+'_'+mode,seed)]=raw
                        pred[h][(arm+'_'+mode+'_shrink',seed)]=shrunk
    folds={};fold_report=[]
    for h in HORIZONS:
        guard();task=f'minute_h{h}_future_log_return'
        mean,alpha=mean_alpha(data.scaler['y_mean'],inner,h)
        losses,path_losses,absolute=independent_losses(pred[h],actual[:,:h],mean,alpha,legacy if h==60 else None)
        receipt=read(base/'analysis/folds'/task/f'fold_{fold}.json')
        close(alpha,receipt['mean_control_alpha'],('mean_alpha',fold,h))
        for directory,recomputed,metadata in (('paired_losses',losses,META),('path_losses',path_losses,{'origins'}),('absolute_errors',absolute,{'origins'})):
            with np.load(base/'analysis'/directory/task/f'fold_{fold}.npz',allow_pickle=False) as expected:
                assert set(expected.files)-metadata==set(recomputed)
                assert np.array_equal(expected['origins'],vi)
                for name,value in recomputed.items():close(value,expected[name],(directory,fold,h,name))
        metric={r['method']:r for r in receipt['metrics']}
        assert set(metric)==set(path_losses)
        for name in path_losses:
            close(float(np.mean(losses[name])),metric[name]['endpoint_mse'],('metric_mse',fold,h,name))
            close(float(np.mean(path_losses[name])),metric[name]['path_mse'],('metric_path',fold,h,name))
            close(float(np.mean(absolute[name])),metric[name]['endpoint_mae'],('metric_mae',fold,h,name))
        folds[h]={'origins':vi.copy(),'session_date':dates,'day_grid':grid,'losses':losses}
        fold_report.append({'fold':fold,'horizon':h,'origins':len(vi),'mean_control_alpha_independently_recomputed':alpha,
            'loss_columns':len(losses),'metrics_recomputed':len(path_losses),'status':'passed'})
        del path_losses,absolute
    return folds,fold_report


def run_market(base):
    started=time.time();rules,evaluation,analysis,receipts=identity_gate(base)
    checkpoint_metadata(base,rules,receipts)
    # Import the frozen data loader only after every completion/hash guard passed.
    sys.path.insert(0,str(base/'code'));import experiment as train
    assert train.training_sources()==rules['sources']
    assert train.source_hashes()==rules['preceding_sources']
    assert train.sha(train.PRIOR/'protocol/DATA.json')==rules['data_identity']
    checkpoint=base/'review/INDEPENDENT_STATISTICS_PROGRESS.json'
    folds_by_h={h:[] for h in HORIZONS};fold_reports=[]
    for fold in range(1,6):
        guard();verified,rows=verify_fold(base,train,fold,rules,receipts)
        for h in HORIZONS:folds_by_h[h].append(verified[h])
        fold_reports+=rows
        atomic(checkpoint,{'status':'in_progress','completed_fold_checks':fold_reports,
            'rules_sha256':sha(base/'protocol/RULES.json'),'deadline_epoch':DEADLINE,
            'statistics_complete':False,'at_utc':datetime.now(timezone.utc).isoformat()})
    official=read(base/'analysis/STATS.json');official_results=read(base/'analysis/RESULTS.json');tasks={}
    assert official_results['status']=='completed' and official_results['bundles']==30 and official_results['fold_horizon_cells']==20
    assert official_results['statistics_sha256']==sha(base/'analysis/STATS.json')
    assert official_results['Q3_values_read'] is False and official_results['independent_confirmation'] is False
    expected_metric_receipts=[read(base/'analysis/folds'/f'minute_h{h}_future_log_return'/f'fold_{f}.json')
        for h in HORIZONS for f in range(1,6)]
    assert official_results['metrics']==expected_metric_receipts
    for h in HORIZONS:
        guard();task=f'minute_h{h}_future_log_return';result=independent_summary(folds_by_h[h],h)
        compare_summary(result,official['tasks'][task],task);tasks[task]=result
        del folds_by_h[h]
        atomic(checkpoint,{'status':'in_progress','completed_fold_checks':fold_reports,
            'completed_statistics_tasks':list(tasks),'rules_sha256':sha(base/'protocol/RULES.json'),
            'deadline_epoch':DEADLINE,'statistics_complete':False,'at_utc':datetime.now(timezone.utc).isoformat()})
    main=tasks['minute_h60_future_log_return']
    assert main['mechanism']['passed']==official['primary_mechanism_gate_passed']
    assert main['candidate_performance_gate_passed']==official['primary_candidate_performance_gate_passed']==analysis['predictive_gate_passed']
    assert official['research_goal_achieved'] is False and official['independent_confirmation'] is False
    result={'status':'passed','scope':'independent numerical replication of exposed historical development; not independent market confirmation',
        'rules_sha256':sha(base/'protocol/RULES.json'),'analysis_complete_sha256':sha(base/'analysis/ANALYSIS_COMPLETE.json'),
        'evaluation_freeze_sha256':sha(base/'protocol/EVALUATION_FREEZE.json'),
        'script_sha256':sha(__file__),'tasks':tasks,'fold_checks':fold_reports,
        'numerical_comparisons':COMPARISONS,'maximum_absolute_numerical_difference':MAX_ERROR,
        'model_alpha_full_stop_reinference_performed':False,
        'model_alpha_validation':'checkpoint/scaler provenance, [0,1] bounds and exact raw-times-recorded-alpha shrink relation; training-source inner-only calibration reviewed separately',
        'fit_mean_control_alpha_independently_recomputed':True,
        'bootstrap':'independent sampled-day multiplicities; block1/5; 5000 draws; RNG2026; original observation weights',
        'CPU_affinity':sorted(os.sched_getaffinity(0)),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'wall_seconds':time.time()-started,'deadline_epoch':DEADLINE,'Q3_values_read':False,
        'independent_confirmation':False,'research_goal_achieved':False,'at_utc':datetime.now(timezone.utc).isoformat()}
    atomic(base/'review/INDEPENDENT_STATISTICS_REVIEW.json',result)
    return {'status':'passed','wall_seconds':result['wall_seconds'],'comparisons':COMPARISONS,
        'maximum_absolute_difference':MAX_ERROR,'primary_mechanism_passed':main['mechanism']['passed'],
        'primary_candidate_passed':main['candidate_performance_gate_passed']}


def self_test():
    """Small generated fixture and direct original-row bootstrap oracle."""
    rng=np.random.default_rng(91);folds=[]
    for f in range(5):
        n=24;h=5;y=rng.normal(0,.01,(n,h));pred={}
        for m in METHODS:
            factor=.95 if m==CANDIDATE else .8 if m=='tied_last' else 1.4 if m=='untied_last' else .5
            for s in SEEDS:pred[(m,s)]=factor*y+rng.normal(0,.0001,(n,h))
        mean=np.full(h,.02);inner=np.zeros((16,60));inner[:,h-1]=.01
        _,alpha=mean_alpha(mean,inner,h);assert alpha==.5
        losses,paths,absolute=independent_losses(pred,y,mean,alpha)
        assert len(losses)==43
        close(losses[CANDIDATE],np.mean([(pred[(CANDIDATE,s)][:,-1]-y[:,-1])**2 for s in SEEDS],axis=0),'seed_loss_mean')
        close(losses[CANDIDATE+'_prediction_ensemble'],(np.mean([pred[(CANDIDATE,s)] for s in SEEDS],axis=0)[:,-1]-y[:,-1])**2,'prediction_ensemble')
        grid=np.array([f'2026-01-{d+1:02d}' for d in range(8)])
        folds.append({'origins':np.arange(n),'day_grid':grid,'session_date':np.repeat(grid,3),'losses':losses})
    for block in (1,5):
        repetitions=200;point,samples=independent_bootstrap(folds,block,repetitions)
        direct=np.zeros(repetitions);counts=np.zeros(repetitions);oracle=np.random.default_rng(2026)
        for f in folds:
            d=len(f['day_grid']);rows=[np.flatnonzero(f['session_date']==day) for day in f['day_grid']]
            delta=f['losses']['untied_last']-f['losses']['tied_last']
            for offset in range(0,repetitions,100):
                starts=oracle.integers(0,d-block+1,size=(100,(d+block-1)//block))
                indices=(starts[:,:,None]+np.arange(block)).reshape(100,-1)[:,:d]
                for k,days in enumerate(indices):
                    ids=np.concatenate([rows[x] for x in days]);direct[offset+k]+=delta[ids].sum();counts[offset+k]+=len(ids)
        names=list(folds[0]['losses']);computed=samples[:,names.index('untied_last')]-samples[:,names.index('tied_last')]
        close(computed,direct/counts,('original_observation_oracle',block),rtol=1e-12,atol=1e-16)
    summary=independent_summary(folds,5,repetitions=200)
    assert summary['mechanism']['passed'] and summary['candidate_performance_gate_passed']
    assert len(summary['intervals'])==16
    from copy import deepcopy
    bad=deepcopy(folds)
    for f in bad:
        f['losses'][CANDIDATE+'_seed_2026']=f['losses']['zero'].copy()
        f['losses'][CANDIDATE]=sum(f['losses'][CANDIDATE+'_seed_'+str(s)] for s in SEEDS)/3
    rejected=independent_summary(bad,5,repetitions=100)
    assert not rejected['candidate_checks']['three_positive_seeds_over_zero'] and not rejected['candidate_performance_gate_passed']
    long_folds=[]
    for f in range(5):
        n=24;y=rng.normal(0,.01,(n,60));pred={}
        for m in METHODS:
            factor=.95 if m==CANDIDATE else .8 if m=='tied_last' else 1.4 if m=='untied_last' else .5
            for s in SEEDS:pred[(m,s)]=factor*y+rng.normal(0,.0001,(n,60))
        legacy={'zero':y[:,-1]**2}
        legacy.update({f'control_{i:02d}':(2+i/10)*y[:,-1]**2+1e-9 for i in range(32)})
        losses,_,_=independent_losses(pred,y,np.full(60,.02),.5,legacy)
        long_folds.append({'origins':np.arange(n),'day_grid':folds[f]['day_grid'],
            'session_date':folds[f]['session_date'],'losses':losses})
    long_summary=independent_summary(long_folds,60,repetitions=200)
    assert len(long_summary['summary'])==76 and len(long_summary['candidate_control_directions'])==40
    assert len(long_summary['intervals'])==82 and long_summary['primary_for_study']
    assert long_summary['mechanism']['passed'] and long_summary['candidate_performance_gate_passed']
    zero_folds=deepcopy(folds)
    for f in zero_folds:
        for name in f['losses']:f['losses'][name]=np.zeros(24)
    zero_summary=independent_summary(zero_folds,5,repetitions=100)
    assert all(r['status']=='degenerate' and not r['supports_improvement'] for r in zero_summary['intervals'])
    assert not zero_summary['mechanism']['passed'] and not zero_summary['candidate_performance_gate_passed']
    with tempfile.TemporaryDirectory(prefix='aapl_independent_gate_') as folder:
        try:required_completion_paths(Path(folder))
        except AssertionError:refuses_incomplete=True
        else:refuses_incomplete=False
    assert refuses_incomplete
    return {'status':'passed','scope':'generated arrays only; running market study not inspected',
        'independent_observation_oracle_blocks':[1,5],'oracle_repetitions':200,
        'default_market_repetitions':5000,'seed_loss_vs_prediction_ensemble_separated':True,
        'mean_alpha_recomputed':True,'strict_nonpositive_seed_rejected':True,
        'H60_76_loss_columns_40_controls_82_intervals_checked':True,
        'degenerate_zero_denominator_intervals_reject_success':True,
        'completion_gate_refuses_absent_30_DONE_and_COMPLETE_and_ANALYSIS_COMPLETE':True,
        'market_arrays_opened':False,'GPU_used':False,'Q3_values_read':False,
        'script_sha256':sha(__file__),'numerical_comparisons':COMPARISONS,
        'maximum_absolute_numerical_difference':MAX_ERROR,'at_utc':datetime.now(timezone.utc).isoformat()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    assert set(os.sched_getaffinity(0)).issubset({24,26})
    for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM):signal.signal(sig,on_signal)
    try:
        if args.self_test:
            result=self_test();atomic(BASE/'review/INDEPENDENT_STATISTICS_SYNTHETIC_ACCEPTANCE.json',result)
        else:result=run_market(BASE)
        print(json.dumps(result,indent=2))
    except (InterruptedError,AssertionError) as exc:
        name='INDEPENDENT_STATISTICS_SYNTHETIC_FAILURE.json' if args.self_test else 'INDEPENDENT_STATISTICS_PAUSED_OR_FAILED.json'
        atomic(BASE/'review'/name,{'status':'paused' if isinstance(exc,InterruptedError) else 'failed',
            'reason':str(exc),'completed_progress_retained':True,'deadline_epoch':DEADLINE,
            'independent_review_complete':False,'Q3_values_read':False,'at_utc':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__':main()
