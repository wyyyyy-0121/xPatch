"""Synthetic-only evaluator contract tests; does not load market artifacts."""
from pathlib import Path
import hashlib
import json
import os
import sys
import numpy as np

REVIEW = Path(__file__).resolve().parent
BASE = REVIEW.parent
sys.path.insert(0, str(BASE/'code'))
import evaluate_shared as ev

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '-1'
assert len(os.sched_getaffinity(0)) <= 2
SEEDS = ev.SEEDS
HORIZONS = ev.HORIZONS


def create_fold(fold, horizon, rng):
    n_days=24; per_day=3; n=n_days*per_day
    y=rng.normal(0,.01,(n,horizon))
    means=np.full(60,.2)
    stop=rng.normal(0,.01,(80,60))
    mean,alpha=ev.mean_control(means,stop,horizon)
    predictions={}
    for method in ev.METHODS:
        for seed in SEEDS:
            if method == 'tied_selected_shrink': factor=.95
            elif method == 'tied_last': factor=.8
            elif method.startswith('tied'): factor=.7
            elif method == 'untied_last': factor=1.5
            else: factor=1.25
            predictions[(method,seed)]=factor*y+rng.normal(0,.0002,(n,horizon))
    legacy=None
    if horizon==60:
        legacy={'zero':y[:,-1]**2}
        for i in range(32):
            value=(4+i/50)*y[:,-1]**2+rng.uniform(0,1e-8,n)
            legacy[f'legacy_control_{i:02d}']=value
    losses,paths,absolute=ev.fold_losses(predictions,y,mean,alpha,legacy)
    assert np.allclose(losses['tied_last'],np.mean([
        (predictions[('tied_last',s)][:,-1]-y[:,-1])**2 for s in SEEDS],axis=0))
    assert np.allclose(losses['tied_last_prediction_ensemble'],
        (np.mean([predictions[('tied_last',s)] for s in SEEDS],axis=0)[:,-1]-y[:,-1])**2)
    assert 'fit_mean_raw' in paths and 'fit_mean_shrink' in absolute
    day_grid=np.array([f'2026-01-{i+1:02d}' for i in range(n_days)])
    dates=np.repeat(day_grid,per_day)
    return {'origins':np.arange(fold*10000,fold*10000+n),'session_date':dates,
        'day_grid':day_grid,'losses':losses,'paths':paths,'absolute_errors':absolute}


rng=np.random.default_rng(20261109)
folds5=[create_fold(f,5,rng) for f in range(1,6)]
folds60=[create_fold(f,60,rng) for f in range(1,6)]
summary5=ev.summarize(folds5,5,repetitions=5000)
summary60=ev.summarize(folds60,60,repetitions=5000)
assert len(summary5['summary']) == 3+len(ev.METHODS)* (1+len(SEEDS)+1)
assert len(summary60['summary']) == 3+len(ev.METHODS)*(1+len(SEEDS)+1)+33
assert summary5['primary_for_study'] is False and summary60['primary_for_study'] is True
assert summary5['mechanism']['passed'] and summary60['mechanism']['passed']
assert summary5['candidate_performance_gate_passed'] and summary60['candidate_performance_gate_passed']
assert len(summary5['intervals'])==2*(1+3+4)
assert len(summary60['intervals'])==2*(1+3+4+33)
assert all(len(v['seed_positive'])==len(SEEDS) for v in summary60['candidate_control_directions'].values())
assert all(sum(name.startswith('legacy_v3b_') for name in f['losses'])==33 for f in folds60)

# Independent bootstrap arithmetic: enumerate original observations on each
# sampled day instead of sampling the evaluator's preaggregated sums matrix.
for block in (1,5):
    sample_rng=np.random.default_rng(2026)
    numerator=np.zeros(5000); denominator=np.zeros(5000)
    for fold in folds60:
        grid=fold['day_grid']; dates=fold['session_date']; n=len(grid)
        delta=fold['losses']['untied_last']-fold['losses']['tied_last']
        row_ids=[np.flatnonzero(dates==day) for day in grid]
        for offset in range(0,5000,100):
            starts=sample_rng.integers(0,n-block+1,size=(100,(n+block-1)//block))
            sampled=(starts[:,:,None]+np.arange(block)).reshape(100,-1)[:,:n]
            for k,days in enumerate(sampled):
                rows=np.concatenate([row_ids[d] for d in days])
                numerator[offset+k]+=delta[rows].sum()
                denominator[offset+k]+=len(rows)
    expected=np.quantile(numerator/denominator,[.025,.975])
    actual=next(row['difference_ci95'] for row in summary60['intervals']
        if row['block']==block and row['primary']=='tied_last' and row['reference']=='untied_last')
    assert np.allclose(actual,expected,rtol=1e-12,atol=1e-16)

# A seed equal to zero must fail the strict three-positive-seed requirement,
# even if the other seeds produce a favorable pooled candidate.
from copy import deepcopy
failed=deepcopy(folds5)
for fold in failed:
    fold['losses'][ev.CANDIDATE+'_seed_2026']=fold['losses']['zero'].copy()
    fold['losses'][ev.CANDIDATE]=np.mean([fold['losses'][ev.CANDIDATE+'_seed_'+str(seed)]
        for seed in SEEDS],axis=0)
negative=ev.summarize(failed,5,repetitions=100)
assert negative['candidate_checks']['three_positive_seeds_over_zero'] is False
assert negative['candidate_performance_gate_passed'] is False

# The real-input gate must reject the frozen non-admitted study before reading paths.
try:
    ev.complete_training_manifest({'admitted':False})
except AssertionError:
    refused=True
else:
    refused=False
assert refused

# Mean calibration uses each H's own inner endpoint, not an H60 scalar reused.
inner=np.zeros((8,60)); fit_mean=np.full(60,.2)
for h,alpha in zip(HORIZONS,(.1,.3,.6,.9)):
    inner[:,h-1]=.2*alpha
    _,got=ev.mean_control(fit_mean,inner,h)
    assert np.isclose(got,alpha)

# Build tiny, explicitly synthetic 30-receipt/checkpoint fixtures. This checks
# the full entry manifest independently of the market data loader.
import torch
fixture=REVIEW/'evaluation_manifest_synthetic'
suffix=1
while fixture.exists():
    fixture=REVIEW/('evaluation_manifest_synthetic_'+str(suffix));suffix+=1
fixture.mkdir()
original_base=ev.BASE; original_prior=ev.train.PRIOR; original_sources=ev.train.training_sources
try:
    ev.BASE=fixture;ev.train.PRIOR=fixture/'prior'
    (fixture/'protocol').mkdir()
    (fixture/'PLAN_ZH.md').write_text('Synthetic manifest fixture only\n')
    meta={'folds':{}};parts={}
    for fold in range(1,6):
        split=fixture/f'synthetic_split_{fold}.npz'
        np.savez(split,validation=np.arange(fold*100,fold*100+3))
        meta['folds'][str(fold)]={'split_file':str(split)}
        parts[str(fold)]={'split_npz_sha256':ev.train.sha(split)}
    mpath=ev.train.PRIOR/'cache/minute_h60_future_log_return/TASK.json'
    mpath.parent.mkdir(parents=True);mpath.write_text(json.dumps(meta))
    jobs=[{'id':f'{a}_f{f}_s{s}','arm':a,'fold':f,'seed':s}
        for a in ('tied','untied') for f in range(1,6) for s in SEEDS]
    rules={'admitted':True,'fixed_epochs':6,'jobs':jobs,'sources':{'synthetic':True},
        'plan_sha256':ev.train.sha(fixture/'PLAN_ZH.md'),'horizons':list(HORIZONS),
        'seeds':list(SEEDS),'folds':[1,2,3,4,5],'all_cells_required':True,'partitions':parts}
    ev.train.training_sources=lambda:{'synthetic':True}
    (fixture/'protocol/RULES.json').write_text(json.dumps(rules))
    rulehash=ev.train.sha(fixture/'protocol/RULES.json')
    for job in jobs:
        folder=fixture/'runs'/job['id'];folder.mkdir(parents=True)
        identity=ev.canonical_hash({'job':job,'rule':rulehash})
        hashes={}
        for name,epoch in (('initial.pt',0),('last.pt',6),('best.pt',4)):
            p=folder/name
            torch.save({'identity':identity,'rules_sha256':rulehash,'job':job,
                'scaler':{'synthetic_only':True},'epoch':epoch},p)
            hashes[name]=ev.train.sha(p)
        outputs={mode+'_h'+str(h):np.zeros((3,h)) for mode in ev.MODES for h in HORIZONS}
        np.savez(folder/'predictions.npz',origins=np.arange(job['fold']*100,job['fold']*100+3),**outputs)
        done={'status':'completed','job':job,'identity':identity,'rules_sha256':rulehash,
            'fixed_epochs':6,'Q3_values_read':False,'independent_confirmation':False,
            'prediction_sha256':ev.train.sha(folder/'predictions.npz'),'checkpoint_hashes':hashes,
            'alphas':{mode:{str(h):.5 for h in HORIZONS} for mode in ('last','selected')},
            'scaler':{'synthetic_only':True},'best_epoch':4,'steps':12,
            'history':[{'epoch':i,'steps':i*2} for i in range(1,7)]}
        (folder/'DONE.json').write_text(json.dumps(done))
    inputs,receipts=ev.complete_training_manifest(rules)
    assert len(receipts)==30 and len(inputs)==150
    donepath=fixture/'runs'/jobs[-1]['id']/'DONE.json'
    content=donepath.read_text();bad=json.loads(content);bad['steps']=13
    donepath.write_text(json.dumps(bad))
    try:ev.complete_training_manifest(rules)
    except AssertionError:reject_mismatch=True
    else:reject_mismatch=False
    donepath.write_text(content)
    assert reject_mismatch
finally:
    ev.BASE=original_base;ev.train.PRIOR=original_prior;ev.train.training_sources=original_sources

sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'code').glob('*.py')}
result={'status':'passed','scope':'synthetic losses only; no market data/targets/checkpoints opened',
    'horizons':{'5':{'summary_rows':len(summary5['summary']),'intervals':len(summary5['intervals']),
        'mechanism':summary5['mechanism'],'candidate_checks':summary5['candidate_checks']},
        '60':{'summary_rows':len(summary60['summary']),'intervals':len(summary60['intervals']),
        'control_count':len(summary60['candidate_control_directions']),
        'legacy_control_columns':33,'mechanism':summary60['mechanism'],
        'candidate_checks':summary60['candidate_checks']}},
    'per_origin_seed_loss_and_prediction_ensemble_are_separate':True,
    'all_three_seed_directions_computed_against_each_control':True,
    'independent_original_observation_bootstrap_matches_both_5000_draw_intervals':True,
    'one_nonpositive_seed_blocks_candidate_gate_even_with_favorable_pooled_loss':True,
    'refuses_nonadmitted_study_before_market_io':refused,
    'all_30_synthetic_receipt_checkpoint_prediction_manifests_verified':True,
    'rejects_mismatched_steps_in_synthetic_manifest':True,
    'each_H_mean_control_uses_its_own_inner_endpoint_alpha':True,
    'market_data_loaded':False,'outer_target_constructed':False,'Q3_values_read':False,
    'sources_sha256':sources}
(REVIEW/'SYNTHETIC_EVALUATION_REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
