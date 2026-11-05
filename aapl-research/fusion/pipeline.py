"""Bounded, resumable OOF -> fusion selection -> one Q2 confirmation pipeline."""
import argparse
import copy
import gc
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aapl.common import ROOT, digest, identity, read_json, runtime, utc_now, write_json
from aapl.data import Market
from aapl.models import chronos_pipeline, chronos_predict
from aapl.resources import ResourceMonitor
from aapl.runner import require_lock
from aapl.statistics import paired_blocks, holm
from .config import OUT, EXPERTS, RESEARCH_ID, SEEDS, source_hash, candidate_grid, selected_experts
from .experts import expert_prediction
from .features import price_features, context_tensor, encode_context, representation
from . import head
from .accounting import Guard


def progress(stage, **details):
    write_json(OUT / 'progress.json', dict(stage=stage, updated_at_utc=utc_now(), **details))


def load_market():
    # This loader is confined to the old development CSV, ending 2026-06-30.
    protocol = require_lock()
    return Market.load(protocol)


def slice_data(data, mask):
    return {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == len(data['y'])}


def join_data(parts):
    keys = ('origins', 'dates', 'y', 'experts', 'features', 'embedding', 'volatility')
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}


def dates_for(market, quarter):
    q = pd.Period(quarter, freq='Q')
    dates = np.asarray(sorted(set(market.dates)))
    evaluation = dates[(dates >= str(q.start_time.date())) & (dates <= str(q.end_time.date()))]
    prior = dates[dates < evaluation[0]]
    if len(prior) < 100:
        raise ValueError('Insufficient expanding expert history')
    return prior[:-1].tolist(), prior[-1:].tolist(), evaluation.tolist()


def assert_protocol():
    p = read_json(OUT / 'protocol.json')
    if p['source_sha256'] != source_hash():
        raise ValueError('Fusion implementation changed after lock')
    if p['previous_protocol_sha256'] != digest(ROOT / 'protocol_locked.json'):
        raise ValueError('Previous research identity changed')
    if p['candidate_grid'] != candidate_grid() or p['expert_selections'] != selected_experts():
        raise ValueError('Fusion search or expert choices changed')
    require_lock()
    return p


def admission(market):
    """Use 2021/2022 only; no fusion validation or confirmation scores."""
    if (OUT / 'protocol.json').exists():
        return assert_protocol()
    progress('admission')
    first = market.select(sorted(set(market.dates[(market.dates >= '2022-01-01') & (market.dates <= '2022-12-31')])))
    origins = first[np.linspace(0, len(first)-1, 8, dtype=int)]
    checks = {}
    with ResourceMonitor() as monitor:
        started = time.perf_counter()
        pipe, receipt = chronos_pipeline(device='cuda')
        contexts = context_tensor(market, origins)
        z, projection = encode_context(pipe, contexts)
        zs = np.concatenate([encode_context(pipe, context[None], projection)[0] for context in contexts])
        checks['encoder_batch_isolation'] = bool(np.allclose(z, zs, rtol=1e-4, atol=1e-4))
        reverse = encode_context(pipe, contexts[::-1].copy(), projection)[0][::-1]
        checks['encoder_order_invariance'] = bool(np.allclose(z, reverse, rtol=1e-4, atol=1e-4))
        features, volatility, names = price_features(market, origins)
        # Change future prices, volume and returns of an observed origin.
        clone = copy.deepcopy(market)
        origin = origins[0]
        clone.bars.iloc[origin+1:, :4] *= 1.7
        clone.bars.iloc[origin+1:, 4] *= 11
        clone.returns[origin+1:] += .1
        changed = price_features(clone, origins[:1])[0]
        checks['price_features_future_invariance'] = bool(np.array_equal(changed, features[:1]))
        checks['foundation_context_future_invariance'] = bool(np.array_equal(context_tensor(clone, origins[:1]), contexts[:1]))
        torch.cuda.synchronize()
        timings = []
        for _ in range(10):
            t = time.perf_counter();encode_context(pipe, contexts[:1], projection);torch.cuda.synchronize()
            timings.append(time.perf_counter()-t)
        t=time.perf_counter();representation(pipe, market, first[:1024], projection=projection);torch.cuda.synchronize()
        per_origin=(time.perf_counter()-t)/1024
        checks['encoder_latency_below_minute'] = max(timings) < 60
        n=len(first[:1024]);rng=np.random.default_rng(20261105)
        smoke={'features':np.tile(features,(128,1)), 'embedding':np.tile(z,(128,1)),
               'experts':rng.normal(0,.0001,(1024,len(EXPERTS))), 'volatility':np.tile(volatility,128),
               'y':market.y(first[:1024])}
        # Serialization and deterministic training for both kinds of fusion layer.
        for kind in ('ridge','gate'):
            cfg=next(c for c in candidate_grid() if c['kind']==kind and c['basis']=='foundation')
            state=head.fit(cfg,smoke,2026);pred=head.predict(state,smoke)
            path=OUT/f'admission/{kind}.pt';head.save(path,state)
            checks[kind+'_serialization_exact']=bool(np.array_equal(pred,head.predict(head.load(path),smoke)))
            checks[kind+'_seed_repeatable']=bool(np.array_equal(pred,head.predict(head.fit(cfg,smoke,2026),smoke)))
        monitor.check()
        resources=monitor.summary()
        del pipe,clone
        gc.collect();torch.cuda.empty_cache()
    report={'passed':all(checks.values()),'completed_at_utc':utc_now(),'checks':checks,
            'resources':resources,'chronos_weight_identity':receipt,'feature_names':names,
            'encoder_projection_shape':list(projection.shape),
            'encoder_seconds_per_origin':per_origin,'encoder_single_origin_p95_seconds':float(np.quantile(timings,.95)),
            'admission_wall_seconds':time.perf_counter()-started,'scoring_period':'2021-2022 only'}
    write_json(OUT/'admission.json',report)
    if not report['passed']:
        raise RuntimeError('Fusion admission failed; evidence retained')
    np.save(OUT/'projection.npy',projection)
    protocol={'research_id':RESEARCH_ID,'locked_at_utc':utc_now(),'source_sha256':source_hash(),
              'previous_protocol_sha256':digest(ROOT/'protocol_locked.json'),
              'data_sha256':market.audit['source_sha256'],'candidate_grid':candidate_grid(),
              'expert_selections':selected_experts(),'expert_seeds':[2026],'fusion_seeds':list(SEEDS),
              'projection_sha256':digest(OUT/'projection.npy'),'admission_sha256':digest(OUT/'admission.json'),
              'oof_quarters':[str(p) for p in pd.period_range('2023Q1','2026Q1',freq='Q')],
              'validation_quarters':['2025Q1','2025Q3','2026Q1'],'confirmation_quarter':'2026Q2',
              'holdout_access':'not_implemented_in_development_pipeline',
              'budget':{'wall_hours_max':8,'gpu_hours_max':8,'artifacts_gib_max':20},
              'selection':'pooled forward validation mean per-seed MSE; main candidates require foundation representation',
              'qualification':{'relative_gain_vs_zero_at_least':.01,'relative_gain_vs_strongest_control_at_least':.01,
                               'holm_one_sided_p_below':.05,'months_improved_at_least':2,'all_seeds_improved':True},
              'primary_aggregation':'mean of per-seed losses; never ensemble MSE substitution'}
    write_json(OUT/'protocol.json',protocol)
    return protocol


def build_quarter(market, protocol, quarter):
    directory=OUT/'oof'/quarter;directory.mkdir(parents=True,exist_ok=True)
    path=directory/'data.npz';manifest_path=directory/'manifest.json'
    if manifest_path.exists():
        receipt=read_json(manifest_path)
        if receipt['data_sha256']!=digest(path) or receipt['protocol_sha256']!=digest(OUT/'protocol.json'):
            raise ValueError('Cached OOF data changed')
        return
    fit_dates,purge,evaluation=dates_for(market,quarter)
    origins=market.select(evaluation);features,volatility,names=price_features(market,origins)
    predictions=[];started=time.perf_counter()
    for name in EXPERTS[:-1]:
        progress('expert_oof',quarter=quarter,expert=name)
        expert_dir=directory/name;cached=expert_dir/'fusion_prediction.npz';meta=expert_dir/'fusion_receipt.json'
        if meta.exists():
            r=read_json(meta)
            if r['prediction_sha256']!=digest(cached) or r['fit_dates']!=fit_dates or r['evaluation_dates']!=evaluation:
                raise ValueError('OOF expert cached identity changed')
            with np.load(cached) as saved:
                np.testing.assert_array_equal(saved['origins'],origins);pred=saved['prediction'].copy()
        else:
            with ResourceMonitor() as monitor:
                pred=expert_prediction(name,protocol['expert_selections'][name],market,fit_dates,evaluation,expert_dir)
                monitor.check();resources=monitor.summary()
            np.savez_compressed(cached,origins=origins,prediction=pred)
            write_json(meta,{'prediction_sha256':digest(cached),'fit_dates':fit_dates,
                            'purge_dates':purge,'evaluation_dates':evaluation,'resources':resources})
        predictions.append(pred)
    progress('foundation_oof',quarter=quarter)
    projection=np.load(OUT/'projection.npy')
    if digest(OUT/'projection.npy')!=protocol['projection_sha256']:
        raise ValueError('Foundation projection changed')
    with ResourceMonitor() as monitor:
        pipe,_=chronos_pipeline(device='cuda')
        config=protocol['expert_selections']['chronos2_frozen']['trial']
        predictions.append(chronos_predict(pipe,market,origins,config))
        embedding,_=representation(pipe,market,origins,projection=projection)
        monitor.check();resources=monitor.summary()
        del pipe
        gc.collect();torch.cuda.empty_cache()
    experts=np.column_stack(predictions)
    if not np.isfinite(experts).all():
        raise ValueError('Invalid OOF expert prediction')
    np.savez_compressed(path,origins=origins,dates=market.dates[origins],y=market.y(origins),
                        experts=experts,features=features,volatility=volatility,embedding=embedding)
    write_json(manifest_path,{'quarter':quarter,'data_sha256':digest(path),
               'protocol_sha256':digest(OUT/'protocol.json'),'fit_dates':fit_dates,'purge_dates':purge,
               'evaluation_dates':evaluation,'experts':list(EXPERTS),'features':names,
               'origins':len(origins),'resources':resources,'wall_seconds':time.perf_counter()-started,
               'finished_at_utc':utc_now()})


def load_oof(quarters):
    parts=[]
    for quarter in quarters:
        directory=OUT/'oof'/quarter;path=directory/'data.npz'
        if read_json(directory/'manifest.json')['data_sha256']!=digest(path):
            raise ValueError('OOF artifact identity mismatch')
        with np.load(path) as saved:
            parts.append({k:saved[k].copy() for k in saved.files})
    return join_data(parts)


def forward_blocks(data, protocol):
    blocks=[]
    all_dates=np.asarray(sorted(set(data['dates'])))
    for quarter in protocol['validation_quarters']:
        q=pd.Period(quarter,freq='Q');start=str(q.start_time.date());end=str(q.end_time.date())
        previous=all_dates[all_dates<start]
        train=slice_data(data,data['dates']<previous[-1])
        val=slice_data(data,(data['dates']>=start)&(data['dates']<=end))
        if len(train['y'])<1000 or len(val['y'])<1000:
            raise ValueError('Insufficient forward fusion samples')
        blocks.append((quarter,train,val))
    return blocks


def evaluate_candidate(config,blocks,seeds):
    losses=[];seed_values=[];details=[]
    for seed in seeds:
        values=[]
        for quarter,train,val in blocks:
            state=head.fit(config,train,seed);pred=head.predict(state,val)
            values.extend((pred-val['y'])**2)
            directory=OUT/'search'/config['candidate_id']/f'seed{seed}'/quarter
            directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory/'predictions.npz',origins=val['origins'],prediction=pred)
            details.append({'quarter':quarter,'seed':seed,'mse':float(np.mean((pred-val['y'])**2)),
                            'train_max_date':str(max(train['dates'])),'validation_min_date':str(min(val['dates'])),
                            'prediction_sha256':digest(directory/'predictions.npz')})
        seed_values.append(float(np.mean(values)))
    return {'config':config,'mse':float(np.mean(seed_values)),'per_seed_mse':seed_values,'blocks':details}


def train(protocol):
    destination=OUT/'selection.json'
    if destination.exists():
        return read_json(destination)
    data=load_oof(protocol['oof_quarters']);blocks=forward_blocks(data,protocol)
    ranking=[]
    for i,config in enumerate(protocol['candidate_grid']):
        progress('fusion_search',completed=i,total=len(protocol['candidate_grid']),candidate=config['candidate_id'])
        path=OUT/'search'/config['candidate_id']/'initial.json'
        if path.exists():row=read_json(path)
        else:
            row=evaluate_candidate(config,blocks,[2026]);write_json(path,row)
        ranking.append(row)
    ranking.sort(key=lambda r:(r['mse'],r['config']['candidate_id']))
    main=[r for r in ranking if r['config']['basis']=='foundation'][:2]
    controls={basis:min((r for r in ranking if r['config']['basis']==basis),key=lambda r:r['mse'])
              for basis in ('without_foundation','static')}
    reviewed=[]
    for i,row in enumerate(main):
        progress('fusion_seed_review',candidate=i+1,total=len(main))
        reviewed.append(evaluate_candidate(row['config'],blocks,SEEDS))
    reviewed.sort(key=lambda r:(r['mse'],r['config']['candidate_id']))
    zeros=np.concatenate([v['y'] for _,_,v in blocks]);zero=float(np.mean(zeros**2))
    selected={'selected_at_utc':utc_now(),'protocol_sha256':digest(OUT/'protocol.json'),
              'validation_zero_mse':zero,'main':reviewed[0],'reviewed':reviewed,'controls':controls,
              'ranking':ranking,'selection_sha256':identity(reviewed[0]),'confirmation_scores_read':False}
    write_json(destination,selected)
    return selected


def confirmation(protocol,selection):
    path=OUT/'confirmation.json'
    if path.exists():
        saved=read_json(path)
        if saved['selection_file_sha256']!=digest(OUT/'selection.json'):
            raise ValueError('Completed confirmation selection changed')
        return saved
    data=load_oof(protocol['oof_quarters']);val=load_oof([protocol['confirmation_quarter']])
    last=max(data['dates']);data=slice_data(data,data['dates']<last)
    preds=[]
    for seed in SEEDS:
        state=head.fit(selection['main']['config'],data,seed)
        directory=OUT/'confirmation'/f'seed{seed}'
        head.save(directory/'head.pt',state)
        pred=head.predict(state,val);preds.append(pred)
        np.savez_compressed(directory/'predictions.npz',origins=val['origins'],prediction=pred)
    y=val['y'];loss=np.mean((np.stack(preds)-y)**2,axis=0)
    controls={'zero':np.zeros_like(y)}
    controls.update({name:val['experts'][:,j] for j,name in enumerate(EXPERTS)})
    controls['equal_four_supervised']=val['experts'][:,:4].mean(axis=1)
    controls['equal_all_five']=val['experts'].mean(axis=1)
    for basis,row in selection['controls'].items():
        state=head.fit(row['config'],data,2026);head.save(OUT/'confirmation'/basis/'head.pt',state)
        controls[basis]=head.predict(state,val)
    mse={name:float(np.mean((p-y)**2)) for name,p in controls.items()}
    # Test against every declared control, avoiding an uncorrected choice of best control.
    stats={name:paired_blocks(loss-(p-y)**2,val['dates']) for name,p in controls.items()}
    adjusted=holm({name:s['one_sided_p'] for name,s in stats.items()})
    for name in stats:stats[name]['holm_p']=adjusted[name]
    primary_mse=float(np.mean(loss));best=min(mse,key=mse.get)
    monthly={str(month):1-float(np.mean(loss[np.char.startswith(val['dates'],str(month))]))/
                float(np.mean(y[np.char.startswith(val['dates'],str(month))]**2))
             for month in ('2026-04','2026-05','2026-06')}
    gains=1-np.mean((np.stack(preds)-y)**2,axis=1)/mse['zero']
    checks={'validation_improves_zero':selection['main']['mse']<selection['validation_zero_mse'],
            'confirmation_gain_vs_zero_at_least1pct':1-primary_mse/mse['zero']>=.01,
            'confirmation_gain_vs_best_control_at_least1pct':1-primary_mse/mse[best]>=.01,
            'holm_significant_against_all_controls':all(p<.05 for p in adjusted.values()),
            'at_least_two_months_improve':sum(g>0 for g in monthly.values())>=2,
            'all_head_seeds_improve_zero':bool(np.all(gains>0)),
            'no_technical_failure':not list(OUT.rglob('failure.json'))}
    report={'completed_at_utc':utc_now(),'protocol_sha256':digest(OUT/'protocol.json'),
            'selection_file_sha256':digest(OUT/'selection.json'),'mse':primary_mse,
            'zero_mse':mse['zero'],'relative_gain_vs_zero':1-primary_mse/mse['zero'],
            'controls_mse':mse,'strongest_control':best,'gain_vs_strongest_control':1-primary_mse/mse[best],
            'per_seed_gain_vs_zero':gains.tolist(),'monthly_gain_vs_zero':monthly,
            'paired_tests':stats,'qualification_checks':checks,'qualified':all(checks.values()),
            'origins':len(y),'days':len(set(val['dates'])),'holdout_accessed':False,
            'training_max_date':str(max(data['dates'])),'evaluation_min_date':str(min(val['dates']))}
    write_json(path,report)
    return report


def run(stage):
    runtime();OUT.mkdir(parents=True,exist_ok=True)
    with Guard():
        return run_guarded(stage)


def run_guarded(stage):
    started=time.time();market=load_market()
    try:
        protocol=admission(market) if stage in ('admit','all') else assert_protocol()
        if stage=='admit':return
        for quarter in protocol['oof_quarters']:
            if time.time()-started>8*3600:raise RuntimeError('Fusion wall budget exceeded')
            build_quarter(market,protocol,quarter)
        if stage=='build':return
        selection=train(protocol)
        if stage=='train':return
        build_quarter(market,protocol,protocol['confirmation_quarter'])
        result=confirmation(protocol,selection)
        from .inference import package
        package(protocol,selection,result)
        from .report import report
        report(protocol,selection,result)
        progress('complete',qualified=result['qualified'],report='FUSION_RESULTS.md',holdout_accessed=False)
    except Exception as exc:
        write_json(OUT/'failure.json',{'failed_at_utc':utc_now(),'error':repr(exc),
                                      'source_sha256':source_hash(),'artifacts_preserved':True})
        progress('failed',error=repr(exc))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['admit','build','train','confirm','all'],default='admit')
    run(parser.parse_args().stage)
