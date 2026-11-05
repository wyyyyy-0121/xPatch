"""F6 pre-confirmation system freeze; does not open Q2 labels or any Q3 values."""
import numpy as np

from .common import OUT,PLAN,digest,identity,read_json,write_json,utc_now,source_identity
from .evaluation import point_scores,seed_system,minimum_detectable_improvement
from .experiment import admitted,score_rows,read_prediction
from .panel import Panel
from .splits import VALIDATION
from .splits import fold
from .accounting import totals


def validation_identity(panel):
    rows=[];quarters=[]
    for quarter in VALIDATION:
        _,part=score_rows(panel,quarter);rows.append(part);quarters.extend([quarter]*len(part))
    rows=np.concatenate(rows)
    return rows,np.asarray(quarters),panel.targets['AAPL'][rows]


def recheck_seed_predictions(panel,selected):
    """Recompute the gate from exact common origins, labels and three saved seeds."""
    namespace=selected['prediction_namespace'];config=selected['config']
    path=OUT/'seed_verification'/namespace/config['candidate_id']/'predictions.npz'
    if digest(path)!=selected['prediction_sha256']:raise ValueError('Selected seed predictions changed')
    rows,quarters,target=validation_identity(panel)
    with np.load(path,allow_pickle=False) as saved:
        expected={'rows':rows,'timestamps_ns':panel.stamps[rows].as_unit('ns').asi8,'quarters':quarters,'target':target}
        for key,value in expected.items():
            if not np.array_equal(saved[key],value):raise ValueError('Selected seed result changed validation '+key)
        seeds=saved['predictions'].copy();stored_ensemble=saved['ensemble'].copy()
    report,prediction=seed_system(target,seeds)
    if not np.array_equal(prediction,stored_ensemble):raise ValueError('Saved ensemble is not the three-seed prediction mean')
    report['quarter_scores']={q:point_scores(target[quarters==q],prediction[quarters==q]) for q in VALIDATION}
    for key in ('system','individual_seeds','quarter_scores'):
        if report[key]!=selected[key]:raise ValueError('Selected score summary differs from raw predictions: '+key)
    return {**selected,**report},rows,target,prediction


def recheck_controls(panel):
    """Check final-rung rankings against their raw predictions before fixing controls."""
    single=[]
    for name in ('ridge','lightgbm','xpatch','patchtst'):
        selection=read_json(OUT/'selections'/('single_'+name+'.json'));ranked=[]
        for candidate in selection['rungs'][-1]['rankings']:
            cfg=candidate['config'];fold_scores=[]
            for quarter in VALIDATION:
                f,rows=score_rows(panel,quarter)
                details={'config':cfg,'fold':f,'seed':2026,'source':source_identity()}
                directory=OUT/'trials'/('single_'+name)/cfg['candidate_id']/quarter
                if name in ('xpatch','patchtst'):
                    details['updates']=cfg['total_updates'];directory=directory/'seed_2026'/f"updates_{cfg['total_updates']}"
                report,prediction=read_prediction(directory,rows,details)
                scores=point_scores(panel.targets['AAPL'][rows],prediction)
                if scores!=report['scores']:raise ValueError('Single-control score differs from saved raw predictions')
                fold_scores.append(scores)
            mse=sum(s['mse']*s['n'] for s in fold_scores)/sum(s['n'] for s in fold_scores)
            if not np.isclose(mse,candidate['mse'],rtol=1e-12,atol=0):raise ValueError('Single-control ranking score changed')
            ranked.append({'family':name,'config':cfg,'mse':mse})
        ranked.sort(key=lambda row:(row['mse'],row['config']['candidate_id']))
        if ranked[0]['config']!=selection['selected']:raise ValueError('Single-control selection differs from recomputed winner')
        single.append(ranked[0])
    single.sort(key=lambda row:(row['mse'],row['family']))
    selection=read_json(OUT/'selections/simple.json');ranked=[]
    for candidate in selection['rankings']:
        scores=[]
        for quarter in VALIDATION:
            f,rows=score_rows(panel,quarter);directory=OUT/'trials/simple'/str(candidate['index'])/quarter
            details=read_json(directory/'result.json')['details']
            if details['config']!=candidate['config'] or details['fold']!=f:raise ValueError('Simple-control recipe changed')
            report,prediction=read_prediction(directory,rows,details)
            score=point_scores(panel.targets['AAPL'][rows],prediction)
            if score!=report['scores']:raise ValueError('Simple-control scores differ from raw predictions')
            scores.append(score)
        mse=sum(s['mse']*s['n'] for s in scores)/sum(s['n'] for s in scores)
        if not np.isclose(mse,candidate['mse'],rtol=1e-12,atol=0):raise ValueError('Simple-control ranking score changed')
        ranked.append({**candidate,'mse':mse})
    ranked.sort(key=lambda row:(row['mse'],row['index']))
    if ranked[0]['config']!=selection['selected']['config'] or ranked[0]['index']!=selection['selected']['index']:
        raise ValueError('Simple-control selection differs from recomputed winner')
    return single,ranked[0]


def prepare():
    specification=admitted();results=read_json(OUT/'F5_complete.json');panel=Panel.load()
    if (OUT/'system_freeze.json').exists():raise ValueError('A system is already frozen; Q2 cannot select a replacement')
    selected,rows,y,prediction=recheck_seed_predictions(panel,results['selected'])
    namespace=selected['prediction_namespace'];single,best_simple=recheck_controls(panel);best_single=single[0]
    gate={'pooled_improvement_at_least_1pct':selected['system']['mse_improvement_vs_zero']>=.01,
          'beats_fixed_strongest_single':selected['system']['mse']<best_single['mse'],
          'beats_fixed_strongest_simple':selected['system']['mse']<best_simple['mse'],
          'at_least_two_validation_quarters_improve':sum(r['mse_improvement_vs_zero']>0 for r in selected['quarter_scores'].values())>=2,
          'all_three_seeds_improve':all(r['mse_improvement_vs_zero']>0 for r in selected['individual_seeds'].values())}
    y64=y.astype(np.float64);zero_delta=(prediction-y64)**2-y64*y64
    power=minimum_detectable_improvement(zero_delta,y64,panel.dates[rows])
    value={'frozen_at_utc':utc_now(),'source_identity':source_identity(),'plan_sha256':digest(PLAN),
           'search_manifest_sha256':digest(OUT/'search_manifest.json'),'F5_sha256':digest(OUT/'F5_complete.json'),
           'selected_system':selected,'selected_namespace':namespace,'fixed_strongest_single':best_single,
           'fixed_strongest_simple':best_simple,'all_single_controls':single,
           'development_gate':gate,'development_gate_passed':all(gate.values()),
           'Q2_status':'known process confirmation; prior CSV access/calibration; one fixed system only',
           'Q3_access_authorized_by_this_freeze':False,'final_horizon_minutes':12,'primary_metric':'raw MSE',
           'final_fit_policy':{'artifact':'deployment/pre_Q2','fit_end':fold(panel.dates,'2026Q2',confirmation=True)['fit_end'],
                               'retrain_after_Q2':False,'reason':'Q2 confirms the exact already-fitted system; Q3 evaluates that same artifact'},
           'statistics':{'block_days':5,'repeats':5000,'null_centered':True,'holm_controls':['zero','single','simple'],
                        'sensitivities':[1,10],'relative_improvement_threshold':.01,'alpha':.05,'minimum_months_improved':2},
           'minimum_detectable_improvement':power,'usage_at_freeze':totals(),
           'stop_rule':'If development or one Q2 confirmation fails, keep Q3 sealed and deliver the runnable frozen system and negative evidence'}
    value['identity']=identity(value);write_json(OUT/'system_freeze.json',value)
    return value


if __name__=='__main__':
    result=prepare();print({'development_gate_passed':result['development_gate_passed'],'checks':result['development_gate']})
