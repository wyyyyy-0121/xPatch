"""Raw-return scores, exact seed ensembles, and prespecified paired day-block inference."""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .common import OUT,digest,write_json


def point_scores(target,prediction):
    target=np.asarray(target,dtype=np.float64);prediction=np.asarray(prediction,dtype=np.float64)
    if target.shape!=prediction.shape or target.ndim!=1 or not len(target):raise ValueError('Scores require equal nonempty vectors')
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():raise ValueError('Missing predictions cannot be dropped')
    error=prediction-target;zero=float(np.mean(target*target));mse=float(np.mean(error*error))
    active=(prediction!=0)&(target!=0);centered=prediction-prediction.mean();variance=np.mean(centered*centered)
    slope=float(np.mean(centered*(target-target.mean()))/variance) if variance>1e-20 else None
    return {'n':len(target),'mse':mse,'zero_mse':zero,'mse_improvement_vs_zero':1-mse/zero if zero else None,
            'mae':float(np.mean(np.abs(error))),
            'direction_hit_excluding_abstentions':float(np.mean(np.sign(prediction[active])==np.sign(target[active]))) if active.any() else None,
            'direction_coverage':float(active.mean()),'direction_zero_policy':'zero forecasts abstain; zero realized returns excluded',
            'calibration_slope':slope,'calibration_intercept':float(target.mean()-slope*prediction.mean()) if slope is not None else None,
            'mean_prediction':float(prediction.mean()),'mean_realized':float(target.mean())}


def seed_system(target,predictions):
    values=np.asarray(predictions,dtype=np.float64)
    if values.shape!=(3,len(target)):raise ValueError('Exactly seeds 2026/2027/2028 on identical origins are required')
    ensemble=values.mean(0)
    return {'primary_aggregation':'MSE of the three-seed mean prediction',
            'system':point_scores(target,ensemble),
            'individual_seeds':{str(seed):point_scores(target,values[i]) for i,seed in enumerate((2026,2027,2028))}},ensemble


def paired_blocks(delta,dates,repeats=5000,block=5,seed=20261105):
    delta=np.asarray(delta,dtype=np.float64);dates=np.asarray(dates,dtype=str)
    if delta.shape!=dates.shape or not np.isfinite(delta).all():raise ValueError('Incomplete paired losses')
    days,index=np.unique(dates,return_inverse=True);n=len(days)
    if n<2*block:raise ValueError('At least two full day blocks required')
    sums=np.bincount(index,weights=delta);counts=np.bincount(index);observed=float(delta.mean())
    null_sums=sums-observed*counts;rng=np.random.default_rng(seed)
    bootstrap=np.empty(repeats);null=np.empty(repeats);blocks=int(np.ceil(n/block))
    # Noncircular moving blocks do not join the final market day to the first day.
    for first in range(0,repeats,128):
        size=min(128,repeats-first);starts=rng.integers(n-block+1,size=(size,blocks))
        draw=(starts[...,None]+np.arange(block)).reshape(size,-1)[:,:n]
        denominator=counts[draw].sum(1)
        bootstrap[first:first+size]=sums[draw].sum(1)/denominator
        null[first:first+size]=null_sums[draw].sum(1)/denominator
    return {'loss_difference':observed,'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),
            'one_sided_p':float((1+np.sum(null<=observed))/(repeats+1)),
            'block_days':block,'days':n,'repeats':repeats,'seed':seed,
            'method':'null-centered noncircular moving trading-day block bootstrap',
            'bootstrap_standard_error':float(bootstrap.std(ddof=1))}


def holm(values):
    ordered=sorted(values,key=values.get);result={};last=0.
    for i,key in enumerate(ordered):last=max(last,min(1.,(len(ordered)-i)*values[key]));result[key]=last
    return result


def compare(target,prediction,controls,dates,decision_times):
    if set(controls)!={'zero','single','simple'}:raise ValueError('Exactly three fixed primary controls required')
    target=np.asarray(target,dtype=np.float64);prediction=np.asarray(prediction,dtype=np.float64)
    dates=np.asarray(dates,dtype=str);system=point_scores(target,prediction);loss=(prediction-target)**2
    tests={};sensitivities={}
    for name,values in controls.items():
        point_scores(target,values);delta=loss-(np.asarray(values)-target)**2
        tests[name]=paired_blocks(delta,dates)
        sensitivities[name]={str(block):paired_blocks(delta,dates,block=block) for block in (1,10)}
    adjusted=holm({name:v['one_sided_p'] for name,v in tests.items()})
    for name,value in tests.items():value['holm_adjusted_p']=adjusted[name]
    monthly={month:point_scores(target[dates.astype('U7')==month],prediction[dates.astype('U7')==month]) for month in np.unique(dates.astype('U7'))}
    local=pd.DatetimeIndex(decision_times).tz_convert('America/New_York');minute=local.hour.to_numpy()*60+local.minute.to_numpy()
    masks={'opening_0931_1029':minute<630,'midday_1030_1429':(minute>=630)&(minute<870),'closing_1430_1548':minute>=870}
    by_time={name:point_scores(target[mask],prediction[mask]) for name,mask in masks.items() if mask.any()}
    success=(system['mse_improvement_vs_zero'] is not None and system['mse_improvement_vs_zero']>=.01
             and all(test['loss_difference']<0 and test['holm_adjusted_p']<.05 for test in tests.values())
             and sum(value['mse_improvement_vs_zero']>0 for value in monthly.values())>=2)
    return {'system':system,'fixed_controls':{name:point_scores(target,values) for name,values in controls.items()},
            'paired_primary_tests':tests,'block_sensitivities':sensitivities,'monthly':monthly,'time_of_day':by_time,
            'research_success':bool(success),'primary_metric':'raw cumulative-12-minute log-return MSE'}


def minimum_detectable_improvement(development_delta,development_target,dates,planned_final_days=64):
    inference=paired_blocks(development_delta,dates)
    se=inference['bootstrap_standard_error']*np.sqrt(inference['days']/planned_final_days)
    zero=float(np.mean(np.asarray(development_target)**2))
    threshold=(norm.ppf(1-.05/3)+norm.ppf(.8))*se
    return {'approximate_relative_MSE_improvement_for_80pct_power':float(threshold/zero),
            'planned_final_days':planned_final_days,'basis':'development paired day-block residuals; Bonferroni conservative alpha .05/3',
            'assumption':'future loss-difference variance resembles development; not guaranteed power'}
