"""Artificial arrays only; maximum shape, schedule boundaries, numerical decisions."""
import hashlib
import time
from runtime import BASE,setup,require,read,write,sha,qualify_dependencies

def main():
    setup(24)
    import numpy as np
    from main_worker import calculate,decision,classify,make_support,planned_grid,MINUTE,CONTROLS
    started=time.monotonic();spec=read(BASE/'DESIGN_SPEC.json')
    dependencies=qualify_dependencies(spec)
    n,h=25753,60;i=np.arange(n,dtype=float)[:,None];j=np.arange(1,h+1,dtype=float)[None,:]
    y=np.sin(i*j/197)*.003;b=np.broadcast_to(j*1e-5,(n,h)).copy()
    arrays={'origins':np.arange(n,dtype=np.int64),'labels':y,'prefix_update_mean':b,
            'zero':np.zeros_like(y),'same_information_Ridge':b+np.cos(i*j/163)*.0002}
    mask=np.zeros((n,h),dtype=bool);mask[::330,-1]=True
    fake={'state':'artificial_maximum','count':n,'horizon':h,
          'tail_sha256':hashlib.sha256(arrays['origins'].astype('<i8').tobytes()).hexdigest()}
    r=calculate(arrays,fake,mask,spec)
    oracle=np.where(mask,y,b)
    truth=np.mean((y-oracle)**2,axis=0)
    np.testing.assert_allclose(truth,[x['floor_MSE'] for x in r['all_horizons']],rtol=1e-13,atol=0.)
    relaxed=mask.copy();relaxed[::30,:]=True
    r2=calculate(arrays,fake,relaxed,spec)
    require(all(a['floor_MSE']>=b['floor_MSE'] for a,b in zip(r['all_horizons'],r2['all_horizons'])),'Superset relax monotonicity')
    for control in CONTROLS:
        for metric in ['endpoint','path']:
            require(r['metrics'][control][metric+'_safe_upper_maximum_improvement_pct']>=r['metrics'][control][metric+'_maximum_improvement_pct'],'Safe upper contains estimate')
    # Hand-calculable separate support denominator: only first of two rows can change.
    small={'origins':np.arange(2,dtype=np.int64),'labels':np.array([[2.],[4.]]),
           'zero':np.zeros((2,1)),'prefix_update_mean':np.ones((2,1)),'same_information_Ridge':np.zeros((2,1))}
    e={'state':'artificial','count':2,'horizon':1,'tail_sha256':hashlib.sha256(small['origins'].astype('<i8').tobytes()).hexdigest()}
    v=calculate(small,e,np.array([[True],[False]]),spec)
    require(v['metrics']['floor']['endpoint_MSE']==4.5,'Floor divides by full original N')
    require(v['all_horizons'][0]['support_b_MSE']==.5,'Support contribution divides by full N')
    # Actual non-market calendar schedules test half days, DST, complete-hour residual.
    c={'calendar':{'name':'XNYS','start':'2021-11-05','end':'2021-11-26'}}
    minute=planned_grid(c,'minute');hour=planned_grid(c,'hourly')
    for g,step in [(minute,MINUTE),(hour,60*MINUTE)]:
        require((g['end_ns']-g['start_ns']==step).all(),'Planned bar duration')
        last=g['session_date']=='2021-11-26'
        require(int(last.sum())==(210 if step==MINUTE else 3),'Early close full groups')
    require(minute['open_ns'][0]% (24*60*MINUTE) != minute['open_ns'][-1]%(24*60*MINUTE),'DST UTC boundary')
    # Original minute full-H60 structure, no values or actual sample selection.
    ends=np.flatnonzero((minute['end_ns']+60*MINUTE)<=minute['close_ns'])
    m=make_support(ends,60,minute,np.ones(len(minute['end_ns']),dtype=bool),'minute')
    require(not m[:,:59].any() and m[:,-1].sum()==len(np.unique(minute['session_date'])),'Minute one eligible clock per session')
    hos=np.arange(len(hour['end_ns'])-8,dtype=np.int64)
    hs=make_support(hos,8,hour,np.ones(len(hour['end_ns']),dtype=bool),'hourly')
    require(hs.any() and not hs.all(),'Hourly cross-session support')
    records=[]
    for entry in spec['states']:
        metrics={'floor':{'endpoint_MSE':.5,'path_MSE':.5}}
        for key in CONTROLS:
            metrics[key]={'endpoint_MSE':1.,'path_MSE':1.,'endpoint_maximum_improvement_pct':50.,'path_maximum_improvement_pct':50.,
                          'endpoint_safe_upper_maximum_improvement_pct':50.00000001,'path_safe_upper_maximum_improvement_pct':50.00000001}
        records.append({'state':entry['state'],'metrics':metrics})
    require(decision(records,spec)['excluded_necessary_conditions']==0,'Known all-possible gate')
    records[0]['metrics']['zero'].update(endpoint_maximum_improvement_pct=0.,endpoint_safe_upper_maximum_improvement_pct=1e-9)
    require(decision(records,spec)['excluded_necessary_conditions']==1,'Known exactly one excluded necessary gate')
    records[0]['metrics']['zero'].update(endpoint_maximum_improvement_pct=50.,endpoint_safe_upper_maximum_improvement_pct=50.)
    records[1]['metrics']['zero'].update(endpoint_maximum_improvement_pct=-1.,endpoint_safe_upper_maximum_improvement_pct=-1.)
    require(decision(records,spec)['tasks'][0]['equal_fold_endpoint_maximum_improvement_over_zero_pct']==24.5,'Equal fold, not pooled risk')
    require(classify(.1,.1,.1,1e-8)=='within_numerical_margin','Equality not rejected')
    require(classify(.1-1e-10,.1-1e-10,.1,1e-8)=='within_numerical_margin','Near boundary not rejected')
    require(classify(-1.,.2,.1,1e-8)=='within_numerical_margin','Unresolved arithmetic not rejected')
    elapsed=time.monotonic()-started;require(elapsed<30,'Artificial engineering cap')
    write(BASE/'ENGINEERING_ACCEPTANCE.json',{'status':'passed_artificial_only','seconds':elapsed,'maximum_shape':[n,h],
          'local_code_hashes':{str(BASE/name):sha(BASE/name) for name in ['runtime.py','main_worker.py','engineering.py']},
          'actual_dependency_receipt':dependencies,
          'full_N_denominator_verified':True,'oracle_floor_verified':True,'superset_monotonicity_verified':True,
          'minute_structural_zero_verified':True,'early_close_DST_hour_crosssession_verified':True,
          'conservative_roundoff_and_equal_fold_decisions_verified':True,'market_payloads_decoded':False,
          'calendar_only_schedule_constructed':True,'market_fits':0,'GPU':False,'CPU':[24]})

if __name__=='__main__':main()
