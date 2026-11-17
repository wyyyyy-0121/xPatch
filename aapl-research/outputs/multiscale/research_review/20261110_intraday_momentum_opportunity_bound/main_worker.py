"""Oracle opportunity floor on fixed exposed labels. No price decoding or fitting."""
import hashlib
import importlib.metadata
import io
import os
import signal
import sys
import time
import zipfile
from runtime import BASE, require, sha, read, write, setup, load_protocol, guard, qualify_dependencies

PAUSE=False
MINUTE=60_000_000_000
CONTROLS=['zero','prefix_update_mean','same_information_Ridge']

def request_pause(signum,frame):
    global PAUSE
    PAUSE=True

def load_selected(path,expected,all_members,selected):
    import numpy as np
    permitted={'origins','labels','zero','prefix_update_mean','same_information_Ridge','start_ns','end_ns','observed','session_date'}
    require(len(set(selected))==len(selected) and set(selected)<=permitted,'Forbidden decoded member')
    require(sha(path)==expected,'Input container identity changed')
    opened=[]
    with zipfile.ZipFile(path) as z:
        require(sorted(z.namelist())==sorted(all_members),'Unexpected or duplicate archive member')
        result={}
        for key in selected:
            member=key+'.npy'
            require(member in all_members,'Unapproved member')
            result[key]=np.load(io.BytesIO(z.read(member)),allow_pickle=False)
            opened.append(member)
    return result,opened

def planned_grid(spec,frequency):
    import numpy as np
    import pandas as pd
    import exchange_calendars as xc
    c=spec['calendar']
    cal=xc.get_calendar(c['name'],start=c['start'],end=c['end'])
    sch=cal.schedule.loc[c['start']:c['end']]
    starts=[];ends=[];dates=[];opens=[];closes=[]
    step=MINUTE if frequency=='minute' else 60*MINUTE
    for day,row in sch.iterrows():
        op=pd.Timestamp(row['open']).as_unit('ns').value
        cl=pd.Timestamp(row['close']).as_unit('ns').value
        n=(cl-op)//step
        require(n>0 and (cl-op)%MINUTE==0,'Invalid calendar duration')
        start=op+step*np.arange(n,dtype=np.int64)
        starts.append(start);ends.append(start+step)
        dates.append(np.full(n,day.strftime('%Y-%m-%d'),dtype='<U10'))
        opens.append(np.full(n,op,dtype=np.int64));closes.append(np.full(n,cl,dtype=np.int64))
    return {k:np.concatenate(v) for k,v in [('start_ns',starts),('end_ns',ends),('session_date',dates),('open_ns',opens),('close_ns',closes)]}

def qualify_bars(arrays,grid,entry,frequency):
    import numpy as np
    n=entry['rows']
    for key in ['start_ns','end_ns','session_date']:
        a=arrays[key]
        require(a.shape==(n,) and np.array_equal(a,grid[key]),'Full planned bar identity '+key)
    require(arrays['start_ns'].dtype==np.dtype('int64') and arrays['end_ns'].dtype==np.dtype('int64'),'Nanosecond dtype')
    require(arrays['observed'].shape==(n,) and arrays['observed'].dtype==np.dtype('bool'),'Observed identity')
    require(arrays['session_date'].dtype.kind=='U','Session string dtype')
    require(str(arrays['session_date'][0])=='2021-01-04' and str(arrays['session_date'][-1])=='2026-06-30','Fixed dates')
    require(int(grid['end_ns'][-1])<1782878400000000000,'No July New York payload')
    return {'frequency':frequency,'rows':n,'observed_rows':int(arrays['observed'].sum()),
            'full_planned_grid_equal':True,'decoded_members':['start_ns.npy','end_ns.npy','observed.npy','session_date.npy'],
            'close_member_opened':False}

def make_support(origins,h,grid,observed,frequency):
    import numpy as np
    require(origins.min()>=0 and origins.max()+h<len(observed),'Origin range')
    target=origins[:,None]+np.arange(1,h+1,dtype=np.int64)
    require(observed[origins].all() and observed[target].all(),'Original complete label identity')
    # Identity check only; this does not filter or create support.
    n=len(origins)
    if frequency=='minute':
        require((grid['session_date'][origins]==grid['session_date'][origins+h]).all(),'Original same-session minute path')
        require(np.all(grid['end_ns'][target]-grid['end_ns'][origins,None]==MINUTE*np.arange(1,h+1)),'Minute continuity')
    t=grid['end_ns'][origins]
    op=grid['open_ns'][origins];cl=grid['close_ns'][origins]
    support=((t>=op+30*MINUTE)&(t<=cl-30*MINUTE))[:,None] & (grid['end_ns'][target]>=cl[:,None])
    if frequency=='minute':
        require(h==60 and not support[:,:59].any(),'Minute h1..59 structural zero')
        require(np.array_equal(support[:,-1],t==cl-60*MINUTE),'Minute H60 singleton-clock identity')
    require(support.shape==(n,h),'Support shape')
    return support

def calculate(arrays,entry,support,spec):
    import numpy as np
    n,h=entry['count'],entry['horizon']
    origins=arrays['origins']
    require(origins.dtype==np.dtype('int64') and origins.shape==(n,) and np.all(np.diff(origins)>0),'Origin shape/order')
    require(hashlib.sha256(origins.astype('<i8').tobytes()).hexdigest()==entry['tail_sha256'],'Complete original tail identity')
    for key in ['labels']+CONTROLS:
        a=arrays[key]
        require(a.dtype==np.dtype('float64') and a.shape==(n,h) and np.isfinite(a).all(),'Saved array identity '+key)
    require(np.count_nonzero(arrays['zero'])==0,'Saved zero must be zero')
    b=arrays['prefix_update_mean'];y=arrays['labels']
    require(np.array_equal(b,np.broadcast_to(b[0],b.shape)),'Fixed prefix mean constant')
    require(support.dtype==np.dtype('bool') and support.shape==y.shape,'Support type')
    e2=(y-b)**2
    floor=np.mean(np.where(support,0.,e2),axis=0,dtype=np.float64)
    inrisk=np.mean(np.where(support,e2,0.),axis=0,dtype=np.float64)
    risk={key:np.mean((y-arrays[key])**2,axis=0,dtype=np.float64) for key in CONTROLS}
    require(all((v>0).all() and np.isfinite(v).all() for v in risk.values()),'Positive finite reference risks')
    scale=np.maximum.reduce([*risk.values(),np.full(h,np.finfo(float).tiny)])
    tol=spec['numeric_identity_eps_factor']*np.finfo(float).eps*scale
    residual=risk['prefix_update_mean']-floor-inrisk
    require(np.all(np.abs(residual)<=tol),'Raw b-risk partition identity')
    factor=spec['fp64_error_factor']*np.finfo(float).eps*(n+h+16)
    error={key:factor*np.maximum(np.mean((np.abs(y)+np.abs(arrays[key]))**2,axis=0),np.finfo(float).tiny) for key in CONTROLS}
    error['floor']=error['prefix_update_mean'].copy()
    maximum={key:100*(1-floor/risk[key]) for key in CONTROLS}
    safe={key:100*(1-np.maximum(0,floor-error['floor'])/(risk[key]+error[key])) for key in CONTROLS}
    metrics={'floor':{'endpoint_MSE':float(floor[-1]),'path_MSE':float(floor.mean()),
                      'endpoint_error_bound':float(error['floor'][-1]),'path_error_bound':float(error['floor'].mean())}}
    for key in CONTROLS:
        metrics[key]={'endpoint_MSE':float(risk[key][-1]),'path_MSE':float(risk[key].mean()),
          'endpoint_maximum_improvement_pct':float(maximum[key][-1]),'path_maximum_improvement_pct':float(100*(1-floor.mean()/risk[key].mean())),
          'endpoint_safe_upper_maximum_improvement_pct':float(safe[key][-1]),
          'path_safe_upper_maximum_improvement_pct':float(100*(1-max(0.,floor.mean()-error['floor'].mean())/(risk[key].mean()+error[key].mean())))}
    if entry['state'].startswith('minute'):
        require(not support[:,:59].any(),'Minute first59 unchanged')
        require(abs((risk['prefix_update_mean'].mean()-floor.mean())-(risk['prefix_update_mean'][-1]-floor[-1])/60)<=max(tol),'Minute full-path dilution identity')
    rows=[{'horizon':j+1,'support_count':int(support[:,j].sum()),'b_MSE':float(risk['prefix_update_mean'][j]),
           'support_b_MSE':float(inrisk[j]),'floor_MSE':float(floor[j]),'identity_residual':float(residual[j]),'identity_tolerance':float(tol[j]),
           'MSE':{k:float(v[j]) for k,v in risk.items()},'risk_error_bounds':{k:float(v[j]) for k,v in error.items()},
           'maximum_improvement_pct':{k:float(v[j]) for k,v in maximum.items()},
           'safe_upper_maximum_improvement_pct':{k:float(v[j]) for k,v in safe.items()}} for j in range(h)]
    return {'state':entry['state'],'count':n,'horizon':h,'tail_sha256':entry['tail_sha256'],
            'support_sha256':hashlib.sha256(support.astype('u1').tobytes()).hexdigest(),
            'metrics':metrics,'all_horizons':rows,'identity_qualified':True}

def classify(value,safe,threshold,margin):
    if safe<threshold-margin:return 'excluded'
    if value<=threshold+margin:return 'within_numerical_margin'
    return 'not_ruled_out'

def decision(records,spec):
    by={r['state']:r for r in records}
    require(list(by)==[x['state'] for x in spec['states']],'All four states required')
    checks=[]
    for r in records:
        for key in CONTROLS:
            for metric in ['endpoint','path']:
                value=r['metrics'][key][metric+'_maximum_improvement_pct']; safe=r['metrics'][key][metric+'_safe_upper_maximum_improvement_pct']
                checks.append({'state':r['state'],'control':key,'metric':metric,'maximum_improvement_pct':value,'safe_upper_maximum_improvement_pct':safe,
                     'threshold_pct':.1,'status':classify(value,safe,.1,spec['percentage_point_margin'])})
    tasks=[]
    for name in ['minute','hourly']:
        value=sum(by[name+'_f'+str(i)]['metrics']['zero']['endpoint_maximum_improvement_pct'] for i in [1,2])/2
        safe=sum(by[name+'_f'+str(i)]['metrics']['zero']['endpoint_safe_upper_maximum_improvement_pct'] for i in [1,2])/2
        tasks.append({'task':name,'equal_fold_endpoint_maximum_improvement_over_zero_pct':value,
                      'safe_upper_maximum_improvement_pct':safe,'threshold_pct':1.,
                      'status':classify(value,safe,1.,spec['percentage_point_margin'])})
    n=sum(x['status']=='excluded' for x in checks+tasks)
    return {'necessary_checks':checks,'tasks':tasks,'necessary_check_count':len(checks)+len(tasks),
            'excluded_necessary_conditions':n,'fixed_support_class_ruled_out_on_development_gates':n>0,
            'model_fitting_admitted':False,'candidate_admitted':False,'independent_market_confirmation':False,
            'original_58_gate_recipe_changed':False,'full_research_goal_achieved':False}

def main():
    setup(24);signal.signal(signal.SIGUSR1,request_pause);signal.signal(signal.SIGTERM,request_pause)
    p=load_protocol();spec=read(BASE/'DESIGN_SPEC.json');output=BASE/'run'
    dependencies=qualify_dependencies(spec)
    write(output/'DEPENDENCY_RECEIPT.json',dependencies)
    identity=sha(BASE/'EXECUTION_PROTOCOL.json');checkpoint=output/'CHECKPOINT.json'
    saved=read(checkpoint) if checkpoint.exists() else {'protocol_sha256':identity,'completed':[],'worker_pids':[]}
    require(saved['protocol_sha256']==identity,'Checkpoint protocol changed')
    require([x['state'] for x in saved['completed']]==[x['state'] for x in spec['states'][:len(saved['completed'])]],'Checkpoint state order')
    for item in saved['completed']:require(sha(output/(item['state']+'.json'))==item['sha256'],'Committed state changed')
    saved['worker_pids'].append(os.getpid());write(checkpoint,saved)
    bars={};grids={};bar_receipts={}
    for entry in spec['states'][len(saved['completed']):]:
        guard(p,35);frequency=entry['state'].split('_')[0]
        if frequency not in bars:
            bars[frequency],opened=load_selected(spec['bars'][frequency]['path'],spec['bars'][frequency]['sha256'],[k+'.npy' for k in spec['bar_archive_members']],spec['bar_decode_members'])
            grids[frequency]=planned_grid(spec,frequency)
            bar_receipts[frequency]=qualify_bars(bars[frequency],grids[frequency],spec['bars'][frequency],frequency)
            bar_receipts[frequency]['opened_members']=opened
        arrays,opened=load_selected(entry['path'],entry['sha256'],spec['prediction_archive_members'],spec['prediction_decode_members'])
        support=make_support(arrays['origins'],entry['horizon'],grids[frequency],bars[frequency]['observed'],frequency)
        result=calculate(arrays,entry,support,spec);result['opened_prediction_members']=opened;result['bar_identity']=bar_receipts[frequency]
        del arrays,support
        write(output/(entry['state']+'.json'),result)
        saved['completed'].append({'state':entry['state'],'sha256':sha(output/(entry['state']+'.json'))});write(checkpoint,saved)
        guard(p,25)
        if len(saved['completed'])==1:
            write(output/'FIRST_COMMIT.json',{'pid':os.getpid(),'checkpoint_sha256':sha(checkpoint)})
            while not PAUSE:guard(p,35);time.sleep(.01)
        if PAUSE:
            write(output/'PAUSED.json',{'pid':os.getpid(),'protocol_sha256':identity,'checkpoint_sha256':sha(checkpoint),'completed_states':len(saved['completed'])});return 75
    records=[read(output/(x['state']+'.json')) for x in saved['completed']]
    write(output/'RESULTS.json',{'protocol_sha256':identity,'records':records,'market_fits':0,'training_updates':0,'model_inference':0,'close_member_opened':False,'Q3_read':False,'GPU':False,'CPU':[24]})
    write(output/'DECISION.json',decision(records,spec));write(output/'WORKER_DONE.json',{'pid':os.getpid(),'completed_states':4,'protocol_sha256':identity})
    return 0

if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        write(BASE/'run'/'WORKER_FAILURE.json',{'pid':os.getpid(),'error':repr(exc)});raise
