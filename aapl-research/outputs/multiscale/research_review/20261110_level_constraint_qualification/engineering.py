"""Artificial-only same-source mapping and maximum saved-array-shape cost probe."""
import ast
from dataclasses import dataclass
import hashlib
import time
from runtime import BASE, MEMBERS, setup, require, read, write, sha, decision

def main():
    setup(24)
    import numpy as np
    from main_worker import calculate
    started=time.monotonic(); spec=read(BASE/'DESIGN_SPEC.json')
    for path,expected in spec['source_files'].items():
        require(sha(path)==expected,'Source changed before engineering')
    scope={'__name__':__name__,'np':np,'dataclass':dataclass,'CloseWindows':object,'require':require}
    for name,names in [('features.py',['Scaler','fit_scaler']),('native_core.py',['fixed_ridge','ridge_predict'])]:
        path=next(p for p in spec['source_files'] if p.endswith('/'+name))
        tree=ast.parse(open(path).read())
        for wanted in names:
            nodes=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name==wanted]
            require(len(nodes)==1,'Exact engineering AST missing')
            exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),path,'exec'),scope)
    n=8203; k=np.arange(n,dtype=float)[:,None]; j=np.arange(1,6,dtype=float)[None,:]
    X=np.sin(k*j/31)+.03*j
    Y=np.sin(k*np.arange(1,61)[None,:]/151)*.002 + np.arange(1,61)[None,:]*1e-5
    class Fixture:
        lookback=4
        bars=None
        def arrays(self,origins): return X[origins,:4],X[origins,4:5]
    scope['targets']=lambda bars,idx,horizon,mode:Y[idx]
    scaler=scope['fit_scaler'](Fixture(),np.arange(n),60,'future_log_return',batch_size=4096)
    xx,aa=scaler.transform_x(X[:,:4],X[:,4:5])
    fitted=scope['fixed_ridge'](np.column_stack((xx,aa)),Y)
    require(np.array_equal(np.array(scaler.y_mean),fitted['y_mean']),'Scaler/Ridge prefix means differ')
    normalized=(np.column_stack((xx,aa)).astype(float)-fitted['x_mean'])/fitted['x_std']
    prediction=scope['ridge_predict'](fitted,np.column_stack((xx,aa)))
    centered=prediction-np.array(scaler.y_mean)
    np.testing.assert_allclose(centered,normalized@fitted['coefficients'],rtol=1e-11,atol=1e-17)
    # Exercise all loss computations at the largest formal saved shape, no market values.
    n=25753; h=60; origins=np.arange(n,dtype=np.int64)
    i=np.arange(n,dtype=float)[:,None]; j=np.arange(h,dtype=float)[None,:]+1
    y=np.sin(i*j/197)*.003
    b=np.broadcast_to(j*1e-5,(n,h)).copy()
    p=np.cos(i*j/163)*.0002+b
    arrays={'origins':origins,'labels':y,'D':p.copy(),'Z':p.copy(),'C':p.copy(),
            'zero':np.zeros_like(y),'prefix_update_mean':b,'same_information_Ridge':p}
    fake={'state':'artificial_maximum','count':n,'horizon':h,
          'tail_sha256':hashlib.sha256(origins.astype('<i8').tobytes()).hexdigest()}
    result=calculate(arrays,fake)
    for key,metric in result['metrics'].items():
        expected=sum(row['MSE'][key] for row in result['all_horizons'])/h
        require(abs(metric['path_MSE']-expected)<=1e-11*expected,'All-horizon equal weighting')
    # Sign of the exact intervention under simple positive/negative cases.
    for y0,b0 in [(0.,.002),(.003,.002)]:
        delta=(y0-b0)**2-y0**2
        require(abs(delta-(b0*b0-2*b0*y0))<1e-20,'Identity sign fixture')
    # Known all-pass then exactly one failed comparison; no data-dependent threshold.
    controls=['same_information_Ridge','zero','prefix_update_mean','half_Ridge','half_g']
    rows=[{'state':name,'metrics':{key:{'endpoint_MSE':1. if key=='g' else 2.,
           'path_MSE':1. if key=='g' else 2.} for key in controls+['g']}}
           for name in ['minute_f1','minute_f2','hourly_f1','hourly_f2']]
    require(decision(rows)['passed_of42']==42 and decision(rows)['continue_to_full_development_design_review'], 'Known 42-gate pass')
    rows[0]['metrics']['same_information_Ridge']['endpoint_MSE']=1.
    require(decision(rows)['passed_of42']==41 and not decision(rows)['continue_to_full_development_design_review'], 'Single failure must close continuation')
    # Unequal fold risk: equal-fold gain is +24.5%, unlike the pooled-risk ratio.
    rows[0]['metrics']['zero']['endpoint_MSE']=1.
    rows[1]['metrics']['zero']['endpoint_MSE']=100.
    rows[0]['metrics']['g']['endpoint_MSE']=.5
    rows[1]['metrics']['g']['endpoint_MSE']=101.
    require(abs(decision(rows)['tasks'][0]['equal_fold_endpoint_improvement_over_zero_pct']-24.5)<1e-12,'Equal-fold relative gain fixture')
    elapsed=time.monotonic()-started
    require(elapsed<30,'Engineering cost exceeds cap')
    write(BASE/'ENGINEERING_ACCEPTANCE.json',{'status':'passed_artificial_only','seconds':elapsed,
        'local_code_hashes':{str(BASE/name):sha(BASE/name) for name in ['runtime.py','main_worker.py','engineering.py']},
        'exact_AST_source_hashes':spec['source_files'],'same_4096_prefix_mean_bitwise_equal':True,
        'g_mapping_verified':True,'maximum_shape':[n,h],'new_paired_identity_horizons':len(result['all_horizons']),
        'all_horizon_path_weighting_tested':True,'gate_all_pass_and_single_failure_tested':True,
        'equal_fold_relative_improvement_tested':True,'external_timeout_seconds':30,
        'market_payloads_decoded':False,'market_fits':0,'synthetic_Ridge_solve':1,'GPU':False,'CPU':[24]})

if __name__=='__main__': main()
