"""Artificial-only numerical and genuine external-process lifecycle acceptance."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import resource
import subprocess
import sys
import time
import types
import numpy as np
from screen_lifecycle import atomic_json,atomic_npz,save_checkpoint,load_checkpoint,file_sha
from screen_math import empty_stats,update,solve,predict,route,score
from source_io import SourceState,InputError,mask_for_history
BASE=Path(__file__).resolve().parents[1]

def numerical_tests(directory):
    rng=np.random.default_rng(20261110)
    x=rng.normal(size=(128,979))+rng.normal(size=979);x[:,0]=0.;x[:,1]=x[:,2]
    y=rng.normal(size=(128,60))
    stats=empty_stats(979,60);update(stats,x[:55],y[:55])
    save_checkpoint(directory/'NUMERIC_CHECKPOINT.npz',{'fixture':True,'cursor':55},stats,{}, {})
    meta,restored,models,extras=load_checkpoint(directory/'NUMERIC_CHECKPOINT.npz')
    assert meta['cursor']==55 and not models and not extras
    update(restored,x[55:],y[55:]);model=solve(restored)
    mean=x.mean(axis=0);sd=x.std(axis=0);sd=np.where(sd>1e-12,sd,1.)
    Z=(x-mean)/sd
    direct=np.linalg.solve(Z.T@Z/len(x)+.1*np.eye(979),Z.T@(y-y.mean(axis=0))/len(x))
    np.testing.assert_allclose(model['coef'],direct,rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(predict(model,x),y.mean(axis=0)+Z@direct,rtol=1e-9,atol=1e-11)
    # A sole final Q scaler: missing rows are raw complete-SPY mean, without a first std.
    z=rng.normal(.001,.0002,size=(201,479));complete=np.arange(len(z))%3!=0
    mean_z=z[complete].mean(axis=0);encoded=z.copy();encoded[~complete]=mean_z
    expected=np.sqrt(complete.mean())*z[complete].std(axis=0)
    np.testing.assert_allclose(encoded.std(axis=0),expected,rtol=1e-12,atol=1e-15)
    standardized=(encoded-encoded.mean(axis=0))/encoded.std(axis=0)
    np.testing.assert_allclose(standardized[~complete],0.,rtol=0,atol=1e-12)
    A=np.array([[-0.,1.],[2.,3.],[4.,5.],[6.,7.]],dtype=np.float64)
    v=np.asarray([False,False,True,True]);s=np.asarray([False,True,False,True])
    base,plus=route(A,np.array([[8.,9.],[10.,11.]]),np.array([[12.,13.]]),v,s)
    assert np.array_equal(plus[~(v&s)].view(np.uint64),base[~(v&s)].view(np.uint64))
    assert plus[0,0].view(np.uint64)==np.asarray(-0.).view(np.uint64)
    Y=np.ones((4,2));pred={m:np.zeros((4,2)) for m in ('A_full','base','plus','zero','prefix_mean')}
    metrics,gates,groups=score(Y,pred,v,s)
    assert not gates['endpoint_below_zero'] and gates['path_not_above_base']
    assert sum(g['count'] for g in groups.values())==4 and all(g['count']==1 for g in groups.values())
    try:score(Y,{**pred,'plus':np.full((4,2),np.nan)},v,s)
    except ValueError:pass
    else:raise AssertionError('nonfinite_prediction_accepted')
    # Worst dictionary shape includes construction, stats update and atomic fsync checkpoint.
    base=rng.normal(0,.01,size=(2048,500));spy=rng.normal(0,.001,size=(2048,479));increments=rng.normal(0,.001,size=(2048,60))
    timed=empty_stats(979,60);began=time.perf_counter()
    X=np.column_stack((base.copy(),spy.copy()));Y=np.cumsum(increments.copy(),axis=1)
    update(timed,X,Y);save_checkpoint(directory/'MAX_SHAPE_CHECKPOINT.npz',{'fixture':True,'cursor':2048},timed,{}, {})
    update_seconds=time.perf_counter()-began
    began=time.perf_counter();solve(timed);solve_seconds=time.perf_counter()-began
    return {'direct_p979_max_coef_error':float(np.max(np.abs(model['coef']-direct))),
            'max_shape_update_seconds':update_seconds,'max_shape_solve_seconds':solve_seconds,
            'maximum_shape':[2048,979,60],'direct_shape':[128,979,60],
            'single_final_Q_scaler_test':True,'bitwise_all_missing_routes_test':True,
            'strict_boundary_and_nonfinite_score_tests':True,'serialized_stats_continuation_test':True}

def source_formula_tests():
    import pandas as pd
    # Only old source code is read; neither constructor nor Predictor is called.
    path=BASE.parents[1]/'mechanism_repair'/'20261109_v3b'/'code'/'predictor.py'
    # BASE=.../research_review/<stage>; sibling mechanism_repair is at multiscale.
    path=BASE.parent.parent/'mechanism_repair'/'20261109_v3b'/'code'/'predictor.py'
    module=types.ModuleType('synthetic_exact_context');module.__file__=str(path)
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    n=1170;starts=[];dates=[]
    for day in ('2026-01-05','2026-01-06','2026-01-07'):
        first=pd.Timestamp(day+'T14:30:00Z').value
        starts.extend(first+np.arange(390,dtype=np.int64)*60_000_000_000);dates.extend([day]*390)
    starts=np.asarray(starts,dtype=np.int64);dates=np.asarray(dates,dtype='U10')
    closes=np.exp(4.5+.0001*np.arange(n));closes[390:]/=4
    split=np.zeros(n);split[390:]=np.log(4.)
    bars={'close':closes,'log_close':np.log(closes),'log_split':split,'observed':np.ones(n,dtype=bool),
          'start_ns':starts,'end_ns':starts+60_000_000_000,'session_date':dates,
          'abnormal':np.arange(n,dtype=float)/1000}
    bars['abnormal_sum']=np.r_[0.,np.cumsum(bars['abnormal'])];bars['abnormal_bad']=np.zeros(n+1,dtype=np.int64)
    src=types.SimpleNamespace(guard=lambda:None,context=module,spy={'minute':100+np.arange(n,dtype=float)*.01},phase='no_labels')
    state=object.__new__(SourceState);state.source=src;state.bars=bars;state.L=480;state.H=60;state.frequency='minute'
    state.lag_column=module.feature_names(60,480).index('return_lag_60')
    state.P=np.array([600,650],dtype=np.int64);state.T=np.array([700],dtype=np.int64)
    origins=state.P;X=state.A(origins)
    history=np.asarray([[np.log(closes[j])+split[j]-split[i]-np.log(closes[i]) for j in range(i-479,i+1)] for i in origins])
    expected=np.column_stack((history,closes[origins,None],module.features(history,bars['end_ns'][origins],60)))
    np.testing.assert_array_equal(X,expected)
    u=np.asarray([bars['abnormal'][i-59:i+1].mean() for i in origins])
    B=state.B(origins)
    np.testing.assert_allclose(B[:,-2],u,rtol=0,atol=1e-14)
    np.testing.assert_allclose(B[:,-1],X[:,481+state.lag_column]*u,rtol=0,atol=1e-14)
    spy=src.spy['minute'];expected_z=np.asarray([np.diff(np.log(spy[i-480:i])) for i in origins])
    np.testing.assert_array_equal(state.z(origins),expected_z)
    for phase in ('prefix','tail'):
        try:state.labels(origins,phase)
        except InputError:pass
        else:raise AssertionError('source_label_phase_admission_missing')
    src.phase='prefix'
    np.testing.assert_array_equal(state.labels(origins,'prefix'),np.log(closes[origins[:,None]+np.arange(1,61)]/closes[origins,None]))
    class RefuseIndex:
        def __len__(self):return n
        def __getitem__(self,_):raise AssertionError('illegal_origin_read_before_rejection')
    src.spy['minute']=RefuseIndex()
    try:state.z(np.array([479],dtype=np.int64))
    except InputError:pass
    else:raise AssertionError('SPY_lower_bound_not_rejected')
    observed=np.ones(12,dtype=bool);observed[3]=False
    np.testing.assert_array_equal(mask_for_history(observed,np.array([3,4,5,7,8]),4),[False,False,False,False,True])
    return {'old_split_effective_history_and_context17':True,'current_completed_volume_H_window':True,
            'SPY_delayed_Lclose_Lminus1diff':True,'label_phase_and_early_index_rejection':True,
            'source_code_only_path':str(path),'source_code_sha256':file_sha(path)}

def process_fixture(directory,benchmark):
    run=directory/'lifecycle';run.mkdir()
    code_names=('screen_worker.py','screen_math.py','screen_lifecycle.py','source_io.py','synthetic_support.py')
    start=time.time()
    protocol={'kind':'synthetic_SPY_screen_fixture_v1','market_arrays_admitted':False,
        'started_epoch':start,'deadline_epoch':start+900,'worker_cpu':26,'GPU':False,'Q3_allowed':False,
        'science_rules_path':str(BASE/'SCIENCE_RULES.json'),'science_rules_sha256':file_sha(BASE/'SCIENCE_RULES.json'),
        'synthetic_acceptance_path':str(benchmark),'synthetic_acceptance_sha256':file_sha(benchmark),
        'supervisor_sha256':file_sha(__file__),'run_directory':str(run),
        'code_hashes':{str(BASE/'code'/name):file_sha(BASE/'code'/name) for name in code_names}}
    path=directory/'FIXTURE_PROTOCOL.json';atomic_json(path,protocol);identity=file_sha(path)
    command=[sys.executable,'-B',str(BASE/'code'/'screen_worker.py'),'--protocol',str(path),'--run']
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
    with (directory/'FIRST_PROCESS.log').open('x') as stream:
        child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        while not (run/'READY_FOR_EXTERNAL_PAUSE.json').exists():
            if child.poll() is not None:raise AssertionError('fixture_exited_before_external_pause:'+str(child.returncode))
            if time.time()>start+120:child.kill();child.wait();raise AssertionError('fixture_no_ready_marker')
            time.sleep(.02)
        marker=json.loads((run/'READY_FOR_EXTERNAL_PAUSE.json').read_bytes());checkpoint=Path(marker['checkpoint_path'])
        before=file_sha(checkpoint);assert before==marker['checkpoint_sha256'] and marker['pid']==child.pid
        os.kill(child.pid,signal.SIGUSR1);code=child.wait(timeout=15);after=file_sha(checkpoint)
        assert code==75 and before==after
    atomic_json(run/'EXTERNAL_PAUSE_RECEIPT.json',{'kind':'screen_external_pause_v1','protocol_sha256':identity,
        'pid':child.pid,'signal':'SIGUSR1','exit_code':75,'checkpoint_path':str(checkpoint),
        'checkpoint_sha256_before':before,'checkpoint_sha256_after':after,'cursor':2048,
        'supervisor_source_sha256':protocol['supervisor_sha256']})
    with (directory/'RESUME_PROCESS.log').open('x') as stream:
        resumed=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        assert resumed.pid!=child.pid
        code=resumed.wait(timeout=180)
        if code!=0:raise AssertionError('fixture_resume_failed:'+str(code))
    result=json.loads((run/'RESULTS.json').read_bytes());manifest=json.loads((run/'FROZEN_MODELS.json').read_bytes())
    assert len(manifest['models'])==12 and result['total_fits']==12 and len(result['rows'])==4
    assert result['market_data_used'] is False and result['goal_achieved'] is False
    for row in result['rows']:
        with np.load(row['tail_path'],allow_pickle=False) as data:
            mask=~(data['mV']&data['mS'])
            np.testing.assert_array_equal(data['plus'][mask].view(np.uint64),data['base'][mask].view(np.uint64))
    # A same-protocol new process reports completed state without retraining.
    again=subprocess.run(command,env=env,capture_output=True,text=True,timeout=15)
    assert again.returncode==0 and 'already_complete' in again.stdout
    return {'actual_external_SIGUSR1':True,'external_child_pid':child.pid,'new_resume_pid':resumed.pid,
        'different_PID_verified':True,'exit75_verified':True,'first_checkpoint_byte_preservation':True,
        'same_immutable_protocol_sha256':identity,'protocol_path':str(path),'full_twelve_fit_barrier_and_four_tail_files':True,
        'completed_checkpoint_no_retrain':True,'fixture_results_path':str(run/'RESULTS.json'),
        'fixture_results_sha256':file_sha(run/'RESULTS.json'),'seconds':time.time()-start}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
    if os.sched_getaffinity(0)!={26}:raise RuntimeError('acceptance_requires_CPU26')
    resource.setrlimit(resource.RLIMIT_AS,(3221225472,3221225472))
    directory=args.output.resolve();directory.mkdir(parents=True,exist_ok=False)
    numbers=numerical_tests(directory);source=source_formula_tests()
    benchmark=directory/'BENCHMARK_ACCEPTANCE.json'
    atomic_json(benchmark,{'status':'passed','market_data_used':False,**numbers})
    lifecycle=process_fixture(directory,benchmark)
    result={'kind':'fixed_SPY_artificial_acceptance_v1','status':'passed','market_data_used':False,
        'source_market_arrays_read':False,'GPU_used':False,'Q3_read':False,'numeric_threads':1,'CPU_affinity':[26],
        **numbers,'source_formula_tests':source,'lifecycle':lifecycle,
        'code_hashes':{str(p):file_sha(p) for p in (BASE/'code').glob('*.py')}}
    atomic_json(directory/'ACCEPTANCE.json',result);print(json.dumps(result))

if __name__=='__main__':main()
