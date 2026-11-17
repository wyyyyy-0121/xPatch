"""At most30s, inner-fit-only cost probe for a17-feature path Ridge model."""
from pathlib import Path
import hashlib,json,os,sys,time
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
V3=OUT.parents[1]/'20261109_v3'
sys.path.insert(0,str(V3/'code'))
from engine import Data
from study import make_job,source_hashes
from multiscale.bars import targets

DEADLINE=1794205909.6228158

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def features(data,idx):
    h=data.cfg['horizon'];L=data.cfg['lookback']
    lags=list(dict.fromkeys([1,max(2,h//4),max(3,h//2),h,2*h,4*h,L-1]))
    x,_=data.windows.arrays(idx)
    lagret=np.column_stack([x[:,-1]-x[:,-1-k] for k in lags])
    delta=np.diff(x,axis=1)
    rms=np.column_stack([np.sqrt(np.mean(delta[:,-k:]**2,axis=1)) for k in (h,4*h)])
    # UTC completion to New York civil clock; no future close/bounds or prices.
    ends=pd.to_datetime(data.bars.end_ns[idx],utc=True).tz_convert('America/New_York')
    u=(ends.hour.to_numpy()*60+ends.minute.to_numpy()-570)/390.
    clock=np.column_stack([fn(2*np.pi*k*u) for k in (1,2) for fn in (np.sin,np.cos)])
    dow=ends.dayofweek.to_numpy()
    weekday=np.column_stack([(dow==k).astype(float) for k in (1,2,3,4)])
    return np.column_stack([lagret,rms,clock,weekday])

def main():
    start=time.time();stop=start+28.
    assert sorted(os.sched_getaffinity(0))==[24,26] and os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    assert source_hashes()==read(V3/'protocol/CODE_FREEZE.json')['hashes']
    measurements=[];all_counts={}
    for profile,h in [('minute',60),('hourly',8)]:
        cfg=make_job(profile,h,'future_log_return','xpatch','anchor_baseline',5,2026)
        init=time.perf_counter();data=Data(cfg);load_elapsed=time.perf_counter()-init
        # Counts inspect partitions, never score or read stopping/outer targets.
        meta=data.meta;counts={}
        for f,rec in meta['folds'].items():
            with np.load(rec['split_file'],allow_pickle=False) as a:counts[f]=len(a['fit'])
        all_counts[profile]=counts
        idx=data.splits['fit'];sx=np.zeros(17);sxx=np.zeros((17,17));sxy=np.zeros((17,h));sy=np.zeros(h);n=0
        fit_begin=time.perf_counter();feature_seconds=0.;target_seconds=0.;gram_seconds=0.
        for pos in range(0,len(idx),4096):
            if time.time()>=stop:break
            chunk=idx[pos:pos+4096]
            tick=time.perf_counter();x=features(data,chunk);feature_seconds+=time.perf_counter()-tick
            tick=time.perf_counter();y=targets(data.bars,chunk,h,'future_log_return');target_seconds+=time.perf_counter()-tick
            assert x.shape==(len(chunk),17) and y.shape==(len(chunk),h) and np.isfinite(x).all() and np.isfinite(y).all()
            tick=time.perf_counter();sx+=x.sum(0);sxx+=x.T@x;sxy+=x.T@y;sy+=y.sum(0);n+=len(chunk);gram_seconds+=time.perf_counter()-tick
        full=n==len(idx);solve_begin=time.perf_counter()
        mean=sx/n;scale=np.sqrt(np.maximum(np.diag(sxx)/n-mean**2,0));scale=np.where(scale>1e-12,scale,1.)
        covariance=(sxx/n-np.outer(mean,mean))/np.outer(scale,scale)
        cross=(sxy/n-np.outer(mean,sy/n))/scale[:,None]
        beta=np.linalg.solve(covariance+.1*np.eye(17),cross)
        assert beta.shape==(17,h) and np.isfinite(beta).all()
        solve_seconds=time.perf_counter()-solve_begin;elapsed=time.perf_counter()-fit_begin
        row={'task':data.task,'fit_origins_total':len(idx),'benchmarked_origins':n,'full_fold5_fit':full,'data_load_seconds':load_elapsed,
             'feature_seconds':feature_seconds,'target_seconds':target_seconds,'gram_seconds':gram_seconds,'solve_seconds':solve_seconds,'fit_seconds':elapsed,
             'seconds_per_origin':elapsed/n,'features':17,'horizon':h,'lambda':.1,'matrix':'XTX/n + lambdaI on fit-standardized centered features','intercept':'unpenalized inner-fit target mean',
             'stop_target_reads':0,'outer_target_reads':0,'scores_computed':False}
        measurements.append(row)
        if time.time()>=stop:break
    # Conservative ten-fold/four-cell budget: count four complete passes even
    # though one streamed moment pass can supply all four nested feature cells.
    projected=[]
    for row in measurements:
        profile=row['task'].split('_',1)[0];count=sum(all_counts[profile].values())
        point=count*row['seconds_per_origin']
        projected.append({'profile':profile,'all_five_fold_fit_origins':count,'single_shared_moment_pass_seconds_estimate':point,
            'four_separate_fit_passes_seconds_estimate':4*point,'tenfold_margin_four_passes_seconds':40*point,
            'excludes':'process startup, stopping calibration, outer inference, intervals, replay and manuscript updates'})
    receipt={'status':'completed_inner_fit_cost_probe','measurements':measurements,'fit_counts':all_counts,'projected_cost':projected,
        'total_wall_seconds':time.time()-start,'wall_cap_seconds':30,'affinity':sorted(os.sched_getaffinity(0)),'GPU':False,
        'deadline_epoch':DEADLINE,'now_epoch':time.time(),'remaining_wall_seconds_at_completion':DEADLINE-time.time(),
        'benchmarked_parameters_are_not_selected_from_scores':True,'formal_inner_or_outer_scoring':False,'Q3_values_read':False,
        'source_sha256':sha(Path(__file__))}
    assert receipt['total_wall_seconds']<30,receipt['total_wall_seconds']
    (OUT/'FIT_COST.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=='__main__':main()
