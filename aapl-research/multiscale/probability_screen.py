"""Frozen, CPU-only probability direction screen. No model search or Q3 access."""
import os
for _k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import argparse,csv,hashlib,json,math,signal,subprocess,sys,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import optimize,special,stats,integrate,signal as sig
from multiscale.historical_design_audit import (load_metadata,complete_origins,reconstruct_folds,original_jobs,index_digest)
from multiscale.historical_results import ADMISSION
from multiscale.historical_coverage import digest,read_json,require

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/multiscale/probability_direction_screen/20261111_fast_screen'
RECIPES=['B0','B1','B2','C','A_clock','A_gap']
TASKS=[('minute',60,'minute'),('hourly',8,'hour')]
DF=5
CONST=special.gammaln(3)-special.gammaln(2.5)-.5*np.log(5*np.pi)
T0='2026-11-11T07:14:50+00:00'
STAGE1='2026-11-11T10:14:50+00:00'
TOTAL='2026-11-11T16:14:50+00:00'

def now():return datetime.now(timezone.utc).isoformat()
def atomic(path,obj):
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False));tmp.replace(path)
def npatomic(path,**kw):
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**kw)
    tmp.replace(path)
def deadline(stage=1):
    end=STAGE1 if stage==1 else TOTAL
    if datetime.now(timezone.utc)>=datetime.fromisoformat(end):raise TimeoutError('Fixed deadline reached')
def status(stage,**kw):
    atomic(OUT/'STATUS.json',dict(T0=T0,stage1_deadline=STAGE1,total_deadline=TOTAL,updated=now(),stage=stage,Q3_read=False,**kw))
def affinity():
    cpus=sorted(os.sched_getaffinity(0))[:2];os.sched_setaffinity(0,cpus);return cpus

def structure(meta,h):
    sess,starts,counts=np.unique(meta.sessions,return_index=True,return_counts=True)
    ends=starts+counts-1;day=np.repeat(np.arange(len(sess)),counts)
    # Complete native-bar sessions; first increment uses previous session's last retained close.
    good=np.array([meta.observed[max(0,s-1):e+1].all() and s>0 for s,e in zip(starts,ends)])
    full=np.zeros(len(sess),bool)
    for d in range(21,len(sess)):full[d]=good[d-20:d].all()
    missing=np.r_[0,np.cumsum(~meta.observed)]
    idx=np.arange(len(day));left=np.maximum(0,idx-h)
    support=(idx>=h)&(missing[idx+1]==missing[left])&full[day]
    return day,starts,ends,support

def freeze():
    path=OUT/'EXECUTION_PROTOCOL.json'
    old=read_json(path)
    if old.get('frozen'):return old
    admission=read_json(ADMISSION/'admission.json');jobs,protocols=original_jobs(digest(ADMISSION/'admission.json'))
    items=[];sources={}
    for task,h,freq in TASKS:
        meta,source=load_metadata(ADMISSION,admission,freq);sources[freq]=source
        day,starts,ends,support=structure(meta,h)
        j=jobs[task,h,'future_log_return',1]
        origins=complete_origins(meta,j['lookback'],h,task=='minute')
        parts=reconstruct_folds(meta,origins,h,'rolling')
        for fold,part in enumerate(parts,1):
            orig=jobs[task,h,'future_log_return',fold]
            require({k:index_digest(v) for k,v in part.items()}==orig['hashes'],'Original split identity mismatch')
            selected={k:v[support[v]] for k,v in part.items()}
            require(len(selected['fit'])>=24 and len(selected['validation'])>0,'History support insufficient')
            fn=f'{task}_f{fold}_origins.npz';npatomic(OUT/fn,**selected)
            items.append(dict(task=task,horizon=h,frequency=freq,fold=fold,indices_file=fn,
                original_split_sha256=orig['hashes'],selected_sha256={k:index_digest(v) for k,v in selected.items()},
                counts={k:len(v) for k,v in selected.items()},original_counts={k:len(v) for k,v in part.items()},
                original_protocol=orig['protocol']))
    p=dict(T0=T0,stage1_deadline=STAGE1,total_deadline=TOTAL,frozen=True,frozen_at=now(),
        historical_development_only=True,Q3_read=False,sources=sources,original_protocols=protocols,items=items,
        code_sha256=digest(Path(__file__)),df=5,location=0,L2=.01,
        optimizer={'method':'L-BFGS-B','maxiter':1000,'ftol':1e-12,'gtol':1e-8,'maxls':50,'initial':'fit sample variance intercept, zero slopes'},
        numerical={'variance_floor':1e-12,'variance_ceiling':1.,'log_feature_floor':1e-16,'scale_squared_to_variance':.6},
        history={'short':'mean of last H native log-return increments squared, including increment ending at origin',
          'day':'mean squared native increments in last completed session strictly before origin session',
          'long':'mean squared native increments in 20 completed sessions strictly before origin session',
          'missing':'no fill; common mask requires H increments plus 20 full prior native sessions and previous close',
          'early_support':'exclude unsupported fit and evaluation origins by metadata-only common mask; keep original fold boundaries; never shorten windows',
          'overnight':'first increment each session uses previous session last retained close; includes overnight; hourly also includes discarded 30-minute residual in first next-session increment',
          'hour_residual':'original 60 minute full bars only, residual excluded as separate bars',
          'split':'known-at-origin cumulative split adjustments to historical increments; no admitted 2021+ split events',
          'clock':'completed bar end NY, fraction (minutes since 09:30)/390, including shortened days on same denominator',
          'gap':'count session-boundary transitions in next H calendar bar positions, metadata only',
          'EWMA':'recursive .94 previous + .06 latest squared native increment, initialized at first valid squared increment; zero-history support prevents early initialization artifacts'},
        standardization='fit-only log historical means, unpenalized intercept; clock/gap also fit standardized; constant columns omitted',
        fit='original purged inner-fit intersect common support only; stop never used',
        scoring='all stage recipes fitted and committed before any stage outer labels materialized',
        screen={'each_fold_C_below':['B0','B1'],'pooled_NLL_reduction_vs_B0_B1_min':.01,'pooled_C_no_worse_B2':True,'C_coverage90':[.87,.93]},
        resources={'cpu_ids':affinity(),'logical_cpu_limit':16,'actual_fit_processes':1,'threads_per_process':2,'GPU':False,'paid':False},
        sources_review_required=True)
    atomic(path,p);status('protocol_frozen',completed=0,planned=24)
    return p

def features(item,p):
    freq=item['frequency'];path=ADMISSION/f'bars_{freq}.npz'
    require(digest(path)==p['sources'][freq]['sha256'],'Bars identity changed')
    with np.load(path,allow_pickle=False) as z:
        close=z['close'];observed=z['observed'];sessions=z['session_date'];end=z['end_ns']
    require(sessions[-1]<='2026-06-30','Q3 cutoff')
    logp=np.log(close);ret=np.r_[np.nan,np.diff(logp)]
    # No split inside admitted intraday support, verified against reviewed action list.
    actions=read_json(ADMISSION/'admission.json')['split_actions']
    require(not any(sessions[0]<=a['date']<=sessions[-1] for a in actions),'Unexpected in-support split')
    good=observed & np.r_[False,observed[:-1]]
    r2=np.where(good,ret*ret,np.nan)
    valid=np.isfinite(r2);sq=np.r_[0,np.cumsum(np.where(valid,r2,0))]
    sess,starts,counts=np.unique(sessions,return_index=True,return_counts=True);ends=starts+counts-1
    day=np.repeat(np.arange(len(sess)),counts);idx=np.arange(len(close));h=item['horizon']
    short=(sq[idx+1]-sq[np.maximum(0,idx-h+1)])/h
    dstat=np.full(len(sess),np.nan);lstat=dstat.copy()
    for d in range(21,len(sess)):
        dstat[d]=(sq[starts[d]]-sq[starts[d-1]])/counts[d-1]
        lstat[d]=(sq[starts[d]]-sq[starts[d-20]])/(starts[d]-starts[d-20])
    local=pd.to_datetime(end,utc=True).tz_convert('America/New_York')
    u=(local.hour.to_numpy()*60+local.minute.to_numpy()-570)/390
    gap=day[np.minimum(idx+h,len(idx)-1)]-day
    raw=np.column_stack([np.log(np.maximum(short,1e-16)),np.log(np.maximum(dstat[day],1e-16)),np.log(np.maximum(lstat[day],1e-16)),np.sin(2*np.pi*u),np.cos(2*np.pi*u),gap])
    safe=np.where(valid,r2,0);first=np.flatnonzero(valid)[0]
    ew=np.zeros(len(close));ew[first:]=sig.lfilter([.06],[1,-.94],safe[first:],zi=[.94*safe[first]])[0]
    ew=np.maximum(h*ew,1e-12)
    return raw,ew,logp,sessions,gap

def design(recipe,raw,fit,ev,ew):
    if recipe=='B0':cols=[]
    elif recipe=='B1':cols=[]
    elif recipe=='B2':cols=[0,1,2]
    elif recipe=='C':cols=list(range(6))
    elif recipe=='A_clock':cols=[0,1,2,5]
    else:cols=[0,1,2,3,4]
    mean=raw[fit][:,cols].mean(axis=0);sd=raw[fit][:,cols].std(axis=0)
    keep=sd>1e-12;cols=np.array(cols,dtype=int)[keep];mean=mean[keep];sd=sd[keep]
    def x(ii):return np.column_stack([np.ones(len(ii)),(raw[ii][:,cols]-mean)/sd])
    xf,xe=x(fit),x(ev)
    require(np.isfinite(xf).all() and np.isfinite(xe).all(),'Nonfinite supported features')
    offsets=(np.log(ew[fit]),np.log(ew[ev])) if recipe=='B1' else (np.zeros(len(fit)),np.zeros(len(ev)))
    return xf,xe,offsets,dict(columns=cols.tolist(),mean=mean.tolist(),std=sd.tolist())

def loss_and_grad(b,x,offset,y):
    eta=x@b+offset;clipped=np.clip(eta,np.log(1e-12),0);var=np.exp(clipped);s2=.6*var
    q=y*y/(5*s2)
    nll=-CONST+.5*np.log(s2)+3*np.log1p(q)
    g=(.5-3*q/(1+q))*((eta>np.log(1e-12))&(eta<0))
    return nll.mean()+.01*np.dot(b[1:],b[1:]),x.T@g/len(y)+np.r_[0,.02*b[1:]]

def fit_unit(item,recipe,p):
    start=time.monotonic();deadline(1 if item['fold']<=2 else 2)
    key=f"{item['task']}_f{item['fold']}_{recipe}";fn=OUT/(key+'.json')
    if fn.exists():return read_json(fn)
    if item['task']=='minute' and recipe=='A_gap':
        c=read_json(OUT/f"minute_f{item['fold']}_C.json");c=dict(c,recipe=recipe,reused_from='C');atomic(fn,c);return c
    raw,ew,logp,sessions,gap=features(item,p)
    with np.load(OUT/item['indices_file']) as z:fit=z['fit'];ev=z['validation']
    xf,xe,offset,transform=design(recipe,raw,fit,ev,ew)
    # Only the inner-fit target is constructed here.
    y=logp[fit+item['horizon']]-logp[fit]
    initial=np.zeros(xf.shape[1]);initial[0]=np.log(max(np.mean(y*y),1e-12))-np.mean(offset[0])
    result=optimize.minimize(loss_and_grad,initial,args=(xf,offset[0],y),jac=True,method='L-BFGS-B',callback=lambda b:deadline(1 if item['fold']<=2 else 2),options={'maxiter':1000,'ftol':1e-12,'gtol':1e-8,'maxls':50})
    require(result.success and np.isfinite(result.fun),'Optimizer qualification failed: '+str(result.message))
    var=np.exp(np.clip(xe@result.x+offset[1],np.log(1e-12),0))
    require(np.isfinite(var).all() and np.all((var>1e-12)&(var<1)),'Variance touches numerical bound')
    npatomic(OUT/(key+'.npz'),origins=ev,variance=var)
    record=dict(task=item['task'],fold=item['fold'],recipe=recipe,status='fitted',beta=result.x.tolist(),transform=transform,objective=float(result.fun),iterations=int(result.nit),gradient_inf=float(np.max(np.abs(result.jac))),seconds=time.monotonic()-start,completed_at=now(),protocol_sha256=digest(OUT/'EXECUTION_PROTOCOL.json'),prediction_sha256=digest(OUT/(key+'.npz')))
    atomic(fn,record);status('fit_checkpoint',last_committed=key,completed=len(list(OUT.glob('*_B*.json')))+len(list(OUT.glob('*_C.json')))+len(list(OUT.glob('*_A*.json'))),planned=24 if item['fold']<=2 else 60)
    return record

# Closed-form CRPS for Student t, with a separately integrated validation.
CRPS_K=2*np.sqrt(5)/(4)*special.beta(.5,4.5)/special.beta(.5,2.5)**2

def metric_arrays(y,var):
    s=np.sqrt(.6*var);z=y/s;pdf=stats.t.pdf(z,5);cdf=stats.t.cdf(z,5)
    nll=-stats.t.logpdf(z,5)+np.log(s)
    crps=s*(z*(2*cdf-1)+2*pdf*(5+z*z)/4-CRPS_K)
    q50=stats.t.ppf(.75,5);q90=stats.t.ppf(.95,5)
    return dict(NLL=nll,CRPS=crps,coverage50=(np.abs(z)<=q50).astype(float),coverage90=(np.abs(z)<=q90).astype(float),width50=2*q50*s,width90=2*q90*s,PIT=cdf)

def numerical():
    xx=np.array([[1.,.2,-.6],[1.,-.8,.7],[1.,.3,.4]])
    yy=np.array([.003,-.02,.01]);bb=np.array([-9.,.1,-.2]);off=np.zeros(3)
    fun=lambda b:loss_and_grad(b,xx,off,yy)[0]
    analytic=loss_and_grad(bb,xx,off,yy)[1];numeric=optimize._numdiff.approx_derivative(fun,bb,method='3-point').ravel()
    require(np.max(np.abs(analytic-numeric))<1e-7,'NLL gradient mismatch')
    checks=[]
    for z in [-4.,-1.,0.,.3,2.]:
        actual=metric_arrays(np.array([z]),np.array([5/3]))['CRPS'][0]
        independent=integrate.quad(lambda x:stats.t.cdf(x,5)**2,-np.inf,z,epsabs=1e-10)[0]+integrate.quad(lambda x:stats.t.sf(x,5)**2,z,np.inf,epsabs=1e-10)[0]
        require(abs(actual-independent)<1e-8,'CRPS quadrature mismatch')
        checks.append({'z':z,'closed':actual,'quadrature':independent})
    for q in [.05,.25,.75,.95]:require(abs(stats.t.cdf(stats.t.ppf(q,5),5)-q)<1e-10,'Quantile inversion failed')
    out=dict(passed=True,gradient_max_error=float(np.max(np.abs(analytic-numeric))),CRPS=checks,quantiles=True,checked_at=now())
    atomic(OUT/'NUMERICAL_QUALIFICATION.json',out);return out

def score(items,p):
    rows=[];bygap=[];replays=[]
    for item in items:
        raw,ew,logp,sessions,gap=features(item,p)
        with np.load(OUT/item['indices_file']) as z:fit=z['fit'];ev=z['validation']
        y=logp[ev+item['horizon']]-logp[ev]
        for recipe in RECIPES:
            key=f"{item['task']}_f{item['fold']}_{recipe}";record=read_json(OUT/(key+'.json'))
            xf,xe,off,transform=design(recipe,raw,fit,ev,ew)
            var=np.exp(np.clip(xe@np.array(record['beta'])+off[1],np.log(1e-12),0))
            predkey=key if recipe!='A_gap' or item['task']!='minute' else f"minute_f{item['fold']}_C"
            with np.load(OUT/(predkey+'.npz')) as z:
                require(np.array_equal(z['origins'],ev) and np.allclose(z['variance'],var,rtol=1e-13,atol=0),'Parameter replay failed')
            m=metric_arrays(y,var)
            # Independent density expression and quantile CDF identity, every saved recipe.
            z=y/np.sqrt(.6*var)
            independent=.5*np.log(5*np.pi*.6*var)+math.lgamma(2.5)-math.lgamma(3)+3*np.log1p(z*z/5)
            err=float(np.max(np.abs(m['NLL']-independent)));require(err<1e-10,'Independent NLL mismatch')
            replays.append(dict(task=item['task'],fold=item['fold'],recipe=recipe,NLL_max_abs_error=err,variance_replayed=True))
            npatomic(OUT/(key+'_scored.npz'),origins=ev,y=y,session=sessions[ev],gap=gap[ev],variance=var,**m)
            row=dict(task=item['task'],fold=item['fold'],recipe=recipe,n_origins=len(ev),**{k:float(v.mean()) for k,v in m.items()});rows.append(row)
            for g in np.unique(gap[ev]):
                mask=gap[ev]==g;bygap.append(dict(task=item['task'],fold=item['fold'],recipe=recipe,n_gap=int(g),n_origins=int(mask.sum()),**{k:float(v[mask].mean()) for k,v in m.items()}))
    for name,data in [('RESULTS.csv',rows),('RESULTS_BY_GAP.csv',bygap)]:
        with (OUT/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    atomic(OUT/'INDEPENDENT_REPLAY.json',dict(passed=True,checks=replays))
    return rows

def assess(rows):
    gates={};pooled={}
    for task,h,freq in TASKS:
        rr=[r for r in rows if r['task']==task];means={}
        for recipe in RECIPES:
            a=[r for r in rr if r['recipe']==recipe];n=sum(r['n_origins'] for r in a)
            means[recipe]={k:sum(r[k]*r['n_origins'] for r in a)/n for k in ['NLL','CRPS','coverage50','coverage90','width50','width90']};means[recipe]['n_origins']=n
        c=means['C'];g={}
        for base in ['B0','B1']:
            g['C_below_'+base+'_every_fold']=all(r['NLL']<next(b['NLL'] for b in rr if b['fold']==r['fold'] and b['recipe']==base) for r in rr if r['recipe']=='C')
            g['pooled_reduction_'+base+'_at_least_0.01']=means[base]['NLL']-c['NLL']>=.01
        g['pooled_C_no_worse_B2']=c['NLL']<=means['B2']['NLL'];g['coverage90_in_87_93']=.87<=c['coverage90']<=.93
        g['passed']=all(g.values());gates[task]=g;pooled[task]=means
    return all(g['passed'] for g in gates.values()),gates,pooled

def block_ci(task,base,block,metric='NLL',coverage=False):
    rng=np.random.default_rng(2026);rep=np.zeros(5000);n=0
    for fold in range(1,6):
        with np.load(OUT/f'{task}_f{fold}_C_scored.npz') as z:days=z['session'];x=z[metric]
        if not coverage:
            with np.load(OUT/f'{task}_f{fold}_{base}_scored.npz') as z:x=z[metric]-x
        _,inv=np.unique(days,return_inverse=True);sums=np.bincount(inv,weights=x);counts=np.bincount(inv);nd=len(sums)
        # Circular moving trading-day blocks within fold strata; pool sums/counts.
        bcount=math.ceil(nd/block);values=np.zeros(5000);weights=np.zeros(5000)
        for r in range(5000):
            starts=rng.integers(nd,size=bcount);ix=((starts[:,None]+np.arange(block))%nd).ravel()[:nd]
            values[r]=sums[ix].sum();weights[r]=counts[ix].sum()
        # Total sampled origin weights accumulated across strata below.
        if fold==1:totals=values;denom=weights
        else:totals+=values;denom+=weights
    return np.quantile(totals/denom,[.025,.975]).tolist()

def full_assess(rows,pooled):
    details={}
    for task,h,freq in TASKS:
        g={};c=pooled[task]['C'];blocks=[1,5] if task=='minute' else [2,5]
        for base in ['B0','B1','B2']:
            cis={str(b):block_ci(task,base,b) for b in blocks}
            wins=sum(next(r['NLL'] for r in rows if r['task']==task and r['fold']==f and r['recipe']==base)>next(r['NLL'] for r in rows if r['task']==task and r['fold']==f and r['recipe']=='C') for f in range(1,6))
            g[base]=dict(NLL_reduction=pooled[task][base]['NLL']-c['NLL'],positive_folds=wins,block_CI95=cis,CRPS_ratio=c['CRPS']/pooled[task][base]['CRPS'])
            g[base]['passed']=g[base]['NLL_reduction']>=.01 and wins>=4 and all(ci[0]>0 for ci in cis.values()) and g[base]['CRPS_ratio']<=1.01
        g['coverage50_passed']=.47<=c['coverage50']<=.53;g['coverage90_passed']=.87<=c['coverage90']<=.93
        g['coverage_CI95']={m:{str(b):block_ci(task,None,b,m,True) for b in blocks} for m in ['coverage50','coverage90']}
        g['ablations']={base:{str(b):block_ci(task,base,b) for b in blocks} for base in ['A_clock','A_gap']}
        g['passed']=all(g[b]['passed'] for b in ['B0','B1','B2']) and g['coverage50_passed'] and g['coverage90_passed'];details[task]=g
    return all(v['passed'] for v in details.values()),details

def deliver(p,rows,screen,gates,pooled,full=None,error=None):
    completed=len(rows);end=now();elapsed=(datetime.fromisoformat(end)-datetime.fromisoformat(T0)).total_seconds()
    numerical_ok=(OUT/'NUMERICAL_QUALIFICATION.json').exists() and (OUT/'RECOVERY_TEST.json').exists() and (OUT/'INDEPENDENT_REPLAY.json').exists()
    decision=dict(numerical_qualified=numerical_ok,screen_passed=screen,screen_gates=gates,pooled=pooled,full_development_gates=full,
       completed=completed,planned=60 if screen else 24,stage='engineering_incomplete' if error else 'completed_stage2' if screen else 'completed_stage1_stop',
       failure_reason=error or (None if screen else 'At least one frozen per-task screen condition failed'),historical_development_only=True,Q3_read=False,
       original_mean_goal_completed=False,T0=T0,finished_at=end,elapsed_seconds=elapsed,resources=p.get('resources',{}),action='STOP',
       scientific_status='not_evaluable' if error else 'full_development_pass' if full and all(x['passed'] for x in full.values()) else 'candidate_failed_frozen_gate')
    atomic(OUT/'DECISION.json',decision)
    lines=['# 概率预测有限筛查结果','', '**停止本轮实验。** '+('工程或资格未完成：'+error if error else '候选未通过冻结筛查；不运行剩余三折。' if not screen else '固定五折执行完毕；本轮不启动独立确认。'),'',f'T0：{T0}；结束：{end}；耗时 {elapsed/60:.2f} 分钟。完成 {completed}/{decision["planned"]} 个位置（分钟 A_gap 复用 C）。数值、参数重放与真实中断恢复资格：{numerical_ok}。','']
    for task,means in pooled.items():
        lines += [f'## {task}（按 origin 加权；NLL 单位 nat/origin）','','|配方|NLL|相对 B0 的 NLL 降低|CRPS|50% 覆盖|90% 覆盖|50% 宽度|90% 宽度|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for recipe,m in means.items():lines.append(f'|{recipe}|{m["NLL"]:.8f}|{means["B0"]["NLL"]-m["NLL"]:.8f}|{m["CRPS"]:.8f}|{m["coverage50"]:.2%}|{m["coverage90"]:.2%}|{m["width50"]:.8f}|{m["width90"]:.8f}|')
        lines+=['', 'C 相对 B0/B1/B2 的 NLL 降低：'+', '.join(f'{b}: {means[b]["NLL"]-means["C"]["NLL"]:.8f}' for b in ['B0','B1','B2'])+'。','', '筛查门：`'+json.dumps(gates.get(task,{}),ensure_ascii=False)+'`。','']
        signals=', '.join(f'{b} 相对 B0 降低 {means["B0"]["NLL"]-means[b]["NLL"]:.8f}' for b in ['B1','B2'])
        lines+=['经典模型开发信号：'+signals+'；此处为描述性历史差值，未通过独立确认。','']
    lines+=['## 范围与局限','','全部评分是原始已曝光 outer 的历史开发证据；Q3 未读。收益中心固定为零，概率结果不能替代原收益均值目标，原均值目标仍未完成。',
      '消融只包含已冻结的 A_clock、A_gap；分钟跨日变量恒零，A_gap 直接复用 C；小时跨日次数按已知日历计算。筛查失败时不开展五折 bootstrap，也不宣称时钟或跨日机制成立。',
      '若经典模型优于常数分布，只说明本开发片段有条件尺度信号，不等于候选具有增量。未通过本轮候选门时不值得继续本配方的追加拟合；不据此断言收益或波动本质不可预测。',
      '来源限制：沿用本地个人非商业开发准入，公开发表权未确认。小时只保留完整小时，首个次日增量含隔夜和被排除尾段；窗口无静默缩短。',
      '资源：单拟合进程、最多 2 个逻辑 CPU、BLAS 2 线程；GPU 关闭，无付费，无子代理，未重启其他研究。','', '明细见 RESULTS.csv、RESULTS_BY_GAP.csv、EXECUTION_PROTOCOL.json、INDEPENDENT_REPLAY.json 和单元参数/逐 origin 结果。']
    (OUT/'FINAL_RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    status('finished',decision=decision['stage'],completed=completed,planned=decision['planned'],failure_reason=decision['failure_reason'])

def run():
    affinity();p=freeze();rows=[];screen=False;gates={};pooled={};full=None
    try:
        deadline();numerical()
        require((OUT/'SOURCE_REVIEW.md').exists(),'Original source review not completed')
        require((OUT/'RECOVERY_TEST.json').exists(),'Actual interruption/recovery qualification missing')
        # Cost probe is one actual first-fold unit, no extra fitting or tuning.
        first=next(i for i in p['items'] if i['task']=='minute' and i['fold']==2);t=time.monotonic();probe=fit_unit(first,'C',p)
        atomic(OUT/'COST_PROBE.json',dict(elapsed_seconds=time.monotonic()-t,fit_origins=first['counts']['fit'],evaluation_origins=first['counts']['validation'],probe_recipe='C',maximum_stage1_shape=True,probe_actual_fit_seconds=probe['seconds'],maximum_planned_fit_origins=max(i['counts']['fit'] for i in p['items']),selection='one process two cores; no concurrency needed',included_in_budget=True))
        stage=[i for i in p['items'] if i['fold']<=2]
        for item in stage:
            for recipe in RECIPES:fit_unit(item,recipe,p)
        status('all_stage1_fitted_before_scoring',completed=24,planned=24)
        deadline();rows=score(stage,p);screen,gates,pooled=assess(rows)
        atomic(OUT/'STAGE1_DECISION.json',dict(screen_passed=screen,gates=gates,pooled=pooled,decided_at=now()))
        status('stage1_pass' if screen else 'stage1_stop',completed=24,planned=60 if screen else 24)
        if screen:
            fit_start=time.monotonic()
            for item in [i for i in p['items'] if i['fold']>2]:
                for recipe in RECIPES:
                    require(time.monotonic()-fit_start<4*3600,'Remaining fit four-hour ceiling reached');fit_unit(item,recipe,p)
            deadline(2);rows=score(p['items'],p);_,_,pooled=assess(rows);passed,full=full_assess(rows,pooled)
        deliver(p,rows,screen,gates,pooled,full)
    except Exception as e:
        atomic(OUT/'FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),time=now()))
        deliver(p,rows,screen,gates,pooled,full,repr(e));raise

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--freeze',action='store_true');a.add_argument('--recovery-child',action='store_true');a.add_argument('--recovery-test',action='store_true');args=a.parse_args()
    if args.freeze:freeze()
    elif args.recovery_child:
        affinity();p=freeze();fit_unit(p['items'][0],'B0',p)
        print('CHECKPOINT_COMMITTED',flush=True)
        # Actual process gets terminated by the parent, then restarted with the same protocol/deadline.
        time.sleep(60)
    elif args.recovery_test:
        affinity();freeze();cmd=[sys.executable,'-m','multiscale.probability_screen','--recovery-child']
        child=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        line=child.stdout.readline();require('CHECKPOINT_COMMITTED' in line,'Child failed checkpoint')
        checkpoint=OUT/'minute_f1_B0.json';before=digest(checkpoint);child.terminate();child.wait(timeout=10)
        p=freeze();t=time.monotonic();fit_unit(p['items'][0],'B0',p);require(digest(checkpoint)==before,'Committed checkpoint refitted')
        # Restart a real second process; verify it also resumes without refit.
        resumed=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        require('CHECKPOINT_COMMITTED' in resumed.stdout.readline(),'Restart failed');resumed.terminate();resumed.wait(timeout=10)
        require(digest(checkpoint)==before,'Restart rewrote committed cell')
        atomic(OUT/'RECOVERY_TEST.json',dict(passed=True,real_child_terminated=True,returncode=child.returncode,restarted_process=True,checkpoint_unchanged=True,resume_seconds=time.monotonic()-t,original_T0=T0,original_deadline=TOTAL,tested_at=now()))
    else:run()
