"""Fit-only fixed-weight BatchNorm moment intervention; original study read-only."""
from pathlib import Path
import sys, os, time, signal, json, copy, hashlib
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
V3=HERE.parent.parent/'20261109_v3'
sys.path.insert(0,str(V3/'code'))
from study import read, atomic, npz, sha, now, make_job, task_key, source_hashes
from engine import Data
from networks import create
from multiscale.bars import targets
from multiscale.training import atomic_torch_save
from portable_stats import day_groups, paired_intervals

RULES=HERE/'BN_RULES.json'
DEADLINE=1794205909.6228158
LAYERS=('backbone.bn1','backbone.bn2','backbone.bn3')
STOP=False

class Pause(Exception): pass
class Captured(Exception): pass

def interrupt(*_):
    global STOP
    STOP=True

def init():
    assert sorted(os.sched_getaffinity(0))==[24,26]
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,interrupt)
    frozen=HERE/'SOURCE_FREEZE.json'
    identity={'diagnostic.py':sha(__file__),'BN_RULES.json':sha(RULES),
              'original_source_hashes':source_hashes(),
              'original_code_freeze_sha256':sha(V3/'protocol/CODE_FREEZE.json'),
              'original_budget_sha256':sha(V3/'protocol/BUDGET.json'),
              'deadline_epoch':DEADLINE}
    assert identity['original_source_hashes']==read(V3/'protocol/CODE_FREEZE.json')['hashes']
    assert read(V3/'protocol/BUDGET.json')['deadline_epoch']==DEADLINE
    if frozen.exists():assert read(frozen)['identity']==identity
    else:atomic(frozen,{'identity':identity,'frozen_at_utc':now(),'frozen_epoch':time.time(),
                       'adapter_outer_evaluation_seen':False})

def cfg_for(profile,h,fold,seed):
    return make_job(profile,h,'future_log_return','xpatch','anchor_baseline',fold,seed)

def source(cfg):
    folder=V3/'runs'/cfg['id'];done=read(folder/'DONE.json')
    assert done['status']=='completed'
    assert sha(folder/'best_trained.pt')==done['checkpoint_sha256']
    saved=torch.load(folder/'best_trained.pt',map_location='cpu',weights_only=True)
    assert saved['cfg']==cfg
    return folder,done,saved

def load_net(cfg,saved,data):
    net=create(cfg,data.scaler);net.load_state_dict(saved['model']);net.eval()
    assert all(not module.training for module in net.modules())
    return net

def args_only(data,indices):
    # Deliberately bypass Data.batch: it constructs future labels.
    x,a=data.windows.arrays(indices)
    x=((x-data.scaler['x_mean'])/data.scaler['x_std']).astype('float32')
    a=((a-data.scaler['anchor_mean'])/data.scaler['anchor_std']).astype('float32')
    return torch.from_numpy(x[:,:,None]),torch.from_numpy(a)

def forward(data,net,indices,batch=128):
    out=[]
    with torch.inference_mode():
        for start in range(0,len(indices),batch):
            if STOP or time.time()>=DEADLINE:raise Pause()
            idx=indices[start:start+batch]
            out.append(data.invert(net(*args_only(data,idx)).numpy(),idx))
    return np.concatenate(out)

def compare_states(original,adapted):
    allowed={name+'.'+suffix for name in LAYERS for suffix in ('running_mean','running_var')}
    changed=[]
    for key,value in original.items():
        equal=torch.equal(value,adapted[key])
        if not equal:
            assert key in allowed,key
            changed.append(key)
    return changed

def calibrate(data,pred,indices):
    truth=targets(data.bars,indices,data.cfg['horizon'],data.cfg['target'])[:,-1]
    p=pred[:,-1];num=float(p@truth);den=float(p@p);base=float(truth@truth)
    alpha=float(np.clip(num/den,0,1)) if den else 0.
    return {'n':len(truth),'raw_mse':float(np.mean((p-truth)**2)),
            'shrunk_mse':float(np.mean((alpha*p-truth)**2)), 'zero_mse':base/len(truth),
            'alpha':alpha,'forecast_energy':den/len(truth),'alignment':num/len(truth)}

def cell(cfg):
    out=HERE/'cells'/cfg['id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'INNER_DONE.json').exists():return read(out/'INNER_DONE.json')
    start=time.time();folder,done,saved=source(cfg);data=Data(cfg)
    net=load_net(cfg,saved,data);original={k:v.clone() for k,v in net.state_dict().items()}
    identity={'cfg':cfg,'checkpoint_sha256':done['checkpoint_sha256'],
              'source_freeze_sha256':sha(HERE/'SOURCE_FREEZE.json'),'rules_sha256':sha(RULES)}
    latest=out/'moments_latest.pt'
    state={'identity':identity,'layer':0,'cursor':0,'count':0,'mean':None,'M2':None,
           'model':copy.deepcopy(net.state_dict()),'completed_layers':[],'elapsed_seconds':0.}
    if latest.exists():
        state=torch.load(latest,map_location='cpu',weights_only=True)
        assert state['identity']==identity
        net.load_state_dict(state['model'])
    previous_elapsed=float(state['elapsed_seconds'])
    def save():
        state['model']=copy.deepcopy(net.state_dict())
        state['elapsed_seconds']=previous_elapsed+time.time()-start
        atomic_torch_save(state,latest)
        atomic(out/'STATE.json',{'identity':identity,'layer':state['layer'],'cursor':state['cursor'],
              'count':state['count'],'fit_origins':len(data.splits['fit']),
              'elapsed_seconds':state['elapsed_seconds'],'deadline_epoch':DEADLINE,'at_utc':now()})
    indices=data.splits['fit']
    while state['layer']<len(LAYERS):
        name=LAYERS[state['layer']];layer=dict(net.named_modules())[name]
        capture={}
        def hook(module,args):
            capture['x']=args[0].detach()
            raise Captured()
        handle=layer.register_forward_pre_hook(hook)
        try:
            with torch.inference_mode():
                while state['cursor']<len(indices):
                    if STOP or time.time()>=DEADLINE:
                        save();raise Pause()
                    end=min(state['cursor']+128,len(indices));idx=indices[state['cursor']:end]
                    try:net(*args_only(data,idx))
                    except Captured:pass
                    assert 'x' in capture
                    x=capture.pop('x').to(torch.float64)
                    assert x.ndim==3 and x.shape[1]==layer.num_features
                    count=x.shape[0]*x.shape[2]
                    mean=x.mean((0,2));M2=((x-mean[None,:,None])**2).sum((0,2))
                    if state['count']==0:
                        state['mean']=mean;state['M2']=M2;state['count']=count
                    else:
                        old=state['count'];total=old+count;delta=mean-state['mean']
                        state['mean']+=delta*(count/total)
                        state['M2']+=M2+delta**2*(old*count/total)
                        state['count']=total
                    state['cursor']=end
                    save()
            assert state['count']>1
            variance=state['M2']/(state['count']-1)
            old_mean=layer.running_mean.clone();old_var=layer.running_var.clone()
            layer.running_mean.copy_(state['mean'].to(layer.running_mean.dtype))
            layer.running_var.copy_(variance.to(layer.running_var.dtype))
            state['completed_layers'].append({'layer':name,'fit_origins':len(indices),
                'activation_count_per_channel':state['count'],'old_running_mean':old_mean.tolist(),
                'old_running_var':old_var.tolist(),'pooled_mean_float64':state['mean'].tolist(),
                'pooled_unbiased_var_float64':variance.tolist(),
                'running_mean_rms_change':float(((old_mean-layer.running_mean)**2).mean().sqrt()),
                'running_var_rms_change':float(((old_var-layer.running_var)**2).mean().sqrt())})
            state.update(layer=state['layer']+1,cursor=0,count=0,mean=None,M2=None);save()
        finally:handle.remove()
    changed=compare_states(original,net.state_dict())
    # Predicting completed windows reads no labels. Compare numerical batching.
    check=data.splits['stop'][:19];p19=forward(data,net,check,19)
    errors={}
    for batch in (1,7):
        value=forward(data,net,check,batch)
        assert np.allclose(value,p19,rtol=1e-6,atol=1e-9)
        errors[str(batch)]=float(np.max(abs(value-p19)))
    adapted=copy.deepcopy(saved);adapted['model']=copy.deepcopy(net.state_dict())
    adapted['bn_intervention']={'identity':identity,'changed_buffers':changed,'moment_labels_read':False}
    atomic_torch_save(adapted,out/'adapted.pt')
    atomic(out/'MOMENTS_DONE.json',{'status':'completed','identity':identity,'layers':state['completed_layers'],
        'changed_buffers':changed,'all_parameters_byte_equal':True,'all_other_buffers_byte_equal':True,
        'moment_labels_read':False,'moment_estimation_seconds':state['elapsed_seconds'],
        'batch_independence_allclose':True,'batch_max_abs_difference':errors,
        'adapted_checkpoint_sha256':sha(out/'adapted.pt'),'at_utc':now()})
    # First label use: original inner-stop partition, after moments are saved.
    si=data.splits['stop'];pa=forward(data,net,si)
    net.load_state_dict(original);po=forward(data,net,si)
    inner={'status':'completed','identity':identity,'fit_origins':len(indices),
           'stop_origins':len(si),'original':calibrate(data,po,si),'adapted':calibrate(data,pa,si),
           'elapsed_seconds':previous_elapsed+time.time()-start,'outer_labels_read':False,
           'adapted_checkpoint_sha256':sha(out/'adapted.pt'),'at_utc':now()}
    old=read(V3/'calibration_candidate'/cfg['id']/'DONE.json')
    assert np.isclose(inner['original']['alpha'],old['alpha'],rtol=1e-8,atol=1e-10)
    assert np.isclose(inner['original']['raw_mse'],old['unshrunk_inner_stop_mse'],rtol=1e-8,atol=1e-10)
    atomic(out/'INNER_DONE.json',inner)
    print(json.dumps({'inner_done':cfg['id'],'original':inner['original']['raw_mse'],
                     'adapted':inner['adapted']['raw_mse']}),flush=True)
    return inner

def frozen_gate():
    path=HERE/'GATE_AND_ADMISSION.json'
    if path.exists():return read(path)
    records={};screen=[];admitted=[];remaining_cost=0.
    for profile,h in (('minute',60),('hourly',8)):
        task=task_key(profile,h,'future_log_return')
        rows=[cell(cfg_for(profile,h,f,2026)) for f in (1,2)]
        screen+=rows
        passed=all(r['adapted']['raw_mse']<r['original']['raw_mse'] for r in rows)
        rate=max(r['elapsed_seconds']/r['fit_origins'] for r in rows)
        remaining=[cfg_for(profile,h,f,s) for f in range(1,6) for s in (2026,2027,2028)
                   if not (f in (1,2) and s==2026)]
        # Fit counts vary only by fold, known before any adapter outer score.
        counts={f:len(Data(cfg_for(profile,h,f,2026)).splits['fit']) for f in range(1,6)}
        complete_estimate=1.5*rate*sum(counts[c['fold']] for c in remaining)
        outer_estimate=1.5*rate*sum(counts[f]*3 for f in range(1,6))
        cost=complete_estimate+outer_estimate
        records[task]={'inner_direction_gate':passed,'screen_raw_directions':[
            r['adapted']['raw_mse']<r['original']['raw_mse'] for r in rows],
            'max_screen_seconds_per_fit_origin':rate,'remaining_cell_estimate_seconds':complete_estimate,
            'outer_and_statistics_estimate_seconds':outer_estimate,'estimated_seconds':cost,
            'fit_counts_by_fold':counts}
        if passed:admitted.append(task);remaining_cost+=cost
    compute_ok=time.time()+remaining_cost+300<DEADLINE
    gate={'status':'frozen_before_adapter_outer_evaluation','tasks':records,
          'mechanism_admitted_tasks':admitted,'compute_admitted':compute_ok,
          'full_tasks':admitted if compute_ok else [],'estimated_seconds':remaining_cost,
          'reserve_seconds':300,'remaining_seconds_at_freeze':DEADLINE-time.time(),
          'deadline_epoch':DEADLINE,'screen_inner_receipt_hashes':{
              r['identity']['cfg']['id']:sha(HERE/'cells'/r['identity']['cfg']['id']/'INNER_DONE.json') for r in screen},
          'rules_sha256':sha(RULES),'source_freeze_sha256':sha(HERE/'SOURCE_FREEZE.json'),
          'outer_labels_used_for_admission':False,'at_utc':now(),'at_epoch':time.time()}
    atomic(path,gate);return gate

def outer_cell(cfg):
    out=HERE/'cells'/cfg['id']
    if (out/'OUTER_DONE.json').exists():return read(out/'OUTER_DONE.json')
    inner=cell(cfg);folder,done,saved=source(cfg);data=Data(cfg)
    adapted=torch.load(out/'adapted.pt',map_location='cpu',weights_only=True)
    net=load_net(cfg,adapted,data);idx=data.splits['validation']
    p=forward(data,net,idx)
    with np.load(folder/'predictions.npz',allow_pickle=False) as a:
        assert np.array_equal(idx,a['origins']);po=a['trained']
    y=targets(data.bars,idx,cfg['horizon'],cfg['target'])[:,-1]
    preds={'original':po[:,-1],'original_shrink':inner['original']['alpha']*po[:,-1],
           'adapted':p[:,-1],'adapted_shrink':inner['adapted']['alpha']*p[:,-1],
           'zero':np.zeros(len(y))}
    losses={name:(value-y)**2 for name,value in preds.items()}
    decomposition={name:{'mse':float(losses[name].mean()),'truth_energy':float(np.mean(y*y)),
        'forecast_energy':float(np.mean(value*value)),'twice_alignment':float(2*np.mean(value*y)),
        'improvement_over_zero_pct':float(100*(1-losses[name].mean()/losses['zero'].mean()))}
        for name,value in preds.items()}
    for v in decomposition.values():
        assert np.isclose(v['mse'],v['truth_energy']+v['forecast_energy']-v['twice_alignment'],rtol=1e-10,atol=1e-14)
    npz(out/'outer_losses.npz',origins=idx,**losses)
    npz(out/'adapted_predictions.npz',origins=idx,prediction=p)
    result={'status':'completed','job':cfg,'decomposition':decomposition,
            'gate_sha256':sha(HERE/'GATE_AND_ADMISSION.json'),
            'predictions_sha256':sha(out/'adapted_predictions.npz'),'losses_sha256':sha(out/'outer_losses.npz'),
            'outer_labels_used_for_fitting':False,'Q3_values_read':False,'at_utc':now()}
    atomic(out/'OUTER_DONE.json',result)
    print(json.dumps({'outer_done':cfg['id']}),flush=True)
    return result

def summarize(gate):
    summary={'status':'completed','gate':gate,'tasks':{},'Q3_values_read':False,
             'independent_confirmation':False,'weights_retrained':False,'method_success_claimed':False,
             'at_utc':now()}
    for profile,h in (('minute',60),('hourly',8)):
        task=task_key(profile,h,'future_log_return')
        complete=task in gate['full_tasks']
        cfgs=[cfg_for(profile,h,f,s) for f in range(1,6) for s in (2026,2027,2028)] if complete else [cfg_for(profile,h,f,2026) for f in (1,2)]
        rows=[outer_cell(cfg) for cfg in cfgs]
        groups=[];all_losses={};fold_gains=[]
        for f in (range(1,6) if complete else (1,2)):
            selected=[cfg for cfg in cfgs if cfg['fold']==f]
            losses={};origins=None
            for cfg in selected:
                with np.load(HERE/'cells'/cfg['id']/'outer_losses.npz',allow_pickle=False) as a:
                    if origins is None:origins=a['origins']
                    assert np.array_equal(origins,a['origins'])
                    for name in a.files:
                        if name!='origins':losses[name]=losses.get(name,0)+a[name]/len(selected)
            for name,value in losses.items():all_losses.setdefault(name,[]).append(value)
            fold_gains.append(100*(1-losses['adapted_shrink'].mean()/losses['zero'].mean()))
            bars=Data(selected[0]).bars;groups.append(day_groups(bars,origins,losses))
        pooled={name:float(np.concatenate(values).mean()) for name,values in all_losses.items()}
        intervals=[]
        if complete:
            for block in ((1,5) if profile=='minute' else (2,5)):
                intervals+=paired_intervals(groups,[(a,b) for a in ('adapted','adapted_shrink') for b in ('original','original_shrink','zero') if a!=b],block,task)
        summary['tasks'][task]={'scope':'five_fold_three_seed' if complete else 'two_fold_single_seed_screen_only',
            'cells':len(cfgs),'pooled_mse':pooled,'improvement_over_zero_pct':{
                name:100*(1-pooled[name]/pooled['zero']) for name in pooled},
            'adapted_shrink_positive_fold_count':sum(x>0 for x in fold_gains),'fold_improvements':fold_gains,
            'paired_intervals':intervals,'cell_receipts':rows,
            'remaining_controls':'Same-information Ridge and original default-recipe comparison not included in this isolated diagnostic; cannot assert full performance gate from this alone.'}
    atomic(HERE/'RESULTS.json',summary)
    print(json.dumps({'status':'completed','tasks':{t:r['improvement_over_zero_pct'] for t,r in summary['tasks'].items()}}),flush=True)

def main():
    init()
    try:
        gate=frozen_gate()
        for task in gate['full_tasks']:
            profile,rest=task.split('_h',1);h=int(rest.split('_',1)[0])
            for f in range(1,6):
                for s in (2026,2027,2028):cell(cfg_for(profile,h,f,s))
        # Gate and cost identity exist before this first adapter outer scoring.
        atomic(HERE/'OUTER_START.json',{'gate_sha256':sha(HERE/'GATE_AND_ADMISSION.json'),
               'source_freeze_sha256':sha(HERE/'SOURCE_FREEZE.json'),'at_epoch':time.time(),'at_utc':now()})
        summarize(gate)
    except Pause:
        atomic(HERE/'PAUSED.json',{'status':'paused','deadline_epoch':DEADLINE,'signal':STOP,'at_utc':now()})
        print(json.dumps({'status':'paused','checkpoints_preserved':True}),flush=True)

if __name__=='__main__':main()
