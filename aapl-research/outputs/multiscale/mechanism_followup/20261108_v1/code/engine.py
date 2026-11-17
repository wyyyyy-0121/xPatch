from common import *
from dataclasses import asdict
from datetime import datetime,timezone
import signal,time,copy
import torch
from models import create,PROBES
from multiscale.features import CloseWindows,fit_scaler
from multiscale.training import atomic_torch_save

STOP=False
def request_stop(*_):
    global STOP;STOP=True

class Data:
    def __init__(self,fold):
        self.bars,_,folds,admission=bars_and_folds();self.fold=folds[fold-1]
        self.windows=CloseWindows(self.bars,480,admission['split_actions'])
        self.fit=self.fold['inner']['fit'];self.stop=self.fold['inner']['stop'];self.val=self.fold['validation']
        self.scaler=fit_scaler(self.windows,self.fit,60,'future_log_return')
        self.cache=cache_arrays()
        self.rows={k:rows_for(v) for k,v in [('fit',self.fit),('stop',self.stop),('val',self.val)]}
        self.feature_stats={}
        for k in ('trend','gate'):
            a=np.asarray(self.cache[k][self.rows['fit']],dtype=np.float64)
            mean=a.mean(0);std=a.std(0);std=np.where(std>1e-12,std,1.)
            self.feature_stats[k]={'mean':mean.tolist(),'std':std.tolist()}
        self.vol_threshold=float(np.median(self.cache['volatility'][self.rows['fit']]))
    def batch(self,part,positions,kind):
        rr=self.rows[part][positions]
        if kind in PROBES or kind=='lstm':
            origins={'fit':self.fit,'stop':self.stop,'val':self.val}[part][positions]
            x,anchor=self.windows.arrays(origins)
            x,anchor=self.scaler.transform_x(x,anchor)
            inputs=(torch.from_numpy(x[:,:,None]),torch.from_numpy(anchor))
        else:
            values=[]
            for k in ('trend','residual','gate'):
                x=np.array(self.cache[k][rr],copy=True)
                if k in self.feature_stats:
                    st=self.feature_stats[k];x=((x-np.array(st['mean']))/np.array(st['std'])).astype('float32')
                values.append(torch.from_numpy(x))
            inputs=tuple(values)
        y=self.scaler.transform_y(self.cache['actual'][rr])
        return inputs,torch.from_numpy(y)

def predict(model,data,part,kind):
    out=[];model.eval()
    with torch.inference_mode():
        for start in range(0,len(data.rows[part]),512):
            args,_=data.batch(part,np.arange(start,min(start+512,len(data.rows[part]))),kind)
            out.append(data.scaler.invert_y(model(*args).numpy()))
    return np.concatenate(out)

def stop_score(model,data,kind):
    # Original probes/LSTM keep the original 128 prediction batch for early stopping.
    total=0.;model.eval()
    with torch.inference_mode():
        for start in range(0,len(data.stop),128):
            pos=np.arange(start,min(start+128,len(data.stop)))
            args,_=data.batch('stop',pos,kind)
            raw=data.scaler.invert_y(model(*args).numpy())[:,-1]
            truth=data.cache['actual'][data.rows['stop'][pos],-1]
            total+=float(np.square(raw-truth).sum())
    return total/len(data.stop)

def metrics(p,y):
    delta=p-y;base=float(np.mean(y[:,-1]**2));mse=float(np.mean(delta[:,-1]**2))
    return {'origins':len(y),'endpoint_mse':mse,'zero_mse':base,'improvement_pct':100*(1-mse/base),
        'path_mse':float(np.mean(delta**2)),'per_step_mse':np.mean(delta**2,axis=0).tolist(),
        'endpoint_mae':float(np.mean(np.abs(delta[:,-1])))}

def run(cfg,bench_seconds=None,stop_after_steps=None):
    global STOP;STOP=False
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and 1<=len(os.sched_getaffinity(0))<=2
    torch.set_num_threads(min(2,len(os.sched_getaffinity(0))));torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,request_stop)
    kind,fold,seed=cfg['kind'],cfg['fold'],cfg['seed'];folder=BASE/'runs'/cfg['id'];folder.mkdir(exist_ok=True)
    frozen=read(BASE/'protocol/CODE_FREEZE.json')
    assert code_hashes()==frozen['hashes'],'Training source changed after freeze'
    identity=hashlib.sha256(json.dumps({'job':cfg,'code':frozen['hashes'],'data':sha(BASE/'protocol/DATA.json')},sort_keys=True).encode()).hexdigest()
    budget=read(BASE/'protocol/BUDGET.json');deadline=budget['deadline_epoch']
    started=time.time();limited=min(deadline,started+bench_seconds) if bench_seconds else deadline
    cpu_started=time.process_time();data=Data(fold)
    torch.manual_seed(seed)
    model=create(kind,-np.asarray(data.scaler.y_mean)/np.asarray(data.scaler.y_std))
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
    rng=np.random.default_rng(seed)
    s={'epoch':0,'cursor':0,'permutation':None,'steps':0,'best_loss':float('inf'),'best_epoch':None,
       'stale':0,'epoch_sum':0.,'epoch_seen':0,'history':[],'wall_seconds':0.,'cpu_seconds':0.}
    latest=folder/'latest.pt'
    def saved_model(epoch):
        return {'identity':identity,'cfg':cfg,'model':model.state_dict(),'scaler':asdict(data.scaler),
            'feature_stats':data.feature_stats,'vol_threshold':data.vol_threshold,'epoch':epoch,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'total_parameters':sum(p.numel() for p in model.parameters())}
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=True)
        assert ck['identity']==identity and ck['scaler']==asdict(data.scaler)
        model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);s=ck['state']
        rng.bit_generator.state=ck['rng'];torch.set_rng_state(ck['torch_rng'])
    elif kind not in PROBES and kind!='lstm':
        # Explicit epoch-zero zero-return option, as declared in the follow-up plan.
        s['best_loss']=float(np.mean(data.cache['actual'][data.rows['stop'],-1]**2));s['best_epoch']=0
        atomic_torch_save(saved_model(0),folder/'best.pt')
    def checkpoint():
        record=saved_model(s['epoch']);record.update(optimizer=optimizer.state_dict(),state=s,
            rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        atomic_torch_save(record,latest)
        atomic(folder/'STATE.json',{'status':'checkpointed','job':cfg,'epoch':s['epoch'],'steps':s['steps'],
            'cursor':s['cursor'],'best_epoch':s['best_epoch'],'history':s['history'],
            'at_utc':datetime.now(timezone.utc).isoformat(),'identity':identity,
            'affinity':sorted(os.sched_getaffinity(0)),'deadline_epoch':deadline})
    if not latest.exists():checkpoint()
    steps_this=0;interrupted=False
    while s['epoch']<20 and s['stale']<3:
        if s['permutation'] is None:s['permutation']=torch.from_numpy(rng.permutation(len(data.fit)));s['cursor']=0
        while s['cursor']<len(data.fit):
            if STOP or time.time()>=limited or (stop_after_steps is not None and steps_this>=stop_after_steps):
                interrupted=True;break
            end=min(s['cursor']+128,len(data.fit))
            if len(data.fit)-end==1:end=len(data.fit)
            pos=s['permutation'][s['cursor']:end].numpy();args,y=data.batch('fit',pos,kind)
            model.train();optimizer.zero_grad(set_to_none=True);pred=model(*args);loss=(pred-y).square().mean()
            assert torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            optimizer.step();s['steps']+=1;steps_this+=1;s['cursor']=end
            s['epoch_sum']+=float(loss.detach())*len(pos);s['epoch_seen']+=len(pos)
            if s['steps']%100==0:checkpoint()
        if interrupted:break
        score=stop_score(model,data,kind)
        s['history'].append({'epoch':s['epoch']+1,'training_scaled_path_loss':s['epoch_sum']/s['epoch_seen'],
            'inner_stop_mse':score,'steps':s['steps']})
        if score<s['best_loss']:
            s['best_loss']=score;s['best_epoch']=s['epoch']+1;s['stale']=0
            atomic_torch_save(saved_model(s['best_epoch']),folder/'best.pt')
        else:s['stale']+=1
        s['epoch']+=1;s['cursor']=0;s['permutation']=None;s['epoch_sum']=0.;s['epoch_seen']=0
        checkpoint()
    s['wall_seconds']+=time.time()-started;s['cpu_seconds']+=time.process_time()-cpu_started;checkpoint()
    if interrupted:
        atomic(folder/'PAUSE.json',{'reason':'signal_or_bound','at_epoch':s['epoch'],'steps':s['steps'],
            'wall_seconds_this':time.time()-started,'cpu_seconds_this':time.process_time()-cpu_started})
        return {'status':'paused','job':cfg,'steps':s['steps'],'epochs':s['epoch']}
    best=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(best['model'])
    p=predict(model,data,'val',kind)
    pp=folder/'prediction.npz';np.savez_compressed(pp,prediction=p,origins=data.val,decision_ns=data.bars.end_ns[data.val])
    y=np.asarray(data.cache['actual'][data.rows['val']])
    result={'status':'completed','job':cfg,'identity':identity,'epochs':s['epoch'],'steps':s['steps'],
        'best_epoch':s['best_epoch'],'history':s['history'],'metrics':metrics(p,y),'checkpoint_sha256':sha(folder/'best.pt'),
        'prediction_sha256':sha(pp),'trainable_parameters':best['trainable_parameters'],'total_parameters':best['total_parameters'],
        'wall_seconds':s['wall_seconds'],
        'training_wall_seconds':s['wall_seconds'],'training_cpu_seconds':s['cpu_seconds'],
        'total_last_invocation_seconds':time.time()-started,'last_invocation_cpu_seconds':time.process_time()-cpu_started,
        'Q3_values_read':False,'independent_confirmation':False,
        'affinity':sorted(os.sched_getaffinity(0)),'code_hashes':frozen['hashes']}
    atomic(folder/'DONE.json',result);print(json.dumps({'job':cfg['id'],'status':'completed','epochs':s['epoch'],'improvement_pct':result['metrics']['improvement_pct']}),flush=True)
    return result
