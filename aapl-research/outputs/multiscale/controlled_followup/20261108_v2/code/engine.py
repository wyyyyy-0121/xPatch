from common import *
from datetime import datetime,timezone
import signal,time
import torch
from models import create,objective
from multiscale.features import CloseWindows,Scaler
from multiscale.bars import targets
from multiscale.training import atomic_torch_save

STOP=False
def request_stop(*_):
    global STOP
    STOP=True

class Data:
    def __init__(self,cfg):
        self.cfg=cfg;self.task=key(cfg['profile'],cfg['horizon'],cfg['target'])
        self.meta=read(BASE/'cache'/self.task/'TASK.json')
        rec=self.meta['folds'][str(cfg['fold'])]
        assert sha(rec['scaler_file'])==rec['scaler_sha256']
        self.bars=admitted_bars(cfg['frequency'])
        self.windows=CloseWindows(self.bars,cfg['lookback'],read(SOURCE/'admission.json')['split_actions'])
        self.splits=dict(np.load(rec['split_file']))
        self.scaler=read(rec['scaler_file'])[cfg['family']=='original' and 'original' or 'tiny']
        self.all_origins=np.load(BASE/'cache'/cfg['profile']/'origins.npy',mmap_mode='r')
        self.x=np.load(BASE/'cache'/cfg['profile']/'x.npy',mmap_mode='r')
        self.rows={k:rows_for(v,self.all_origins) for k,v in self.splits.items()}
        self.ym=np.asarray(self.scaler['y_mean']);self.ys=np.asarray(self.scaler['y_std'])
    def batch(self,part,pos):
        idx=self.splits[part][pos]
        y=targets(self.bars,idx,self.cfg['horizon'],self.cfg['target'])
        if self.cfg['family']=='tiny':
            x=np.array(self.x[self.rows[part][pos]],copy=True)
            x=((x-self.scaler['x_mean'])/self.scaler['x_std']).astype('float32')
            if self.cfg['target']=='future_close':y-=self.bars.close[idx,None]
            args=(torch.from_numpy(x),)
        else:
            x,a=self.windows.arrays(idx)
            x,a=Scaler(**self.scaler).transform_x(x,a)
            args=(torch.from_numpy(x[:,:,None]),torch.from_numpy(a))
        yy=((y-self.ym)/self.ys).astype('float32')
        return args,torch.from_numpy(yy)
    def invert(self,p,indices):
        v=np.asarray(p)*self.ys+self.ym
        if self.cfg['family']=='tiny' and self.cfg['target']=='future_close':v+=self.bars.close[indices,None]
        return v

def stop_score(model,data):
    model.eval();total=0.;indices=data.splits['stop']
    with torch.inference_mode():
        for i in range(0,len(indices),128):
            pos=np.arange(i,min(i+128,len(indices)));args,_=data.batch('stop',pos)
            p=data.invert(model(*args).numpy(),indices[pos])
            y=targets(data.bars,indices[pos],data.cfg['horizon'],data.cfg['target'])
            total+=float(((p[:,-1]-y[:,-1])**2).sum())
    return total/len(indices)

def predict(model,data,part='validation'):
    model.eval();out=[];indices=data.splits[part]
    with torch.inference_mode():
        for i in range(0,len(indices),128):
            pos=np.arange(i,min(i+128,len(indices)));args,_=data.batch(part,pos)
            out.append(data.invert(model(*args).numpy(),indices[pos]))
    return np.concatenate(out)

def run(cfg,bench_seconds=None,stop_after_steps=None,output_override=None):
    global STOP
    STOP=False
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    assert len(os.sched_getaffinity(0))<=2
    torch.set_num_threads(min(2,len(os.sched_getaffinity(0))));torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,request_stop)
    folder=Path(output_override) if output_override else BASE/'runs'/cfg['id']
    folder.mkdir(parents=True,exist_ok=True)
    frozen=read(BASE/'protocol/CODE_FREEZE.json')
    assert code_hashes()==frozen['hashes'],'Frozen training source changed'
    identity=hashlib.sha256(json.dumps({'cfg':cfg,'code':frozen['hashes'],
        'data':sha(BASE/'protocol/DATA.json')},sort_keys=True).encode()).hexdigest()
    budget=read(BASE/'protocol/BUDGET.json');deadline=budget['deadline_epoch']
    started=time.time();cpu=time.process_time();limited=min(deadline,started+bench_seconds) if bench_seconds else deadline
    data=Data(cfg);torch.manual_seed(cfg['seed'])
    model=create(cfg,-data.ym/data.ys);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    rng=np.random.default_rng(cfg['seed'])
    state={'epoch':0,'cursor':0,'permutation':None,'steps':0,'best_loss':float('inf'),'best_epoch':None,
           'stale':0,'loss_sum':0.,'seen':0,'history':[],'wall_seconds':0.,'cpu_seconds':0.}
    latest=folder/'latest.pt'
    def best_record(epoch):
        return {'identity':identity,'cfg':cfg,'model':model.state_dict(),'scaler':data.scaler,
                'epoch':epoch,'parameters':sum(p.numel() for p in model.parameters())}
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['identity']==identity and saved['scaler']==data.scaler
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);state=saved['state']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
    elif cfg['family']=='tiny':
        idx=data.splits['stop'];y=targets(data.bars,idx,cfg['horizon'],cfg['target'])
        baseline=np.zeros(len(idx)) if cfg['target']=='future_log_return' else data.bars.close[idx]
        state['best_loss']=float(((baseline-y[:,-1])**2).mean());state['best_epoch']=0
        atomic_torch_save(best_record(0),folder/'best.pt')
    def checkpoint():
        record=best_record(state['epoch']);record.update(optimizer=optimizer.state_dict(),state=state,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        atomic_torch_save(record,latest)
        atomic(folder/'STATE.json',{'job':cfg,'epoch':state['epoch'],'cursor':state['cursor'],
            'steps':state['steps'],'best_epoch':state['best_epoch'],'history':state['history'],
            'identity':identity,'deadline_epoch':deadline,'affinity':sorted(os.sched_getaffinity(0)),
            'at_utc':datetime.now(timezone.utc).isoformat()})
    if not latest.exists():checkpoint()
    initial_steps=state['steps'];paused=False
    while state['epoch']<20 and state['stale']<3:
        if state['permutation'] is None:
            state['permutation']=torch.from_numpy(rng.permutation(len(data.splits['fit'])));state['cursor']=0
        while state['cursor']<len(data.splits['fit']):
            if STOP or time.time()>=limited or stop_after_steps is not None and state['steps']-initial_steps>=stop_after_steps:
                paused=True;break
            end=min(state['cursor']+128,len(data.splits['fit']))
            if len(data.splits['fit'])-end==1:end=len(data.splits['fit'])
            pos=state['permutation'][state['cursor']:end].numpy();args,y=data.batch('fit',pos)
            model.train();optimizer.zero_grad(set_to_none=True);p=model(*args)
            loss=objective(p,y,cfg['loss']);assert torch.isfinite(loss)
            loss.backward();assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in model.parameters())
            optimizer.step();state['cursor']=end;state['steps']+=1
            state['loss_sum']+=float(loss.detach())*len(pos);state['seen']+=len(pos)
            if state['steps']%100==0:checkpoint()
        if paused:break
        score=stop_score(model,data)
        state['history'].append({'epoch':state['epoch']+1,'training_objective':state['loss_sum']/state['seen'],
            'inner_stop_mse':score,'steps':state['steps']})
        if score<state['best_loss']:
            state['best_loss']=score;state['best_epoch']=state['epoch']+1;state['stale']=0
            atomic_torch_save(best_record(state['best_epoch']),folder/'best.pt')
        else:state['stale']+=1
        state['epoch']+=1;state['cursor']=0;state['permutation']=None;state['loss_sum']=0.;state['seen']=0
        checkpoint()
    state['wall_seconds']+=time.time()-started;state['cpu_seconds']+=time.process_time()-cpu;checkpoint()
    if paused:
        result={'status':'paused','job':cfg,'steps':state['steps'],'epochs':state['epoch'],
            'invocation_seconds':time.time()-started,'invocation_steps':state['steps']-initial_steps,
            'cpu_seconds':time.process_time()-cpu,'deadline_epoch':deadline}
        atomic(folder/'PAUSE.json',result);return result
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(saved['model'])
    # Prediction may be restarted from best.pt; latest optimizer/RNG state is already safe.
    p=predict(model,data);idx=data.splits['validation']
    save_npz(folder/'prediction.npz',prediction=p,origins=idx,decision_ns=data.bars.end_ns[idx])
    y=targets(data.bars,idx,cfg['horizon'],cfg['target'])
    result={'status':'completed','job':cfg,'identity':identity,'epochs':state['epoch'],'steps':state['steps'],
        'best_epoch':state['best_epoch'],'history':state['history'],'parameters':saved['parameters'],
        'metrics':metric(p,y,data.bars.close[idx],cfg['target']),'checkpoint_sha256':sha(folder/'best.pt'),
        'prediction_sha256':sha(folder/'prediction.npz'),'training_wall_seconds':state['wall_seconds'],
        'training_cpu_seconds':state['cpu_seconds'],'last_invocation_seconds':time.time()-started,
        'affinity':sorted(os.sched_getaffinity(0)),'Q3_values_read':False,'independent_confirmation':False}
    atomic(folder/'DONE.json',result)
    print(json.dumps({'job':cfg['id'],'status':'completed','epochs':state['epoch']}),flush=True)
    return result
