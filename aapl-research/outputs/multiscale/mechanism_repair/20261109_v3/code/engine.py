"""Registered paired training; baseline selection and patience are recorded separately."""
from study import *
import signal, time
import torch
from networks import create, objective
from multiscale.features import CloseWindows
from multiscale.matrix_executor import read_bars
from multiscale.bars import targets
from multiscale.training import atomic_torch_save

STOP=False
def stop(*_):
    global STOP
    STOP=True

class Data:
    def __init__(self,cfg):
        self.cfg=cfg;self.task=task_key(cfg['profile'],cfg['horizon'],cfg['target'])
        path=PRIOR/'cache'/self.task/'TASK.json'
        assert sha(path)==read(PRIOR/'protocol/DATA.json')['task_files'][self.task]
        self.meta=read(path);self.rec=self.meta['folds'][str(cfg['fold'])]
        self.bars=read_bars(SOURCE,read(SOURCE/'admission.json'),cfg['frequency'])
        assert self.bars.session_date[-1]<='2026-06-30'
        self.windows=CloseWindows(self.bars,cfg['lookback'],read(SOURCE/'admission.json')['split_actions'])
        with np.load(self.rec['split_file'],allow_pickle=False) as a:self.splits={k:a[k] for k in a.files}
        assert sha(self.rec['scaler_file'])==self.rec['scaler_sha256']
        scalers=read(self.rec['scaler_file']);self.scaler=dict(scalers['tiny' if cfg['model']=='tiny' else 'original'])
        if cfg['output']=='residual':
            self.scaler.update(y_mean=scalers['tiny']['y_mean'],y_std=scalers['tiny']['y_std'])
        self.ym=np.asarray(self.scaler['y_mean']);self.ys=np.asarray(self.scaler['y_std'])
        if cfg['model']=='tiny':
            self.origins=np.load(PRIOR/'cache'/cfg['profile']/'origins.npy',mmap_mode='r')
            self.x=np.load(PRIOR/'cache'/cfg['profile']/'x.npy',mmap_mode='r')
        fit,tail,val=(self.splits[k] for k in ('fit','stop','validation'))
        h=cfg['horizon']
        assert self.bars.end_ns[fit[-1]+h]<self.bars.end_ns[tail[0]]
        assert self.bars.end_ns[tail[-1]+h]<self.bars.end_ns[val[0]]

    def batch(self,part,pos):
        indices=self.splits[part][pos]
        y=targets(self.bars,indices,self.cfg['horizon'],self.cfg['target'])
        if self.cfg['model']=='tiny':
            rows=np.searchsorted(self.origins,indices)
            assert np.array_equal(self.origins[rows],indices)
            x=np.asarray(self.x[rows],dtype='float32')
            x=((x-self.scaler['x_mean'])/self.scaler['x_std']).astype('float32')
            args=(torch.from_numpy(x),)
        else:
            x,a=self.windows.arrays(indices)
            x=((x-self.scaler['x_mean'])/self.scaler['x_std']).astype('float32')
            a=((a-self.scaler['anchor_mean'])/self.scaler['anchor_std']).astype('float32')
            if self.cfg['anchor']=='constant':a=np.zeros_like(a)
            args=(torch.from_numpy(x[:,:,None]),torch.from_numpy(a))
        if self.cfg['output']=='residual':y=y-self.bars.close[indices,None]
        return args,torch.from_numpy(((y-self.ym)/self.ys).astype('float32'))

    def invert(self,p,indices):
        raw=np.asarray(p)*self.ys+self.ym
        if self.cfg['output']=='residual':raw+=self.bars.close[indices,None]
        return raw

    def baseline(self,indices):
        return np.zeros((len(indices),self.cfg['horizon'])) if self.cfg['target']=='future_log_return' else \
            np.broadcast_to(self.bars.close[indices,None],(len(indices),self.cfg['horizon']))

def prediction(net,data,part):
    net.eval();indices=data.splits[part];out=[]
    with torch.inference_mode():
        for i in range(0,len(indices),128):
            pos=np.arange(i,min(i+128,len(indices)));args,_=data.batch(part,pos)
            out.append(data.invert(net(*args).numpy(),indices[pos]))
    value=np.concatenate(out)
    assert np.isfinite(value).all()
    return value

def stop_score(net,data):
    net.eval();indices=data.splits['stop'];total=0.
    with torch.inference_mode():
        for i in range(0,len(indices),128):
            pos=np.arange(i,min(i+128,len(indices)));args,_=data.batch('stop',pos)
            p=data.invert(net(*args).numpy(),indices[pos])
            y=targets(data.bars,indices[pos],data.cfg['horizon'],data.cfg['target'])
            total+=float(((p[:,-1]-y[:,-1])**2).sum())
    return total/len(indices)

def run(cfg,folder=None,steps_limit=None,seconds_limit=None):
    global STOP
    STOP=False
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1' and len(os.sched_getaffinity(0))<=2
    torch.set_num_threads(min(2,len(os.sched_getaffinity(0))));torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)
    folder=Path(folder) if folder else BASE/'runs'/cfg['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'DONE.json').exists():return read(folder/'DONE.json')
    frozen=read(BASE/'protocol/CODE_FREEZE.json');assert source_hashes()==frozen['hashes']
    identity=hashlib.sha256(json.dumps({'cfg':cfg,'code':frozen['hashes'],
                'data':sha(PRIOR/'protocol/DATA.json')},sort_keys=True).encode()).hexdigest()
    deadline=read(BASE/'protocol/BUDGET.json')['deadline_epoch'];start=time.time();cpu=time.process_time()
    until=min(deadline,start+seconds_limit) if seconds_limit else deadline
    data=Data(cfg);torch.manual_seed(cfg['seed']);net=create(cfg,data.scaler)
    opt=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate']);rng=np.random.default_rng(cfg['seed'])
    si=data.splits['stop'];sy=targets(data.bars,si,cfg['horizon'],cfg['target'])
    baseline_stop=float(np.mean((data.baseline(si)[:,-1]-sy[:,-1])**2))
    state={'epoch':0,'cursor':0,'permutation':None,'steps':0,'best_score':float('inf'),
           'best_epoch':None,'stale':0,'history':[],'loss_sum':0.,'seen':0,
           'shadow_best':baseline_stop,'shadow_epoch':0,'shadow_stale':0,'shadow_stop_epoch':None,
           'wall_seconds':0.,'cpu_seconds':0.,'gradients_first_two':[]}

    def record(epoch):
        return {'identity':identity,'cfg':cfg,'model':net.state_dict(),'scaler':data.scaler,'epoch':epoch,
                'parameters':sum(p.numel() for p in net.parameters())}

    latest=folder/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location='cpu',weights_only=True)
        assert saved['identity']==identity and saved['scaler']==data.scaler
        net.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);state=saved['state']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
    else:
        atomic_torch_save(record(0),folder/'initial.pt')
        idx=np.arange(min(16,len(si)));args,_=data.batch('stop',idx)
        net.eval()
        with torch.inference_mode():initial=data.invert(net(*args).numpy(),si[idx])
        initial_error=float(np.max(abs(initial-data.baseline(si[idx]))))
        if cfg['initialization']=='baseline':
            assert np.allclose(initial,data.baseline(si[idx]),rtol=1e-6,atol=1e-6),initial_error
        atomic(folder/'INITIALIZATION.json',{'initial_function_max_abs_difference_from_baseline':initial_error,
              'baseline_function_required':cfg['initialization']=='baseline','raw_units':cfg['target'],
              'initial_checkpoint_sha256':sha(folder/'initial.pt'),'parameters':sum(p.numel() for p in net.parameters())})

    def checkpoint():
        value=record(state['epoch']);value.update(optimizer=opt.state_dict(),state=state,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state())
        atomic_torch_save(value,latest)
        atomic(folder/'STATE.json',{'job':cfg,'identity':identity,'epoch':state['epoch'],
            'cursor':state['cursor'],'steps':state['steps'],'best_epoch':state['best_epoch'],
            'shadow_stop_epoch':state['shadow_stop_epoch'],'deadline_epoch':deadline,
            'affinity':sorted(os.sched_getaffinity(0)),'at_utc':now()})

    if not latest.exists():checkpoint()
    initial_steps=state['steps'];paused=False
    while state['epoch']<cfg['max_epochs'] and state['stale']<cfg['patience']:
        if state['permutation'] is None:
            state['permutation']=torch.from_numpy(rng.permutation(len(data.splits['fit'])));state['cursor']=0
        while state['cursor']<len(data.splits['fit']):
            if STOP or time.time()>=until or steps_limit is not None and state['steps']-initial_steps>=steps_limit:
                paused=True;break
            end=min(state['cursor']+cfg['batch_size'],len(data.splits['fit']))
            if len(data.splits['fit'])-end==1:end=len(data.splits['fit'])
            pos=state['permutation'][state['cursor']:end].numpy();args,y=data.batch('fit',pos)
            net.train();opt.zero_grad(set_to_none=True);p=net(*args);loss=objective(p,y,cfg['loss'])
            assert torch.isfinite(loss)
            loss.backward()
            assert all(q.grad is not None and torch.isfinite(q.grad).all() for q in net.parameters())
            if state['steps']<2:
                state['gradients_first_two'].append({n:float(q.grad.norm()) for n,q in net.named_parameters()})
            opt.step();state['cursor']=end;state['steps']+=1
            state['loss_sum']+=float(loss.detach())*len(pos);state['seen']+=len(pos)
            if state['steps']%100==0:checkpoint()
        if paused:break
        score=stop_score(net,data);ep=state['epoch']+1
        state['history'].append({'epoch':ep,'objective':state['loss_sum']/state['seen'],
                                'inner_stop_mse':score,'baseline_stop_mse':baseline_stop,'steps':state['steps']})
        if score<state['best_score']:
            state['best_score']=score;state['best_epoch']=ep;state['stale']=0
            atomic_torch_save(record(ep),folder/'best_trained.pt')
        else:state['stale']+=1
        if state['shadow_stop_epoch'] is None:
            if score<state['shadow_best']:
                state['shadow_best']=score;state['shadow_epoch']=ep;state['shadow_stale']=0
                atomic_torch_save(record(ep),folder/'coupled_best.pt')
            else:state['shadow_stale']+=1
            if state['shadow_stale']>=cfg['patience']:state['shadow_stop_epoch']=ep
        state['epoch']=ep;state['cursor']=0;state['permutation']=None;state['loss_sum']=0.;state['seen']=0
        checkpoint()
    state['wall_seconds']+=time.time()-start;state['cpu_seconds']+=time.process_time()-cpu;checkpoint()
    if paused or time.time()>=deadline:
        value={'status':'paused','steps':state['steps'],'epoch':state['epoch'],'deadline_epoch':deadline}
        atomic(folder/'PAUSE.json',value);return value
    saved=torch.load(folder/'best_trained.pt',map_location='cpu',weights_only=True);net.load_state_dict(saved['model'])
    pred=prediction(net,data,'validation');vi=data.splits['validation']
    truth=targets(data.bars,vi,cfg['horizon'],cfg['target']);base=data.baseline(vi)
    selected_base=baseline_stop<=state['best_score'];selected=base if selected_base else pred
    if state['shadow_epoch']==0:coupled=base
    elif state['shadow_epoch']==state['best_epoch']:coupled=pred
    else:
        saved_shadow=torch.load(folder/'coupled_best.pt',map_location='cpu',weights_only=True)
        net.load_state_dict(saved_shadow['model']);coupled=prediction(net,data,'validation')
    npz(folder/'predictions.npz',origins=vi,trained=pred,coupled=coupled)
    result={'status':'completed','identity':identity,'job':cfg,'epochs':state['epoch'],'steps':state['steps'],
        'trained_best_epoch':state['best_epoch'],'selected_baseline':selected_base,
        'baseline_stop_mse':baseline_stop,'best_trained_stop_mse':state['best_score'],
        'shadow_stop_epoch':state['shadow_stop_epoch'] or state['epoch'],'shadow_best_epoch':state['shadow_epoch'],
        'history':state['history'],'parameters':saved['parameters'],
        'metrics':{'trained':metrics(pred,truth,data.bars.close[vi],cfg['target']),
                   'selected':metrics(selected,truth,data.bars.close[vi],cfg['target']),
                   'coupled':metrics(coupled,truth,data.bars.close[vi],cfg['target'])},
        'prediction_sha256':sha(folder/'predictions.npz'),'checkpoint_sha256':sha(folder/'best_trained.pt'),
        'gradients_first_two':state['gradients_first_two'],'training_wall_seconds':state['wall_seconds'],
        'training_cpu_seconds':state['cpu_seconds'],'deadline_epoch':deadline,'affinity':sorted(os.sched_getaffinity(0)),
        'Q3_values_read':False,'independent_confirmation':False,'at_utc':now()}
    atomic(folder/'DONE.json',result)
    print(json.dumps({'job':cfg['id'],'epochs':state['epoch'],'status':'completed'}),flush=True)
    return result
