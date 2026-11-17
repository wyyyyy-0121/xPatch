"""Independent GPU audit using only generated arrays and isolated subprocesses."""
from copy import deepcopy
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch

REVIEW = Path(__file__).resolve().parent
BASE = REVIEW.parent
sys.path.insert(0, str(BASE/'code'))
exp = importlib.import_module('experiment')
from shared_network import Bundle, HORIZONS, native_reference
bars_module = importlib.import_module('multiscale.bars')


class SyntheticData:
    """Batch and windows API compatible with the real loader, no files read."""
    def __init__(self):
        rng = np.random.default_rng(119)
        self.scaler = {'x_mean':0., 'x_std':1., 'anchor_mean':0., 'anchor_std':1.,
            'y_mean':np.linspace(-.0001,.0001,60).tolist(),
            'y_std':np.linspace(.003,.02,60).tolist()}
        self.ym = np.asarray(self.scaler['y_mean'])
        self.ys = np.asarray(self.scaler['y_std'])
        self.splits = {'fit':np.arange(257), 'stop':np.arange(257,274),
            'validation':np.arange(274,293)}
        self.x = rng.normal(0,.25,(293,480,1)).astype('float32')
        self.anchor = rng.normal(0,.25,(293,1)).astype('float32')
        self.y = rng.normal(0,1,(293,60)).astype('float32')
        self.raw = self.y*self.ys+self.ym
        self.bars = SimpleNamespace(synthetic=True,synthetic_targets=self.raw)
        self.windows = SimpleNamespace(arrays=lambda ix:(self.x[ix,:,0],self.anchor[ix]))
        self.reads = []

    def batch(self,part,pos):
        assert part in ('fit','stop'), 'Outer labels must not be constructed'
        self.reads.append(part)
        # Exercise both RNG streams, including across the real process exit.
        torch.rand(7)
        torch.rand(11,device='cuda')
        if part=='fit' and getattr(self,'signal_after_first_fit',False):
            self.signal_after_first_fit=False
            os.kill(os.getpid(),signal.SIGUSR1)
        ix=self.splits[part][pos]
        return (torch.from_numpy(self.x[ix]),torch.from_numpy(self.anchor[ix])),torch.from_numpy(self.y[ix])


def exact(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(exact(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    return a==b


def mapped_state(bundle,h):
    encoder=bundle.encoder if bundle.tied else bundle.encoders[str(h)]
    head=bundle.heads[str(h)]
    out={}
    for name,value in encoder.state_dict().items():
        out[name if name.startswith('ema.') else 'backbone.'+name]=value
    for name,value in head.state_dict().items():
        out[name if name.startswith('anchor_head.') else 'backbone.'+name]=value
    return out


def hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (BASE/'code').glob('*.py')}


def invariants():
    data=SyntheticData()
    tied=Bundle(480,data.scaler,2026,True)
    untied=Bundle(480,data.scaler,2026,False)
    assert all(exact(mapped_state(tied,h),mapped_state(untied,h)) for h in HORIZONS)
    gpu=deepcopy(tied).cuda()
    assert exact(tied.state_dict(),gpu.state_dict())
    args,y=data.batch('fit',np.arange(16))
    cpu_rng=torch.get_rng_state().clone(); cuda_rng=torch.cuda.get_rng_state().clone()
    tied.eval(); untied.eval(); gpu.eval()
    with torch.no_grad():
        pc=tied(*args);pu=untied(*args);pg=gpu(*(a.cuda() for a in args))
    assert all(torch.equal(pc[h],pu[h]) and torch.equal(pc[h],pg[h].cpu()) for h in HORIZONS)
    raw_error=max(float(np.abs(pc[h].numpy()*data.ys[:h]+data.ym[:h]).max()) for h in HORIZONS)
    # A nonzero original function ensures device comparison is not only zero output.
    native=native_reference(480,60,2026,None).eval()
    native_gpu=deepcopy(native).cuda().eval()
    with torch.no_grad():
        a=native(*args);b=native_gpu(*(x.cuda() for x in args)).cpu()
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    restoration_error=float((a-b).abs().max())
    # Untied bundle must optimize exactly as four separate native Adam problems.
    bundle=untied.cuda()
    opt=exp.make_optimizer(bundle)
    natives={h:native_reference(480,h,2026,data.scaler).cuda() for h in HORIZONS}
    opts={h:exp.make_optimizer(natives[h]) for h in HORIZONS}
    max_error=0.; bitwise=True
    gradients=[]
    for step in range(3):
        args,y=data.batch('fit',np.arange(step*16,(step+1)*16))
        args=tuple(a.cuda() for a in args);y=y.cuda()
        bundle.train();opt.zero_grad(set_to_none=True)
        loss=exp.loss(bundle(*args),y);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in bundle.parameters())
        gradients.append(sum(bool(torch.count_nonzero(p.grad)) for p in bundle.parameters()))
        opt.step()
        for h in HORIZONS:
            n=natives[h];n.train();opts[h].zero_grad(set_to_none=True)
            value=(n(*args)-y[:,:h]).square().mean();value.backward();opts[h].step()
            state=mapped_state(bundle,h);ref=n.state_dict()
            for k in state:
                bitwise=bitwise and torch.equal(state[k],ref[k])
                if state[k].dtype.is_floating_point:
                    max_error=max(max_error,float((state[k]-ref[k]).abs().max()))
                torch.testing.assert_close(state[k],ref[k],atol=1e-7,rtol=1e-6)
    bn={k:int(v) for k,v in bundle.state_dict().items() if k.endswith('num_batches_tracked')}
    assert len(bn)==12 and set(bn.values())=={3}
    # Shared encoder BN counts each batch once, despite four decoder uses.
    shared=Bundle(480,data.scaler,2026,True).cuda().train()
    with torch.no_grad():
        for _ in range(3):shared(*args)
    shared_bn={k:int(v) for k,v in shared.state_dict().items() if k.endswith('num_batches_tracked')}
    assert len(shared_bn)==3 and set(shared_bn.values())=={3}
    return {'same_head_arm_initial_states_bitwise_equal':True,
        'CPU_GPU_initial_state_and_baseline_outputs_bitwise_equal':True,
        'raw_zero_initialization_max_error':raw_error,
        'nonzero_native_CPU_GPU_max_abs_error':restoration_error,
        'CPU_GPU_nonzero_forward_atol':2e-5,'CPU_GPU_nonzero_forward_rtol':2e-5,
        'GPU_native_Adam_updates':3,'GPU_untied_native_state_bitwise_equal':bitwise,
        'GPU_untied_native_state_max_abs_error':max_error,
        'GPU_untied_native_state_atol':1e-7,'GPU_untied_native_state_rtol':1e-6,
        'nonzero_gradient_parameter_tensors_by_update':gradients,
        'untied_BN_counters':bn,'tied_BN_counters':shared_bn}


def graph_tail_equivalence():
    """Verify warmup rollback and static-gradient recovery after eager tail."""
    data=SyntheticData(); rows=[]
    for arm in ('tied','untied'):
        graph_net=Bundle(480,data.scaler,2026,arm=='tied').cuda().train()
        eager=deepcopy(graph_net).train()
        graph_opt=exp.make_optimizer(graph_net);eager_opt=exp.make_optimizer(eager)
        args,y=data.batch('fit',np.arange(128));args=tuple(a.cuda() for a in args);y=y.cuda()
        weights=deepcopy(graph_net.state_dict());cpu_rng=torch.get_rng_state().clone()
        cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()]
        runner=exp.GraphBackward(graph_net,graph_opt,exp.loss,args,y)
        assert exact(weights,graph_net.state_dict()),'Capture changed weights or BN'
        assert torch.equal(cpu_rng,torch.get_rng_state()),'Capture changed CPU RNG'
        assert exact(cuda_rng,torch.cuda.get_rng_state_all()),'Capture changed CUDA RNG'
        assert not graph_opt.state,'Capture advanced Adam optimizer'
        max_error=0.;bitwise=True;update_rows=[]
        for step,positions in enumerate((np.arange(128),np.arange(128,257),np.arange(128))):
            args,y=data.batch('fit',positions);args=tuple(a.cuda() for a in args);y=y.cuda()
            value,runner=exp.training_backward(graph_net,graph_opt,args,y,runner)
            assert exp.finite_gradients(graph_net)
            if len(y)==128:
                assert all(p.grad is g for p,g in zip(runner.parameters,runner.gradients)), 'Graph gradient references lost after eager tail'
            graph_opt.step()
            eager.train();eager_opt.zero_grad(set_to_none=True)
            exp.loss(eager(*args),y).backward();eager_opt.step()
            for k,a in graph_net.state_dict().items():
                b=eager.state_dict()[k]
                bitwise=bitwise and torch.equal(a,b)
                if a.dtype.is_floating_point:max_error=max(max_error,float((a-b).abs().max()))
                torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
            assert exact(exp.cpu_snapshot(graph_opt.state_dict()),exp.cpu_snapshot(eager_opt.state_dict())), 'Graph/eager Adam state differs'
            update_rows.append({'step':step+1,'batch_size':len(y),'backend':'graph' if len(y)==128 else 'eager_tail'})
        counts={k:int(v) for k,v in graph_net.state_dict().items() if k.endswith('num_batches_tracked')}
        assert set(counts.values())=={3},'Capture or tail overcounted BatchNorm'
        rows.append({'arm':arm,'warmup_weights_BN_CPU_CUDA_RNG_rollback_bitwise_equal':True,
            'warmup_does_not_advance_Adam':True,'updates':update_rows,
            'graph_static_gradient_references_restored_after_eager_tail':True,
            'graph_eager_model_BN_bitwise_equal':bitwise,'max_abs_error':max_error,
            'graph_eager_Adam_state_bitwise_equal':True,'BN_counters':counts})
    return {'scope':'generated_arrays_only; fused Adam in both graph/eager arms',
        'comparison_atol':1e-6,'comparison_rtol':1e-5,'rows':rows}


def child(work,arm,limit):
    exp.setup()
    exp.BASE=Path(work)
    exp.STOP=False
    exp.freeze=lambda:{'admitted':True,'fixed_epochs':3}
    made=[]
    def factory(_):
        data=SyntheticData();data.signal_after_first_fit=limit is not None
        made.append(data);return data
    def blocked(*a,**k):raise AssertionError('Real market loader prohibited')
    exp.Data=blocked;exp.data_for=factory
    def targets(bars,origins,horizon,target):
        assert getattr(bars,'synthetic',False)
        assert horizon==60 and target=='future_log_return'
        assert np.isin(origins,np.arange(257,274)).all()
        return bars.synthetic_targets[origins,:horizon]
    bars_module.targets=targets
    job={'id':arm+'_synthetic_gpu','arm':arm,'fold':1,'seed':2026}
    out=exp.run(job)
    assert all('validation' not in d.reads for d in made)
    print(json.dumps({'child_pid':os.getpid(),'status':out['status'],
        'steps':out.get('steps'),'batch_reads':[d.reads for d in made]}),flush=True)


def process_resume():
    root=REVIEW/('gpu_synthetic_process_'+str(time.time_ns()))
    rows=[]
    for arm in ('tied','untied'):
        dirs={}; children=[]
        for mode in ('uninterrupted','interrupted'):
            work=root/arm/mode;(work/'protocol').mkdir(parents=True)
            (work/'protocol/RULES.json').write_text('{"synthetic_only":true,"fixed_epochs":3}\n')
            cmd=[sys.executable,str(Path(__file__).resolve()),'--child',str(work),'--arm',arm]
            def invoke(extra):
                run=subprocess.run(cmd+extra,text=True,capture_output=True,timeout=90)
                children.append({'returncode':run.returncode,'stdout':run.stdout,'stderr':run.stderr})
                assert run.returncode==0,run.stderr+run.stdout
                return json.loads(run.stdout.splitlines()[-1])
            if mode=='interrupted':
                paused=invoke(['--limit','1']);assert paused['status']=='paused'
                saved=torch.load(work/'runs'/f'{arm}_synthetic_gpu/latest.pt',map_location='cpu',weights_only=True)
                assert saved['state']['steps']==1 and saved['state']['cursor']==128
                assert 'cuda_rng' in saved or 'cuda_rng_all' in saved or 'cuda_rng_states' in saved
            done=invoke([]);assert done['status']=='completed' and done['steps']==6
            dirs[mode]=work/'runs'/f'{arm}_synthetic_gpu'
        a=torch.load(dirs['uninterrupted']/'latest.pt',map_location='cpu',weights_only=True)
        b=torch.load(dirs['interrupted']/'latest.pt',map_location='cpu',weights_only=True)
        keys=['identity','model','optimizer','numpy_rng','torch_rng','epoch','scaler']
        keys += [k for k in a if 'cuda' in k and 'rng' in k]
        assert any('cuda' in k for k in keys)
        for k in keys:assert exact(a[k],b[k]),(arm,k)
        for k in ('epoch','cursor','permutation','steps','best_score','best_epoch','history','loss_sum','seen'):
            assert exact(a['state'][k],b['state'][k]),(arm,'state',k)
        for name in ('initial','last','best'):
            ca=torch.load(dirs['uninterrupted']/(name+'.pt'),map_location='cpu',weights_only=True)
            cb=torch.load(dirs['interrupted']/(name+'.pt'),map_location='cpu',weights_only=True)
            assert exact(ca,cb),(arm,name)
        with np.load(dirs['uninterrupted']/'predictions.npz') as pa,np.load(dirs['interrupted']/'predictions.npz') as pb:
            assert set(pa.files)==set(pb.files) and len(pa.files)==17
            assert all(np.array_equal(pa[k],pb[k]) for k in pa.files)
            assert all(pa[f'{mode}{suffix}_h{h}'].shape==(19,h)
                for mode in ('last','selected') for suffix in ('','_shrink') for h in HORIZONS)
        bn={k:int(v) for k,v in a['model'].items() if k.endswith('num_batches_tracked')}
        assert set(bn.values())=={6} and len(bn)==(3 if arm=='tied' else 12)
        rows.append({'arm':arm,'actual_child_processes':3,'fixed_epochs':3,'steps':6,
            'interruption':'actual OS SIGUSR1 in first fit batch, saved at next boundary, process exits, fresh process resumes',
            'model_optimizer_CPU_CUDA_RNG_cursor_history_bitwise_equal':True,
            'initial_last_best_checkpoints_bitwise_equal':True,
            'all_16_prediction_arrays_bitwise_equal':True,'compared_rng_keys':keys,
            'batchnorm_counters':bn,'children':children})
    return {'evidence_root':str(root),'rows':rows}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child');parser.add_argument('--arm');parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    assert set(os.sched_getaffinity(0)).issubset({24,26})
    if args.child:return child(args.child,args.arm,args.limit)
    exp.setup()
    before=hashes()
    result={'scope':'synthetic_only_no_market_Data_stop_outer_or_Q3',
        'source_hashes_before':before,'GPU':True,'GPU_name':torch.cuda.get_device_name(0),
        'invariants':invariants(),'graph_tail_equivalence':graph_tail_equivalence(),
        'actual_process_resume':process_resume()}
    result['source_hashes_after']=hashes()
    assert before==result['source_hashes_after'],'Source changed during audit'
    result.update(status='passed',Q3_values_read=False,formal_market_training=False,
        same_sources_during_audit=True)
    (REVIEW/'GPU_SYNTHETIC_ACCEPTANCE.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
