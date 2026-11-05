"""Resume the actual optimizer/RNG/sampler stream, and audit information boundaries."""
from types import SimpleNamespace
import random
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from . import checkpoint
from .data import CHANNELS, causal_transform, eligible_origins
from .panel import Panel, BalancedSampler,CoverageSampler
from .splits import fold
from .common import OUT,digest
from .expert_models import NativeExpert
from .trainer import Trainer
from .trainer import loss_for
from .models import NeuralFusion
from .tests_data import two_sessions


def seed():
    torch.manual_seed(19);random.seed(19);np.random.seed(19)


def toy_sampler():
    panel=SimpleNamespace(stocks=['AAPL','MSFT'],dates=np.array(['2021-01-04']*20+['2021-01-05']*20),
                          origins={'AAPL':np.arange(40),'MSFT':np.arange(40)})
    return BalancedSampler(panel,'2021-01-05',83)


def model():
    net=nn.Sequential(nn.Linear(3,8),nn.BatchNorm1d(8),nn.Dropout(.3),nn.Linear(8,1))
    # An immutable parameter must be reconstructed, not duplicated in each checkpoint.
    net.register_parameter('fixed',nn.Parameter(torch.ones(1),requires_grad=False))
    return net


def update(net,opt,sampler):
    stocks,rows=sampler.sample(8)
    x=torch.column_stack([torch.as_tensor(rows).float()/40,torch.as_tensor(stocks).float(),torch.randn(8)])
    x=x+random.random()*.01+float(np.random.random())*.01
    opt.zero_grad();loss=(net(x)[:,0]-torch.sin(x[:,0])).square().mean();loss.backward();opt.step()
    return float(loss.detach())


def test_interrupted_training_matches_continuous_including_batchnorm_and_dropout(tmp_path):
    seed();a=model();oa=torch.optim.AdamW(a.parameters(),lr=.01);sa=toy_sampler()
    expected=[update(a,oa,sa) for _ in range(12)]
    seed();b=model();ob=torch.optim.AdamW(b.parameters(),lr=.01);sb=toy_sampler()
    for _ in range(6):update(b,ob,sb)
    path=tmp_path/'resume.pt';recipe={'seed':19,'total_updates':12,'base':'frozen identity'}
    checkpoint.save(path,b,ob,{'main':sb},6,recipe)
    assert 'fixed' not in checkpoint.changing_state(b)
    # New model initialization intentionally consumes RNG before restore.
    c=model();oc=torch.optim.AdamW(c.parameters(),lr=.01);sc=toy_sampler()
    step,_=checkpoint.load(path,c,oc,{'main':sc},recipe);assert step==6
    actual=[update(c,oc,sc) for _ in range(6)]
    np.testing.assert_array_equal(actual,expected[6:])
    for name,value in a.state_dict().items():torch.testing.assert_close(value,c.state_dict()[name],rtol=0,atol=0)
    with pytest.raises(ValueError,match='recipe'):
        checkpoint.load(path,c,oc,{'main':sc},{**recipe,'total_updates':13})


def test_full_trading_day_is_purged_and_Q2_Q3_have_no_development_entry():
    dates=np.array(['2022-12-28','2022-12-29','2022-12-30','2023-01-03'])
    value=fold(dates,'2023Q1')
    assert value['fit_end']=='2022-12-29' and value['purged_full_session']=='2022-12-30'
    with pytest.raises(ValueError,match='confirmation'):fold(dates,'2026Q2')
    with pytest.raises(ValueError,match='Q3'):fold(dates,'2026Q3')


def make_panel(frame,starts,ends):
    features,observed,close,_=causal_transform(frame,starts);origins=eligible_origins(observed,ends)
    y=np.full(len(frame),np.nan,dtype=np.float32);y[origins]=np.log(close[origins+12]/close[origins])
    return Panel(['AAPL'],['SPY'],frame.index.strftime('%Y-%m-%d').to_numpy(),frame.index,
                 {n:features.copy() for n in ('AAPL','SPY')},{n:observed.copy() for n in ('AAPL','SPY')},
                 {n:origins.copy() for n in ('AAPL','SPY')},{n:y.copy() for n in ('AAPL','SPY')})


def test_close_only_path_does_not_retain_open_high_low_or_volume():
    frame,starts,ends=two_sessions();first=make_panel(frame,starts,ends)
    changed=frame.copy();changed['high']*=1.05;changed['low']*=.95;changed['open']*=1.01;changed['volume']*=99
    second=make_panel(changed,starts,ends)
    config={'lookback':240,'information':'close'};rows=np.array([390,430,700]);stocks=np.zeros(3,dtype=np.int64)
    a=first.windows(stocks,rows,config,first.scaler('2021-01-05'))
    b=second.windows(stocks,rows,config,second.scaler('2021-01-05'))
    for field in ('local','market','foundation_context','state','sigma12','target'):
        torch.testing.assert_close(a[field],b[field],rtol=0,atol=0)
    # An overnight close-to-close move is its own signal, never minute volatility.
    assert a['local'].shape[-1]==4 and a['foundation_context'].shape[-1]==2
    assert a['foundation_context'][0,-1,0]==0
    assert a['foundation_context'][0,-1,1]==pytest.approx(float(np.log(1.1)))
    assert a['sigma12'][0]==pytest.approx(1e-6)


def test_prediction_window_never_reads_targets_or_future_rows():
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends);scalers=panel.scaler('2021-01-04')
    class NoTargets:
        def __getitem__(self,key):raise AssertionError('Inference read a future target')
    class PastOnly:
        def __init__(self,array):self.array=array
        def __getitem__(self,index):
            assert np.max(index)<=450,'Inference read a future feature'
            return self.array[index]
    panel.targets=NoTargets();panel.features={k:PastOnly(v) for k,v in panel.features.items()}
    result=panel.windows([0],[450],{'lookback':240},scalers,include_labels=False)
    assert 'target' not in result and 'step_targets' not in result
    with pytest.raises(ValueError,match='history'):panel.windows([0],[5],{'lookback':48},scalers,include_labels=False)


@pytest.mark.parametrize('sampler_policy',[None,CoverageSampler.policy])
@pytest.mark.parametrize('prefetch',[0,2])
def test_actual_trainer_halving_and_mid_forward_failure_preserve_exact_resume(sampler_policy,prefetch):
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends)
    config={'family':'xpatch','lookback':48,'lr':.001,'batch_size':4,'total_updates':4,'loss':'raw'}
    if sampler_policy:config['sampler_policy']=sampler_policy
    config['prefetch_batches']=prefetch
    parent=OUT/'training';parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='unit_resume_',dir=parent) as directory:
        from pathlib import Path
        prefix=Path(directory).name
        torch.manual_seed(31)
        continuous=Trainer(panel,NativeExpert(config,24,1),{**config,'prefetch_batches':0},'2021-01-05',prefix+'/continuous',device='cpu')
        continuous.run_until(4);expected=checkpoint.changing_state(continuous.model)
        torch.manual_seed(31)
        first=Trainer(panel,NativeExpert(config,24,1),config,'2021-01-05',prefix+'/resumed',device='cpu')
        first.run_until(2);before=digest(first.path)
        def fault(*args):raise RuntimeError('injected after forward changed BatchNorm buffers')
        hook=first.model.register_forward_hook(fault)
        with pytest.raises(RuntimeError,match='injected'):first.run_until(4)
        hook.remove();assert digest(first.path)==before
        restored=Trainer(panel,NativeExpert(config,24,1),config,'2021-01-05',prefix+'/resumed',device='cpu')
        restored.resume();assert restored.step==2
        restored.run_until(4)
        for name,value in expected.items():torch.testing.assert_close(value,checkpoint.changing_state(restored.model)[name],rtol=0,atol=0)
        from .inference import checkpoint_model
        loaded,scalers,recipe=checkpoint_model(panel,restored.path,config,'cpu')
        batch=panel.windows([0,0],[430,700],config,scalers,include_labels=False)
        restored.model.eval()
        torch.testing.assert_close(loaded(batch)['prediction'],restored.model(batch)['prediction'],rtol=0,atol=0)


def test_coverage_epoch_visits_all_rows_balances_stocks_and_resumes_mid_epoch(tmp_path):
    dates=np.array(['2021-01-04']*13+['2021-01-05']*7+['2021-01-06']*8)
    panel=SimpleNamespace(stocks=['AAPL','MSFT'],dates=dates,
        origins={'AAPL':np.arange(28),'MSFT':np.r_[np.arange(10),np.arange(13,19)]})
    allowed={0:np.arange(28)!=3,1:np.arange(28)!=15}
    sampler=CoverageSampler(panel,'2021-01-05',83,available=allowed)
    first=sampler.sample(11);saved=sampler.state_dict()
    # Test the actual safe checkpoint serialization format, not an in-memory copy.
    path=tmp_path/'sampler.pt';torch.save(saved,path)
    restored=CoverageSampler(panel,'2021-01-05',83,available=allowed)
    restored.load_state_dict(torch.load(path,weights_only=True))
    tail=sampler.sample(sampler.epoch_size-11)
    actual=restored.sample(restored.epoch_size-11)
    for left,right in zip(tail,actual):np.testing.assert_array_equal(left,right)
    stocks=np.r_[first[0],tail[0]];rows=np.r_[first[1],tail[1]]
    expected={(sid,int(row)) for sid,name in enumerate(panel.stocks) for row in panel.origins[name]
              if dates[row]<='2021-01-05' and allowed[sid][row]}
    assert set(zip(stocks,rows))==expected
    assert np.sum(stocks==0)==np.sum(stocks==1)
    report=sampler.summary();assert report['unique_windows_seen']==len(expected)
    assert report['unique_coverage_fraction']==1. and report['completed_coverage_epochs']==1
    for left,right in zip(sampler.sample(sampler.epoch_size+3),restored.sample(restored.epoch_size+3)):
        np.testing.assert_array_equal(left,right)


def test_fusion_uses_one_day_balanced_sampler_for_early_and_OOF_history():
    frame,starts,ends=two_sessions()
    later=frame.copy();later.index=later.index+pd.DateOffset(years=2)
    all_frame=pd.concat([frame,later]);all_starts=np.tile(starts,2);all_ends=np.r_[ends,ends+780]
    panel=make_panel(all_frame,all_starts,all_ends)
    class Experts:
        available={0:panel.dates>='2023-01-01'}
        sources={'test':'known causal OOF'}
        def predict(self,stocks,rows):
            assert np.all(panel.dates[rows]>='2023-01-01')
            return np.full((len(rows),4),.0001,dtype=np.float32)
    config={'family':'fusion','lookback':48,'hidden':32,'projection':64,'market_encoder':'tcn',
            'residual_cap':.25,'foundation':False,'market':True,'lr':.001,'batch_size':32,'total_updates':2,'loss':'raw'}
    parent=OUT/'training';parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='unit_balance_',dir=parent) as directory:
        from pathlib import Path
        net=NeuralFusion(config,12,12,1,device='cpu')
        trainer=Trainer(panel,net,config,'2023-01-05',Path(directory).name,experts=Experts(),device='cpu')
        sampler=trainer.samplers['main'];_,rows=sampler.sample(20000)
        assert np.mean(panel.dates[rows]<'2023-01-01')==pytest.approx(.5,abs=.025)
        seen=[]
        hook=net.register_forward_pre_hook(lambda module,args:seen.append(args[0]['early_history_mask'].clone()))
        trainer.run_until(2);hook.remove()
        assert any(mask.any() and (~mask).any() for mask in seen)


def test_early_history_objective_routes_gradients_to_auxiliary_readout_only():
    main=torch.tensor([2.,2.],requires_grad=True);aux=torch.tensor([3.,3.],requires_grad=True)
    batch={'early_history_mask':torch.tensor([True,False]),'target':torch.zeros(2),'target_scale':torch.ones(2)}
    loss=loss_for({'prediction':main,'auxiliary_prediction':aux},batch,{'loss':'raw','raw_loss_unit':1.})
    loss.backward()
    torch.testing.assert_close(main.grad,torch.tensor([0.,2.]));torch.testing.assert_close(aux.grad,torch.tensor([3.,0.]))


def test_checkpoint_commit_survives_interruption_between_receipt_and_alias(monkeypatch):
    from pathlib import Path
    with TemporaryDirectory(prefix='unit_checkpoint_commit_',dir=OUT) as temporary:
        path=Path(temporary)/'resume.pt';recipe={'seed':19};seed()
        net=model();optimizer=torch.optim.AdamW(net.parameters(),lr=.01);sampler=toy_sampler()
        for _ in range(2):update(net,optimizer,sampler)
        checkpoint.save(path,net,optimizer,{'main':sampler},2,recipe)
        update(net,optimizer,sampler);expected=checkpoint.changing_state(net)
        replace=checkpoint.os.replace
        def interrupt_alias(source,destination):
            if Path(destination)==path:raise OSError('injected after atomic receipt commit')
            return replace(source,destination)
        monkeypatch.setattr(checkpoint.os,'replace',interrupt_alias)
        with pytest.raises(OSError,match='receipt commit'):checkpoint.save(path,net,optimizer,{'main':sampler},3,recipe)
        restored=model();new_optimizer=torch.optim.AdamW(restored.parameters(),lr=.01);new_sampler=toy_sampler()
        step,_=checkpoint.load(path,restored,new_optimizer,{'main':new_sampler},recipe)
        assert step==3
        for name,value in expected.items():torch.testing.assert_close(value,checkpoint.changing_state(restored)[name],rtol=0,atol=0)
        monkeypatch.setattr(checkpoint.os,'replace',replace)
        checkpoint.save(path,restored,new_optimizer,{'main':new_sampler},3,recipe)
        checkpoint.save(path,restored,new_optimizer,{'main':new_sampler},3,recipe)
        assert len(list((path.parent/('.'+path.stem+'.checkpoints')).glob('*.pt')))==2


def test_failed_checkpoint_receipt_publication_keeps_previous_prefix(monkeypatch):
    from pathlib import Path
    with TemporaryDirectory(prefix='unit_checkpoint_old_prefix_',dir=OUT) as temporary:
        path=Path(temporary)/'resume.pt';recipe={'seed':19};seed()
        net=model();optimizer=torch.optim.AdamW(net.parameters(),lr=.01);sampler=toy_sampler()
        checkpoint.save(path,net,optimizer,{'main':sampler},0,recipe);before=checkpoint.changing_state(net)
        update(net,optimizer,sampler)
        def fail(*args,**kwargs):raise OSError('injected receipt publication failure')
        monkeypatch.setattr(checkpoint,'write_json',fail)
        with pytest.raises(OSError):checkpoint.save(path,net,optimizer,{'main':sampler},1,recipe)
        restored=model();new_optimizer=torch.optim.AdamW(restored.parameters(),lr=.01);new_sampler=toy_sampler()
        step,_=checkpoint.load(path,restored,new_optimizer,{'main':new_sampler},recipe)
        assert step==0
        for name,value in before.items():torch.testing.assert_close(value,checkpoint.changing_state(restored)[name],rtol=0,atol=0)
