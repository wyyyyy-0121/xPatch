import torch
import pytest

from .models import MarketBranch,NeuralFusion


@pytest.mark.parametrize('kind',['tcn','transformer'])
def test_market_branch_cannot_read_future_prefix(kind):
    torch.manual_seed(33);net=MarketBranch(6,32,kind).eval()
    x=torch.randn(3,48,6)
    # Evaluate the full sequence so future tokens really enter the encoder.
    before=net.sequence(x)
    changed=x.clone();changed[:,32:]+=100
    after=net.sequence(changed)
    torch.testing.assert_close(before[:,:32],after[:,:32],rtol=0,atol=0)
    assert not torch.equal(before[:,32:],after[:,32:])


def test_joint_gate_zero_experts_and_zero_residual_are_zero():
    cfg={'lookback':48,'hidden':32,'projection':64,'market_encoder':'tcn',
         'residual_cap':0.,'foundation':False,'market':True}
    net=NeuralFusion(cfg,12,24,16,device='cpu').eval()
    batch={'local':torch.randn(4,48,12),'market':torch.randn(4,48,24),
           'stock':torch.tensor([0,1,2,3]),'state':torch.randn(4,8),'sigma12':torch.full((4,),.001),
           'experts':torch.zeros(4,4)}
    output=net(batch)
    assert torch.equal(output['prediction'],torch.zeros(4))
    torch.testing.assert_close(output['weights'].sum(-1),torch.ones(4))
    assert torch.all((output['shrink']>=0)&(output['shrink']<=1))


def test_joint_predictor_is_origin_isolated_in_eval():
    cfg={'lookback':48,'hidden':32,'projection':64,'market_encoder':'tcn',
         'residual_cap':.25,'foundation':False,'market':True}
    net=NeuralFusion(cfg,12,24,16,device='cpu').eval()
    batch={'local':torch.randn(3,48,12),'market':torch.randn(3,48,24),
           'stock':torch.tensor([0,1,2]),'state':torch.randn(3,8),'sigma12':torch.full((3,),.001),
           'experts':torch.randn(3,4)*.0001}
    batched=net(batch)['prediction']
    isolated=torch.cat([net({k:v[i:i+1] for k,v in batch.items()})['prediction'] for i in range(3)])
    torch.testing.assert_close(batched,isolated,rtol=1e-5,atol=1e-7)
