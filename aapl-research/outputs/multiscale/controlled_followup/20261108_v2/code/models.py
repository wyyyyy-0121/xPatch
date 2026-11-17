import torch
from torch import nn
from torch.nn import functional as F
from multiscale.models import make_model

class TinyPath(nn.Module):
    def __init__(self,horizon,field,zero_scaled):
        super().__init__()
        self.dilations=[1]*8 if field=='short' else [2**i for i in range(8)]
        self.layers=nn.ModuleList([nn.Conv1d(1 if i==0 else 8,8,3,dilation=d)
                                 for i,d in enumerate(self.dilations)])
        self.head=nn.Linear(16,horizon)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.register_buffer('zero_scaled',torch.as_tensor(zero_scaled,dtype=torch.float32))
        assert sum(p.numel() for p in self.parameters())==1432+17*horizon
    def representation(self,x):
        v=x[:,None,:]
        for layer,d in zip(self.layers,self.dilations):
            v=F.relu(layer(F.pad(v,(2*d,0))))
        return v
    def forward(self,x,anchor=None):
        v=self.representation(x)
        return self.head(torch.cat((v[:,:,-1],v.mean(2)),1))+self.zero_scaled

def create(cfg,zero_scaled):
    if cfg['family']=='original':return make_model(cfg['model'],cfg['lookback'],cfg['horizon'])
    return TinyPath(cfg['horizon'],cfg['field'],zero_scaled)

def objective(pred,y,loss_kind):
    sq=(pred-y).square()
    return sq.mean() if loss_kind=='path' else .5*sq.mean()+.5*sq[:,-1].mean()
