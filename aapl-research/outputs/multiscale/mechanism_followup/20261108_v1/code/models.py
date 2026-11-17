import torch
from torch import nn
from common import BASE
from multiscale.models import make_model

PROBES={'mlp_noanchor':'mlp','gru_mean':'gru','lstm_mean':'lstm',
        'xpatch_noanchor':'xpatch','xpatch_notrend':'xpatch','xpatch_noresidual':'xpatch'}

class Probe(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind;self.base=make_model(PROBES[kind],480,60)
    def forward(self,x,anchor):
        if self.kind.endswith('_noanchor'):return self.base(x,torch.zeros_like(anchor))
        if self.kind.endswith('_mean'):
            hidden=self.base.rnn(x)[0].mean(dim=1)
            return self.base.head(torch.cat((hidden,anchor),dim=1))
        t=self.base.ema(x);s=x-t
        if self.kind=='xpatch_notrend':t=t*0
        if self.kind=='xpatch_noresidual':s=s*0
        return (self.base.backbone(s,t)[:,:,0]+self.base.anchor_head(anchor))[:,:60]

class Small(nn.Module):
    def __init__(self,kind,zero_scaled):
        super().__init__();self.kind=kind
        self.trend_head=nn.Linear(4,60)
        self.conv1=nn.Conv1d(1,8,3,dilation=1,padding=2)
        self.conv2=nn.Conv1d(8,8,3,dilation=2,padding=4)
        self.local_head=nn.Linear(16,60)
        self.gating=nn.Sequential(nn.Linear(4,8),nn.ReLU(),nn.Linear(8,2))
        self.constants=nn.Parameter(torch.zeros(2))
        self.register_buffer('zero_scaled',torch.as_tensor(zero_scaled,dtype=torch.float32))
        nn.init.zeros_(self.trend_head.weight);nn.init.zeros_(self.trend_head.bias)
        nn.init.zeros_(self.local_head.weight);nn.init.zeros_(self.local_head.bias)
        active={'trend'} if kind=='trend' else {'short'} if kind in ('short','short_plain','short_const','short_state') else {'trend','short'}
        for p in self.trend_head.parameters():p.requires_grad_('trend' in active)
        for m in (self.conv1,self.conv2,self.local_head):
            for p in m.parameters():p.requires_grad_('short' in active)
        for p in self.gating.parameters():p.requires_grad_(kind in ('full','mix_noshrink','short_state'))
        self.constants.requires_grad_(kind in ('fixed_mix','short_const'))
        assert sum(p.numel() for p in self.parameters())<5000
    def forward(self,trend,residual,gate):
        t=self.trend_head(trend)
        v=torch.relu(self.conv1(residual[:,None,:])[...,:480])
        v=torch.relu(self.conv2(v)[...,:480])
        s=self.local_head(torch.cat((v[:,:,-1],v.mean(dim=2)),dim=1))
        if self.kind=='trend':out=t
        elif self.kind in ('short','short_plain'):out=s
        elif self.kind=='short_const':out=torch.sigmoid(self.constants[1])*s
        elif self.kind=='short_state':out=torch.sigmoid(self.gating(gate)[:,1:2])*s
        else:
            z=torch.sigmoid(self.constants)[None,:].expand(len(t),-1) if self.kind=='fixed_mix' else torch.sigmoid(self.gating(gate))
            g=z[:,0:1];lam=z[:,1:2] if self.kind!='mix_noshrink' else 1
            out=lam*(g*t+(1-g)*s)
        return out+self.zero_scaled

def create(kind,zero_scaled=None):
    if kind in PROBES:return Probe(kind)
    if kind=='lstm':return make_model('lstm',480,60)
    return Small(kind,zero_scaled)
