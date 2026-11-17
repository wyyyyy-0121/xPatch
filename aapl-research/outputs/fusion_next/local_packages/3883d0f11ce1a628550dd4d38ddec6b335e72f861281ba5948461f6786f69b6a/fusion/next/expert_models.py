"""Same-information supervised experts for the new all-day task."""
import numpy as np
import torch
from torch import nn

from aapl.models import CausalEMA
from aapl.vendor.xpatch_network import Network
from aapl.vendor.patchtst_backbone import PatchTST_backbone


class NativeExpert(nn.Module):
    def __init__(self,config,channels,stocks):
        super().__init__();self.config=config
        if config['family']=='xpatch':
            self.ema=CausalEMA(config.get('alpha',.1))
            self.native=Network(config['lookback'],12,config.get('patch',8),config.get('stride',4),'end')
        elif config['family']=='patchtst':
            d=config.get('d_model',32)
            self.native=PatchTST_backbone(c_in=channels,context_window=config['lookback'],target_window=12,
                    patch_len=config.get('patch',8),stride=config.get('stride',4),n_layers=1,d_model=d,n_heads=4,d_ff=2*d,
                    dropout=config.get('dropout',.1),fc_dropout=0.,head_dropout=0.,padding_patch='end',revin=False,
                    affine=True,subtract_last=False,individual=False)
        else:raise ValueError(config['family'])
        self.output=nn.Linear(channels,1)
        self.stock=nn.Embedding(stocks,12)
        nn.init.zeros_(self.stock.weight)

    def forward(self,batch):
        x=torch.cat([batch['local'],batch['market']],dim=-1)
        if self.config['family']=='xpatch':
            trend=self.ema(x);values=self.native(x-trend,trend)
        else:values=self.native(x.transpose(1,2)).transpose(1,2)
        steps=self.output(values)[:,:,0]+self.stock(batch['stock'])
        steps=steps*batch['target_scale'][:,None]
        return {'steps':steps,'prediction':steps.sum(1)}


def tabular(batch,stock_count=None):
    """Deterministic summaries of the same completed local and market channels."""
    x=torch.cat([batch['local'],batch['market']],dim=-1)
    columns=[x[:,-1]]
    for length in (12,48,96,240):
        z=x[:,-min(length,x.shape[1]):]
        columns.extend([z.mean(1),z.std(1,unbiased=False)])
    stock=torch.nn.functional.one_hot(batch['stock'],stock_count).float() if stock_count else batch['stock'][:,None].float()
    columns.extend([batch['state'],stock])
    return torch.cat(columns,dim=1).detach().cpu().numpy().astype(np.float32)
