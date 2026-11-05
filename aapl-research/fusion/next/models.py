"""User-plan neural architecture: native local branch, market encoder, learned FM projection."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from aapl.models import CausalEMA
from aapl.vendor.xpatch_network import Network
from .foundation import load_foundation


class LocalBranch(nn.Module):
    def __init__(self,lookback,channels,width=64):
        super().__init__()
        self.ema=CausalEMA(.1)
        self.native=Network(lookback,12,8,4,'end')
        self.projection=nn.Sequential(nn.Linear(12*channels,width),nn.LayerNorm(width),nn.GELU())

    def forward(self,x):
        trend=self.ema(x)
        values=self.native(x-trend,trend)
        return self.projection(values.flatten(1))


class CausalConv(nn.Module):
    def __init__(self,incoming,outgoing,dilation):
        super().__init__();self.padding=4*dilation
        self.conv=nn.Conv1d(incoming,outgoing,5,dilation=dilation)
        self.norm=nn.LayerNorm(outgoing)

    def forward(self,x):
        # No right-side padding or dependence on unavailable time steps.
        x=self.conv(F.pad(x,(self.padding,0)))
        return F.gelu(self.norm(x.transpose(1,2))).transpose(1,2)


class MarketBranch(nn.Module):
    def __init__(self,channels,width,kind='tcn'):
        super().__init__();self.kind=kind
        if kind=='tcn':
            self.net=nn.Sequential(CausalConv(channels,width,1),CausalConv(width,width,4),CausalConv(width,width,16))
        elif kind=='transformer':
            self.input=nn.Linear(channels,width)
            layer=nn.TransformerEncoderLayer(width,4,2*width,dropout=0.,batch_first=True,norm_first=True)
            self.net=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        else:raise ValueError(kind)

    def sequence(self,x):
        if self.kind=='tcn':return self.net(x.transpose(1,2)).transpose(1,2)
        n=x.shape[1];x=self.input(x)
        position=torch.arange(n,device=x.device,dtype=x.dtype)[:,None]
        scale=torch.exp(torch.arange(0,x.shape[-1],2,device=x.device,dtype=x.dtype)*(-np.log(10000.)/x.shape[-1]))
        code=torch.zeros(n,x.shape[-1],device=x.device,dtype=x.dtype)
        code[:,::2]=torch.sin(position*scale);code[:,1::2]=torch.cos(position*scale)
        mask=torch.ones(n,n,dtype=torch.bool,device=x.device).triu(1)
        return self.net(x+code[None],mask=mask)

    def forward(self,x):return self.sequence(x)[:,-1]


class FoundationBranch(nn.Module):
    def __init__(self,projection_width,device='cuda',mode='frozen',rank=8,random_seed=6026):
        super().__init__()
        pipe,receipt=load_foundation(device=device)
        self.backbone=pipe.model
        self.receipt=receipt;self.mode=mode
        if mode=='random_frozen':
            with torch.random.fork_rng(devices=[]):
                # Construct on CPU without changing any caller CUDA RNG stream.
                torch.random.default_generator.manual_seed(random_seed)
                self.backbone=type(self.backbone)(self.backbone.config).to(device)
        for p in self.backbone.parameters():p.requires_grad=False
        if mode=='lora':
            from peft import LoraConfig,get_peft_model
            targets=['self_attention.q','self_attention.k','self_attention.v','self_attention.o']
            self.backbone=get_peft_model(self.backbone,LoraConfig(r=rank,lora_alpha=2*rank,lora_dropout=0.,target_modules=targets))
            self.trainable_names=[n for n,p in self.backbone.named_parameters() if p.requires_grad]
            if not self.trainable_names:raise RuntimeError('LoRA attached no trainable parameters')
            if any('lora_' not in n for n in self.trainable_names):raise RuntimeError('Unexpected full-parameter unfreeze')
        elif mode not in ('frozen','random_frozen'):raise ValueError(mode)
        width=self.backbone.config.d_model
        self.projection=nn.Sequential(nn.Linear(width,projection_width),nn.LayerNorm(projection_width),nn.GELU())

    def latent(self,context):
        # context: [B,L,3], no future-target or future-covariate arguments exist.
        batch,length,variables=context.shape
        groups=torch.arange(batch,device=context.device).repeat_interleave(variables)
        series=context.transpose(1,2).reshape(batch*variables,length)
        self.backbone.eval()
        with torch.set_grad_enabled(self.training and self.mode=='lora'):
            encoded,*_=self.backbone.encode(context=series,group_ids=groups,num_output_patches=1)
            # Symmetric variable pooling; the dimensionality reduction is learned below.
            latent=encoded[0][:,-1].reshape(batch,variables,-1).mean(1)
        return latent

    def forward(self,context,cached=None):
        if cached is not None:
            if self.mode!='frozen':raise ValueError('Representation cache is invalid for changing/random backbones')
            latent=cached.float()
        else:latent=self.latent(context)
        return self.projection(latent)


class NeuralFusion(nn.Module):
    def __init__(self,config,local_channels,market_channels,stocks,device='cuda'):
        super().__init__();self.config=config
        self.local=LocalBranch(config['lookback'],local_channels,64)
        self.use_foundation=config.get('foundation',True)
        self.use_market=config.get('market',True)
        self.stock=nn.Embedding(stocks,8)
        self.foundation=FoundationBranch(config['projection'],device,config.get('foundation_mode','frozen'),config.get('rank',8),config.get('seed',2026)+4000) if self.use_foundation else None
        self.market=MarketBranch(market_channels,32,config['market_encoder']) if self.use_market else None
        width=64+8+8+4+2+(config['projection'] if self.use_foundation else 0)+(32 if self.use_market else 0)
        hidden=config['hidden']
        self.joint=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Dropout(config.get('dropout',0.)),
                                 nn.Linear(hidden,hidden),nn.GELU())
        self.gate=nn.Linear(hidden,5)
        self.shrink=nn.Linear(hidden,1)
        self.residual=nn.Linear(hidden,1)
        self.plain=nn.Linear(hidden,1)
        # Early-history pretraining is supervised without inventing in-sample expert predictions.
        self.auxiliary=nn.Linear(hidden,1)
        inactive=(self.gate,self.shrink,self.residual) if config.get('fusion_type','gate')=='plain_mlp' else (self.plain,)
        for head in inactive:
            for parameter in head.parameters():parameter.requires_grad=False
        if config.get('fusion_type','gate')!='plain_mlp' and config['residual_cap']==0:
            for parameter in self.residual.parameters():parameter.requires_grad=False
        nn.init.zeros_(self.residual.weight);nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        with torch.no_grad():self.gate.bias[-1]=1.5
        trainable=sum(p.numel() for p in self.parameters() if p.requires_grad)
        if trainable>5_000_000:raise ValueError(f'New trainable parameters exceed user-plan limit: {trainable}')

    def forward(self,batch):
        experts=batch['experts'];scale=batch['sigma12'].clamp_min(1e-6)
        parts=[self.local(batch['local']),self.stock(batch['stock']),batch['state'],
               experts/scale[:,None],experts.std(1,keepdim=True)/scale[:,None],
               experts.abs().mean(1,keepdim=True)/scale[:,None]]
        if self.use_foundation:parts.append(self.foundation(batch['foundation_context'],batch.get('foundation_cached')))
        if self.use_market:parts.append(self.market(batch['market']))
        z=self.joint(torch.cat(parts,dim=1))
        if self.config.get('fusion_type','gate')=='plain_mlp':
            prediction=scale*self.plain(z)[:,0]
            weights=torch.zeros(len(z),5,device=z.device)
            shrink=torch.ones(len(z),device=z.device)
        else:
            weights=self.gate(z).softmax(-1);shrink=self.shrink(z)[:,0].sigmoid()
            choices=torch.cat([experts,torch.zeros_like(experts[:,:1])],dim=1)
            residual=self.config['residual_cap']*scale*torch.tanh(self.residual(z)[:,0])
            prediction=shrink*(weights*choices).sum(1)+residual
        return {'prediction':prediction,'weights':weights,'shrink':shrink,
                'auxiliary_prediction':scale*self.auxiliary(z)[:,0]}
