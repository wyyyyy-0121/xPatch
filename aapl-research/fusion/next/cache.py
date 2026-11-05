"""Sparse on-demand frozen latent cache, isolated by weights, data, time and source."""
from pathlib import Path

import numpy as np
import torch

from .common import OUT, identity, digest, source_identity, read_json, write_json


class RepresentationCache:
    def __init__(self,panel,branch,config,allowed_end='2026-03-31',storage_dtype='float32'):
        if branch.mode!='frozen':raise ValueError('Changing/LoRA/random encoder representations cannot use the pretrained cache')
        if allowed_end>'2026-03-31':raise ValueError('Confirmation representations require a separately frozen entry')
        if storage_dtype not in ('float32','float16'):raise ValueError('Unadmitted cache dtype')
        if storage_dtype=='float16' and not config.get('fp16_cache_admitted',False):
            raise ValueError('Float16 storage must first pass F2 numerical admission')
        self.panel=panel;self.branch=branch;self.allowed_end=allowed_end;self.dtype=np.dtype(storage_dtype)
        self.base={'weights':branch.receipt,'source':source_identity(),'information':config.get('information','ohlcv'),
                   'lookback':config['lookback'],'start':'2021-01-04','end':allowed_end,
                   'calendar_sha256':digest(OUT/'data/timestamps_ns.npy'),'dtype':storage_dtype,
                   'encoder_compute_precision':'fp32',
                   'head_compute_precision':config.get('precision','fp32')}
        self.width=branch.backbone.config.d_model;self.arrays={};self.filled={};self.paths={}

    def shard(self,sid):
        if sid in self.arrays:return self.arrays[sid],self.filled[sid]
        name=self.panel.stocks[sid]
        spec={**self.base,'stock':name,'data_manifest_sha256':digest(OUT/'data'/name/'manifest.json')}
        key=identity(spec);directory=OUT/'representation_cache'/key;directory.mkdir(parents=True,exist_ok=True)
        receipt=directory/'identity.json'
        if receipt.exists() and read_json(receipt)!=spec:raise ValueError('Cache key collision')
        write_json(receipt,spec)
        values=directory/'latent.npy';flags=directory/'filled.npy';n=len(self.panel.dates)
        if values.exists()!=flags.exists():raise ValueError('Incomplete cache allocation; rebuild explicitly')
        if not values.exists():
            np.lib.format.open_memmap(values,mode='w+',dtype=self.dtype,shape=(n,self.width)).flush()
            filled=np.lib.format.open_memmap(flags,mode='w+',dtype=np.bool_,shape=(n,));filled[:]=False;filled.flush()
        self.arrays[sid]=np.lib.format.open_memmap(values,mode='r+')
        self.filled[sid]=np.lib.format.open_memmap(flags,mode='r+')
        self.paths[sid]=str(values)
        return self.arrays[sid],self.filled[sid]

    @torch.no_grad()
    def get(self,batch):
        stocks=batch['stock'].detach().cpu().numpy();rows=batch['row'].detach().cpu().numpy()
        if np.any(self.panel.dates[rows]>self.allowed_end):raise ValueError('Cache request crosses its declared time boundary')
        result=np.empty((len(rows),self.width),dtype=np.float32)
        for sid in np.unique(stocks):
            select=np.flatnonzero(stocks==sid);values,filled=self.shard(int(sid))
            unknown=select[~filled[rows[select]]]
            if len(unknown):
                # Duplicate sampled rows are harmless; no labels are read by the encoder.
                latent=self.branch.latent(batch['foundation_context'][unknown]).float().cpu().numpy()
                if not np.isfinite(latent).all():raise FloatingPointError('Nonfinite foundation latent')
                values[rows[unknown]]=latent.astype(self.dtype);values.flush()
                filled[rows[unknown]]=True;filled.flush()
            result[select]=values[rows[select]]
        return torch.as_tensor(result,device=batch['local'].device)
