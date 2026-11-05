"""Small, hash-verified checkpoints including optimizer, sampler and every RNG stream."""
import os
from pathlib import Path
import random

import numpy as np
import torch

from .common import digest, read_json, utc_now, write_json


def capture_rng():
    name,values,pos,gaussian,cached=np.random.get_state()
    return {'python':random.getstate(),'numpy':(name,values.tolist(),pos,gaussian,cached),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    name,values,pos,gaussian,cached=state['numpy']
    np.random.set_state((name,np.asarray(values,dtype=np.uint32),pos,gaussian,cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def changing_state(model):
    # Keep training BatchNorm buffers; avoid duplicating a frozen 120M parameter backbone.
    names={name for name,p in model.named_parameters() if p.requires_grad}
    names.update(name for name,_ in model.named_buffers())
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items() if name in names}


def save(path,model,optimizer,samplers,step,recipe,extra=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema':1,'recipe':recipe,'step':int(step),'model':changing_state(model),
             'optimizer':optimizer.state_dict(),'samplers':{k:v.state_dict() for k,v in samplers.items()},
             'rng':capture_rng(),'extra':extra or {}}
    temporary=path.with_suffix(path.suffix+'.tmp');torch.save(payload,temporary);os.replace(temporary,path)
    write_json(path.with_suffix(path.suffix+'.json'),{'sha256':digest(path),'step':int(step),
               'recipe':recipe,'created_at_utc':utc_now(),'bytes':path.stat().st_size})


def load(path,model,optimizer,samplers,recipe):
    path=Path(path);receipt=read_json(path.with_suffix(path.suffix+'.json'))
    if digest(path)!=receipt['sha256'] or receipt['recipe']!=recipe:
        raise ValueError('Checkpoint identity or immutable training recipe changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['recipe']!=recipe:raise ValueError('Checkpoint recipe differs from receipt')
    expected=set(changing_state(model))
    if set(state['model'])!=expected:raise ValueError('Checkpoint does not cover all changing parameters and buffers')
    result=model.load_state_dict(state['model'],strict=False)
    if result.unexpected_keys:raise ValueError('Unexpected checkpoint parameters')
    optimizer.load_state_dict(state['optimizer'])
    if set(samplers)!=set(state['samplers']):raise ValueError('Checkpoint sampler set differs')
    for name,sampler in samplers.items():sampler.load_state_dict(state['samplers'][name])
    restore_rng(state['rng'])
    return state['step'],state['extra']
