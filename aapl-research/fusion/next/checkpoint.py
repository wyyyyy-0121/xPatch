"""Small, hash-verified checkpoints including optimizer, sampler and every RNG stream."""
import os
from pathlib import Path
import random
import uuid

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
    generations=path.parent/('.'+path.stem+'.checkpoints');generations.mkdir(exist_ok=True)
    generation=generations/(uuid.uuid4().hex+'.pt')
    with generation.open('wb') as stream:
        torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
    directory_fd=os.open(generations,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    receipt_path=path.with_suffix(path.suffix+'.json')
    previous=read_json(receipt_path).get('generation') if receipt_path.exists() else None
    # This single atomic receipt commits the new generation. The convenient .pt
    # alias is never the authority, so a crash between receipt and alias is safe.
    write_json(receipt_path,{'sha256':digest(generation),'step':int(step),'recipe':recipe,
               'generation':str(generation.relative_to(path.parent)),
               'created_at_utc':utc_now(),'bytes':generation.stat().st_size})
    alias=path.parent/('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
    alias.symlink_to(generation.relative_to(path.parent));os.replace(alias,path)
    directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    keep={generation.resolve()}
    if previous:keep.add((path.parent/previous).resolve())
    # Two complete generations suffice for recovery; avoid accumulating every
    # 100-update snapshot and exhausting the user's 50 GiB training disk.
    pruned_path=path.with_suffix('.pruning.json')
    pruning=read_json(pruned_path) if pruned_path.exists() else {'files':{}}
    for relative,event in pruning['files'].items():
        if event['status']=='planned' and not (path.parent/relative).exists():
            event.update(status='deleted_after_interruption',verified_at_utc=utc_now());write_json(pruned_path,pruning)
    for old in generations.glob('*.pt'):
        if old.resolve() in keep:continue
        relative=str(old.relative_to(path.parent));event={'sha256':digest(old),'bytes':old.stat().st_size,
            'status':'planned','at_utc':utc_now(),'reason':'superseded checkpoint generation; current and previous committed generations retained'}
        pruning['files'][relative]=event;write_json(pruned_path,pruning)
        try:old.unlink();event['status']='deleted'
        except OSError as exc:event.update(status='failed',error=str(exc))
        write_json(pruned_path,pruning)


def resolve_checkpoint(path):
    """Resolve the last committed generation, including portable exported copies."""
    path=Path(path);receipt=read_json(path.with_suffix(path.suffix+'.json'))
    if path.exists() and digest(path)==receipt['sha256']:return path,receipt
    if receipt.get('generation'):
        generation=(path.parent/receipt['generation']).resolve()
        if not generation.is_relative_to(path.parent.resolve()):raise ValueError('Checkpoint generation leaves its model directory')
        if generation.is_file() and digest(generation)==receipt['sha256']:return generation,receipt
    raise ValueError('Checkpoint identity changed or committed generation is missing')


def load(path,model,optimizer,samplers,recipe):
    path,receipt=resolve_checkpoint(path)
    if receipt['recipe']!=recipe:
        raise ValueError('Checkpoint identity or immutable training recipe changed')
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state['recipe']!=recipe or state['step']!=receipt['step']:raise ValueError('Checkpoint recipe or step differs from receipt')
    expected=set(changing_state(model))
    if set(state['model'])!=expected:raise ValueError('Checkpoint does not cover all changing parameters and buffers')
    result=model.load_state_dict(state['model'],strict=False)
    if result.unexpected_keys:raise ValueError('Unexpected checkpoint parameters')
    optimizer.load_state_dict(state['optimizer'])
    if set(samplers)!=set(state['samplers']):raise ValueError('Checkpoint sampler set differs')
    for name,sampler in samplers.items():sampler.load_state_dict(state['samplers'][name])
    restore_rng(state['rng'])
    return state['step'],state['extra']
