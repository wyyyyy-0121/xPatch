"""Portable, hash-verified local/offline loading of the already selected Chronos-2."""
import os
from pathlib import Path
import torch

from .common import ROOT,OUT,digest,read_json,write_json


def manifest():
    path=OUT/'foundation_weights.json'
    if path.exists():return read_json(path)
    old=Path('/home/ayaka/xPatch')
    row=read_json(old/'configs/aapl_v2_model_lock.json')['models']['chronos2']
    result={k:row[k] for k in ('repository','revision','license','files')}
    write_json(path,result)
    return result


def weight_directory():
    override=os.environ.get('FUSION_CHRONOS_WEIGHTS')
    if override:directory=Path(override).expanduser().resolve()
    elif (ROOT/'weights/chronos2').exists():directory=ROOT/'weights/chronos2'
    else:
        old=Path('/home/ayaka/xPatch')
        row=read_json(old/'configs/aapl_v2_model_lock.json')['models']['chronos2']
        directory=old/row['directory']
    return directory


def load_foundation(device='cuda'):
    from chronos import Chronos2Pipeline
    receipt=manifest();directory=weight_directory()
    for name,row in receipt['files'].items():
        if digest(directory/name)!=row['sha256']:raise ValueError('Chronos-2 weight identity differs from pinned manifest')
    pipe=Chronos2Pipeline.from_pretrained(str(directory),device_map=device,local_files_only=True,torch_dtype=torch.float32)
    return pipe,receipt
