"""Independent complete typed payload comparison, no worker/native-core imports."""
import hashlib
import json
from pathlib import Path
import struct

import torch


def read_committed(directory):
    directory=Path(directory)
    manifest=json.loads((directory/'CHECKPOINT.json').read_text())
    path=directory/manifest['payload']
    if path.parent!=directory or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['sha256']:
        raise RuntimeError('Independent comparator: invalid payload commitment')
    value=torch.load(path,map_location='cpu',weights_only=True)
    if value['identity']!=manifest['identity'] or value['state']['steps']!=manifest['steps']:
        raise RuntimeError('Independent comparator: manifest semantics changed')
    return value


def tensor_bytes(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def compare_recursive(left,right,path,checks):
    if type(left) is not type(right): raise RuntimeError('Payload type differs at '+path)
    if isinstance(left,torch.Tensor):
        if left.dtype!=right.dtype or left.shape!=right.shape or left.stride()!=right.stride() or left.layout!=right.layout or tensor_bytes(left)!=tensor_bytes(right):
            raise RuntimeError('Full tensor payload differs at '+path)
        checks.append({'path':path,'kind':'tensor','dtype':str(left.dtype),'shape':list(left.shape),'bytes':len(tensor_bytes(left))})
    elif isinstance(left,dict):
        if list(left)!=list(right): raise RuntimeError('Payload keys/order differ at '+path)
        for key in left: compare_recursive(left[key],right[key],path+'/'+str(key),checks)
    elif isinstance(left,(list,tuple)):
        if len(left)!=len(right): raise RuntimeError('Payload sequence size differs at '+path)
        for index,(a,b) in enumerate(zip(left,right)): compare_recursive(a,b,path+'/'+str(index),checks)
    elif isinstance(left,float):
        if struct.pack('>d',left)!=struct.pack('>d',right): raise RuntimeError('Float payload bytes differ at '+path)
        checks.append({'path':path,'kind':'float'})
    elif left is None or isinstance(left,(str,int,bool,bytes)):
        if left!=right: raise RuntimeError('Payload scalar differs at '+path)
        checks.append({'path':path,'kind':type(left).__name__})
    else:
        raise RuntimeError('Unsupported hidden payload type at '+path+':'+str(type(left)))


def canonical(value,hasher):
    def feed(raw):
        hasher.update(struct.pack('>Q',len(raw))); hasher.update(raw)
    feed((type(value).__module__+'.'+type(value).__qualname__).encode())
    if isinstance(value,torch.Tensor):
        feed(str(value.dtype).encode()); feed(str(tuple(value.shape)).encode()); feed(str(value.stride()).encode()); feed(tensor_bytes(value))
    elif isinstance(value,dict):
        for key,item in value.items(): canonical(key,hasher); canonical(item,hasher)
    elif isinstance(value,(tuple,list)):
        feed(str(len(value)).encode())
        for item in value: canonical(item,hasher)
    elif isinstance(value,float): feed(struct.pack('>d',value))
    elif value is None: feed(b'none')
    elif isinstance(value,(str,int,bool)): feed(str(value).encode())
    elif isinstance(value,bytes): feed(value)
    else: raise RuntimeError('Unsupported canonical payload type')


def compare_all(root):
    root=Path(root)
    reference=read_committed(root/'continuous')
    rows=[]
    for name in ['signal','deadline']:
        candidate=read_committed(root/name); checks=[]
        compare_recursive(reference,candidate,'payload',checks)
        rows.append({'recovery':name,'passed':True,'all_leaf_checks':len(checks),'checks':checks})
    h=hashlib.sha256(); canonical(reference,h)
    return {'status':'passed_independent_full_typed_payload_equality','canonical_payload_sha256':h.hexdigest(),
            'included_top_level_keys':list(reference),'recoveries':rows,
            'exclusions':'Only data outside payload: pids, wallclock, local injected deadline, output paths and pause kind. No payload member is excluded.',
            'comparison_algorithm_imported_worker':False,'market_read':False,'Q3_read':False,'forecast_success':False}
