"""Atomic committed-batch checkpoints and an immutable absolute execution guard."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import tempfile
import time

class PreserveStop(Exception):pass

def digest(body):return hashlib.sha256(body).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(65536),b''):h.update(block)
    return h.hexdigest()

def _sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)

def atomic_bytes(path,body):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);name=None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as stream:
            name=stream.name;stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path);name=None;_sync_directory(path.parent)
    finally:
        if name is not None:os.unlink(name)

def atomic_json(path,value):
    atomic_bytes(path,(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())

def atomic_npz(path,arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);name=None
    try:
        with tempfile.NamedTemporaryFile('wb',dir=path.parent,delete=False) as stream:
            name=stream.name;np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(name,path);name=None;_sync_directory(path.parent)
    finally:
        if name is not None:os.unlink(name)

def save_checkpoint(path,meta,stats,models,extras):
    import numpy as np
    arrays={'metadata':np.asarray(json.dumps(meta,sort_keys=True,allow_nan=False))}
    if stats is not None:
        for key,value in stats.items():arrays['stats__'+key]=np.asarray(value)
    for index,model in models.items():
        for key,value in model.items():arrays['model__'+str(index)+'__'+key]=np.asarray(value)
    for key,value in extras.items():arrays['extra__'+key]=np.asarray(value)
    atomic_npz(path,arrays)

def load_checkpoint(path):
    import numpy as np
    with np.load(path,allow_pickle=False) as saved:
        meta=json.loads(str(saved['metadata']));stats={};models={};extras={}
        for key in saved.files:
            if key.startswith('stats__'):stats[key[7:]]=saved[key].copy()
            elif key.startswith('model__'):
                _,index,name=key.split('__',2);models.setdefault(int(index),{})[name]=saved[key].copy()
            elif key.startswith('extra__'):extras[key[7:]]=saved[key].copy()
    if stats:stats['n']=int(stats['n'])
    return meta,stats or None,models,extras

class Guard:
    def __init__(self,protocol):
        self.deadline=protocol['deadline_epoch'];self.maximum=3221225472;self.stop=None
        signal.signal(signal.SIGUSR1,self._signal);signal.signal(signal.SIGTERM,self._signal)
        signal.signal(signal.SIGINT,self._signal)
    def _signal(self,signum,_frame):self.stop=signal.Signals(signum).name
    def __call__(self):
        if self.stop is not None:raise PreserveStop('external_'+self.stop)
        if time.time()>=self.deadline-1:raise PreserveStop('original_deadline_reached')
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>self.maximum:
            raise PreserveStop('maximum_RSS_reached')

def validate_external_pause(run,protocol,identity,checkpoint,meta):
    run=Path(run)
    marker=json.loads((run/'READY_FOR_EXTERNAL_PAUSE.json').read_bytes())
    receipt=json.loads((run/'EXTERNAL_PAUSE_RECEIPT.json').read_bytes())
    timing_body=(run/'FIRST_BATCH_TIMING.json').read_bytes();timing=json.loads(timing_body)
    expected=marker['checkpoint_sha256']
    if not (marker['protocol_sha256']==identity and marker['cursor']==2048
            and Path(marker['checkpoint_path']).resolve()==Path(checkpoint).resolve()
            and receipt['kind']=='screen_external_pause_v1' and receipt['protocol_sha256']==identity
            and receipt['pid']==marker['pid'] and receipt['pid']!=os.getpid()
            and receipt['signal']=='SIGUSR1' and receipt['exit_code']==75 and receipt['cursor']==2048
            and Path(receipt['checkpoint_path']).resolve()==Path(checkpoint).resolve()
            and receipt['checkpoint_sha256_before']==expected==receipt['checkpoint_sha256_after']
            and receipt['supervisor_source_sha256']==protocol['supervisor_sha256']
            and timing['protocol_sha256']==identity and timing['pid']==marker['pid']
            and digest(timing_body)==marker['timing_sha256']
            and timing['actual_first_A_update_seconds']>0):
        raise ValueError('external_pause_receipt_invalid')
    if not meta.get('external_pause_accepted') and file_sha(checkpoint)!=expected:
        raise ValueError('first_checkpoint_bytes_not_preserved')
    if meta.get('external_pause_accepted') and meta['external_pause_receipt_sha256']!=file_sha(run/'EXTERNAL_PAUSE_RECEIPT.json'):
        raise ValueError('accepted_pause_receipt_changed')
    return marker,receipt,timing
