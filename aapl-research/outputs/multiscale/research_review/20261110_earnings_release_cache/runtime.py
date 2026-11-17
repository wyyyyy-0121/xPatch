"""Durable source-cache helpers; no numerical/market imports."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

BASE=Path(__file__).resolve().parent
def require(ok,message):
    if not ok:raise RuntimeError(message)
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,dir=path.parent)
    try:
        with os.fdopen(fd,'w') as f:
            json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
        d=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(d)
        finally:os.close(d)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
def setup():
    os.sched_setaffinity(0,{24});os.environ['CUDA_VISIBLE_DEVICES']=''
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
def protocol():
    p=read(BASE/'COLLECTION_PROTOCOL.json')
    require(p['source_collection_only'] and not p['market_execution_admitted'],'Wrong scope')
    require(p['deadline_epoch']-p['started_epoch']==180,'Do not reset source deadline')
    require(time.time()<p['deadline_epoch'],'Source window closed')
    for file,expected in p['locked_files'].items():require(sha(file)==expected,'Frozen identity changed')
    return p
