"""Two analysis CPUs, immutable absolute deadline and atomic local records."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

BASE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(fd)
        finally: os.close(fd)
    finally:
        if os.path.exists(temp): os.unlink(temp)

def setup(cpu):
    require(cpu in (24,26), 'Only allocated analysis CPUs')
    os.sched_setaffinity(0, {cpu})
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
        os.environ[k] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cap = 768*1024*1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(cap,hard) if hard != resource.RLIM_INFINITY else cap
    resource.setrlimit(resource.RLIMIT_AS,(cap,cap))

def guard(protocol, reserve=0):
    require(time.time() < protocol['deadline_epoch']-reserve, 'Shared absolute deadline expired')

def qualify_dependencies(spec):
    require(sys.version == spec['python']['version'], 'Python version drift')
    require(str(Path(sys.executable).resolve()) == spec['python']['executable'], 'Python executable drift')
    for p,h in spec['calendar_source_files'].items(): require(sha(p)==h,'Calendar source drift')
    result={}
    for name,entry in spec['modules'].items():
        mod=importlib.import_module(entry['module'])
        require(importlib.metadata.version(name)==mod.__version__==spec['package_versions'][name], 'Actual dependency version drift')
        require(str(Path(mod.__file__).resolve())==entry['file'], 'Actual module path drift')
        result[name]={'version':mod.__version__,'file':str(Path(mod.__file__).resolve())}
    return result

def protocol():
    p=read(BASE/'EXECUTION_PROTOCOL.json')
    require(p['schema']=='earnings_event_support_execution_v1' and p['execution_admitted'] is True,'Execution not admitted')
    require(p['deadline_epoch']-p['started_epoch']==180,'Absolute window changed')
    for path,h in p['locked_files'].items(): require(sha(path)==h,'Frozen file drift '+path)
    guard(p)
    return p
