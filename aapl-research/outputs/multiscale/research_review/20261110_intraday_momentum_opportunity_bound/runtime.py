"""Scope, resource and durable checkpoint helpers. No numerical imports."""
import hashlib
import json
import os
from pathlib import Path
import resource
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
    fd, tmp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def setup(cpu):
    require(cpu in (24,26), 'Only analysis CPUs24/26')
    os.sched_setaffinity(0, {cpu})
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS']:
        os.environ[key] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    cap = 768*1024*1024
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = min(cap, hard) if hard != resource.RLIM_INFINITY else cap
    resource.setrlimit(resource.RLIMIT_AS, (cap,cap))

def load_protocol():
    path = BASE/'EXECUTION_PROTOCOL.json'
    p = read(path)
    require(p['schema']=='intraday_opportunity_bound_v1' and p['execution_admitted'] is True, 'No reviewed execution')
    require(p['deadline_epoch']-p['started_epoch']==180, 'Deadline must not reset')
    require(p['GPU'] is False and p['Q3_allowed'] is False, 'Scope differs')
    for file, expected in p['locked_files'].items():
        require(sha(file)==expected, 'Frozen source/design identity changed '+file)
    guard(p)
    return p

def guard(p, reserve=0):
    require(time.time() < p['deadline_epoch']-reserve, 'Absolute shared deadline reached')

def qualify_dependencies(spec):
    import importlib
    import importlib.metadata
    import sys
    require(sys.version==spec['python']['version'] and str(Path(sys.executable).resolve())==spec['python']['executable'],'Python identity changed')
    for path,expected in spec['calendar_source_files'].items():
        require(sha(path)==expected,'Calendar source changed')
    receipt={}
    for name,entry in spec['modules'].items():
        module=importlib.import_module(entry['module'])
        require(importlib.metadata.version(name)==spec['package_versions'][name] and module.__version__==spec['package_versions'][name],'Actual module version changed')
        require(str(Path(module.__file__).resolve())==entry['file'],'Actual module path changed')
        receipt[name]={'version':module.__version__,'file':str(Path(module.__file__).resolve())}
    return receipt

