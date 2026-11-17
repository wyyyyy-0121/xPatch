"""Exact file admission for the single cap intervention; source review required.

Python audit hook only, not an OS sandbox or ZIP-member monitor. Runtime modules
are imported before installation; no market/tensor inputs are read by imports.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parents[1]
NUMERIC={'.npy','.npz','.pt','.pth','.safetensors','.csv','.parquet','.pkl','.pickle','.h5','.hdf5','.arrow','.feather'}


def install_guard(protocol, protocol_path):
    if os.sched_getaffinity(0)!={24,26} or os.environ.get('CUDA_VISIBLE_DEVICES')!='-1':
        raise RuntimeError('Requires CPU24+26 and hidden GPU')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='2': raise RuntimeError('Two numerical threads required')
    if protocol.get('Q3_allowed') is not False or protocol.get('network_allowed') is not False:
        raise ValueError('Forbidden Q3 or network admission')
    output=Path(protocol['output_directory']).resolve()
    if output==BASE or not output.is_relative_to(BASE): raise ValueError('Wrong output')
    allowed={Path(protocol_path).resolve()}
    for mapping in ('code_sha256','metadata_sha256','input_sha256'):
        allowed.update(Path(p).resolve() for p in protocol.get(mapping,{}))
    runtime={Path(sys.prefix).resolve()/'lib',Path(sys.base_prefix).resolve()/'lib'}
    runtime.update(Path(p).resolve() for p in sys.path if p and Path(p).name=='site-packages')
    runtime.update(Path(p) for p in ('/usr/lib','/usr/local/lib','/lib','/usr/share/zoneinfo'))
    state={'guard_installed':True,'CPU':[24,26],'GPU_hidden':True,'source_open_counts':{},'rejected':[],
           'scope':'Python audit hook plus inspected source, not OS sandbox or ZIP-member monitor'}
    def reject(event,path):
        state['rejected'].append({'event':event,'path':str(path)})
        raise PermissionError('Cap diagnostic rejected '+str(path))
    def hook(event,args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):reject(event,'network/subprocess')
        if event!='open' or isinstance(args[0],int):return
        path=Path(os.fsdecode(args[0])).resolve()
        if any('sealed' in p.lower() or p.lower() in ('q3','2026q3') for p in path.parts):reject(event,path)
        mode=args[1];flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if path.is_relative_to(output):
            if path.suffix.lower() not in ('','.json','.pt','.npz'):reject(event,path)
            return
        if writing:reject(event,path)
        if path in allowed:
            key=str(path);state['source_open_counts'][key]=state['source_open_counts'].get(key,0)+1;return
        if path.suffix.lower() in NUMERIC:reject(event,path)
        if any(path.is_relative_to(p) for p in runtime):return
        reject(event,path)
    sys.addaudithook(hook)
    return state
