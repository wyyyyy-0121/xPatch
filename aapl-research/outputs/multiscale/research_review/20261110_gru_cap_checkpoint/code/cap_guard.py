"""Python file/network guard for the no-update checkpoint verification.

This supplements reviewed source. It is not an OS sandbox or a tensor archive
member monitor. Only the two registered original checkpoints may be read.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parents[1]
ROOT = Path('/home/ayaka/wslcode/aapl-research').resolve()
NUMERIC = {'.npy', '.npz', '.pt', '.pth', '.safetensors', '.csv', '.parquet',
           '.pkl', '.pickle', '.h5', '.hdf5', '.feather', '.arrow'}


def install_guard(protocol, protocol_path):
    output = Path(protocol['output_directory']).resolve()
    if output == BASE or not output.is_relative_to(BASE):
        raise ValueError('Output must be a specific child of this stage')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or os.sched_getaffinity(0) != {26}:
        raise RuntimeError('CPU26 and hidden GPU are required')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if os.environ.get(key) != '1':
            raise RuntimeError('Numerical thread count must be1: '+key)
    if protocol.get('market_arrays_admitted') is not False or protocol.get('training_updates_admitted') is not False:
        raise ValueError('No-update scope changed')
    allowed = {Path(protocol_path).resolve()}
    allowed.update(Path(p).resolve() for p in protocol.get('code_sha256', {}))
    allowed.update(Path(p).resolve() for p in protocol.get('metadata_sha256', {}))
    for record in protocol.get('checkpoints', {}).values():
        allowed.add(Path(record['path']).resolve())
    runtime = {Path(sys.base_prefix).resolve()/'lib', Path(sys.prefix).resolve()/'lib'}
    runtime.update(Path(p).resolve() for p in sys.path if p and Path(p).name == 'site-packages')
    runtime.update(Path(p) for p in ('/usr/lib', '/usr/local/lib', '/lib', '/usr/share/zoneinfo'))
    state = {'guard_installed': True, 'CPU_affinity':[26], 'GPU_hidden':True,
             'source_open_counts':{}, 'rejected_attempts':[],
             'scope':'Python audit hook plus reviewed source, not OS sandbox or tensor-member monitor',
             'runtime_roots':sorted(map(str,runtime))}

    def reject(event, path):
        state['rejected_attempts'].append({'event':event,'path':str(path)})
        raise PermissionError('Checkpoint-only scope rejected: '+str(path))

    def hook(event, args):
        if event in ('socket.connect','socket.getaddrinfo','subprocess.Popen','os.system'):
            reject(event,'network/subprocess')
        if event != 'open' or isinstance(args[0],int):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if any('sealed' in p.lower() or p.lower() in ('q3','2026q3') for p in path.parts):
            reject(event,path)
        mode=args[1]
        flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
        writing=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
        if path.is_relative_to(output):
            if path.suffix.lower() not in ('','.json','.pt'):
                reject(event,path)
            if path.name.lower().startswith(('bars','prediction','target','close')):
                reject(event,path)
            return
        if writing:
            reject(event,path)
        if path in allowed:
            key=str(path)
            state['source_open_counts'][key]=state['source_open_counts'].get(key,0)+1
            return
        if path.suffix.lower() in NUMERIC:
            reject(event,path)
        if any(path.is_relative_to(p) for p in runtime):
            return
        reject(event,path)

    sys.addaudithook(hook)
    return state


def synthetic_guard_acceptance():
    import socket
    # Match the admitted executable: numerical runtimes import before the hook,
    # but no input checkpoint or market file is opened during those imports.
    import numpy as np
    import torch
    import torch._dynamo  # Resolve Adam's lazy runtime cache before the hook.
    output=BASE/'guard_acceptance'
    output.mkdir(parents=True,exist_ok=True)
    protocol={'output_directory':str(output),'code_sha256':{str(Path(__file__).resolve()):'synthetic'},
              'metadata_sha256':{},'checkpoints':{},'market_arrays_admitted':False,'training_updates_admitted':False}
    state=install_guard(protocol,Path(__file__).resolve())
    import encodings.cp437
    assert Path(encodings.cp437.__file__).read_bytes()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert np.unique(np.array(['a','a','b'])).tolist()==['a','b']
    model=torch.nn.GRU(1,32,batch_first=True)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    with torch.inference_mode():
        result=model(torch.zeros(128,96,1))[0]
    assert tuple(result.shape)==(128,96,32)
    cases=[ROOT/'data/2026Q3/anything.json',ROOT/'data/disguised_market.json',
           ROOT/'outputs/multiscale/admission/alpaca_20261107T052228573242Z/bars_hour.npz',
           ROOT/'outputs/fusion_next/data/close.npy',ROOT/'outputs/multiscale/unregistered.pt',
           output/'anything.npy',output/'anything.npz',output/'bars.json',
           Path(np.__file__).parent/'market_disguised.npy',
           Path(torch.__file__).parent/'unregistered.pt',output/'sealed/stats.json']
    for path in cases:
        try:
            open(path,'rb')
        except PermissionError:
            pass
        else:
            raise AssertionError('Forbidden path accepted: '+str(path))
    for action in (lambda:socket.getaddrinfo('example.com',443),lambda:socket.socket().connect(('127.0.0.1',9))):
        try:
            action()
        except PermissionError:
            pass
        else:
            raise AssertionError('Network accepted')
    receipt={'status':'passed','forbidden_file_cases':len(cases),'network_cases':2,
             'synthetic_GRU_Adam_import_forward':True,'actual_checkpoint_or_market_access':False,'guard':state}
    (output/'GUARD_ACCEPTANCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='guard'}))


if __name__=='__main__':
    synthetic_guard_acceptance()
