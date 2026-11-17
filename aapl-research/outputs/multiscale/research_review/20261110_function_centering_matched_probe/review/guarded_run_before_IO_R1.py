"""Hash-pinned Python audit boundary. Not an OS sandbox or malicious-code proof."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--protocol',required=True,type=Path); parser.add_argument('--mode',required=True,choices=['qualify','cost','trajectory','market','compare']); parser.add_argument('--output',required=True,type=Path); parser.add_argument('--resume',action='store_true'); parser.add_argument('--pause-kind',choices=['signal','deadline']); args=parser.parse_args()
    protocol_path=args.protocol.resolve(); protocol=json.loads(protocol_path.read_text())
    stage=protocol_path.parent.resolve()
    repository=next(p for p in stage.parents if (p/'MULTISCALE_CURRENT_PLAN_ZH.md').is_file())
    assert protocol['phase'] in ('synthetic','market')
    assert protocol['rules_sha256']==digest(protocol['rules_path'])
    rules=json.loads(Path(protocol['rules_path']).read_text())
    assert time.time()<protocol['deadline_epoch']
    if protocol['phase']=='synthetic':
        assert protocol['deadline_epoch']==rules['deadline_epoch']
        assert not protocol.get('market_admission') and not protocol.get('market_files')
        assert args.mode!='market'
    else:
        assert protocol.get('market_admission') is True and args.mode=='market'
        assert protocol['maximum_seconds']<=7200
        assert protocol['deadline_epoch']==protocol['started_epoch']+protocol['maximum_seconds']
        assert protocol.get('preparation_acceptance_path') and protocol.get('preparation_acceptance_sha256')
        assert digest(protocol['preparation_acceptance_path'])==protocol['preparation_acceptance_sha256']
    for group in ['code_files','source_files']:
        for path,expected in protocol[group].items(): assert digest(path)==expected,(group,path)
    assert str(Path(__file__).resolve()) in protocol['code_files']
    worker=Path(__file__).parent/'matched_worker.py'
    assert str(worker.resolve()) in protocol['code_files']
    assert str(Path(protocol.get('design_path',rules['design']['path'])).resolve()) == str(Path(rules['design']['path']).resolve())
    if protocol.get('input_metadata_path'):
        assert protocol.get('input_metadata_sha256') and digest(protocol['input_metadata_path']) == protocol['input_metadata_sha256']
    for role in ['design','binding_clarifications','independent_design_review','synthetic_acceptance']:
        assert digest(rules[role]['path'])==rules[role]['sha256']
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    assert os.environ.get('OMP_NUM_THREADS')=='1' and os.environ.get('OPENBLAS_NUM_THREADS')=='1'
    assert set(os.sched_getaffinity(0)) <= {24}
    output_root=Path(protocol['output_directory']).resolve()
    args.output.resolve().relative_to(output_root)
    assert output_root.is_relative_to(stage)
    allowed={protocol_path,Path(protocol['rules_path']).resolve()}
    for group in ['code_files','source_files']:
        allowed.update(Path(p).resolve() for p in protocol[group])
    allowed.update(Path(rules[k]['path']).resolve() for k in ['design','binding_clarifications','independent_design_review','synthetic_acceptance'])
    for key in ['input_metadata_path','design_path','preparation_acceptance_path']:
        if protocol.get(key): allowed.add(Path(protocol[key]).resolve())
    market_paths=set()
    if protocol['phase']=='market':
        metadata=json.loads(Path(protocol['input_metadata_path']).read_text())
        expected={v['path']:v['sha256'] for v in metadata['bars'].values()}
        expected.update({v['path']:v['file_sha256'] for profile in metadata['fits'].values() for v in profile.values()})
        assert len(expected)==6 and protocol['market_files']==expected
        market_paths={Path(p).resolve() for p in expected}
        assert all('2026Q3' not in str(p) and 'holdout_sealed' not in str(p) for p in market_paths)
    runtime_roots={(repository/'.venv').resolve(),Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()}
    sys.dont_write_bytecode=True
    OS_READ_ROOTS=[Path(p) for p in ['/usr','/lib','/lib64','/opt','/proc','/sys','/dev','/etc','/tmp']]
    def audit(event,values):
        if event in ('socket.connect','socket.bind','socket.getaddrinfo'):
            raise PermissionError('No network admitted')
        if event!='open' or not values or isinstance(values[0],int): return
        raw=values[0]
        if not isinstance(raw,(str,bytes,os.PathLike)): return
        path=Path(os.fsdecode(raw)).resolve()
        mode=values[1] if len(values)>1 else 'r'
        flags=values[2] if len(values)>2 and isinstance(values[2],int) else 0
        write=(isinstance(mode,str) and any(v in mode for v in ['w','a','+','x'])) or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
        if path.is_relative_to(output_root): return
        if write:
            # Runtime temporary caches can be generated; no repository input can be changed.
            if path.is_relative_to(Path('/tmp')): return
            if str(path)=='/dev/null': return
            raise PermissionError('Write outside admitted new output: '+str(path))
        if path in allowed or path in market_paths or any(path.is_relative_to(root) for root in runtime_roots): return
        if any(path.is_relative_to(root) for root in OS_READ_ROOTS): return
        raise PermissionError('Read outside explicit source/runtime/input admission: '+str(path))
    sys.addaudithook(audit)
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(worker.parent.resolve()))
    os.environ['AAPL_MATCHED_GUARDED']='1'
    sys.argv=[str(worker),'--protocol',str(protocol_path),'--mode',args.mode,'--output',str(args.output.resolve())]
    if args.resume: sys.argv.append('--resume')
    if args.pause_kind: sys.argv.extend(['--pause-kind',args.pause_kind])
    runpy.run_path(str(worker),run_name='__main__')

if __name__=='__main__': main()
