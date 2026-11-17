"""Snapshot paused AAPL checkpoints and conservatively preserve queue budgets."""
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil
import sys
from zipfile import ZipFile,is_zipfile


ROOT=Path('/home/ayaka/wslcode/aapl-research/outputs/multiscale')
PARENTS={
 'minute_parallel_folds_45.py':ROOT/'private_minute_parallel_folds/20261107T143936100555Z',
 'minute_breadth_h30_queue.py':ROOT/'private_minute_breadth_development/20261107T150838302360Z',
 'minute_mixed_queue.py':ROOT/'private_minute_mixed_development/20261107T151427101607Z',
 'minute_matrix_queue.py':ROOT/'private_minute_matrix_development/20261107T151850900588Z',
}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    manifest_path=Path(sys.argv[1]).resolve();manifest=json.loads(manifest_path.read_text())
    dest=manifest_path.parent/'checkpoint_snapshot';dest.mkdir(exist_ok=False)
    pause_time=datetime.fromisoformat(manifest['paused_at_utc']);queues={};workers=[];errors=[]
    for process in manifest['processes']:
        cmd=process['cmdline'];script=next((s for s in PARENTS if s in cmd),None)
        if not script:errors.append(f"Unknown process {process['pid']}");continue
        if '--worker' not in cmd:
            root=PARENTS[script];state_path=root/'PRIVATE_QUEUE_STATE.json';plan_path=root/'PRIVATE_PROTOCOL.json'
            state=json.loads(state_path.read_text());plan=json.loads(plan_path.read_text())
            start=datetime.fromisoformat(state['started_at_utc'])
            active=max(0,math.ceil((pause_time-start).total_seconds())+5)
            maximum=int(plan['maximum_wall_seconds'])
            remaining=max(0,maximum-active)
            out=dest/f"queue_{process['pid']}";out.mkdir()
            for src in (state_path,plan_path):
                copied=out/src.name;shutil.copy2(src,copied)
                if digest(src)!=digest(copied):raise ValueError('Queue snapshot hash mismatch')
            queues[str(process['pid'])]={'script':script,'root':str(root),'state':str(state_path),'plan':str(plan_path),
                                         'plan_sha256':digest(plan_path),'status_at_pause':state['status'],
                                         'maximum_wall_seconds':maximum,'active_seconds_before_pause':active,
                                         'remaining_active_seconds':remaining,'snapshot_dir':str(out)}
    for process in manifest['processes']:
        cmd=process['cmdline']
        if '--worker' not in cmd:continue
        script=next((s for s in PARENTS if s in cmd),None)
        if not script:continue
        args=shlex.split(cmd)
        try:
            plan_path=Path(args[args.index('--plan')+1]).resolve();plan=json.loads(plan_path.read_text())
            if '--job' in args:
                key=args[args.index('--job')+1]
                folder=plan_path.parent/('jobs' if script=='minute_matrix_queue.py' else '')/key
            else:
                key=args[args.index('--fold')+1];folder=plan_path.parent/f'fold_{int(key):02d}'
            state_path=folder/'fold_state.json'
            state=json.loads(state_path.read_text())
            if state['config']['protocol_identity']!=digest(plan_path):raise ValueError('Fold protocol identity changed')
            out=dest/f"worker_{process['pid']}";out.mkdir()
            copies=[]
            for src in [state_path]+sorted(folder.glob('*/latest.pt'))+sorted(folder.glob('*/best.pt'))+sorted(folder.glob('*/training_summary.json')):
                rel=src.relative_to(folder);target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(src,target);source_hash=digest(src)
                if source_hash!=digest(target):raise ValueError('Checkpoint copy hash mismatch')
                if src.suffix=='.pt':
                    if not is_zipfile(target):raise ValueError('Checkpoint is not a valid PyTorch archive')
                    with ZipFile(target) as z:
                        if z.testzip() is not None:raise ValueError('Checkpoint archive failed integrity check')
                copies.append({'source':str(src),'copy':str(target),'sha256':source_hash,'bytes':src.stat().st_size})
            for name,details in state.get('models',{}).items():
                pred=folder/f'prediction_{name}.npz'
                if not pred.is_file() or digest(pred)!=details['prediction_sha256']:
                    raise ValueError(f'Completed prediction changed: {name}')
            latest=[x for x in copies if x['source'].endswith('/latest.pt')]
            if not latest:raise ValueError('No persisted training checkpoint found')
            workers.append({'pid':process['pid'],'parent_pid':process['ppid'],'script':script,'job':key,
                            'plan':str(plan_path),'plan_sha256':digest(plan_path),'fold_dir':str(folder),
                            'fold_identity':state['fold_identity'],'device':state['identity_fields']['runtime']['device'],
                            'completed_models':list(state.get('models',{})),'checkpoint_files':copies,
                            'process_start_ticks':process['start_ticks'],'original_affinity':process['affinity']})
        except Exception as exc:errors.append(f"worker {process['pid']}: {type(exc).__name__}: {exc}")
    result={'status':'snapshot_complete' if not errors else 'snapshot_has_errors',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'pause_manifest':str(manifest_path),
            'pause_manifest_sha256':digest(manifest_path),'boot_id':manifest['boot_id'],
            'resume_at_local':manifest['resume_at_local'],'queues':queues,'workers':workers,'errors':errors,
            'notes':['Frozen processes preserve exact in-memory state while the OS session survives.',
                     'On-disk checkpoints are periodic; an OS or WSL shutdown can lose batches since the latest checkpoint.',
                     'The five-second deduction is conservative and does not reset any original wall budget.']}
    path=manifest_path.parent/'CHECKPOINT_AND_BUDGET_SNAPSHOT.json'
    with path.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':result['status'],'path':str(path),'queues':len(queues),'workers':len(workers),
                      'checkpoint_files':sum(len(w['checkpoint_files']) for w in workers),'errors':errors,
                      'remaining_seconds':{q['script']:q['remaining_active_seconds'] for q in queues.values()}},ensure_ascii=False))


if __name__=='__main__':main()
