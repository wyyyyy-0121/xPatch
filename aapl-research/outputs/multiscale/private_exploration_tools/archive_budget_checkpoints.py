"""Copy original periodic checkpoints after the approved budget watchdog freezes workers."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path('/home/ayaka/wslcode/aapl-research')
OUT = ROOT / 'outputs/multiscale'
NIGHT = OUT / 'private_low_load/20261107T153438623158Z'
MATRIX = OUT / 'private_minute_matrix_development/20261107T151850900588Z'


def digest(raw): return hashlib.sha256(raw).hexdigest()
def load(p): return json.loads(p.read_text(encoding='utf8'))
def require(ok, message):
    if not ok: raise ValueError(message)
def process(pid, start):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();parts=raw[raw.rfind(')')+2:].split()
        return parts[0] if int(parts[19])==start else None
    except FileNotFoundError: return None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    output=args.output.resolve();require(output.is_relative_to(OUT/'budget_preserved_checkpoints'), 'Invalid archive path')
    require(not output.exists(), 'Archive destination exists')
    plan_path=MATRIX/'PRIVATE_PROTOCOL.json';plan=load(plan_path)
    require(plan['Q3_values_read'] is False and plan['independent_test'] is False, 'Original task scope changed')
    if args.check:
        print(json.dumps({'status':'checkpoint_archive_preflight_passed','original_protocol_sha256':digest(plan_path.read_bytes()),
                          'output':str(output),'training_started':False,'processes_changed':False}));return
    marker=NIGHT/'BUDGET_STOP_797109.json'
    deadline=time.monotonic()+120
    while not marker.exists():
        state=load(NIGHT/'NIGHT_RESUME_STATE.json')
        if state['groups']['797109']['status']=='completed':
            marker=None;break
        require(time.monotonic()<deadline, 'No original budget-stop evidence; no running files will be archived')
        time.sleep(3)
    if marker is not None:
        stop=load(marker)
        require(stop['reason']=='original_compute_budget_exhausted' and
                Path(stop['queue_plan']).resolve()==plan_path, 'Unexpected stop marker')
        for row in stop['stopped']:
            status=process(row['pid'],row['start_ticks'])
            require(status in ('T','t','Z','X','x',None), 'A matched worker is still running; refuse inconsistent archive')
    output.mkdir(parents=True,exist_ok=False)
    state=load(MATRIX/'PRIVATE_QUEUE_STATE.json');records=[];folds=[]
    for job in plan['jobs']:
        entry=state['jobs'][job['key']]
        if entry['status']=='completed':continue
        folder=MATRIX/'jobs'/job['key']
        if not folder.exists():continue
        paths=[folder/'fold_state.json']+sorted(folder.glob('*/latest.pt'))+sorted(folder.glob('*/best.pt'))+sorted(folder.glob('*/training_summary.json'))
        paths=[p for p in paths if p.is_file()]
        for path in paths:
            require(path.resolve().is_relative_to(folder.resolve()),'Checkpoint path escapes original fold')
            raw=path.read_bytes();expected=digest(raw);dest=output/job['key']/path.relative_to(folder)
            dest.parent.mkdir(parents=True,exist_ok=True)
            with dest.open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            require(digest(dest.read_bytes())==expected and digest(path.read_bytes())==expected,
                    'Checkpoint changed while creating archive')
            records.append({'source':str(path),'copy':str(dest),'bytes':len(raw),'sha256':expected})
        folds.append({'job':job['key'],'status_at_archive':entry['status'],'files':len(paths)})
    receipt={'status':'original_budget_checkpoint_archive_verified','created_at_utc':datetime.now(timezone.utc).isoformat(),
             'original_protocol_sha256':digest(plan_path.read_bytes()),'stop_marker':str(marker) if marker else None,
             'stop_marker_sha256':digest(marker.read_bytes()) if marker else None,
             'source_queue_state_sha256':digest((MATRIX/'PRIVATE_QUEUE_STATE.json').read_bytes()),
             'folds':folds,'files':records,'total_bytes':sum(r['bytes'] for r in records),
             'training_started':False,'processes_resumed':False,'budgets_modified':False,'Q3_values_read':False,
             'limitations':['Copies preserve latest saved periodic checkpoints; the original stopped workers retain any newer in-memory updates.']}
    with (output/'PRESERVATION_RECEIPT.json').open('x',encoding='utf8') as stream:
        json.dump(receipt,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'status':receipt['status'],'folds':len(folds),'files':len(records),
                      'bytes':receipt['total_bytes'],'output':str(output)}))


if __name__=='__main__':main()
