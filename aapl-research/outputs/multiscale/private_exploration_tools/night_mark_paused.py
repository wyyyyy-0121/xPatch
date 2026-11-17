"""Expose the frozen state in queue status files without touching checkpoints."""
from datetime import datetime,timezone
import hashlib,json,os,tempfile
from pathlib import Path
import sys


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path,value):
    with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno());tmp=f.name
    os.replace(tmp,path)


def main():
    snapshot_path=Path(sys.argv[1]).resolve();snapshot=json.loads(snapshot_path.read_text())
    manifest=json.loads(Path(snapshot['pause_manifest']).read_text());changes=[];prepared=[]
    for process in manifest['processes']:
        pid=process['pid'];raw=Path(f'/proc/{pid}/stat').read_text();fields=raw[raw.rfind(')')+2:].split()
        if fields[0] not in ('T','t') or int(fields[19])!=process['start_ticks'] or not os.sched_getaffinity(pid).issubset({24,26}):
            raise ValueError('An AAPL process is not safely frozen and limited')
    for parent,group in snapshot['queues'].items():
        state_path=Path(group['state']);previous=Path(group['snapshot_dir'])/'PRIVATE_QUEUE_STATE.json'
        if digest(state_path)!=digest(previous):raise ValueError('Queue status changed since the paused snapshot')
        state=json.loads(state_path.read_text());before=digest(state_path)
        state['status']='paused_until_0300';state['night_pause']={
            'paused_at_utc':manifest['paused_at_utc'],'resume_at_local':snapshot['resume_at_local'],
            'active_seconds_before_pause':group['active_seconds_before_pause'],
            'remaining_active_seconds_at_resume':group['remaining_active_seconds'],
            'original_status_before_pause':group['status_at_pause'],
            'checkpoint_snapshot':str(snapshot_path)}
        for worker in snapshot['workers']:
            if str(worker['parent_pid'])!=parent:continue
            entry=(state['jobs'][worker['job']] if worker['script']=='minute_matrix_queue.py'
                   else state['workers'][worker['job']])
            if entry['status']!='running' or entry['pid']!=worker['pid']:
                raise ValueError('Active worker status differs from the frozen process')
            entry['status_before_night_pause']=entry['status'];entry['status']='paused_night'
        prepared.append((state_path,state,before))
    for state_path,state,before in prepared:
        atomic_json(state_path,state)
        changes.append({'queue':str(state_path),'sha256_before':before,'sha256_after':digest(state_path)})
    out={'status':'queue_statuses_marked_paused','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'resume_at_local':snapshot['resume_at_local'],'queues':changes,'training_started':False}
    path=snapshot_path.parent/'NIGHT_PAUSE_STATE_RECEIPT.json'
    with path.open('x',encoding='utf8') as f:json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'status':out['status'],'queues':len(changes),'path':str(path)}))


if __name__=='__main__':main()
