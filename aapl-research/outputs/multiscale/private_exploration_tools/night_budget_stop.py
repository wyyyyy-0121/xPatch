"""Freeze one resumed queue at its original remaining active-time limit."""
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import sys
import tempfile


def process(pid):
    try:
        raw=Path(f'/proc/{pid}/stat').read_text();rest=raw[raw.rfind(')')+2:].split()
        argv=[x.decode('utf8','replace') for x in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if x]
        return {'pid':pid,'state':rest[0],'start_ticks':int(rest[19]),'argv':argv}
    except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError):return None


def flag_value(argv,flag):
    if flag not in argv:return None
    index=argv.index(flag)
    return argv[index+1] if index+1<len(argv) else None


def main():
    snapshot_path=Path(sys.argv[1]).resolve();parent_pid=str(int(sys.argv[2]))
    reason=sys.argv[3] if len(sys.argv)>3 else 'original_compute_budget_exhausted'
    snapshot=json.loads(snapshot_path.read_text());queue=snapshot['queues'][parent_pid]
    plan=str(Path(queue['plan']).resolve());members={str(w['pid']) for w in snapshot['workers'] if str(w['parent_pid'])==parent_pid}
    stopped=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        item=process(int(p.name))
        if not item or not item['argv'] or item['pid']==os.getpid():continue
        argv=item['argv'];exe=argv[0]
        if not (exe.endswith('/python') or exe.endswith('/python3') or 'aapl-research/.venv/bin/python' in exe):continue
        original=('--worker' in argv and flag_value(argv,'--plan')==plan)
        fallback_snapshot=flag_value(argv,'--snapshot')
        fallback=(any(x.endswith('/night_resume_worker.py') for x in argv) and fallback_snapshot is not None and
                  str(Path(fallback_snapshot).resolve())==str(snapshot_path) and
                  flag_value(argv,'--pid') in members and '--check' not in argv)
        if not (original or fallback):continue
        again=process(item['pid'])
        if not again or again['start_ticks']!=item['start_ticks']:continue
        try:os.kill(item['pid'],signal.SIGSTOP)
        except ProcessLookupError:continue
        stopped.append({'pid':item['pid'],'start_ticks':item['start_ticks'],'argv':argv})
    result={'status':'queue_workers_frozen','reason':reason,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'parent_pid_at_pause':int(parent_pid),'queue_plan':plan,'stopped':stopped,
            'note':'SIGSTOP preserves in-memory work; existing periodic checkpoints remain on disk.'}
    path=snapshot_path.parent/f'BUDGET_STOP_{parent_pid}.json'
    with tempfile.NamedTemporaryFile('w',encoding='utf8',dir=path.parent,delete=False) as f:
        json.dump(result,f,indent=2,ensure_ascii=False);f.write('\n');tmp=f.name
    os.replace(tmp,path)
    print(json.dumps({'status':result['status'],'parent_pid':int(parent_pid),'stopped':len(stopped),'path':str(path)}))


if __name__=='__main__':main()
