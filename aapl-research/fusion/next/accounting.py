"""Append-only job identities and conservative single-GPU computation accounting."""
from contextlib import AbstractContextManager
from datetime import datetime,timezone
import os
from pathlib import Path
import time
import uuid
import psutil

from .common import OUT, read_json, write_json, utc_now


LIMITS={'performance':1.,'experts_and_cache':5.,'fusion_search':7.,
        'adaptation_and_mechanisms':6.,'final_and_deployment':3.,'recovery':2.}


def recover_interrupted():
    """Recover a missing process only; a heartbeat timeout never proves it stopped."""
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    recovered=[]
    for path in (OUT/'usage_events').glob('*.json'):
        row=read_json(path)
        if row['status']!='running':continue
        same_boot=row.get('boot_id')==boot
        try:alive=same_boot and abs(psutil.Process(row['pid']).create_time()-row.get('process_created_unix',-1))<.01
        except psutil.NoSuchProcess:alive=False
        if alive:continue
        # Without a reliable provider stop receipt, upper-bound the unobserved
        # interval by current wall time rather than erase interrupted GPU work.
        elapsed=max(row['wall_seconds'],(datetime.now(timezone.utc)-datetime.fromisoformat(row['started_at_utc'])).total_seconds())
        row.update(status='interrupted_process_missing',recovered_at_utc=utc_now(),
                   recovery_evidence='boot or exact PID creation identity no longer exists',
                   wall_seconds=elapsed,gpu_card_hours=elapsed/3600 if row.get('gpu') else 0.,
                   conservative_unobserved_interval_included=True)
        write_json(path,row);recovered.append(row['id'])
    return recovered


def totals():
    jobs=[read_json(p) for p in (OUT/'usage_events').glob('*.json')]
    gpu=sum(j['gpu_card_hours'] for j in jobs)
    return {'gpu_card_hours':gpu,'jobs':len(jobs),
            'by_category':{name:sum(j['gpu_card_hours'] for j in jobs if j['category']==name) for name in LIMITS},
            'active_jobs':[j['id'] for j in jobs if j['status']=='running'],
            'active_gpu_jobs':[j['id'] for j in jobs if j['status']=='running' and j.get('gpu',False)],
            'network_install_and_user_wait_excluded':True}


class Usage(AbstractContextManager):
    def __init__(self,category,description,gpu=True):
        if category not in LIMITS:raise ValueError('Unknown frozen budget category')
        self.category=category;self.gpu=gpu;self.description=description
        self.id=uuid.uuid4().hex;self.start=None

    def __enter__(self):
        recover_interrupted()
        current=totals()
        if self.gpu and (current['gpu_card_hours']>=24 or current['by_category'][self.category]>=LIMITS[self.category]):
            raise RuntimeError('Prespecified GPU budget exhausted')
        if self.gpu and current['active_gpu_jobs']:raise RuntimeError('An accounted GPU job is already active; inspect it before starting another')
        self.start=time.monotonic();self.started=utc_now();self.record('running');return self

    def record(self,status='running',error=None):
        elapsed=time.monotonic()-self.start
        row={'id':self.id,'category':self.category,'description':self.description,'pid':os.getpid(),
             'process_created_unix':psutil.Process().create_time(),
             'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
             'started_at_utc':self.started,'updated_at_utc':utc_now(),'status':status,
             'wall_seconds':elapsed,'gpu':self.gpu,'gpu_card_hours':elapsed/3600 if self.gpu else 0.,'error':error}
        write_json(OUT/'usage_events'/f'{self.id}.json',row)
        current=totals();write_json(OUT/'usage_summary.json',current)
        if status=='running' and self.gpu and (current['gpu_card_hours']>24 or current['by_category'][self.category]>LIMITS[self.category]):
            raise RuntimeError('Prespecified GPU budget exceeded; checkpoint and stop')
        return row

    def __exit__(self,kind,error,traceback):
        self.record('failed' if kind else 'completed',str(error) if error else None)
        return False
