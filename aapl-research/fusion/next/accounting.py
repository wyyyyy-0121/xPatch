"""Append-only job identities and conservative single-GPU computation accounting."""
from contextlib import AbstractContextManager
import os
import time
import uuid

from .common import OUT, read_json, write_json, utc_now


LIMITS={'performance':1.,'experts_and_cache':5.,'fusion_search':7.,
        'adaptation_and_mechanisms':6.,'final_and_deployment':3.,'recovery':2.}


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
        current=totals()
        if self.gpu and (current['gpu_card_hours']>=24 or current['by_category'][self.category]>=LIMITS[self.category]):
            raise RuntimeError('Prespecified GPU budget exhausted')
        if self.gpu and current['active_gpu_jobs']:raise RuntimeError('An accounted GPU job is already active; inspect it before starting another')
        self.start=time.monotonic();self.started=utc_now();self.record('running');return self

    def record(self,status='running',error=None):
        elapsed=time.monotonic()-self.start
        row={'id':self.id,'category':self.category,'description':self.description,'pid':os.getpid(),
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
