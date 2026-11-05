"""Fusion-only active wall/storage guard; never touches previous run accounting."""
import threading
import time

from aapl.common import utc_now,read_json,write_json
from .config import OUT


class Guard:
    def __enter__(self):
        self.started=time.time();self.stop=threading.Event();self.violation=None
        self.state=read_json(OUT/'runtime.json') if (OUT/'runtime.json').exists() else {'attempts':[]}
        self.attempt={'started_at_utc':utc_now(),'start_epoch':self.started,'active':True}
        self.state['attempts'].append(self.attempt);write_json(OUT/'runtime.json',self.state)
        self.thread=threading.Thread(target=self.monitor,daemon=True);self.thread.start()
        return self

    def monitor(self):
        while not self.stop.wait(10):
            try:self.check()
            except Exception as exc:self.violation=repr(exc);return

    def check(self):
        if self.violation:raise RuntimeError(self.violation)
        seconds=sum(row.get('end_epoch',time.time())-row['start_epoch'] for row in self.state['attempts'])
        size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        usage={'active_wall_hours':seconds/3600,'gpu_hours_conservative_upper_bound':seconds/3600,
               'artifacts_gib':size/2**30,'updated_at_utc':utc_now()}
        write_json(OUT/'usage.json',usage)
        if seconds>8*3600 or size>20*2**30:
            self.violation='Fusion active wall/storage budget exceeded'
            raise RuntimeError(self.violation)

    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=12)
        self.attempt.update(active=False,end_epoch=time.time(),finished_at_utc=utc_now())
        write_json(OUT/'runtime.json',self.state)
        self.check()
