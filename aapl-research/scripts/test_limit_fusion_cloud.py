"""Exercise process ownership and PID-creation protection with disposable children."""
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory

import psutil

from fusion.next.common import OUT
from scripts import limit_fusion_cloud as limiter


def test_limiter_signals_only_the_same_project_and_exact_process_identity(monkeypatch):
    with TemporaryDirectory(prefix='unit_limit_',dir=OUT) as temporary:
        root=Path(temporary);foreign=root/'other';foreign.mkdir()
        package=root/'fusion/next';package.mkdir(parents=True)
        (root/'fusion/__init__.py').write_text('')
        (package/'__init__.py').write_text('')
        (package/'capfixture.py').write_text("import time\nprint('ready',flush=True)\ntime.sleep(300)\n")
        monkeypatch.setattr(limiter,'ROOT',root)
        env={**os.environ,'PYTHONPATH':str(root)};children=[]
        try:
            for cwd in (root,foreign):
                process=subprocess.Popen([sys.executable,'-m','fusion.next.capfixture'],cwd=cwd,
                    env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                children.append(process)
                assert select.select([process.stdout],[],[],5)[0]
                assert process.stdout.readline().strip()=='ready'
            owned,other=map(lambda p:psutil.Process(p.pid),children)
            assert limiter.project_process(owned) and not limiter.project_process(other)
            row={'pid':owned.pid,'created_unix':owned.create_time()-1}
            assert not limiter.send_exact(row,signal.SIGTERM)
            assert children[0].poll() is None and children[1].poll() is None
            row['created_unix']=owned.create_time()
            assert limiter.send_exact(row,signal.SIGTERM)
            children[0].wait(timeout=5)
            assert children[1].poll() is None
        finally:
            for process in children:
                if process.poll() is None:process.terminate()
                process.wait(timeout=5)
