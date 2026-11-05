"""Start an existing bound entry with the explicit portable inference adapter."""
import os
from pathlib import Path
import runpy
import sys

from fusion.next.common import ROOT, digest
from scripts.portable_inference_runtime import checked_policy, install

policy=checked_policy()
if digest(Path(__file__))!=policy['launcher_sha256']:
    raise ValueError('Portable launcher changed')
if len(sys.argv)!=2 or sys.argv[1] not in ('scripts/run_fusion_100h.py','scripts/finish_fusion_100h.py'):
    raise ValueError('Specify the exact cloud or local completion entry')
entry=ROOT/sys.argv[1]
bootstrap=ROOT/'tmp/portable_inference_bootstrap';bootstrap.mkdir(exist_ok=True)
text='from scripts.portable_inference_runtime import install\ninstall()\n'
path=bootstrap/'sitecustomize.py'
if path.exists() and path.read_text()!=text:raise ValueError('Portable bootstrap changed')
path.write_text(text)
os.environ['PYTHONPATH']=str(bootstrap)+os.pathsep+str(ROOT)
install();sys.argv=[str(entry)]
runpy.run_path(str(entry),run_name='__main__')
