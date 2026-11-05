"""Explicit future-process entry with independently verified input acceleration."""
import os
from pathlib import Path
import runpy
import sys

from fusion.next.common import ROOT
from scripts.portable_inference_runtime import install as portable
from scripts.future_speed_runtime import checked_policy, install


if len(sys.argv) != 2 or sys.argv[1] not in ('scripts/run_fusion_100h.py', 'scripts/finish_fusion_100h.py'):
    raise ValueError('Specify an admitted cloud or local completion entry')
checked_policy()
entry = ROOT/sys.argv[1]
bootstrap = ROOT/'tmp/fast_inference_bootstrap'
bootstrap.mkdir(exist_ok=True)
source = ('from scripts.portable_inference_runtime import install as portable\nportable()\n'
          'import sys\n'
          'if sys.argv[0] == "-m" or sys.argv[0].endswith("/scripts/finish_fusion_100h.py"):\n'
          '    from scripts.future_speed_runtime import install\n    install()\n')
target = bootstrap/'sitecustomize.py'
if target.exists() and target.read_text() != source:
    raise ValueError('Future speed bootstrap changed')
target.write_text(source)
os.environ['PYTHONPATH'] = str(bootstrap) + os.pathsep + str(ROOT)
portable()
install()
sys.argv = [str(entry)]
runpy.run_path(str(entry), run_name='__main__')
