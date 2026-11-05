#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
python - <<'PY'
import torch
if not torch.cuda.is_available():raise SystemExit('CUDA is unavailable in the chosen image')
print('PyTorch:',torch.__version__,'runtime CUDA:',torch.version.cuda)
print('GPU:',torch.cuda.get_device_name(),'capability:',torch.cuda.get_device_capability())
# Exercise actual kernels: CUDA visibility alone does not verify RTX 5090 support.
x=torch.randn(1024,1024,device='cuda');y=(x@x).sum();torch.cuda.synchronize()
if not torch.isfinite(y):raise SystemExit('GPU arithmetic check failed')
PY
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r cloud/requirements.txt
.venv/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
root=Path.cwd();manifest=json.loads((root/'bundle_manifest.json').read_text())
for name,row in manifest['files'].items():
 h=hashlib.sha256()
 with open(root/name,'rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 if h.hexdigest()!=row['sha256']:raise SystemExit('Archive identity failure: '+name)
print('All uploaded files match the local package manifest')
PY
FUSION_EXECUTION_PROFILE=autodl bash cloud/run.sh -m fusion.next.hardware
