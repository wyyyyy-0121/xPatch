"""CPU package entry: apply the declared FP32 inference policy before loading models."""
import os

# Bound affinity and BLAS pools before the adapter imports numerical libraries.
available=set(os.sched_getaffinity(0));selected=set()
if os.environ.get('AAPL_CPU_IDS'):
    for part in os.environ['AAPL_CPU_IDS'].split(','):
        bounds=list(map(int,part.split('-')))
        selected.update(range(bounds[0],bounds[-1]+1))
else:selected=set(sorted(available&set(range(18)))[:4])
if not selected or not selected<=available&set(range(18)):
    raise ValueError('CPU IDs must remain within available research CPUs 0-17')
threads=int(os.environ.get('AAPL_CPU_THREADS',min(4,len(selected))))
if not 1<=threads<=len(selected):raise ValueError('Thread count exceeds CPU allocation')
os.sched_setaffinity(0,selected)
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS',
             'RAYON_NUM_THREADS','ARROW_NUM_THREADS','POLARS_MAX_THREADS','AAPL_CPU_THREADS'):
    os.environ[name]=str(threads)
os.environ['CUDA_VISIBLE_DEVICES']=''

from portable_inference_runtime import install
install()

from pathlib import Path
import fusion.next.cloud_assets.cpu_run as original

# The original entry derives the package directory from __file__. Preserve its
# run.py location while reusing the exact admitted command implementation.
source=Path(original.__file__).read_text()
exec(compile(source,str(Path(__file__).resolve()),'exec'),
     {'__name__':'__main__','__file__':str(Path(__file__).resolve())})
