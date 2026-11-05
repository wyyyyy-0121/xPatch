from pathlib import Path
import json
import os
import tempfile
from aapl.common import ROOT,digest,identity,read_json,utc_now

OUT=ROOT/'outputs/fusion_next'
PLAN=ROOT/'FUSION_IMPLEMENTATION_PLAN.md'
STOCKS=('AAPL','MSFT','AMZN','GOOGL','META','NVDA','AMD','INTC','JPM','BAC','XOM','CVX','WMT','COST','JNJ','UNH')
BACKGROUND=('SPY','QQQ')
DEV_START='2021-01-04'
DEV_END='2026-06-30'


def write_json(path,value):
    # Parallel CPU/GPU jobs may publish the same aggregate status. Unique temporary
    # files avoid one process renaming another process's in-progress write.
    path=Path(path).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('New output must stay under the research root')
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',prefix=path.name+'.',suffix='.tmp',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        try:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
            stream.flush();os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True);raise
    os.replace(temporary,path)


def source_identity():
    files=[p for p in (ROOT/'fusion/next').rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.txt')]
    files+=[ROOT/name for name in ('fusion/__init__.py','aapl/__init__.py','aapl/common.py','aapl/data.py','aapl/models.py')]
    files+=[p for p in (ROOT/'aapl/vendor').glob('*') if p.suffix in ('.py','.json','.txt')]
    return identity({str(p.relative_to(ROOT)):digest(p) for p in sorted(set(files))})


def status(stage,**details):
    write_json(OUT/'progress.json',dict(stage=stage,updated_at_utc=utc_now(),authoritative_plan_sha256=digest(PLAN),**details))
