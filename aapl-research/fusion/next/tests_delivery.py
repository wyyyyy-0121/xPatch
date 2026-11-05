"""Actually exit and resume toy replay in different OS processes; no large model."""
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .common import ROOT,OUT,write_json
from . import delivery
from .tests_data import two_sessions
from .tests_training import make_panel


def configure_toy(directory):
    """Called in fresh subprocesses; all values are fabricated, in known development dates."""
    frame,starts,ends=two_sessions();frame.index=frame.index+(pd.Timestamp('2026-01-05')-pd.Timestamp('2021-01-04'))
    panel=make_panel(frame,starts,ends);cfg={'lookback':240,'information':'ohlcv','batch_size':2}
    scalers=panel.scaler('2026-01-05');directory=Path(directory)
    write_json(directory/'system_freeze.json',{'selected_system':{'config':cfg}})
    write_json(directory/'data/AAPL/manifest.json',{'source':{'source_csv_read_only':'unused_toy_source'}})
    class Predictor:
        model_identity='toy_process_recovery'
        models=[(None,scalers,cfg)]
        def __init__(self,*args,**kwargs):pass
        def predict_origin(self,panel,rows):
            batch=panel.windows(np.zeros(len(rows),dtype=np.int64),rows,cfg,scalers,include_labels=False)
            return {'prediction':batch['local'][:,-1,3].numpy().astype(np.float64)*.001,
                    'weights':np.tile([0.,0.,0.,0.,1.],(len(rows),1))}
    delivery.OUT=directory;delivery.Panel.load=lambda:panel;delivery.FrozenSystem=Predictor
    delivery.read_source=lambda path:frame;delivery.pages_frame=lambda name:(frame,{})
    return directory/'deployment'


def test_acceptance_exits_and_recovers_in_distinct_processes_without_duplicate_outputs():
    with TemporaryDirectory(prefix='unit_delivery_process_',dir=OUT) as temporary:
        directory=Path(temporary)
        program="""import sys
from pathlib import Path
from fusion.next.tests_delivery import configure_toy
from fusion.next.delivery import replay
deployment=configure_toy(Path(sys.argv[1]))
print(replay(deployment,maximum_origins=6,stop_after=3) if sys.argv[2]=='first' else replay(deployment,maximum_origins=6,resume=True))
"""
        for mode in ('first','resume'):
            result=subprocess.run([sys.executable,'-c',program,str(directory),mode],cwd=ROOT,
                                  env=dict(os.environ,PYTHONPATH=str(ROOT)),capture_output=True,text=True,timeout=60)
            assert result.returncode==0,result.stderr
            if mode=='first':
                progress=json.loads((directory/'deployment_acceptance/toy_process_recovery/progress.json').read_text())
                assert progress['completed_origins']==3
        saved=directory/'deployment_acceptance/toy_process_recovery'
        report=json.loads((saved/'report.json').read_text());state=json.loads((saved/'state.json').read_text())
        assert report['distinct_exit_and_resume_processes'] and report['resume_without_duplicate_prediction']
        assert report['origins']==report['timed_origins']==len(state['outputs'])==6
        assert len({r['pid'] for r in report['cold_load_measurements_by_process']})==2
        assert report['batch_stream_max_prediction_difference']==0.
