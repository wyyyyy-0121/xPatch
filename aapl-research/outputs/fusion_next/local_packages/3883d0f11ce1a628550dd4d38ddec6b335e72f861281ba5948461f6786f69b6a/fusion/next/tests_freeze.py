"""The final-entry gate must depend on aligned raw predictions, not reported scores."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from . import freeze
from .common import OUT,digest
from .evaluation import point_scores,seed_system
from .splits import VALIDATION


def test_freeze_recomputes_scores_and_rejects_shifted_origins_labels_or_summary(monkeypatch):
    dates=np.array(['2025-01-02','2025-07-01','2026-01-02'])
    panel=SimpleNamespace(dates=dates,stamps=pd.to_datetime(dates,utc=True),
                          targets={'AAPL':np.array([.1,-.2,.3],dtype=np.float32)})
    monkeypatch.setattr(freeze,'score_rows',lambda panel,q:({},np.array([VALIDATION.index(q)])))
    with TemporaryDirectory(prefix='unit_freeze_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(freeze,'OUT',root)
        directory=root/'seed_verification/fusion/trial';directory.mkdir(parents=True)
        target=panel.targets['AAPL'];predictions=np.tile(target*.1,(3,1))
        scores,ensemble=seed_system(target,predictions)
        selected={**scores,'config':{'candidate_id':'trial'},'prediction_namespace':'fusion',
                  'quarter_scores':{q:point_scores(target[i:i+1],ensemble[i:i+1]) for i,q in enumerate(VALIDATION)}}
        arrays={'rows':np.arange(3),'timestamps_ns':panel.stamps.as_unit('ns').asi8,
                'quarters':np.array(VALIDATION),'target':target,'predictions':predictions,'ensemble':ensemble}
        path=directory/'predictions.npz'
        def save(**changes):
            np.savez(path,**{**arrays,**changes});selected['prediction_sha256']=digest(path)
        save();actual,rows,y,prediction=freeze.recheck_seed_predictions(panel,selected)
        assert actual['system']==scores['system']
        for key,changed in [('rows',np.array([1,2,3])),('target',target*2),
                            ('timestamps_ns',arrays['timestamps_ns']+60_000_000_000),('ensemble',ensemble*2)]:
            save(**{key:changed})
            with pytest.raises(ValueError):freeze.recheck_seed_predictions(panel,selected)
        save();forged={**selected,'system':{**scores['system'],'mse':0.}}
        with pytest.raises(ValueError,match='summary'):freeze.recheck_seed_predictions(panel,forged)
