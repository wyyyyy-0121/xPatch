"""Behavioral checks for expert isolation, all-fold ranking, and model comparisons."""
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from .common import OUT
from .experts import scope_key,fit_cpu,predict_cpu,matrix,balanced_rows
from .experiment import pooled_score,mechanism_configs
from .tests_data import two_sessions
from .tests_training import make_panel
from .simple import fit,predict


def test_information_scopes_do_not_share_OOF_predictions():
    keys={scope_key(),scope_key('close'),scope_key(training_scope='aapl'),scope_key(market_information=False)}
    assert len(keys)==4
    top=[{'candidate_id':'full','market':True,'information':'ohlcv','training_scope':'multistock',
          'lookback':48,'foundation_mode':'frozen','representation_cache':True}]
    no_market=mechanism_configs(top,'no_market')[0]
    assert not no_market['market'] and not no_market['market_information']
    aapl=mechanism_configs(top,'aapl')[0]
    assert aapl['training_scope']=='aapl' and aapl['adaptation_fraction']==0
    random=mechanism_configs(top,'random_frozen')[0]
    assert random['foundation_mode']=='random_frozen' and not random['representation_cache']


def test_no_market_path_never_reads_market_arrays_or_leaves_hidden_features():
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends);scalers=panel.scaler('2021-01-04')
    class Refuse:
        def __getitem__(self,index):raise AssertionError('No-market ablation read background information')
    panel.features['SPY']=Refuse()
    config={'lookback':48,'market_information':False};stocks=np.zeros(4,dtype=np.int64);rows=np.array([390,430,500,700])
    batch=panel.windows(stocks,rows,config,scalers,include_labels=False)
    assert batch['market'].shape==(4,48,0)
    assert matrix(panel,stocks,rows,config,scalers,with_labels=False)[0].shape[1]==12*9+8+1


def test_pooled_ranking_uses_observation_counts():
    results=[{'scores':{'mse':1.,'n':9}},{'scores':{'mse':9.,'n':1}}]
    assert pooled_score(results)==1.8


def test_real_cpu_expert_fits_early_history_and_preserves_balanced_weights():
    frame,starts,ends=two_sessions();panel=make_panel(frame,starts,ends)
    # Production dates are memory-mapped U10 arrays; NumPy cannot reduce them
    # with minimum, unlike the object-string dates in the small panel fixture.
    panel.dates=np.asarray(panel.dates,dtype='U10')
    config={'family':'ridge','lookback':48,'alpha':1000.,'rows_per_day':8,'information':'ohlcv'}
    with TemporaryDirectory(prefix='unit_cpu_',dir=OUT) as path:
        forecast,scalers,report=fit_cpu(panel,config,'2021-01-04',path)
        assert report['fit_start']=='2021-01-04' and report['fit_end']=='2021-01-04'
        value=predict_cpu(forecast,panel,np.zeros(4,dtype=np.int64),np.array([390,430,500,700]),config,scalers)
        assert np.isfinite(value).all()
    stock,rows,weights=balanced_rows(panel,'2021-01-05',2026,per_day=8)
    a=weights[panel.dates[rows]=='2021-01-04'].sum();b=weights[panel.dates[rows]=='2021-01-05'].sum()
    assert a==pytest.approx(b)


def test_convex_control_can_choose_zero_instead_of_adverse_experts():
    y=np.tile([1.,-1.],50)*.001;x=np.column_stack([-y,-2*y,-3*y,-4*y])
    model=fit(x,y,{'family':'convex','alpha':0.})
    assert sum(model['coefficient'])<1e-8 and model['zero_expert_weight']>1-1e-8
    assert np.mean((predict(model,x)-y)**2)<=np.mean(y*y)*(1+1e-8)
