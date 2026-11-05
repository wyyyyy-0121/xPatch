from types import SimpleNamespace
import numpy as np
import pandas as pd

from .budget import work_items
from .design import draft,halving_units,halving_survivors
from . import budget
from .common import OUT,write_json
from pathlib import Path
from tempfile import TemporaryDirectory


def test_budget_enumerates_all_information_scopes_mechanisms_and_conditional_final_work():
    dates=pd.bdate_range('2021-01-04','2026-06-30').strftime('%Y-%m-%d').to_numpy()
    panel=SimpleNamespace(stocks=['AAPL','MSFT'],dates=dates,origins={s:np.arange(len(dates)) for s in ('AAPL','MSFT')})
    spec=draft();spec['groups']['ridge']=[{'family':'ridge'}]
    spec['updates']={family:2000 for family in ('xpatch','patchtst','fusion','lora')}
    for family in ('xpatch','patchtst','fusion','lora'):
        for config in spec['groups'][family]:config['total_updates']=2000
    spec['transfer_configs']=spec['groups']['fusion'][:8]
    items,counts=work_items(panel,spec)
    assert counts['stocks']==2 and counts['all_stock_OOF_origins']==2*counts['AAPL_OOF_origins']
    oof=[item for item in items if item['description'].startswith('quarterly OOF')]
    assert len(oof)==4*4 and all(item['fit_equivalents']==13 for item in oof)
    mechanisms=[item for item in items if item['description'].startswith('mechanism ')]
    assert len(mechanisms)==7 and all(item['fit_equivalents']==2*3*3 for item in mechanisms)
    assert any(item['family']=='fusion_random' for item in mechanisms)
    final=[item for item in items if item['category']=='final_and_deployment']
    assert {item['family'] for item in final}=={'lora','ridge','lightgbm','xpatch','patchtst'}


def test_budget_reduced_search_keeps_two_full_budget_finalists():
    assert halving_survivors(24)==8 and halving_survivors(16)==6
    assert halving_survivors(2)==halving_survivors(3)==2
    assert halving_units(2)==2.


def test_projection_counts_scaler_fits_and_slower_market_encoder(monkeypatch):
    def items(*args):
        return ([{'category':'fusion_search','family':'fusion','fit_equivalents':1.,'updates':2.,'predict_rows':1},
                 {'category':'experts_and_cache','family':'ridge','fit_equivalents':1.,'updates':0.,'predict_rows':0}],
                {'maximum_fit_calendar_days':1000})
    monkeypatch.setattr(budget,'work_items',items)
    monkeypatch.setattr(budget,'totals',lambda:{'by_category':{}})
    with TemporaryDirectory(prefix='unit_budget_rates_',dir=OUT) as temporary:
        root=Path(temporary);monkeypatch.setattr(budget,'OUT',root)
        write_json(root/'performance_admission.json',{});write_json(root/'performance_cpu.json',{})
        measurements={name:{'seconds_per_update':1.} for name in ('xpatch_end_to_end','patchtst_end_to_end',
            'end_to_end_cache_population','end_to_end_raw','frozen_tcn_end_to_end','lora_stress','lora_end_to_end','lora_tcn_end_to_end')}
        measurements.update({name:{'seconds_per_origin':1.} for name in
                             ('xpatch_inference','patchtst_inference','lora_inference','frozen_inference','frozen_tcn_inference','lora_tcn_inference')})
        measurements.update(fusion_initialization_seconds=0.,scaler_fit_seconds=2.,scaler_training_calendar_days=500)
        cpu={'measurements':{name:{'scaler_seconds':2.,'feature_seconds':0.,'fit_seconds':0.,
                                 'training_calendar_days':500,'prediction_seconds_per_origin':0.} for name in ('ridge','lightgbm')}}
        before=budget.project(None,{}, {'measurements':measurements},cpu)
        assert before['CPU_work_seconds_before_margin']==4.
        assert before['work'][0]['projected_seconds_before_margin']==9.  # 2 updates + inference + scaled fit + IO
        measurements['frozen_tcn_end_to_end']['seconds_per_update']=10.
        after=budget.project(None,{}, {'measurements':measurements},cpu)
        assert after['work'][0]['projected_seconds_before_margin']==27.
