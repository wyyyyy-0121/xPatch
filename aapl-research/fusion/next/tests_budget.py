from types import SimpleNamespace
import numpy as np
import pandas as pd

from .budget import work_items
from .design import draft


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
