import copy
import json
from pathlib import Path

import pytest

from scripts.run_fusion_100h import plan


def fixture():
    groups={family:[{'candidate_id':family+str(i),'total_updates':123,'batch_size':256}
        for i in range(count)] for family,count in
        (('ridge',1),('lightgbm',24),('xpatch',16),('patchtst',16),('fusion',24))}
    groups['lora']=[{'candidate_id':f'lora{s}_{r}_{lr}','rank':r,'backbone_lr':lr,'promoted_structure_slot':s}
        for s in (0,1) for r in (8,16) for lr in (1e-5,3e-5)]
    work=[]
    def add(family,description,category='test',hours=10):
        work.append({'family':family,'description':description,'category':category,
            'fit_equivalents':18,'updates':18000,'predict_rows':1800,
            'projected_seconds_before_margin':hours*3600/1.5})
    for family in ('ridge','lightgbm'):add(family,'same-information full CPU search')
    for family in ('xpatch','patchtst'):add(family,'supervised search 10/30/100% prefixes')
    for family in ('ridge','lightgbm','xpatch','patchtst'):add(family,'quarterly OOF main OHLCV')
    for family in ('fusion','lora'):
        add(family,'neural search');add(family,'top two, three seeds')
    add('fusion','mechanism plain_mlp (no speed discount for removed modules)')
    add('fusion','mechanism random_frozen (no speed discount for removed modules)')
    add('fusion','paired transfer; both sides always advance')
    add('lora','pre-Q2 system fits, conservative spare fit allowance and acceptance replay','final_and_deployment')
    return {'identity':'original','groups':groups,'simple':list(range(12)),
        'validation':['2025Q1','2025Q3','2026Q1'],'seeds':[2026,2027,2028],
        'training_budget_policy':{'total_coverage_equivalent_epochs':2,'adaptation_included_in_total':True},
        'projection':{'margin_multiplier':1.5,'work':work}}


def test_scope_retains_epochs_folds_seeds_and_uses_fixed_positions_without_mutating_parent():
    original=fixture();before=copy.deepcopy(original);result=plan(original)
    assert original==before
    assert result['counts']=={'ridge':1,'lightgbm':4,'xpatch':4,'patchtst':4,'fusion':2,'lora':2}
    assert result['seeds']==original['seeds'] and result['validation']==original['validation']
    assert result['training_budget_policy']==original['training_budget_policy']
    for family in ('lightgbm','xpatch','patchtst','fusion'):
        assert result['groups'][family]==original['groups'][family][:result['counts'][family]]
    assert {(cfg['promoted_structure_slot'],cfg['rank'],cfg['backbone_lr']) for cfg in result['groups']['lora']}=={
        (0,8,1e-5),(0,16,1e-5)}
    original['observed_scores']={'bad':-1e99,'good':1e99}
    assert plan(original)==result


def test_budget_keeps_all_main_oof_and_delivery_and_removes_deferred_work():
    result=plan(fixture());rows=result['projection']['work']
    assert len([r for r in rows if r['description']=='quarterly OOF main OHLCV'])==4
    assert not any('random_frozen' in r['description'] or 'paired transfer' in r['description'] for r in rows)
    assert any(r['category']=='final_and_deployment' for r in rows)
    assert result['projection']['hours_with_margin_and_reserve']==pytest.approx(
        sum(r['projected_seconds_before_margin'] for r in rows)*1.5/3600+4)
    for row in rows:
        if row['description'].startswith('one selected structure'):
            assert row['fit_equivalents']==6
        if row['category']=='final_and_deployment':assert row['fit_equivalents']==9


def test_other_epoch_count_is_rejected():
    original=fixture();original['training_budget_policy']['total_coverage_equivalent_epochs']=10
    with pytest.raises(ValueError,match='two total'):plan(original)
