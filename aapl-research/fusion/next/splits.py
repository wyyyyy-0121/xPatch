"""Fixed forward quarters, one complete purged session, and confirmation access gate."""
import numpy as np
import pandas as pd

from .common import OUT, PLAN, digest, identity, read_json, write_json


VALIDATION=('2025Q1','2025Q3','2026Q1')


def fold(dates,quarter,confirmation=False):
    period=pd.Period(quarter,freq='Q')
    if period<pd.Period('2023Q1') or period>pd.Period('2026Q2'):
        raise ValueError('Quarter outside development/confirmation scope; Q3 has a separate frozen entry')
    if quarter=='2026Q2' and not confirmation:
        raise ValueError('Q2 is a one-time frozen-process confirmation, not development selection')
    start=str(period.start_time.date());end=str(period.end_time.date())
    unique=np.unique(dates);earlier=unique[unique<start]
    if len(earlier)<2:raise ValueError('Insufficient earlier sessions')
    return {'quarter':quarter,'score_start':start,'score_end':end,
            'fit_start':'2021-01-04','fit_end':str(earlier[-2]),'purged_full_session':str(earlier[-1]),
            'role':'confirmation' if confirmation else ('main_validation' if quarter in VALIDATION else 'OOF_training')}


def manifest(dates):
    records=[fold(dates,str(q)) for q in pd.period_range('2023Q1','2026Q1',freq='Q')]
    value={'plan_sha256':digest(PLAN),'folds':records,'main_validation':list(VALIDATION),
           'early_history':{'start':'2021-01-04','end':'2022-12-31',
               'expert_use':'All supervised expert fits include the available 2021 onward history',
               'fusion_use':'One common uniform stock/day sampler covers 2021 onward; pre-2023 rows train an auxiliary readout of the shared branches, later rows use earlier-quarter OOF experts; both can appear in the same batch',
               'pretraining_budget_fraction':'determined by the number of eligible days, without a fixed age weight',
               'fabricated_or_in_sample_expert_inputs':False},
           'Q2':'blocked until system/process/control freeze; known process-confirmation status',
           'Q3':'blocked for every symbol; unavailable to this module'}
    value['identity']=identity(value)
    path=OUT/'time_splits.json'
    if path.exists() and read_json(path)!=value:
        previous=read_json(path)
        if previous['folds']!=records or (OUT/'search_manifest.json').exists():
            raise ValueError('Fixed time-split specification changed after freeze')
        write_json(OUT/('pre_score_recipe_correction_'+previous['identity'][:12]+'.json'),{
            'previous_time_splits_sha256':digest(path),'previous_early_history_recipe':previous['early_history'],
            'corrected_early_history_recipe':value['early_history'],
            'reason':'Use the same uniform stock/day sampler for early and OOF periods; a fixed early-history percentage would change relative day weights',
            'new_scope_validation_scores_exist':False,'search_frozen':False})
    write_json(path,value)
    return value
