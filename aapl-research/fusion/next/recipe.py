"""Resolve the pre-score two-epoch update table at a legal fit boundary."""


def for_fit(config,fit_end):
    result=dict(config);table=config.get('updates_by_fit_end')
    if table is not None:
        if fit_end not in table:raise ValueError('Fit boundary is outside the frozen two-epoch table')
        result['total_updates']=int(table[fit_end])
        result['early_stopping_patience_updates']=result['total_updates']
    return result
