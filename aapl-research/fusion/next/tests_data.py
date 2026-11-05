"""Meaningful checks for all-day/session timing and cross-symbol leakage."""
import numpy as np
import pandas as pd
import pytest

from .data import FIELDS,causal_transform,eligible_origins
from .acquire import bounds


def two_sessions():
    times=pd.date_range('2021-01-04T14:30Z',periods=390,freq='min').append(
          pd.date_range('2021-01-05T14:30Z',periods=390,freq='min'))
    close=np.r_[np.full(390,100.),np.full(390,110.)]
    frame=pd.DataFrame({k:close.copy() for k in FIELDS},index=times)
    frame['volume']=1000.;frame['trade_count']=10.
    starts=np.zeros(780,dtype=bool);starts[[0,390]]=True
    ends=np.r_[np.full(390,389),np.full(390,779)]
    return frame,starts,ends


def test_open_uses_previous_session_history_without_crossing_target_close():
    frame,starts,ends=two_sessions();features,observed,close,_=causal_transform(frame,starts)
    origins=eligible_origins(observed,ends)
    assert 390 in origins and 779 not in origins and 767 in origins and 768 not in origins
    # The overnight move has its own channel. It is not a fake one-minute return.
    assert features[390,3]==0 and np.isclose(features[390,8],np.log(1.1))
    assert features[390,7]==1 and features[391,7]==0


def test_missing_fill_is_forward_only_and_future_labels_reject_missing():
    frame,starts,ends=two_sessions();frame.iloc[400]=np.nan
    features,observed,close,_=causal_transform(frame,starts)
    assert not observed[400] and features[400,9]==1 and close[400]==close[399]
    origins=eligible_origins(observed,ends)
    assert 399 not in origins and 390 not in origins
    assert 413 in origins  # missing history is explicitly masked and within fixed allowance


def test_future_rows_do_not_change_current_features():
    frame,starts,ends=two_sessions();before=causal_transform(frame,starts)[0]
    changed=frame.copy();changed.iloc[500:,:4]*=3;changed.iloc[500:,4]*=17
    after=causal_transform(changed,starts)[0]
    np.testing.assert_array_equal(before[:500],after[:500])


def test_request_boundaries_never_include_Q3_for_any_symbol():
    low,high=bounds('2021-01-04','2026-06-30')
    assert pd.Timestamp(high)<pd.Timestamp('2026-07-01T04:00Z')


def test_missing_initial_history_keeps_mask_and_never_backfills():
    frame,starts,ends=two_sessions();frame.iloc[:3]=np.nan
    features,observed,close,_=causal_transform(frame,starts)
    assert np.isnan(close[:3]).all() and not observed[:3].any()
    assert (features[:3,9]==1).all() and features[0,7]==1
    assert np.isfinite(features).all()
