"""Synthetic-only checks of split-neutral paths and past feature windows."""
import numpy as np

from multiscale.bars import Bars,targets
from multiscale.features import CloseWindows
from split_neutral import independent_neutral_return,share_adjusted_bars


def main():
    dates=np.array(['2020-08-27','2020-08-28','2020-08-31','2020-09-01',
                    '2020-09-02','2020-09-03'],dtype='U10')
    start=np.arange(len(dates),dtype=np.int64)*2
    raw=np.array([100.,100.,25.,26.,27.,28.])
    bars=Bars('day',start,start+1,raw,np.ones(len(dates),dtype=bool),dates)
    actions=[{'date':'2020-08-31','ratio':4.0}]
    adjusted,factors=share_adjusted_bars(bars,actions)
    np.testing.assert_allclose(factors,[1,1,4,4,4,4])
    np.testing.assert_allclose(adjusted.close,[100,100,100,104,108,112])
    assert np.array_equal(bars.close,raw)
    origins=np.array([1,2,3],dtype=np.int64)
    direct=independent_neutral_return(bars,origins,2,actions)
    np.testing.assert_allclose(direct,targets(adjusted,origins,2,'future_log_return'),rtol=1e-12,atol=1e-14)
    np.testing.assert_allclose(direct[0],[0,np.log(1.04)],rtol=1e-12,atol=1e-14)
    raw_x,_=CloseWindows(bars,2,actions).arrays(origins)
    adjusted_x,_=CloseWindows(adjusted,2,[]).arrays(origins)
    np.testing.assert_allclose(raw_x,adjusted_x,rtol=1e-12,atol=1e-14)
    unchanged,_=share_adjusted_bars(bars,[])
    np.testing.assert_array_equal(unchanged.close,bars.close)
    try:
        share_adjusted_bars(bars,[actions[0],actions[0]])
    except ValueError:
        pass
    else:
        raise AssertionError('Duplicate split action was accepted')
    print('Synthetic split-neutral checks passed; no market data used.')


if __name__=='__main__':main()
