import numpy as np
import pytest

from .evaluation import point_scores,seed_system,paired_blocks,holm


def test_seed_ensemble_scores_predictions_instead_of_averaging_losses():
    y=np.array([1.,-1.]);predictions=np.array([[2.,-2.],[0.,0.],[1.,-1.]])
    report,prediction=seed_system(y,predictions)
    assert report['system']['mse']==0
    assert np.mean([row['mse'] for row in report['individual_seeds'].values()])>0
    np.testing.assert_array_equal(prediction,y)


def test_null_centered_block_test_uses_whole_days_and_holm_family():
    dates=np.repeat(np.array([f'2026-07-{i:02d}' for i in range(1,25)]),[2,3]*12)
    result=paired_blocks(np.full(len(dates),-1e-6),dates)
    assert result['days']==24 and result['repeats']==5000
    assert result['one_sided_p']==1/5001
    np.testing.assert_allclose(result['ci95'],[-1e-6,-1e-6])
    adjusted=holm({'a':.01,'b':.04,'c':.02})
    assert adjusted=={'a':.03,'c':.04,'b':.04}


def test_missing_predictions_are_not_silently_dropped_and_zero_direction_abstains():
    with pytest.raises(ValueError,match='Missing'):point_scores([1.,2.],[1.,np.nan])
    result=point_scores([1.,-1.],[0.,0.])
    assert result['direction_hit_excluding_abstentions'] is None
    assert result['direction_coverage']==0 and result['mse_improvement_vs_zero']==0
