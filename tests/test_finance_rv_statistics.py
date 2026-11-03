"""Analytical examples for calibration, daily weighting and test direction."""
import numpy as np
import pytest

from finance.calibration import Calibration
from finance.metrics import block_comparison, dm_test, holm, mincer_zarnowitz, qlike, score


def test_smearing_returns_mean_and_empirical_intervals_on_variance_scale():
    raw = np.full(100, 2e-6)
    y = np.repeat([2e-6, 6e-6], 50)
    state = Calibration.fit(y, raw, np.repeat(["2022-01-03", "2022-01-04"], 50))
    mean, quantiles = state.predict(np.array([2e-6, 4e-6]))
    np.testing.assert_allclose(mean, [4e-6, 8e-6])
    np.testing.assert_allclose(quantiles[:, [0, -1]], [[2e-6, 6e-6], [4e-6, 12e-6]])
    restored = Calibration(**state.dictionary())
    np.testing.assert_array_equal(restored.predict(raw)[0], state.predict(raw)[0])
    with pytest.raises(ValueError):
        Calibration.fit(y, np.zeros(100), np.repeat("day", 100))


def test_daily_weighting_gives_short_sessions_equal_weight():
    observed = np.array([1., 1., 1., 2.])
    predicted = np.ones(4)
    result = score(observed, predicted, np.array(["day1"] * 3 + ["day2"]))
    # Three perfect points on day1 and one loss 1-log(2) on day2.
    assert result["qlike"] == pytest.approx((1 - np.log(2)) / 2)
    assert result["log_rv_mse"] == pytest.approx(np.log(2) ** 2 / 2)
    np.testing.assert_allclose(qlike(observed * 1e-5, predicted * 1e-5), qlike(observed, predicted))


def test_dm_and_bootstrap_reject_only_in_improvement_direction():
    improvement = np.tile(np.linspace(-.3, -.1, 40), 5)
    for block in (1, 5):
        better = block_comparison(improvement, block, repeats=5000)
        worse = block_comparison(-improvement, block, repeats=5000)
        assert better["p_less_centered_null"] < .01
        assert better["ci95"][1] < 0
        assert worse["p_less_centered_null"] > .99
        assert worse["ci95"][0] > 0
    assert dm_test(improvement)["p_less"] < .01
    assert dm_test(-improvement)["p_less"] > .99
    assert dm_test(np.zeros(20))["p_less"] == 1
    assert block_comparison(np.zeros(20))["p_less_centered_null"] == 1


def test_holm_keeps_missing_comparisons_in_fixed_family():
    np.testing.assert_allclose(holm([.01, .03, .04, .2]), [.04, .09, .09, .2])
    np.testing.assert_allclose(holm([.001] + [1.] * 9), [.01] + [1.] * 9)


def test_daily_mincer_zarnowitz_recovers_known_calibration_relation():
    predicted = np.linspace(1e-6, 5e-6, 30)
    observed = 2e-6 + 1.5 * predicted
    result = mincer_zarnowitz(np.repeat(observed, 3), np.repeat(predicted, 3), np.repeat(np.arange(30), 3))
    assert result["intercept"] == pytest.approx(2e-6)
    assert result["slope"] == pytest.approx(1.5)
    assert result["sessions"] == 30
    unidentified = mincer_zarnowitz(observed, np.full(30, 2e-6), np.arange(30))
    assert not unidentified["identified"] and unidentified["slope"] is None
