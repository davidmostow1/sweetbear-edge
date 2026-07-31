import numpy as np
import pytest

from sweetbear import calibration as cal


def make_calibrated_sample(n=20_000, seed=7):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.02, 0.98, size=n)
    y = (rng.uniform(size=n) < p).astype(float)
    return y, p


def test_brier_score_bounds():
    assert cal.brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cal.brier_score([1, 0], [0.0, 1.0]) == pytest.approx(1.0)
    assert cal.brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)


def test_log_loss_known_value():
    assert cal.log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
    assert cal.log_loss([1], [1.0]) == pytest.approx(0.0, abs=1e-12)


def test_log_loss_punishes_confident_errors_harder_than_brier():
    mild_brier = cal.brier_score([1], [0.4])
    wild_brier = cal.brier_score([1], [0.01])
    mild_ll = cal.log_loss([1], [0.4])
    wild_ll = cal.log_loss([1], [0.01])
    assert wild_brier / mild_brier < wild_ll / mild_ll


def test_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        cal.brier_score([1, 0], [0.5])
    with pytest.raises(ValueError):
        cal.brier_score([1, 2], [0.5, 0.5])
    with pytest.raises(ValueError):
        cal.brier_score([1, 0], [0.5, 1.5])
    with pytest.raises(ValueError):
        cal.brier_score([], [])


def test_brier_decomposition_identity_holds_for_discrete_forecasts():
    rng = np.random.default_rng(3)
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    p = rng.choice(levels, size=5000)
    y = (rng.uniform(size=5000) < p).astype(float)

    d = cal.brier_decomposition(y, p, n_bins=10)
    assert d.brier == pytest.approx(cal.brier_score(y, p), abs=1e-12)


def test_calibrated_model_has_low_reliability_and_positive_skill():
    y, p = make_calibrated_sample()
    d = cal.brier_decomposition(y, p, n_bins=20)
    assert d.reliability < 1e-3
    assert d.resolution > 0.05
    assert d.skill_score > 0.25


def test_constant_forecast_has_no_resolution():
    rng = np.random.default_rng(11)
    y = (rng.uniform(size=5000) < 0.4).astype(float)
    p = np.full(5000, 0.4)
    d = cal.brier_decomposition(y, p, n_bins=10)
    assert d.resolution == pytest.approx(0.0, abs=1e-9)
    assert d.skill_score == pytest.approx(0.0, abs=1e-2)


def test_reliability_curve_tracks_the_diagonal_when_calibrated():
    y, p = make_calibrated_sample()
    curve = cal.reliability_curve(y, p, n_bins=10)
    assert curve.counts.sum() == y.size
    assert np.max(np.abs(curve.gaps)) < 0.03


def test_reliability_curve_detects_overconfidence():
    y, p = make_calibrated_sample()
    # Push predictions away from 0.5 without changing outcomes.
    overconfident = np.clip((p - 0.5) * 1.6 + 0.5, 0.01, 0.99)
    curve = cal.reliability_curve(y, overconfident, n_bins=10)
    high = curve.mean_predicted > 0.5
    assert np.all(curve.gaps[high] > 0.0)


def test_quantile_strategy_balances_bin_counts():
    y, p = make_calibrated_sample(n=5000)
    curve = cal.reliability_curve(y, p, n_bins=10, strategy="quantile")
    assert curve.counts.max() - curve.counts.min() <= 2


def test_ece_and_mce_are_small_when_calibrated_and_large_when_not():
    y, p = make_calibrated_sample()
    assert cal.expected_calibration_error(y, p) < 0.02
    assert cal.maximum_calibration_error(y, p) < 0.04

    broken = np.clip(p * 0.5 + 0.4, 0.01, 0.99)
    assert cal.expected_calibration_error(y, broken) > 0.05


def test_unknown_binning_strategy_rejected():
    with pytest.raises(ValueError, match="strategy"):
        cal.reliability_curve([1, 0], [0.6, 0.4], strategy="vibes")


def test_platt_leaves_a_calibrated_model_roughly_alone():
    y, p = make_calibrated_sample()
    scaler = cal.PlattScaler().fit(y, p)
    assert scaler.a_ == pytest.approx(1.0, abs=0.1)
    assert scaler.b_ == pytest.approx(0.0, abs=0.1)


def test_platt_repairs_overconfidence():
    y, p = make_calibrated_sample()
    logit = np.log(p / (1 - p))
    overconfident = 1.0 / (1.0 + np.exp(-1.8 * logit))

    before = cal.log_loss(y, overconfident)
    scaler = cal.PlattScaler().fit(y, overconfident)
    after = cal.log_loss(y, scaler.transform(overconfident))

    assert after < before
    # Recovering the inverse of the distortion means a slope near 1/1.8.
    assert scaler.a_ == pytest.approx(1 / 1.8, abs=0.1)


def test_platt_survives_a_perfectly_separable_sample():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    scaler = cal.PlattScaler(regularize=True).fit(y, p)
    out = scaler.transform(p)
    assert np.all(np.isfinite(out))
    assert np.all((out > 0.0) & (out < 1.0))


def test_platt_requires_fitting_first():
    with pytest.raises(RuntimeError):
        cal.PlattScaler().transform([0.5])


def test_isotonic_output_is_monotone_and_bounded():
    y, p = make_calibrated_sample(n=4000, seed=21)
    iso = cal.IsotonicCalibrator().fit(y, p)
    grid = np.linspace(0.0, 1.0, 101)
    out = iso.transform(grid)
    assert np.all(np.diff(out) >= -1e-12)
    assert np.all((out >= 0.0) & (out <= 1.0))


def test_isotonic_never_increases_in_sample_brier():
    y, p = make_calibrated_sample(n=4000, seed=5)
    broken = np.clip(p * 0.5 + 0.4, 0.01, 0.99)
    fitted = cal.IsotonicCalibrator().fit_transform(y, broken)
    assert cal.brier_score(y, fitted) <= cal.brier_score(y, broken) + 1e-12


def test_isotonic_clips_outside_the_training_range():
    y = np.array([0.0, 0.0, 1.0, 1.0])
    p = np.array([0.3, 0.4, 0.6, 0.7])
    iso = cal.IsotonicCalibrator().fit(y, p)
    assert iso.transform([0.0])[0] == pytest.approx(iso.transform([0.3])[0])
    assert iso.transform([1.0])[0] == pytest.approx(iso.transform([0.7])[0])


def test_pool_adjacent_violators_known_case():
    out = cal._pool_adjacent_violators(np.array([1.0, 0.0, 1.0, 0.0]))
    assert out == pytest.approx([0.5, 0.5, 0.5, 0.5])

    out = cal._pool_adjacent_violators(np.array([0.0, 0.0, 1.0, 1.0]))
    assert out == pytest.approx([0.0, 0.0, 1.0, 1.0])


def test_isotonic_requires_fitting_first():
    with pytest.raises(RuntimeError):
        cal.IsotonicCalibrator().transform([0.5])
