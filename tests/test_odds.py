import numpy as np
import pytest

from sweetbear import odds


def test_american_to_decimal_known_values():
    assert odds.american_to_decimal(100) == pytest.approx(2.0)
    assert odds.american_to_decimal(-100) == pytest.approx(2.0)
    assert odds.american_to_decimal(150) == pytest.approx(2.5)
    assert odds.american_to_decimal(-200) == pytest.approx(1.5)
    assert odds.american_to_decimal(-110) == pytest.approx(1.9090909, abs=1e-6)


def test_decimal_to_american_known_values():
    assert odds.decimal_to_american(2.5) == pytest.approx(150.0)
    assert odds.decimal_to_american(1.5) == pytest.approx(-200.0)
    assert odds.decimal_to_american(2.0) == pytest.approx(100.0)


def test_american_decimal_roundtrip():
    american = np.array([-500, -220, -110, 100, 145, 900], dtype=float)
    back = odds.decimal_to_american(odds.american_to_decimal(american))
    assert np.allclose(back, american)


def test_invalid_american_odds_rejected():
    with pytest.raises(ValueError):
        odds.american_to_decimal(50)
    with pytest.raises(ValueError):
        odds.american_to_decimal(0)


def test_invalid_decimal_odds_rejected():
    with pytest.raises(ValueError):
        odds.decimal_to_implied(1.0)
    with pytest.raises(ValueError):
        odds.decimal_to_implied(0.5)


def test_fractional_conversions():
    assert odds.fractional_to_decimal(5, 2) == pytest.approx(3.5)
    assert odds.fractional_to_decimal(1, 4) == pytest.approx(1.25)
    assert odds.decimal_to_fractional(3.5) == [(5, 2)]
    assert odds.decimal_to_fractional([1.25, 3.0]) == [(1, 4), (2, 1)]


def test_implied_probability_roundtrip():
    probs = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    assert np.allclose(odds.decimal_to_implied(odds.implied_to_decimal(probs)), probs)


def test_overround_and_vig_on_standard_juice():
    market = odds.american_to_decimal([-110, -110])
    assert odds.overround(market) == pytest.approx(1.0476190, abs=1e-6)
    assert odds.vig(market) == pytest.approx(0.0454545, abs=1e-6)


def test_fair_market_has_no_vig():
    market = np.array([2.0, 2.0])
    assert odds.overround(market) == pytest.approx(1.0)
    assert odds.vig(market) == pytest.approx(0.0)


def test_overround_supports_batched_markets():
    markets = np.array([[2.0, 2.0], [1.9090909, 1.9090909]])
    book = odds.overround(markets, axis=-1)
    assert book.shape == (2,)
    assert book[0] == pytest.approx(1.0)
    assert book[1] > 1.0
