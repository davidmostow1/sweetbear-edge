import numpy as np
import pytest

from sweetbear import margin as dv
from sweetbear import odds

STANDARD_JUICE = odds.american_to_decimal([-110, -110])
LOPSIDED = odds.american_to_decimal([-400, 320])
THREE_WAY = np.array([1.85, 3.60, 4.50])


@pytest.mark.parametrize("method", dv.METHODS)
def test_all_methods_produce_a_proper_distribution(method):
    for market in (STANDARD_JUICE, LOPSIDED, THREE_WAY):
        p = dv.devig(market, method=method)
        assert np.all(p > 0.0)
        assert np.all(p < 1.0)
        assert p.sum() == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("method", dv.METHODS)
def test_symmetric_market_splits_evenly(method):
    p = dv.devig(STANDARD_JUICE, method=method)
    assert p == pytest.approx([0.5, 0.5], abs=1e-9)


@pytest.mark.parametrize("method", dv.METHODS)
def test_already_fair_market_is_left_alone(method):
    p = dv.devig(np.array([4.0, 4.0, 2.0]), method=method)
    assert p == pytest.approx([0.25, 0.25, 0.5], abs=1e-9)


@pytest.mark.parametrize("method", dv.METHODS)
def test_devig_lowers_every_probability_relative_to_raw(method):
    raw = odds.decimal_to_implied(LOPSIDED)
    fair = dv.devig(LOPSIDED, method=method)
    assert np.all(fair < raw)


def test_devig_preserves_ordering_of_outcomes():
    for method in dv.METHODS:
        fair = dv.devig(THREE_WAY, method=method)
        assert np.all(np.diff(fair) < 0)


def test_shin_assigns_more_margin_to_the_longshot_than_multiplicative():
    mult = dv.devig(LOPSIDED, method="multiplicative")
    shin = dv.devig(LOPSIDED, method="shin")
    # Favourite is index 0. Shin strips proportionally more from the longshot,
    # so the favourite retains more probability mass.
    assert shin[0] > mult[0]
    assert shin[1] < mult[1]


def test_power_exponent_exceeds_one_for_an_overround_book():
    raw = odds.decimal_to_implied(THREE_WAY)
    k = dv.power_exponent(raw)
    assert k > 1.0
    assert np.sum(raw**k) == pytest.approx(1.0, abs=1e-9)


def test_shin_z_is_zero_for_a_fair_book_and_positive_otherwise():
    fair = odds.decimal_to_implied(np.array([2.0, 2.0]))
    assert dv.shin_z(fair) == pytest.approx(0.0, abs=1e-9)

    juiced = odds.decimal_to_implied(THREE_WAY)
    z = dv.shin_z(juiced)
    assert 0.0 < z < 1.0


def test_shin_z_rises_with_the_book_margin():
    light = odds.decimal_to_implied(np.array([1.98, 1.98]))
    heavy = odds.decimal_to_implied(np.array([1.75, 1.75]))
    assert dv.shin_z(heavy) > dv.shin_z(light)


def test_additive_refuses_impossible_markets():
    # A heavy favourite alongside a big longshot: the longshot's share of the
    # margin exceeds its raw price, so additive is not applicable.
    market = np.array([1.2, 3.0, 400.0])
    with pytest.raises(ValueError, match="Additive"):
        dv.devig(market, method="additive")


def test_shin_rejects_an_underround_book():
    underround = odds.decimal_to_implied(np.array([2.2, 2.2]))
    with pytest.raises(ValueError, match="at least 1.0"):
        dv.devig_probabilities(underround, method="shin")


def test_multiplicative_handles_an_underround_book():
    underround = np.array([2.2, 2.2])
    p = dv.devig(underround, method="multiplicative")
    assert p == pytest.approx([0.5, 0.5])


def test_batched_markets_are_devigged_row_wise():
    markets = np.array([[1.9090909, 1.9090909], [1.25, 4.5]])
    fair = dv.devig(markets, method="shin")
    assert fair.shape == (2, 2)
    assert np.allclose(fair.sum(axis=1), 1.0)
    assert fair[0] == pytest.approx([0.5, 0.5], abs=1e-9)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown de-vig method"):
        dv.devig(STANDARD_JUICE, method="wishful-thinking")
