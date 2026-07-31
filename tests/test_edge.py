import numpy as np
import pytest

from sweetbear import edge


def test_expected_value_known_values():
    assert edge.expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert edge.expected_value(0.6, 2.0) == pytest.approx(0.2)
    assert edge.expected_value(0.5, 1.9090909) == pytest.approx(-0.0454545, abs=1e-6)


def test_breakeven_probability():
    assert edge.breakeven_probability(2.0) == pytest.approx(0.5)
    assert edge.breakeven_probability(1.9090909) == pytest.approx(0.5238095, abs=1e-6)


def test_expected_value_rejects_bad_input():
    with pytest.raises(ValueError):
        edge.expected_value(0.5, 1.0)
    with pytest.raises(ValueError):
        edge.expected_value(1.5, 2.0)


def test_kelly_known_values():
    assert edge.kelly_fraction(0.6, 2.0) == pytest.approx(0.2)
    assert edge.kelly_fraction(0.55, 3.0) == pytest.approx((0.55 * 3 - 1) / 2)


def test_kelly_is_zero_without_an_edge():
    assert edge.kelly_fraction(0.5, 2.0) == pytest.approx(0.0)
    assert edge.kelly_fraction(0.4, 2.0) == pytest.approx(0.0)
    assert edge.kelly_fraction(0.5, 1.9090909) == pytest.approx(0.0)


def test_fractional_kelly_scales_and_caps():
    assert edge.fractional_kelly(0.6, 2.0, fraction=0.25, cap=1.0) == pytest.approx(0.05)
    assert edge.fractional_kelly(0.9, 5.0, fraction=1.0, cap=0.05) == pytest.approx(0.05)


def test_fractional_kelly_validates_parameters():
    with pytest.raises(ValueError):
        edge.fractional_kelly(0.6, 2.0, fraction=0.0)
    with pytest.raises(ValueError):
        edge.fractional_kelly(0.6, 2.0, fraction=1.5)
    with pytest.raises(ValueError):
        edge.fractional_kelly(0.6, 2.0, cap=0.0)


def test_growth_rate_peaks_at_full_kelly():
    p, d = 0.6, 2.0
    f_star = float(edge.kelly_fraction(p, d))
    grid = np.linspace(0.001, 0.6, 600)
    growth = edge.growth_rate(p, d, grid)
    assert grid[int(np.argmax(growth))] == pytest.approx(f_star, abs=0.005)


def test_growth_rate_crosses_zero_just_below_double_kelly():
    p, d = 0.6, 2.0
    f_star = float(edge.kelly_fraction(p, d))
    grid = np.linspace(1.0, 2.5, 1501) * f_star
    growth = edge.growth_rate(p, d, grid)
    crossing = grid[np.argmax(growth < 0.0)] / f_star
    assert 1.8 < crossing < 2.0
    assert edge.growth_rate(p, d, 2.2 * f_star) < 0.0


def test_growth_rate_asymmetry_favours_underbetting():
    p, d = 0.6, 2.0
    f_star = float(edge.kelly_fraction(p, d))
    under = edge.growth_rate(p, d, 0.5 * f_star)
    over = edge.growth_rate(p, d, 1.5 * f_star)
    assert under > over


def test_shrink_toward_market_endpoints():
    assert edge.shrink_toward_market(0.7, 0.5, confidence=1.0) == pytest.approx(0.7)
    assert edge.shrink_toward_market(0.7, 0.5, confidence=0.0) == pytest.approx(0.5)


def test_shrink_toward_market_lands_between_the_two():
    blended = float(edge.shrink_toward_market(0.70, 0.50, confidence=0.5))
    assert 0.5 < blended < 0.7
    # Geometric mean of odds ratios: sqrt((0.7/0.3) * (0.5/0.5)).
    expected_ratio = np.sqrt((0.7 / 0.3) * 1.0)
    assert blended == pytest.approx(expected_ratio / (1 + expected_ratio))


def test_shrink_toward_market_validates_confidence():
    with pytest.raises(ValueError):
        edge.shrink_toward_market(0.7, 0.5, confidence=1.5)


def test_simultaneous_kelly_matches_scalar_kelly_for_one_outcome():
    stakes = edge.simultaneous_kelly([0.6], [2.0])
    assert stakes[0] == pytest.approx(0.2, abs=1e-4)


def _log_growth(p, d, f):
    residual = 1.0 - np.sum(p)
    wealth = 1.0 - f.sum() + f * d
    return float(p @ np.log(wealth) + residual * np.log(1.0 - f.sum()))


def test_simultaneous_kelly_beats_naive_per_bet_kelly():
    p = np.array([0.40, 0.35])
    d = np.array([3.0, 3.2])
    joint = edge.simultaneous_kelly(p, d)
    naive = edge.kelly_fraction(p, d)
    assert np.all(joint > 0.0)
    assert _log_growth(p, d, joint) > _log_growth(p, d, naive)


def test_simultaneous_kelly_stakes_more_than_naive_on_a_hedged_market():
    # Mutually exclusive outcomes offset each other, so the joint optimum can
    # carry more total exposure than sizing each leg in isolation.
    p = np.array([0.40, 0.35])
    d = np.array([3.0, 3.2])
    assert edge.simultaneous_kelly(p, d).sum() > edge.kelly_fraction(p, d).sum()


def test_simultaneous_kelly_declines_a_market_with_no_edge():
    stakes = edge.simultaneous_kelly([0.5, 0.5], [1.9090909, 1.9090909])
    assert stakes == pytest.approx([0.0, 0.0], abs=1e-6)


def test_simultaneous_kelly_respects_the_exposure_cap():
    stakes = edge.simultaneous_kelly([0.5, 0.45], [3.0, 3.0], max_total_stake=0.1)
    assert stakes.sum() <= 0.1 + 1e-9


def test_simultaneous_kelly_validates_input():
    with pytest.raises(ValueError):
        edge.simultaneous_kelly([0.6, 0.6], [2.0, 2.0])
    with pytest.raises(ValueError):
        edge.simultaneous_kelly([0.5], [2.0, 2.0])
    with pytest.raises(ValueError):
        edge.simultaneous_kelly([0.5], [1.0])


def test_closing_line_value():
    # Took +150 (decimal 2.5) on something the market closed at a fair 45%.
    assert edge.closing_line_value(2.5, 0.45) == pytest.approx(0.125)
    # Took a price the market then moved against you.
    assert edge.closing_line_value(2.5, 0.35) == pytest.approx(-0.125)


def test_betting_the_closing_price_yields_exactly_minus_the_vig():
    """The invariant that makes CLV meaningful: with no line movement there is
    no edge to find, and CLV collapses to the book's margin. A CLV number that
    hovers at ``-vig`` means the prices being compared are the same prices."""
    from sweetbear import devig, odds

    market = odds.american_to_decimal([-110, -110])
    fair = devig(market, method="shin")
    clv = edge.closing_line_value(market, fair)
    assert clv == pytest.approx(-odds.vig(market), abs=1e-9)


def test_closing_line_value_validates_input():
    with pytest.raises(ValueError):
        edge.closing_line_value(1.0, 0.5)
    with pytest.raises(ValueError):
        edge.closing_line_value(2.0, 1.0)
