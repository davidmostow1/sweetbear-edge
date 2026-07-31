"""Tests for the pitcher strikeout distribution.

The tests that matter most are the coherence ones: a single distribution must
make self-contradictory quotes impossible, and the whole ladder must be
recognised as one correlated observation rather than many independent ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweetbear.backtest import run_backtest, flat_stake
from sweetbear.strikeouts import (
    OpponentProfile,
    PitcherProfile,
    StrikeoutDistribution,
    batters_faced_pmf,
    build_strikeout_distribution,
    matchup_strikeout_rate,
    standard_ladder,
)


@pytest.fixture
def ace() -> PitcherProfile:
    return PitcherProfile(
        name="Ace", strikeout_rate=0.30, expected_batters_faced=24.0
    )


@pytest.fixture
def lineup() -> OpponentProfile:
    return OpponentProfile(name="Lineup", strikeout_rate=0.22)


@pytest.fixture
def dist(ace, lineup) -> StrikeoutDistribution:
    return build_strikeout_distribution(ace, lineup, league_strikeout_rate=0.22)


# --- matchup rate ----------------------------------------------------------


def test_league_average_opponent_returns_pitcher_rate():
    """Facing a league-average lineup, the pitcher's own rate is unchanged."""
    rate = matchup_strikeout_rate(0.30, 0.22, 0.22)
    assert rate == pytest.approx(0.30)


def test_high_strikeout_lineup_raises_the_rate():
    baseline = matchup_strikeout_rate(0.25, 0.22, 0.22)
    whiffy = matchup_strikeout_rate(0.25, 0.30, 0.22)
    contact = matchup_strikeout_rate(0.25, 0.15, 0.22)
    assert contact < baseline < whiffy


def test_matchup_rate_stays_within_bounds():
    """Extreme inputs must not produce a probability outside (0, 1)."""
    for p in (0.01, 0.45, 0.60):
        for b in (0.01, 0.40, 0.55):
            rate = matchup_strikeout_rate(p, b, 0.22)
            assert 0.0 < rate < 1.0


def test_matchup_rate_rejects_degenerate_inputs():
    with pytest.raises(ValueError):
        matchup_strikeout_rate(0.0, 0.22, 0.22)
    with pytest.raises(ValueError):
        matchup_strikeout_rate(0.3, 1.0, 0.22)


# --- batters faced ---------------------------------------------------------


def test_batters_faced_pmf_is_normalised_and_centred():
    pmf = batters_faced_pmf(mean=24.0, sd=3.5)
    assert pmf.sum() == pytest.approx(1.0)
    mean = float(np.dot(np.arange(pmf.size), pmf))
    assert mean == pytest.approx(24.0, abs=0.2)


def test_batters_faced_pmf_rejects_bad_parameters():
    with pytest.raises(ValueError):
        batters_faced_pmf(mean=0.0, sd=3.0)
    with pytest.raises(ValueError):
        batters_faced_pmf(mean=24.0, sd=0.0)


# --- the distribution ------------------------------------------------------


def test_distribution_is_a_valid_pmf(dist):
    assert dist.pmf.sum() == pytest.approx(1.0)
    assert np.all(dist.pmf >= 0.0)


def test_expected_strikeouts_matches_rate_times_batters(ace, lineup, dist):
    """E[K] = E[BF] * p, since K | BF is binomial."""
    rate = matchup_strikeout_rate(0.30, 0.22, 0.22)
    assert dist.mean == pytest.approx(24.0 * rate, rel=0.02)


def test_compounding_widens_the_distribution(ace, lineup):
    """Random workload must produce more spread than a fixed one.

    This is the whole reason for compounding. A fixed batters-faced count
    understates variance, which makes alternate lines look mispriced.
    """
    compound = build_strikeout_distribution(ace, lineup)

    fixed_pmf = np.zeros(46)
    fixed_pmf[24] = 1.0
    fixed = build_strikeout_distribution(ace, lineup, faced_pmf=fixed_pmf)

    assert compound.variance > fixed.variance
    assert compound.mean == pytest.approx(fixed.mean, rel=0.02)


def test_higher_strikeout_pitcher_shifts_distribution_right(lineup):
    weak = build_strikeout_distribution(
        PitcherProfile("Weak", 0.15, 24.0), lineup
    )
    strong = build_strikeout_distribution(
        PitcherProfile("Strong", 0.35, 24.0), lineup
    )
    assert strong.mean > weak.mean
    assert strong.probability_over(6.5) > weak.probability_over(6.5)


# --- coherence: the reason for one distribution ----------------------------


def test_over_probability_is_monotone_across_the_ladder(dist):
    """Higher lines are strictly harder. Separate models can violate this."""
    lines = standard_ladder(dist.mean)
    overs = [dist.probability_over(x) for x in lines]
    assert all(a >= b for a, b in zip(overs, overs[1:]))


def test_under_probability_is_monotone_across_the_ladder(dist):
    lines = standard_ladder(dist.mean)
    unders = [dist.probability_under(x) for x in lines]
    assert all(a <= b for a, b in zip(unders, unders[1:]))


def test_over_under_and_push_partition_probability(dist):
    for line in (4.0, 4.5, 5.0, 6.0, 7.5):
        total = (
            dist.probability_over(line)
            + dist.probability_under(line)
            + dist.probability_push(line)
        )
        assert total == pytest.approx(1.0)


def test_half_lines_never_push(dist):
    for line in (4.5, 5.5, 6.5):
        assert dist.probability_push(line) == 0.0


def test_whole_lines_can_push(dist):
    assert dist.probability_push(6.0) > 0.0


# --- quoting ---------------------------------------------------------------


def test_fair_odds_on_a_half_line_invert_the_probability(dist):
    quote = dist.quote(5.5, "over")
    assert quote.push_probability == 0.0
    assert quote.conditional_probability == pytest.approx(quote.win_probability)
    assert quote.fair_decimal_odds == pytest.approx(1.0 / quote.win_probability)


def test_whole_line_price_is_conditional_on_no_push(dist):
    """A push refunds the stake, so it must not count as a loss."""
    quote = dist.quote(6.0, "over")
    assert quote.push_probability > 0.0
    assert quote.conditional_probability > quote.win_probability
    resolved = quote.win_probability + dist.probability_under(6.0)
    assert quote.conditional_probability == pytest.approx(
        quote.win_probability / resolved
    )


def test_both_sides_of_a_line_are_conditionally_complementary(dist):
    over = dist.quote(6.0, "over")
    under = dist.quote(6.0, "under")
    assert over.conditional_probability + under.conditional_probability == (
        pytest.approx(1.0)
    )


def test_quote_rejects_unknown_side(dist):
    with pytest.raises(ValueError):
        dist.quote(5.5, "middle")


def test_ladder_shares_one_correlation_group(dist):
    """The central safeguard: a ladder is one observation, not many."""
    quotes = dist.ladder()
    assert len(quotes) > 8
    assert len({q.correlation_group for q in quotes}) == 1


# --- market comparison -----------------------------------------------------


def test_no_edge_when_market_matches_the_model(dist):
    """Quoting the model's own fair prices with no margin gives zero EV."""
    line = 5.5
    p_over = dist.probability_over(line)
    market = {line: (1.0 / p_over, 1.0 / (1.0 - p_over))}
    quotes = dist.price_against_market(market)
    for q in quotes:
        assert q.expected_value == pytest.approx(0.0, abs=1e-9)


def test_vig_makes_both_sides_negative(dist):
    """A real book charges margin, so betting both sides must lose."""
    line = 5.5
    p_over = dist.probability_over(line)
    # 4.5% margin split evenly across the two sides
    market = {line: (1.0 / (p_over * 1.045), 1.0 / ((1.0 - p_over) * 1.045))}
    quotes = dist.price_against_market(market)
    assert all(q.expected_value < 0.0 for q in quotes)
    assert not any(q.has_edge for q in quotes)


def test_edge_appears_when_the_market_underrates_the_over(dist):
    """A real book with margin that simply disagrees with the model.

    The book still holds a 4.5% margin -- this is not an arbitrage -- but it
    prices the over 15 points below where the model has it.
    """
    line = 5.5
    model_p = dist.probability_over(line)
    market_p = model_p - 0.15
    assert 0.0 < market_p < 1.0

    market = {
        line: (1.0 / (market_p * 1.045), 1.0 / ((1.0 - market_p) * 1.045))
    }
    # The book is overround, as any real book is.
    assert sum(1.0 / o for o in market[line]) > 1.0

    quotes = dist.price_against_market(market)
    over = next(q for q in quotes if q.side == "over")
    under = next(q for q in quotes if q.side == "under")
    assert over.expected_value > 0.0
    assert over.has_edge
    assert not under.has_edge  # the edge is one-sided, as it must be


def test_market_fair_probabilities_are_devigged(dist):
    """De-vigged probabilities sum to one; raw implied ones would exceed it."""
    market = {5.5: (1.90, 1.90)}
    quotes = dist.price_against_market(market)
    total = sum(q.market_fair_probability for q in quotes)
    assert total == pytest.approx(1.0)
    raw = sum(1.0 / q.market_decimal_odds for q in quotes)
    assert raw > 1.0


def test_push_is_not_treated_as_a_loss(dist):
    """EV on a whole line must beat the naive calculation that ignores pushes."""
    line = 6.0
    market = {line: (2.0, 2.0)}
    quote = next(
        q for q in dist.price_against_market(market) if q.side == "over"
    )
    naive = quote.win_probability * 2.0 - 1.0
    assert quote.expected_value > naive
    assert quote.push_probability > 0.0


def test_price_against_market_requires_both_sides(dist):
    with pytest.raises(ValueError):
        dist.price_against_market({5.5: (1.9,)})


# --- integration with the inference layer ----------------------------------


def test_ladder_backtests_as_one_clustered_observation(dist):
    """A full ladder across many starts must not inflate the sample size."""
    rng = np.random.default_rng(0)
    bets = []
    t = 0
    for start in range(40):
        d = build_strikeout_distribution(
            PitcherProfile(f"P{start}", 0.30, 24.0),
            OpponentProfile("L", 0.22),
            correlation_group=f"start-{start}",
        )
        actual = int(rng.choice(np.arange(d.pmf.size), p=d.pmf))
        for line in (4.5, 5.5, 6.5):
            quote = d.price_against_market({line: (2.0, 2.0)})[0]
            bets.append(
                quote.to_bet(timestamp=t, outcome=int(actual > line))
            )
            t += 1

    result = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    sig = result.significance(n_resamples=1)

    assert result.n_bets == 120
    assert sig.n_clusters == 40  # one per start, not one per ticket
    assert sig.effective_sample_size < result.n_bets


def test_to_bet_requires_a_market_price(dist):
    with pytest.raises(ValueError):
        dist.quote(5.5, "over").to_bet(timestamp=0, outcome=1)


def test_summary_lists_the_ladder(dist):
    text = dist.summary()
    assert "expected K" in text
    assert "correlation group" in text
