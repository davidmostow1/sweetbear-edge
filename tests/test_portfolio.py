"""Tests for correlated-portfolio staking.

The load-bearing tests here are the three that pin the optimiser to known
answers -- one position must reproduce textbook Kelly, mutually exclusive
outcomes must reproduce the simultaneous solution, and a duplicated bet must be
staked once -- plus the one that proves the point of the module: naive
per-position staking on a correlated ladder achieves strictly lower expected
log growth than the joint solution, and can drive a genuine edge negative.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweetbear.batters import (
    LEAGUE_PA_OUTCOME_RATES,
    BatterProfile,
    PitcherAllowedProfile,
    build_batter_distribution,
)
from sweetbear.edge import kelly_fraction, simultaneous_kelly
from sweetbear.portfolio import (
    ScenarioSet,
    batter_scenarios,
    combine_independent,
    count_ladder_scenarios,
    effective_position_count,
    expected_log_growth,
    independent_kelly,
    joint_kelly,
    payoff_correlation,
    return_quantile,
)
from sweetbear.strikeouts import (
    OpponentProfile,
    PitcherProfile,
    build_strikeout_distribution,
)


@pytest.fixture
def coin() -> ScenarioSet:
    """One bet: 55% at even money."""
    return ScenarioSet(
        payoffs=np.array([[1.0], [-1.0]]),
        probabilities=np.array([0.55, 0.45]),
    )


@pytest.fixture
def duplicated() -> ScenarioSet:
    """The same 55% even-money bet listed twice. One risk, two tickets."""
    return ScenarioSet(
        payoffs=np.array([[1.0, 1.0], [-1.0, -1.0]]),
        probabilities=np.array([0.55, 0.45]),
    )


@pytest.fixture
def strikeout_dist():
    return build_strikeout_distribution(
        PitcherProfile(name="Ace", strikeout_rate=0.30, expected_batters_faced=24.0),
        OpponentProfile(name="Lineup", strikeout_rate=0.22),
        league_strikeout_rate=0.22,
    )


@pytest.fixture
def batter_dist():
    return build_batter_distribution(
        BatterProfile("Bat", dict(LEAGUE_PA_OUTCOME_RATES), 4.3, 0.9),
        PitcherAllowedProfile("Opp", dict(LEAGUE_PA_OUTCOME_RATES)),
    )


# --------------------------------------------------------------------------
# ScenarioSet validation
# --------------------------------------------------------------------------


def test_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.5, 0.4]))


def test_probabilities_must_match_scenario_count():
    with pytest.raises(ValueError, match="one entry per scenario"):
        ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.3, 0.3, 0.4]))


def test_rejects_negative_probability_mass():
    with pytest.raises(ValueError, match="negative mass"):
        ScenarioSet(np.array([[1.0], [-1.0]]), np.array([1.2, -0.2]))


def test_cannot_lose_more_than_the_stake():
    with pytest.raises(ValueError, match="more than the stake"):
        ScenarioSet(np.array([[1.0], [-2.0]]), np.array([0.5, 0.5]))


def test_labels_must_match_position_count():
    with pytest.raises(ValueError, match="labels"):
        ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.5, 0.5]), labels=("a", "b"))


def test_marginals_account_for_push_mass():
    # Middle scenario is a push: neither a win nor a loss.
    s = ScenarioSet(
        payoffs=np.array([[1.0], [0.0], [-1.0]]),
        probabilities=np.array([0.5, 0.2, 0.3]),
    )
    win, lose = s.marginal_probabilities()
    assert win[0] == pytest.approx(0.5)
    assert lose[0] == pytest.approx(0.3)


# --------------------------------------------------------------------------
# The optimiser, pinned to answers that are known independently
# --------------------------------------------------------------------------


def test_single_position_reproduces_textbook_kelly(coin):
    assert joint_kelly(coin)[0] == pytest.approx(kelly_fraction(0.55, 2.0), abs=1e-6)


def test_mutually_exclusive_market_reproduces_simultaneous_kelly():
    probs = np.array([0.5, 0.3, 0.2])
    odds = np.array([2.2, 3.6, 5.5])
    payoffs = np.full((3, 3), -1.0)
    for i in range(3):
        payoffs[i, i] = odds[i] - 1.0
    scenarios = ScenarioSet(payoffs=payoffs, probabilities=probs)
    np.testing.assert_allclose(
        joint_kelly(scenarios), simultaneous_kelly(probs, odds), atol=1e-5
    )


def test_duplicated_bet_is_staked_once_in_total(duplicated, coin):
    """Two tickets on one risk must total the stake of one ticket."""
    joint = joint_kelly(duplicated)
    assert joint.sum() == pytest.approx(joint_kelly(coin)[0], abs=1e-5)


def test_naive_staking_doubles_a_duplicated_bet(duplicated):
    naive = independent_kelly(duplicated)
    assert naive.sum() == pytest.approx(2.0 * kelly_fraction(0.55, 2.0), abs=1e-9)


def test_naive_staking_drives_a_real_edge_to_negative_growth(duplicated):
    """The whole point of the module, in one assertion.

    A 55% bet at even money is genuinely profitable. Staked rung by rung as
    though the two tickets were separate risks, expected log growth goes
    negative -- the edge is real and the staking destroys it.
    """
    joint = joint_kelly(duplicated)
    naive = independent_kelly(duplicated)
    assert expected_log_growth(duplicated, joint) > 0.0
    assert expected_log_growth(duplicated, naive) < 0.0


def test_joint_growth_beats_naive_growth_on_a_correlated_ladder(strikeout_dist):
    ladder = [(4.5, "over", 2.10), (5.5, "over", 2.70), (6.5, "over", 3.60)]
    scenarios = count_ladder_scenarios(strikeout_dist.pmf, ladder)
    joint = joint_kelly(scenarios)
    naive = independent_kelly(scenarios)
    assert joint.sum() < naive.sum()
    assert expected_log_growth(scenarios, joint) > expected_log_growth(scenarios, naive)


def test_no_edge_portfolio_is_not_staked():
    # Fair price with no edge: 50% at 2.0.
    s = ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.5, 0.5]))
    assert joint_kelly(s)[0] == pytest.approx(0.0, abs=1e-6)
    assert independent_kelly(s)[0] == pytest.approx(0.0, abs=1e-9)


def test_negative_edge_portfolio_is_not_staked():
    s = ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.4, 0.6]))
    assert joint_kelly(s)[0] == pytest.approx(0.0, abs=1e-6)


def test_max_total_stake_is_respected(strikeout_dist):
    ladder = [(3.5, "over", 4.0), (4.5, "over", 5.0), (5.5, "over", 6.0)]
    scenarios = count_ladder_scenarios(strikeout_dist.pmf, ladder)
    stakes = joint_kelly(scenarios, max_total_stake=0.05)
    assert stakes.sum() <= 0.05 + 1e-6


def test_fraction_scales_stakes_linearly(coin):
    full = joint_kelly(coin)
    quarter = joint_kelly(coin, fraction=0.25)
    np.testing.assert_allclose(quarter, full * 0.25, atol=1e-6)


def test_joint_kelly_rejects_bad_fraction(coin):
    with pytest.raises(ValueError, match="fraction"):
        joint_kelly(coin, fraction=0.0)


def test_joint_kelly_rejects_bad_max_total_stake(coin):
    with pytest.raises(ValueError, match="max_total_stake"):
        joint_kelly(coin, max_total_stake=1.5)


def test_stakes_must_match_position_count(coin):
    with pytest.raises(ValueError, match="one entry per position"):
        expected_log_growth(coin, np.array([0.1, 0.2]))


def test_ruinous_stakes_have_no_growth_rate():
    """A stake vector that can lose everything gets -inf, not a finite number."""
    s = ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.55, 0.45]))
    assert expected_log_growth(s, np.array([1.0])) == float("-inf")


def test_return_quantile_is_worse_for_naive_stakes(strikeout_dist):
    ladder = [(4.5, "over", 2.10), (5.5, "over", 2.70), (6.5, "over", 3.60)]
    scenarios = count_ladder_scenarios(strikeout_dist.pmf, ladder)
    joint = joint_kelly(scenarios)
    naive = independent_kelly(scenarios)
    assert return_quantile(scenarios, naive) < return_quantile(scenarios, joint)


def test_return_quantile_rejects_bad_quantile(coin):
    with pytest.raises(ValueError, match="quantile"):
        return_quantile(coin, np.array([0.1]), quantile=1.0)


# --------------------------------------------------------------------------
# Correlation diagnostics
# --------------------------------------------------------------------------


def test_duplicated_positions_correlate_at_one(duplicated):
    assert payoff_correlation(duplicated)[0, 1] == pytest.approx(1.0)


def test_mutually_exclusive_positions_correlate_negatively():
    payoffs = np.array([[1.2, -1.0], [-1.0, 1.6]])
    s = ScenarioSet(payoffs=payoffs, probabilities=np.array([0.5, 0.5]))
    assert payoff_correlation(s)[0, 1] == pytest.approx(-1.0)


def test_effective_count_collapses_to_one_for_duplicates(duplicated):
    assert effective_position_count(duplicated) == pytest.approx(1.0, abs=1e-9)


def test_effective_count_equals_position_count_when_independent():
    rng = np.random.default_rng(11)
    one = ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.5, 0.5]))
    two = ScenarioSet(np.array([[1.0], [-1.0]]), np.array([0.5, 0.5]))
    combined = combine_independent([one, two], draws=200000, rng=rng)
    assert effective_position_count(combined) == pytest.approx(2.0, abs=0.05)


def test_ladder_positions_are_worth_far_fewer_than_their_ticket_count(strikeout_dist):
    """Five rungs of one ladder are not five positions, and the number says so."""
    ladder = [
        (3.5, "over", 1.6),
        (4.5, "over", 2.1),
        (5.5, "over", 2.7),
        (6.5, "over", 3.6),
        (7.5, "over", 5.2),
    ]
    scenarios = count_ladder_scenarios(strikeout_dist.pmf, ladder)
    assert effective_position_count(scenarios) < 2.0


# --------------------------------------------------------------------------
# Exact enumeration from a count distribution
# --------------------------------------------------------------------------


def test_ladder_scenarios_are_exact_against_the_pmf(strikeout_dist):
    scenarios = count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 2.7)])
    win, _ = scenarios.marginal_probabilities()
    assert win[0] == pytest.approx(strikeout_dist.probability_over(5.5))


def test_whole_number_line_produces_push_scenarios(strikeout_dist):
    scenarios = count_ladder_scenarios(strikeout_dist.pmf, [(6.0, "over", 2.5)])
    win, lose = scenarios.marginal_probabilities()
    push = 1.0 - win[0] - lose[0]
    assert push == pytest.approx(strikeout_dist.probability_push(6.0))
    assert push > 0.0


def test_under_side_is_the_complement_of_over(strikeout_dist):
    scenarios = count_ladder_scenarios(
        strikeout_dist.pmf, [(5.5, "over", 2.0), (5.5, "under", 2.0)]
    )
    win, _ = scenarios.marginal_probabilities()
    assert win[0] + win[1] == pytest.approx(1.0)
    # Opposite sides of one line are perfectly opposed.
    assert payoff_correlation(scenarios)[0, 1] == pytest.approx(-1.0)


def test_ladder_scenarios_recover_decimal_odds(strikeout_dist):
    scenarios = count_ladder_scenarios(
        strikeout_dist.pmf, [(4.5, "over", 2.1), (6.5, "over", 3.6)]
    )
    np.testing.assert_allclose(scenarios.decimal_odds(), [2.1, 3.6])


def test_ladder_scenarios_reject_bad_side(strikeout_dist):
    with pytest.raises(ValueError, match="over.*under"):
        count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "sideways", 2.0)])


def test_ladder_scenarios_reject_bad_odds(strikeout_dist):
    with pytest.raises(ValueError, match="greater than 1.0"):
        count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 1.0)])


def test_ladder_scenarios_reject_unnormalised_pmf():
    with pytest.raises(ValueError, match="sum to 1"):
        count_ladder_scenarios(np.array([0.3, 0.3]), [(0.5, "over", 2.0)])


def test_ladder_scenarios_require_a_position(strikeout_dist):
    with pytest.raises(ValueError, match="at least one position"):
        count_ladder_scenarios(strikeout_dist.pmf, [])


# --------------------------------------------------------------------------
# Monte Carlo scenarios for a batter game
# --------------------------------------------------------------------------


def test_batter_scenario_marginals_match_the_analytic_pmf(batter_dist):
    rng = np.random.default_rng(3)
    specs = [
        ("hits", 0.5, "over", 2.0),
        ("total_bases", 1.5, "over", 2.5),
        ("home_runs", 0.5, "over", 5.0),
    ]
    scenarios = batter_scenarios(batter_dist, specs, draws=200000, rng=rng)
    win, _ = scenarios.marginal_probabilities()
    for (stat, line, _side, _odds), simulated in zip(specs, win):
        assert simulated == pytest.approx(
            batter_dist.probability_over(stat, line), abs=0.01
        )


def test_batter_scenarios_are_cross_stat_consistent(batter_dist):
    """No draw may record zero hits and a positive total-base count.

    This is the property analytic marginals cannot deliver: it constrains the
    joint distribution, not the two margins.
    """
    rng = np.random.default_rng(5)
    scenarios = batter_scenarios(
        batter_dist,
        [("hits", 0.5, "under", 2.0), ("total_bases", 0.5, "over", 2.0)],
        draws=50000,
        rng=rng,
    )
    hitless = scenarios.payoffs[:, 0] > 0.0
    got_bases = scenarios.payoffs[:, 1] > 0.0
    assert not np.any(hitless & got_bases)


def test_batter_props_are_positively_correlated(batter_dist):
    rng = np.random.default_rng(9)
    scenarios = batter_scenarios(
        batter_dist,
        [("hits", 0.5, "over", 2.0), ("total_bases", 1.5, "over", 2.5)],
        draws=50000,
        rng=rng,
    )
    assert payoff_correlation(scenarios)[0, 1] > 0.3


def test_batter_scenarios_reject_unknown_stat(batter_dist):
    with pytest.raises(ValueError, match="unknown or unsupported stat"):
        batter_scenarios(batter_dist, [("rbi", 0.5, "over", 2.0)], draws=100)


def test_batter_scenarios_reject_bad_draws(batter_dist):
    with pytest.raises(ValueError, match="draws must be positive"):
        batter_scenarios(batter_dist, [("hits", 0.5, "over", 2.0)], draws=0)


def test_batter_scenarios_are_reproducible(batter_dist):
    specs = [("hits", 0.5, "over", 2.0)]
    a = batter_scenarios(batter_dist, specs, draws=5000, rng=np.random.default_rng(1))
    b = batter_scenarios(batter_dist, specs, draws=5000, rng=np.random.default_rng(1))
    np.testing.assert_array_equal(a.payoffs, b.payoffs)


# --------------------------------------------------------------------------
# Combining across games
# --------------------------------------------------------------------------


def test_combine_independent_concatenates_positions(strikeout_dist):
    rng = np.random.default_rng(13)
    a = count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 2.7)], "game-a")
    b = count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 2.7)], "game-b")
    combined = combine_independent([a, b], draws=20000, rng=rng)
    assert combined.n_positions == 2
    assert combined.correlation_groups == ("game-a", "game-b")


def test_combine_independent_decorrelates_across_sets(strikeout_dist):
    rng = np.random.default_rng(17)
    a = count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 2.7)], "game-a")
    b = count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 2.7)], "game-b")
    combined = combine_independent([a, b], draws=200000, rng=rng)
    assert abs(payoff_correlation(combined)[0, 1]) < 0.02


def test_combine_independent_preserves_within_set_correlation(strikeout_dist):
    """Joining across games must not destroy the structure inside a game."""
    rng = np.random.default_rng(19)
    ladder = count_ladder_scenarios(
        strikeout_dist.pmf, [(4.5, "over", 2.1), (5.5, "over", 2.7)], "game-a"
    )
    other = count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 2.7)], "game-b")
    exact = payoff_correlation(ladder)[0, 1]
    combined = combine_independent([ladder, other], draws=200000, rng=rng)
    assert payoff_correlation(combined)[0, 1] == pytest.approx(exact, abs=0.02)


def test_combine_independent_requires_a_set():
    with pytest.raises(ValueError, match="at least one scenario set"):
        combine_independent([])


def test_sample_rejects_bad_draws(coin):
    with pytest.raises(ValueError, match="draws must be positive"):
        coin.sample(0)


def test_two_independent_edges_are_staked_more_than_one_correlated_pair(strikeout_dist):
    """Diversification is worth something, and the optimiser should find it.

    The same two edges, once as a correlated ladder and once as separate games,
    should be staked more heavily in total when they genuinely diversify.
    """
    rng = np.random.default_rng(23)
    ladder = count_ladder_scenarios(
        strikeout_dist.pmf, [(4.5, "over", 2.10), (5.5, "over", 2.70)], "one-game"
    )
    a = count_ladder_scenarios(strikeout_dist.pmf, [(4.5, "over", 2.10)], "game-a")
    b = count_ladder_scenarios(strikeout_dist.pmf, [(5.5, "over", 2.70)], "game-b")
    spread = combine_independent([a, b], draws=100000, rng=rng)
    assert joint_kelly(spread).sum() > joint_kelly(ladder).sum()
