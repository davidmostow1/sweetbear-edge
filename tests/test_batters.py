"""Tests for the batter plate-appearance distribution.

As with the strikeout model, the coherence tests matter most: every prop must
come from the same per-PA distribution, so they cannot contradict each other,
and the whole batter-game must be recognised as one correlated observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweetbear.backtest import flat_stake, run_backtest
from sweetbear.batters import (
    CATEGORIES,
    LEAGUE_PA_OUTCOME_RATES,
    STAT_VALUE_MAPS,
    BatterOutcomeDistribution,
    BatterProfile,
    PitcherAllowedProfile,
    build_batter_distribution,
    matchup_batter_rates,
    plate_appearances_pmf,
)
from sweetbear.strikeouts import standard_ladder

GOOD_HITTER_RATES = {
    "strikeout": 0.16,
    "walk": 0.10,
    "hbp": 0.01,
    "single": 0.16,
    "double": 0.06,
    "triple": 0.005,
    "home_run": 0.045,
    "out_in_play": 0.46,
}


@pytest.fixture
def hitter() -> BatterProfile:
    return BatterProfile(name="Slugger", rates=GOOD_HITTER_RATES)


@pytest.fixture
def pitcher() -> PitcherAllowedProfile:
    return PitcherAllowedProfile(name="Arm", rates=dict(LEAGUE_PA_OUTCOME_RATES))


@pytest.fixture
def dist(hitter, pitcher) -> BatterOutcomeDistribution:
    return build_batter_distribution(hitter, pitcher)


# --- rate validation ---------------------------------------------------


def test_league_rates_sum_to_one():
    assert sum(LEAGUE_PA_OUTCOME_RATES.values()) == pytest.approx(1.0)


def test_batter_profile_rejects_rates_not_summing_to_one():
    bad = dict(GOOD_HITTER_RATES)
    bad["walk"] += 0.1
    with pytest.raises(ValueError):
        BatterProfile(name="Bad", rates=bad)


def test_batter_profile_rejects_missing_category():
    bad = dict(GOOD_HITTER_RATES)
    del bad["triple"]
    with pytest.raises(ValueError):
        BatterProfile(name="Bad", rates=bad)


def test_batter_profile_rejects_unknown_category():
    bad = dict(GOOD_HITTER_RATES)
    bad["sacrifice_fly"] = 0.01
    with pytest.raises(ValueError):
        BatterProfile(name="Bad", rates=bad)


def test_batter_profile_rejects_nonpositive_pa():
    with pytest.raises(ValueError):
        BatterProfile(name="X", rates=GOOD_HITTER_RATES, expected_plate_appearances=0.0)


# --- matchup combination ------------------------------------------------


def test_league_average_pitcher_returns_batter_rates_unchanged(hitter):
    combined = matchup_batter_rates(
        hitter.rates, dict(LEAGUE_PA_OUTCOME_RATES), LEAGUE_PA_OUTCOME_RATES
    )
    for cat in CATEGORIES:
        assert combined[cat] == pytest.approx(hitter.rates[cat], abs=1e-9)


def test_combined_rates_always_sum_to_one(hitter):
    tough_pitcher = {
        "strikeout": 0.30,
        "walk": 0.06,
        "hbp": 0.01,
        "single": 0.12,
        "double": 0.03,
        "triple": 0.002,
        "home_run": 0.018,
        "out_in_play": 0.46,
    }
    combined = matchup_batter_rates(hitter.rates, tough_pitcher, LEAGUE_PA_OUTCOME_RATES)
    assert sum(combined.values()) == pytest.approx(1.0)


def test_high_strikeout_pitcher_raises_batter_strikeout_rate(hitter):
    whiffy_pitcher = dict(LEAGUE_PA_OUTCOME_RATES)
    whiffy_pitcher["strikeout"] = 0.32
    whiffy_pitcher["out_in_play"] = (
        LEAGUE_PA_OUTCOME_RATES["out_in_play"]
        - (whiffy_pitcher["strikeout"] - LEAGUE_PA_OUTCOME_RATES["strikeout"])
    )
    combined = matchup_batter_rates(hitter.rates, whiffy_pitcher, LEAGUE_PA_OUTCOME_RATES)
    assert combined["strikeout"] > hitter.rates["strikeout"]


def test_matchup_rejects_mismatched_categories(hitter):
    bad = dict(LEAGUE_PA_OUTCOME_RATES)
    del bad["triple"]
    with pytest.raises(ValueError):
        matchup_batter_rates(hitter.rates, bad, LEAGUE_PA_OUTCOME_RATES)


# --- plate appearance count ---------------------------------------------


def test_plate_appearances_pmf_normalised_and_centred():
    pmf = plate_appearances_pmf(mean=4.3, sd=0.9)
    assert pmf.sum() == pytest.approx(1.0)
    mean = float(np.dot(np.arange(pmf.size), pmf))
    assert mean == pytest.approx(4.3, abs=0.15)


def test_plate_appearances_pmf_rejects_bad_params():
    with pytest.raises(ValueError):
        plate_appearances_pmf(mean=0.0, sd=1.0)
    with pytest.raises(ValueError):
        plate_appearances_pmf(mean=4.0, sd=0.0)


# --- the distribution -----------------------------------------------------


def test_all_supported_stats_are_valid_pmfs(dist):
    for stat in STAT_VALUE_MAPS:
        pmf = dist.pmf_for(stat)
        assert pmf.sum() == pytest.approx(1.0)
        assert np.all(pmf >= 0.0)


def test_hits_mean_matches_hand_calculation(dist, hitter):
    """E[hits] = E[PA] * P(hit per PA), since hits are a compound binomial."""
    p_hit = sum(
        hitter.rates[c] for c in ("single", "double", "triple", "home_run")
    )
    expected = dist.expected_plate_appearances * p_hit
    assert dist.mean("hits") == pytest.approx(expected, rel=0.02)


def test_total_bases_mean_matches_hand_calculation(dist, hitter):
    expected_per_pa = (
        hitter.rates["single"]
        + 2 * hitter.rates["double"]
        + 3 * hitter.rates["triple"]
        + 4 * hitter.rates["home_run"]
    )
    expected = dist.expected_plate_appearances * expected_per_pa
    assert dist.mean("total_bases") == pytest.approx(expected, rel=0.02)


def test_pmf_for_rejects_runs_and_rbi(dist):
    """The explicit guardrail: these require game-state coupling and must not
    be faked from isolated batter rates."""
    with pytest.raises(ValueError):
        dist.pmf_for("runs")
    with pytest.raises(ValueError):
        dist.pmf_for("rbi")
    assert "runs" not in STAT_VALUE_MAPS
    assert "rbi" not in STAT_VALUE_MAPS


def test_compounding_widens_hits_distribution(hitter, pitcher):
    """Random PA count must widen the distribution relative to a fixed count,
    for the same reason it does in the strikeout model."""
    compound = build_batter_distribution(hitter, pitcher)

    fixed_pmf = np.zeros(10)
    fixed_pmf[4] = 1.0
    fixed = build_batter_distribution(hitter, pitcher, pa_pmf=fixed_pmf)

    compound_pmf = compound.pmf_for("hits")
    fixed_pmf_hits = fixed.pmf_for("hits")

    def variance(pmf):
        k = np.arange(pmf.size)
        mean = np.dot(k, pmf)
        return np.dot((k - mean) ** 2, pmf)

    assert variance(compound_pmf) > variance(fixed_pmf_hits)


# --- coherence: the reason for one distribution ----------------------------


def test_over_probability_monotone_across_ladder_for_every_stat(dist):
    for stat in STAT_VALUE_MAPS:
        lines = standard_ladder(dist.mean(stat), span=3)
        overs = [dist.probability_over(stat, x) for x in lines if x >= 0]
        assert all(a >= b - 1e-12 for a, b in zip(overs, overs[1:]))


def test_over_under_push_partition_probability(dist):
    for stat in ("hits", "total_bases", "home_runs", "walks", "strikeouts"):
        for line in (0.5, 1.0, 1.5, 2.0):
            total = (
                dist.probability_over(stat, line)
                + dist.probability_under(stat, line)
                + dist.probability_push(stat, line)
            )
            assert total == pytest.approx(1.0)


def test_half_lines_never_push(dist):
    for stat in STAT_VALUE_MAPS:
        assert dist.probability_push(stat, 1.5) == 0.0


def test_hits_and_total_bases_are_mechanically_linked(dist):
    """A game with zero hits must have zero total bases -- both come from the
    same underlying per-PA outcome, so P(hits=0) == P(total_bases=0)."""
    hits_pmf = dist.pmf_for("hits")
    tb_pmf = dist.pmf_for("total_bases")
    assert hits_pmf[0] == pytest.approx(tb_pmf[0])


# --- quoting ----------------------------------------------------------------


def test_quote_rejects_unknown_side(dist):
    with pytest.raises(ValueError):
        dist.quote("hits", 1.5, "sideways")


def test_whole_line_conditional_probability_excludes_push(dist):
    quote = dist.quote("hits", 1.0, "over")
    assert quote.push_probability > 0.0
    resolved = quote.win_probability + dist.probability_under("hits", 1.0)
    assert quote.conditional_probability == pytest.approx(
        quote.win_probability / resolved
    )


def test_both_sides_of_a_line_are_conditionally_complementary(dist):
    over = dist.quote("total_bases", 1.0, "over")
    under = dist.quote("total_bases", 1.0, "under")
    assert over.conditional_probability + under.conditional_probability == (
        pytest.approx(1.0)
    )


def test_ladder_for_one_stat_shares_one_correlation_group(dist):
    quotes = dist.ladder("total_bases")
    assert len(quotes) > 4
    assert len({q.correlation_group for q in quotes}) == 1


def test_different_stats_from_one_batter_game_share_correlation_group(dist):
    """The stronger claim: hits, total bases, and home runs from one
    batter-game are the same observation, not three."""
    groups = set()
    for stat in ("hits", "total_bases", "home_runs"):
        groups.update(q.correlation_group for q in dist.ladder(stat))
    assert len(groups) == 1


# --- market comparison -------------------------------------------------


def test_no_edge_when_market_matches_model(dist):
    line = 1.5
    p_over = dist.probability_over("hits", line)
    market = {line: (1.0 / p_over, 1.0 / (1.0 - p_over))}
    for q in dist.price_against_market("hits", market):
        assert q.expected_value == pytest.approx(0.0, abs=1e-9)


def test_vig_makes_both_sides_negative(dist):
    line = 1.5
    p_over = dist.probability_over("total_bases", line)
    market = {
        line: (1.0 / (p_over * 1.045), 1.0 / ((1.0 - p_over) * 1.045))
    }
    quotes = dist.price_against_market("total_bases", market)
    assert all(q.expected_value < 0.0 for q in quotes)


def test_price_against_market_requires_both_sides(dist):
    with pytest.raises(ValueError):
        dist.price_against_market("hits", {1.5: (1.9,)})


def test_market_probabilities_are_devigged(dist):
    market = {1.5: (1.90, 1.90)}
    quotes = dist.price_against_market("hits", market)
    total = sum(q.market_fair_probability for q in quotes)
    assert total == pytest.approx(1.0)


# --- integration with the inference layer -----------------------------


def test_multi_stat_ladder_backtests_as_one_clustered_observation():
    """Betting hits, total bases, and home runs on the same 40 batter-games
    must not inflate the effective sample size to 3x the real count."""
    rng = np.random.default_rng(0)
    bets = []
    t = 0
    for game in range(40):
        d = build_batter_distribution(
            BatterProfile(f"B{game}", rates=GOOD_HITTER_RATES),
            PitcherAllowedProfile("P", rates=dict(LEAGUE_PA_OUTCOME_RATES)),
            correlation_group=f"game-{game}",
        )
        # A batter-game with at least one hit has at least one total base too,
        # since both are read off the same per-PA outcomes (see
        # test_hits_and_total_bases_are_mechanically_linked). So "any hit" and
        # "any total base" settle identically here, and one draw covers both.
        hits_pmf = d.pmf_for("hits")
        got_a_hit = int(rng.choice(np.arange(hits_pmf.size), p=hits_pmf)) > 0
        for stat in ("hits", "total_bases"):
            quote = d.price_against_market(stat, {0.5: (2.0, 2.0)})[0]
            bets.append(quote.to_bet(timestamp=t, outcome=int(got_a_hit)))
            t += 1

    result = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    sig = result.significance(n_resamples=1)

    assert result.n_bets == 80
    assert sig.n_clusters == 40


def test_summary_includes_key_stats(dist):
    text = dist.summary()
    assert "hits" in text
    assert "correlation group" in text


def test_walks_stat_combines_walk_and_hbp_as_documented():
    """Locks down the walks = BB + HBP semantics called out in
    STAT_VALUE_MAPS, so a future change to that mapping fails a test instead
    of silently shipping a different definition of the prop."""
    rates = dict(LEAGUE_PA_OUTCOME_RATES)
    fixed_pa = np.zeros(2)
    fixed_pa[1] = 1.0  # exactly one plate appearance, deterministically

    d = BatterOutcomeDistribution(
        per_pa_rates=rates,
        pa_pmf=fixed_pa,
        correlation_group="g",
    )
    walks_pmf = d.pmf_for("walks")
    assert walks_pmf[1] == pytest.approx(rates["walk"] + rates["hbp"])
    assert walks_pmf[0] == pytest.approx(1.0 - rates["walk"] - rates["hbp"])
