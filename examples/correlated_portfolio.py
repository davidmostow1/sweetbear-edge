"""Size a slate of correlated positions instead of pricing them one at a time.

Run: python examples/correlated_portfolio.py

Three portfolios, in increasing order of how badly per-position staking fails:
a pitcher's strikeout ladder, one batter's props across several stats, and the
two together as a slate.
"""

from __future__ import annotations

import numpy as np

from sweetbear.batters import (
    LEAGUE_PA_OUTCOME_RATES,
    BatterProfile,
    PitcherAllowedProfile,
    build_batter_distribution,
)
from sweetbear.portfolio import (
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

RNG = np.random.default_rng(20260801)


def report(name: str, scenarios, labels) -> None:
    """Compare the two staking rules at FULL Kelly.

    Full Kelly is the only setting where the comparison is meaningful, because
    that is where the joint solution is provably the growth-maximising one.
    Scaling both rules down by the same fraction does not preserve the ranking:
    a quarter of an overstaked vector can land closer to the optimum than a
    quarter of the optimum does, and the naive rule then looks better for a
    reason that has nothing to do with correlation. Bet the quarter-Kelly column
    below; judge the rules here.
    """
    joint = joint_kelly(scenarios)
    naive = independent_kelly(scenarios)
    print(f"\n{name}")
    print("-" * len(name))
    print(f"  positions            {scenarios.n_positions}")
    print(f"  effective positions  {effective_position_count(scenarios):.2f}")
    print("\n  position                     joint    per-position     joint @ 1/4")
    for label, j, i in zip(labels, joint, naive):
        print(f"  {label:<26s}  {j:7.4f}      {i:7.4f}         {j * 0.25:7.4f}")
    print(
        f"  {'TOTAL STAKE':<26s}  {joint.sum():7.4f}      {naive.sum():7.4f}"
        f"         {joint.sum() * 0.25:7.4f}"
    )
    print(
        f"\n  expected log growth        {expected_log_growth(scenarios, joint):8.5f}   "
        f"  {expected_log_growth(scenarios, naive):8.5f}"
    )
    print(
        f"  5th-percentile return      {return_quantile(scenarios, joint):+8.4f}   "
        f"  {return_quantile(scenarios, naive):+8.4f}"
    )


# ---------------------------------------------------------------------------
# 1. A pitcher's strikeout ladder -- exact scenarios, no simulation
# ---------------------------------------------------------------------------

dist = build_strikeout_distribution(
    PitcherProfile(name="Sample Ace", strikeout_rate=0.30, expected_batters_faced=24.0),
    OpponentProfile(name="Sample Lineup", strikeout_rate=0.245),
    league_strikeout_rate=0.22,
    correlation_group="2026-08-01-GAME1",
)

# Priced at a genuine 5% edge on each rung, so nothing below is a pricing error.
ladder = [
    (line, "over", 1.05 / dist.probability_over(line)) for line in (4.5, 5.5, 6.5)
]
ladder_scenarios = count_ladder_scenarios(
    dist.pmf, ladder, correlation_group=dist.correlation_group
)
report("Pitcher strikeout ladder (3 rungs, each +5% edge)", ladder_scenarios,
       [f"K over {line}" for line, _, _ in ladder])

print(
    "\n  Every rung is a view of one number, so they are substitutes rather than\n"
    "  a portfolio. Joint staking backs the best rung and declines the rest;\n"
    "  per-position staking backs all three and stakes several times too much."
)

# ---------------------------------------------------------------------------
# 2. One batter, several stats -- Monte Carlo over shared plate appearances
# ---------------------------------------------------------------------------

batter_dist = build_batter_distribution(
    BatterProfile("Sample Bat", dict(LEAGUE_PA_OUTCOME_RATES), 4.3, 0.9),
    PitcherAllowedProfile("Sample Arm", dict(LEAGUE_PA_OUTCOME_RATES)),
    correlation_group="2026-08-01-GAME2",
)

specs = [
    (stat, line, "over", 1.05 / batter_dist.probability_over(stat, line))
    for stat, line in (("hits", 0.5), ("total_bases", 1.5), ("home_runs", 0.5))
]
batter_sc = batter_scenarios(batter_dist, specs, draws=60000, rng=RNG)
report("One batter, three props (each +5% edge)", batter_sc,
       [f"{stat} over {line}" for stat, line, _, _ in specs])

corr = payoff_correlation(batter_sc)
print("\n  Payoff correlation between these props:")
print(f"    hits / total bases   {corr[0, 1]:+.3f}")
print(f"    hits / home runs     {corr[0, 2]:+.3f}")
print(
    "\n  These are not independent bets. They are read off the same plate\n"
    "  appearances, which is why the scenarios are simulated at the PA level\n"
    "  rather than assembled from three separate marginal distributions."
)

# ---------------------------------------------------------------------------
# 3. Both games as one slate -- correlated within, independent across
# ---------------------------------------------------------------------------

slate = combine_independent([ladder_scenarios, batter_sc], draws=60000, rng=RNG)
report("Full slate: two games, six positions", slate,
       [f"K over {line}" for line, _, _ in ladder]
       + [f"{stat} over {line}" for stat, line, _, _ in specs])

print(
    "\n  Across the two games the positions genuinely diversify, so the joint\n"
    "  solution stakes more in total here than on either game alone -- it is not\n"
    "  simply a rule that bets less. It bets less where risk is duplicated and\n"
    "  more where it is spread.\n"
    "\n  The independence ACROSS games is an assumption, not a measurement. Two\n"
    "  games sharing a weather front or a postponement risk are not independent,\n"
    "  and nothing here can detect that for you."
)
