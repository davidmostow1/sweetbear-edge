"""What does staking a correlated ladder rung by rung actually cost?

Every position in this script has a GENUINE POSITIVE EDGE. Nothing here is a
mispricing test and nothing depends on the model being wrong. The only variable
is whether the stake sizes acknowledge that the rungs share fate.

Two measurements, both falsifiable:

1. Simulated bankroll growth, compounding one outing at a time, under joint
   staking versus per-rung staking. This is the number that shows up in a real
   bankroll curve.

2. Expected log growth computed exactly from the outing distribution, which is
   the same quantity without the simulation noise.

If per-rung staking were harmless, the two staking rules would produce the same
growth. The claim is that they do not, that the gap widens with the number of
rungs, and that past a few rungs the per-rung rule turns a real edge into a
losing strategy.
"""

from __future__ import annotations

import numpy as np

from sweetbear.portfolio import (
    count_ladder_scenarios,
    effective_position_count,
    expected_log_growth,
    independent_kelly,
    joint_kelly,
    return_quantile,
)
from sweetbear.strikeouts import (
    OpponentProfile,
    PitcherProfile,
    build_strikeout_distribution,
)

# One ace, one league-average lineup. The distribution is the model's truth.
DIST = build_strikeout_distribution(
    PitcherProfile(name="Ace", strikeout_rate=0.30, expected_batters_faced=24.0),
    OpponentProfile(name="Lineup", strikeout_rate=0.22),
    league_strikeout_rate=0.22,
)

#: Rungs of one over ladder, priced so each carries the same honest edge.
EDGE = 0.06
RUNGS = [3.5, 4.5, 5.5, 6.5, 7.5]


def ladder(n_rungs: int) -> list[tuple[float, str, float]]:
    """The first ``n_rungs`` overs, each priced at a true 6% edge.

    Deriving the price from the model's own probability is what makes this a
    clean experiment: there is no pricing error anywhere, so every difference
    in outcome is attributable to stake sizing alone.
    """
    out = []
    for line in RUNGS[:n_rungs]:
        p = DIST.probability_over(line)
        out.append((line, "over", (1.0 + EDGE) / p))
    return out


def simulate(stakes: np.ndarray, positions, n_outings: int, seed: int) -> float:
    """Compound a bankroll across independent outings at a fixed stake vector."""
    rng = np.random.default_rng(seed)
    draws = rng.choice(DIST.pmf.size, size=n_outings, p=DIST.pmf)
    log_bankroll = 0.0
    for k in draws:
        ret = 0.0
        for f, (line, _side, odds) in zip(stakes, positions):
            if k > line:
                ret += f * (odds - 1.0)
            elif k < line:
                ret -= f
        if 1.0 + ret <= 0.0:
            return float("-inf")
        log_bankroll += np.log1p(ret)
    return log_bankroll


def main() -> None:
    print("Every position below has a true +6% edge. Only the staking differs.\n")
    print(
        "  rungs  n_eff  backed  joint stake   naive stake   joint growth   naive growth"
    )

    for n in range(1, len(RUNGS) + 1):
        positions = ladder(n)
        scenarios = count_ladder_scenarios(DIST.pmf, positions)
        joint = joint_kelly(scenarios)
        naive = independent_kelly(scenarios)
        g_joint = expected_log_growth(scenarios, joint)
        g_naive = expected_log_growth(scenarios, naive)
        n_eff = effective_position_count(scenarios)
        backed = int((joint > 1e-6).sum())
        flag = "  <-- negative" if g_naive < 0.0 else ""
        print(
            f"  {n:5d}  {n_eff:5.2f}  {backed:6d}  {joint.sum():11.3f}   "
            f"{naive.sum():11.3f}   {g_joint:12.5f}   {g_naive:12.5f}{flag}"
        )

    print(
        "\nn_eff is how many independent positions the ladder is really worth."
        "\nThe ticket count is the first column; the honest count is the second."
        "\n'backed' is how many rungs the joint solution actually stakes. It stays"
        "\nat one: rungs this correlated are substitutes, not a portfolio, so the"
        "\noptimiser takes the single best risk-adjusted rung and declines the rest."
        "\nAdding rungs adds variance without adding edge."
    )

    # ---- The same thing as a bankroll curve, which is what people look at ----
    n_rungs, n_outings, n_paths = 5, 400, 200
    positions = ladder(n_rungs)
    scenarios = count_ladder_scenarios(DIST.pmf, positions)
    joint = joint_kelly(scenarios)
    naive = independent_kelly(scenarios)

    print(f"\n{n_paths} simulated bankrolls, {n_outings} outings, {n_rungs} rungs:")
    results = {}
    for name, stakes in (("joint", joint), ("naive", naive)):
        finals = [simulate(stakes, positions, n_outings, seed) for seed in range(n_paths)]
        ruined = sum(1 for x in finals if x == float("-inf"))
        survivors = [x for x in finals if x != float("-inf")]
        median = float(np.median(survivors)) if survivors else float("-inf")
        results[name] = (median, ruined)
        growth = f"{np.exp(median):.3f}x" if survivors else "n/a"
        print(
            f"  {name:6s}  median final bankroll {growth:>10s}   "
            f"wiped out {ruined}/{n_paths}"
        )

    print(
        f"\n  5th-percentile outing return   joint {return_quantile(scenarios, joint):+.3f}"
        f"   naive {return_quantile(scenarios, naive):+.3f}"
    )

    verdict = (
        "CONFIRMED" if results["naive"][0] < results["joint"][0] else "NOT CONFIRMED"
    )
    print(f"\nClaim -- per-rung staking underperforms joint staking: {verdict}")


if __name__ == "__main__":
    main()
