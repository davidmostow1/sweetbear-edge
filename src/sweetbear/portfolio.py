"""Staking a set of positions that share fate.

Everything upstream of this module prices one position at a time. That is the
right decomposition for pricing and the wrong one for staking, because the
positions a prop model emits are not separate risks. Five rungs of a pitcher's
strikeout ladder are five views of one random variable. A batter's hits ladder
and his total-bases ladder are read off the same plate appearances. Size each
of those in isolation and the stakes add while the risk does not diversify.

The direction of the error depends on the sign of the correlation, and the two
cases are opposite -- which is why they need separate machinery and why
conflating them is expensive:

*Mutually exclusive outcomes of one market* partially hedge. Whichever way it
lands, one ticket is live. Per-position Kelly prices each leg as a standalone
all-or-nothing risk and therefore **understakes**. That case is solved by
:func:`sweetbear.edge.simultaneous_kelly`.

*Positively correlated positions* -- a ladder, a same-game group, anything
driven by one latent outcome -- lose together. Per-position Kelly assumes each
loss is independently survivable and therefore **overstakes**, and the error
compounds with the number of rungs. Three copies of one bet at full Kelly each
is three times the correct total stake, which is past the point where expected
log growth turns negative. Nothing in the per-position formula can see this,
because the correlation is not in its inputs.

This module fixes the second case by refusing to work from marginals at all.
Instead of a probability per position it takes a **joint scenario
distribution**: a matrix of what every position pays in every state of the
world, with a probability attached to each state. Stakes are then chosen to
maximise expected log wealth over that joint distribution:

    maximise  sum_s q_s * log(1 + sum_i f_i * payoff[s, i])

The objective is concave in ``f`` and the feasible set is convex, so the
optimum is unique and the solver finds it. Correlation never appears as a
parameter to estimate because it is already carried, exactly, by the scenario
matrix.

Where the scenarios come from matters more than the optimiser. Two builders are
provided, and they differ in kind:

:func:`count_ladder_scenarios` is **exact**. Every position on a pitcher's
strikeout ladder is a deterministic function of one integer, so enumerating that
integer over its pmf enumerates the joint distribution with no sampling error
and no independence assumption anywhere.

:func:`batter_scenarios` is **Monte Carlo**, because a batter's props are a
function of the whole vector of plate-appearance outcomes rather than of one
count. Sampling at the plate-appearance level reproduces the joint structure by
construction: hits and total bases come out consistent because they are read off
the same simulated plate appearances, exactly as they are in
:mod:`sweetbear.batters`. The marginals it produces agree with the analytic
``pmf_for`` to sampling error, which is a test in ``test_portfolio.py`` rather
than an assurance here.

**What this module does not model.** Positions in different games are combined
by :func:`combine_independent`, which assumes exactly what its name says. That
is defensible across unrelated games and wrong for anything sharing a weather
system, an umpire, or a postponement risk. More importantly, it cannot couple a
pitcher's strikeout total to the opposing batters' strikeout totals in the same
game, even though those are mechanically the same events counted twice: the
pricing modules do not share a plate-appearance-level game state, so the
coupling is not available to be extracted. Betting both sides of that pair and
treating them as independent will understate the true correlation. That is a
real limitation and it is stated here rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from .batters import CATEGORIES, STAT_VALUE_MAPS, BatterOutcomeDistribution

__all__ = [
    "ScenarioSet",
    "joint_kelly",
    "independent_kelly",
    "expected_log_growth",
    "return_quantile",
    "payoff_correlation",
    "effective_position_count",
    "count_ladder_scenarios",
    "batter_scenarios",
    "combine_independent",
]


def _over_under_payoff(value: float, line: float, side: str, decimal_odds: float) -> float:
    """Profit per unit staked on one over/under position at a realised value.

    A whole-number line that lands exactly refunds the stake, so its payoff is
    zero rather than a loss. Getting this wrong silently converts every push
    into a loss and understates every whole-number line.
    """
    if value > line:
        return decimal_odds - 1.0 if side == "over" else -1.0
    if value < line:
        return -1.0 if side == "over" else decimal_odds - 1.0
    return 0.0


@dataclass(frozen=True)
class ScenarioSet:
    """What every position pays in every state of the world.

    ``payoffs`` is ``(n_scenarios, n_positions)`` of profit per unit staked:
    ``d - 1`` on a win, ``-1`` on a loss, ``0`` on a push. ``probabilities`` is
    the mass on each scenario and must sum to 1.

    The joint structure lives entirely in the rows. Two positions that always
    win together appear as two identical columns, and the optimiser will stake
    them as the single position they really are. Nothing needs to tell it that
    they are correlated.
    """

    payoffs: NDArray[np.float64]
    probabilities: NDArray[np.float64]
    labels: tuple[str, ...] = ()
    correlation_groups: tuple[Hashable, ...] = ()

    def __post_init__(self) -> None:
        if self.payoffs.ndim != 2 or self.payoffs.size == 0:
            raise ValueError("payoffs must be a non-empty two-dimensional array")
        if self.probabilities.ndim != 1:
            raise ValueError("probabilities must be one-dimensional")
        if self.probabilities.size != self.payoffs.shape[0]:
            raise ValueError(
                "probabilities must have one entry per scenario row: "
                f"{self.probabilities.size} vs {self.payoffs.shape[0]}"
            )
        if np.any(self.probabilities < 0.0):
            raise ValueError("probabilities contains negative mass")
        total = float(self.probabilities.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"probabilities must sum to 1, got {total}")
        if np.any(self.payoffs < -1.0 - 1e-9):
            raise ValueError("a position cannot lose more than the stake")
        if self.labels and len(self.labels) != self.payoffs.shape[1]:
            raise ValueError("labels must have one entry per position")
        if self.correlation_groups and len(self.correlation_groups) != self.payoffs.shape[1]:
            raise ValueError("correlation_groups must have one entry per position")

    @property
    def n_scenarios(self) -> int:
        return int(self.payoffs.shape[0])

    @property
    def n_positions(self) -> int:
        return int(self.payoffs.shape[1])

    def marginal_probabilities(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """``(win, lose)`` mass per position. The remainder is push mass."""
        q = self.probabilities[:, None]
        win = (q * (self.payoffs > 0.0)).sum(axis=0)
        lose = (q * (self.payoffs < 0.0)).sum(axis=0)
        return win, lose

    def decimal_odds(self) -> NDArray[np.float64]:
        """The price implied by each column's winning payoff."""
        odds = np.ones(self.n_positions, dtype=np.float64)
        for i in range(self.n_positions):
            wins = self.payoffs[:, i][self.payoffs[:, i] > 0.0]
            if wins.size:
                odds[i] = float(wins.max()) + 1.0
        return odds

    def sample(self, draws: int, rng: np.random.Generator | None = None) -> "ScenarioSet":
        """Resample into ``draws`` equally weighted scenarios.

        Used by :func:`combine_independent` to put exact and simulated sets on
        the same footing before they are joined side by side.
        """
        if draws <= 0:
            raise ValueError("draws must be positive")
        rng = np.random.default_rng() if rng is None else rng
        idx = rng.choice(self.n_scenarios, size=draws, p=self.probabilities)
        return ScenarioSet(
            payoffs=self.payoffs[idx],
            probabilities=np.full(draws, 1.0 / draws),
            labels=self.labels,
            correlation_groups=self.correlation_groups,
        )


def expected_log_growth(scenarios: ScenarioSet, stakes: NDArray[np.float64]) -> float:
    """Expected log wealth multiple for a stake vector.

    Returns ``-inf`` if any scenario with positive probability wipes the
    bankroll out. That is not a numerical inconvenience: a stake vector that can
    lose everything has no long-run growth rate at any horizon, and reporting a
    finite number for it would be the whole error this module exists to prevent.
    """
    f = np.asarray(stakes, dtype=np.float64).ravel()
    if f.size != scenarios.n_positions:
        raise ValueError("stakes must have one entry per position")
    wealth = 1.0 + scenarios.payoffs @ f
    if np.any(wealth[scenarios.probabilities > 0.0] <= 0.0):
        return float("-inf")
    return float(np.dot(scenarios.probabilities, np.log(wealth)))


def joint_kelly(
    scenarios: ScenarioSet,
    fraction: float = 1.0,
    max_total_stake: float = 1.0,
) -> NDArray[np.float64]:
    """Stakes maximising expected log wealth over the joint distribution.

    This is the generalisation of both per-bet Kelly and
    :func:`sweetbear.edge.simultaneous_kelly`. Feed it a single position and it
    returns the textbook Kelly fraction; feed it the mutually exclusive outcomes
    of one market and it reproduces the simultaneous solution; feed it a
    correlated ladder and it returns something far smaller than the sum of the
    per-rung fractions. All three fall out of the same objective because the
    correlation structure is in the scenarios rather than in the formula.

    ``fraction`` scales the result afterwards, for the same reason
    :func:`sweetbear.edge.fractional_kelly` exists -- the optimum is only
    optimal if the scenario probabilities are exactly right, and they are not.
    Solving jointly removes the correlation error; it does nothing about
    estimation error, and the quadratic penalty for overbetting is unchanged.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if not 0.0 < max_total_stake <= 1.0:
        raise ValueError("max_total_stake must lie in (0, 1]")

    n = scenarios.n_positions
    q = scenarios.probabilities
    payoffs = scenarios.payoffs
    floor = 1e-12

    def objective(f: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        wealth = np.maximum(1.0 + payoffs @ f, floor)
        # Negated: scipy minimises, we want maximum expected log growth.
        value = -float(np.dot(q, np.log(wealth)))
        grad = -(payoffs * (q / wealth)[:, None]).sum(axis=0)
        return value, grad

    result = minimize(
        objective,
        x0=np.full(n, min(0.01, max_total_stake / max(n, 1))),
        jac=True,
        bounds=[(0.0, max_total_stake)] * n,
        constraints=[{"type": "ineq", "fun": lambda f: max_total_stake - f.sum()}],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    stakes = np.maximum(result.x, 0.0)
    stakes[stakes < 1e-9] = 0.0
    return stakes * fraction


def independent_kelly(
    scenarios: ScenarioSet,
    fraction: float = 1.0,
) -> NDArray[np.float64]:
    """Per-position Kelly from the marginals alone, ignoring the joint structure.

    Provided as the comparator, not as a recommendation. This is what staking
    a ladder rung by rung actually does, and the gap between its total stake and
    :func:`joint_kelly`'s is the cost of pretending correlated positions are
    separate bets. On a whole-number line the probability used is conditional on
    the bet resolving, matching the convention in
    :class:`sweetbear.strikeouts.LineQuote`.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    win, lose = scenarios.marginal_probabilities()
    resolved = win + lose
    conditional = np.divide(win, resolved, out=np.zeros_like(win), where=resolved > 0.0)
    d = scenarios.decimal_odds()
    b = np.maximum(d - 1.0, 1e-12)
    return np.maximum((conditional * d - 1.0) / b, 0.0) * fraction


def return_quantile(
    scenarios: ScenarioSet, stakes: NDArray[np.float64], quantile: float = 0.05
) -> float:
    """Portfolio return at a lower quantile of the scenario distribution.

    The number that makes overstaking visible before it happens rather than
    after. Two stake vectors with similar expected growth can have very
    different 5th percentiles, and the correlated one is always worse.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must lie strictly in (0, 1)")
    f = np.asarray(stakes, dtype=np.float64).ravel()
    if f.size != scenarios.n_positions:
        raise ValueError("stakes must have one entry per position")
    returns = scenarios.payoffs @ f
    order = np.argsort(returns)
    cumulative = np.cumsum(scenarios.probabilities[order])
    idx = int(np.searchsorted(cumulative, quantile))
    idx = min(idx, returns.size - 1)
    return float(returns[order][idx])


def payoff_correlation(scenarios: ScenarioSet) -> NDArray[np.float64]:
    """Probability-weighted correlation matrix of the position payoffs.

    A diagnostic, not an input -- :func:`joint_kelly` never consults it. It is
    here so that a portfolio can be inspected before it is staked, and because
    a ladder whose off-diagonal entries sit near 1 is a ladder that should be
    reported as roughly one position.
    """
    q = scenarios.probabilities
    x = scenarios.payoffs
    mean = q @ x
    centred = x - mean
    cov = (centred * q[:, None]).T @ centred
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    denom = np.outer(sd, sd)
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0.0)
    np.fill_diagonal(corr, np.where(sd > 0.0, 1.0, 0.0))
    return corr


def effective_position_count(scenarios: ScenarioSet) -> float:
    """How many independent positions the portfolio is really worth.

    ``n^2 / sum_ij rho_ij``, the same design-effect form
    :func:`sweetbear.significance.effective_sample_size` uses on clustered
    observations, applied to payoffs instead of residuals. A five-rung ladder
    correlated near 1 returns close to 1, which is the honest count to report
    and the one a ticket count will never give you.

    Values above ``n`` are possible and are not an error: genuinely hedging
    positions diversify better than independent ones.
    """
    corr = payoff_correlation(scenarios)
    n = scenarios.n_positions
    total = float(corr.sum())
    if total <= 0.0:
        return float("inf")
    return float(n * n / total)


def count_ladder_scenarios(
    pmf: NDArray[np.float64],
    positions: Sequence[tuple[float, str, float]],
    correlation_group: Hashable = None,
) -> ScenarioSet:
    """Exact joint scenarios for positions driven by one integer count.

    ``positions`` is a sequence of ``(line, side, decimal_odds)``. Every entry
    must be a function of the same underlying count -- a pitcher's strikeouts, a
    batter's hits -- which is what makes the enumeration exact: scenario ``k``
    is "the count came in at ``k``", it carries probability ``pmf[k]``, and
    every position's payoff at ``k`` is determined with no sampling and no
    assumption. Pass :attr:`sweetbear.strikeouts.StrikeoutDistribution.pmf` or
    :meth:`sweetbear.batters.BatterOutcomeDistribution.pmf_for` straight in.

    Mixing counts from different stats or different players here is a modelling
    error the signature cannot prevent: they would all be indexed by the same
    ``k`` and come out perfectly correlated. Use :func:`combine_independent` or
    :func:`batter_scenarios` for anything spanning more than one count.
    """
    pmf = np.asarray(pmf, dtype=np.float64).ravel()
    if pmf.size == 0:
        raise ValueError("pmf must be non-empty")
    if np.any(pmf < 0.0):
        raise ValueError("pmf contains negative mass")
    total = float(pmf.sum())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"pmf must sum to 1, got {total}")
    if not positions:
        raise ValueError("need at least one position")

    counts = np.arange(pmf.size, dtype=np.float64)
    payoffs = np.zeros((pmf.size, len(positions)), dtype=np.float64)
    labels: list[str] = []
    for i, (line, side, odds) in enumerate(positions):
        side = str(side).lower()
        if side not in ("over", "under"):
            raise ValueError(f"side must be 'over' or 'under', got {side!r}")
        if odds <= 1.0:
            raise ValueError("Decimal odds must be greater than 1.0")
        for k_idx, k in enumerate(counts):
            payoffs[k_idx, i] = _over_under_payoff(k, float(line), side, float(odds))
        labels.append(f"{side} {line}")

    return ScenarioSet(
        payoffs=payoffs,
        probabilities=pmf / total,
        labels=tuple(labels),
        correlation_groups=tuple([correlation_group] * len(positions)),
    )


def batter_scenarios(
    distribution: BatterOutcomeDistribution,
    positions: Sequence[tuple[str, float, str, float]],
    draws: int = 20000,
    rng: np.random.Generator | None = None,
) -> ScenarioSet:
    """Monte Carlo joint scenarios across several props for one batter-game.

    ``positions`` is a sequence of ``(stat, line, side, decimal_odds)``. A
    batter's props cannot be enumerated the way a single count can, because
    hits, total bases and home runs are different functions of the same vector
    of plate-appearance outcomes rather than of one number. So the plate
    appearances themselves are simulated -- a count drawn from the PA pmf, then
    a multinomial over the eight outcome categories -- and every stat is read
    off the same draw. That is the identical construction
    :mod:`sweetbear.batters` compounds analytically, which is why the marginals
    agree, and it produces the cross-stat coupling that no amount of work on the
    marginals could recover.

    Sampling error is real and scales as ``1/sqrt(draws)``. The default is large
    enough that stake vectors are stable to well under a percentage point of
    bankroll; raise it if a position sits at long odds, where the tail
    scenarios that matter are rare.
    """
    if draws <= 0:
        raise ValueError("draws must be positive")
    if not positions:
        raise ValueError("need at least one position")
    rng = np.random.default_rng() if rng is None else rng

    pa_pmf = np.asarray(distribution.pa_pmf, dtype=np.float64)
    pa_counts = rng.choice(pa_pmf.size, size=draws, p=pa_pmf / pa_pmf.sum())
    rates = np.array([distribution.per_pa_rates[c] for c in CATEGORIES], dtype=np.float64)
    rates = rates / rates.sum()
    category_counts = rng.multinomial(pa_counts, rates)

    payoffs = np.zeros((draws, len(positions)), dtype=np.float64)
    labels: list[str] = []
    for i, (stat, line, side, odds) in enumerate(positions):
        if stat not in STAT_VALUE_MAPS:
            raise ValueError(
                f"unknown or unsupported stat {stat!r}; available: "
                f"{sorted(STAT_VALUE_MAPS)}"
            )
        side = str(side).lower()
        if side not in ("over", "under"):
            raise ValueError(f"side must be 'over' or 'under', got {side!r}")
        if odds <= 1.0:
            raise ValueError("Decimal odds must be greater than 1.0")
        value_map = STAT_VALUE_MAPS[stat]
        weights = np.array([value_map.get(c, 0) for c in CATEGORIES], dtype=np.float64)
        realised = category_counts @ weights

        over = realised > line
        under = realised < line
        if side == "over":
            payoffs[over, i] = float(odds) - 1.0
            payoffs[under, i] = -1.0
        else:
            payoffs[under, i] = float(odds) - 1.0
            payoffs[over, i] = -1.0
        labels.append(f"{stat} {side} {line}")

    return ScenarioSet(
        payoffs=payoffs,
        probabilities=np.full(draws, 1.0 / draws),
        labels=tuple(labels),
        correlation_groups=tuple([distribution.correlation_group] * len(positions)),
    )


def combine_independent(
    sets: Sequence[ScenarioSet],
    draws: int = 20000,
    rng: np.random.Generator | None = None,
) -> ScenarioSet:
    """Join scenario sets that are assumed independent of one another.

    Each set is resampled and the columns placed side by side, so within a set
    the joint structure survives exactly and across sets it is destroyed by
    assumption. That assumption is the whole content of this function and it is
    only as good as the separation between the sets: two different games on
    different days, fine; two games sharing a bullpen, a weather front, or a
    postponement risk, not fine. Correlation you assume away does not stop
    existing, it stops being staked for.

    A full product of the sets would be exact but grows combinatorially, which
    is why this samples instead. Sampling error scales as ``1/sqrt(draws)``.
    """
    if not sets:
        raise ValueError("need at least one scenario set")
    rng = np.random.default_rng() if rng is None else rng
    sampled = [s.sample(draws, rng) for s in sets]
    payoffs = np.hstack([s.payoffs for s in sampled])
    labels: list[str] = []
    groups: list[Hashable] = []
    for s in sets:
        labels.extend(s.labels if s.labels else [""] * s.n_positions)
        groups.extend(
            s.correlation_groups if s.correlation_groups else [None] * s.n_positions
        )
    return ScenarioSet(
        payoffs=payoffs,
        probabilities=np.full(draws, 1.0 / draws),
        labels=tuple(labels),
        correlation_groups=tuple(groups),
    )
