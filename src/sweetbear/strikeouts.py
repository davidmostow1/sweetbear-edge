"""Pitcher strikeout distribution, and every line priced from it.

The design rule is that there is exactly **one** distribution per outing:
``P(K=0), P(K=1), P(K=2), ...``. Every market -- the main line, the whole
alternate ladder, and any exotic derived from it -- is an integral of that same
distribution. Nothing is priced by a separate classifier.

This matters for two reasons that are easy to conflate.

The first is coherence. If the over 5.5 and the over 6.5 come from different
models, they can disagree with each other: you can end up quoting a higher
probability for the harder outcome, which is an arbitrage against yourself. A
single distribution makes that impossible by construction.

The second is statistical, and it is the one that actually costs money. Every
rung of a ladder is a view of one random variable. Over 4.5, over 5.5, and over
6.5 on the same start are correlated near 1 -- they mostly win and lose
together. Backtest them as independent bets and significance inflates
enormously, and they sit at long odds where that inflation is worst. So every
quote produced here carries the same ``correlation_group``, and the inference
layer treats the ladder as the single observation it really is.

The model itself is a compound binomial. A pitcher faces a random number of
batters, and each faced batter is a strikeout with some matchup-dependent
probability:

    K | BF ~ Binomial(BF, p)      BF ~ (supplied distribution)

Marginalising over ``BF`` is what produces realistic spread. Conditioning on a
fixed number of batters faced would understate the variance badly, because a
short outing and a long one are very different strikeout opportunities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .backtest import Bet
from .margin import devig

__all__ = [
    "PitcherProfile",
    "OpponentProfile",
    "StrikeoutDistribution",
    "LineQuote",
    "matchup_strikeout_rate",
    "batters_faced_pmf",
    "build_strikeout_distribution",
    "standard_ladder",
]


def matchup_strikeout_rate(
    pitcher_rate: float, opponent_rate: float, league_rate: float
) -> float:
    """Combine pitcher and opponent strikeout rates via the odds ratio (log5).

    A pitcher who strikes out 30% of batters facing a lineup that strikes out
    18% of the time, in a league averaging 22%, does not simply average to 24%.
    The odds-ratio method combines them multiplicatively in odds space, which
    respects the boundaries at 0 and 1 and is the standard approach:

        odds = odds(pitcher) * odds(opponent) / odds(league)

    Above-average pitcher against below-average contact lineup pushes the rate
    up; the league term stops the two effects from double-counting.
    """
    for name, value in (
        ("pitcher_rate", pitcher_rate),
        ("opponent_rate", opponent_rate),
        ("league_rate", league_rate),
    ):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must lie strictly in (0, 1): {value}")

    odds = (
        (pitcher_rate / (1.0 - pitcher_rate))
        * (opponent_rate / (1.0 - opponent_rate))
        / (league_rate / (1.0 - league_rate))
    )
    return float(odds / (1.0 + odds))


def batters_faced_pmf(
    mean: float, sd: float, min_faced: int = 0, max_faced: int = 45
) -> NDArray[np.float64]:
    """Discrete distribution over batters faced, indexed from zero.

    A discretised normal, truncated and renormalised. This is deliberately the
    simplest defensible choice, and its limitation should be stated plainly: it
    is symmetric, while real outings are left-skewed because a pitcher who is
    getting hit is removed early, and no pitcher throws a 40-batter complete
    game to balance it out. Symmetry therefore understates the short-outing
    tail, which biases the under slightly.

    Callers with real outing data should estimate this empirically and pass the
    resulting pmf to :func:`build_strikeout_distribution` directly rather than
    accepting this default.
    """
    if mean <= 0.0:
        raise ValueError("mean batters faced must be positive")
    if sd <= 0.0:
        raise ValueError("sd must be positive")
    if not 0 <= min_faced < max_faced:
        raise ValueError("require 0 <= min_faced < max_faced")

    support = np.arange(min_faced, max_faced + 1, dtype=np.float64)
    weights = stats.norm.pdf(support, loc=mean, scale=sd)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("batters-faced distribution collapsed; check mean and sd")

    pmf = np.zeros(max_faced + 1, dtype=np.float64)
    pmf[min_faced : max_faced + 1] = weights / total
    return pmf


@dataclass(frozen=True)
class PitcherProfile:
    """A pitcher's strikeout rate per batter faced, and expected workload."""

    name: str
    strikeout_rate: float
    expected_batters_faced: float
    batters_faced_sd: float = 3.5

    def __post_init__(self) -> None:
        if not 0.0 < self.strikeout_rate < 1.0:
            raise ValueError(f"strikeout_rate out of range: {self.strikeout_rate}")
        if self.expected_batters_faced <= 0.0:
            raise ValueError("expected_batters_faced must be positive")
        if self.batters_faced_sd <= 0.0:
            raise ValueError("batters_faced_sd must be positive")


@dataclass(frozen=True)
class OpponentProfile:
    """An opposing lineup's strikeout rate per plate appearance."""

    name: str
    strikeout_rate: float

    def __post_init__(self) -> None:
        if not 0.0 < self.strikeout_rate < 1.0:
            raise ValueError(f"strikeout_rate out of range: {self.strikeout_rate}")


@dataclass(frozen=True)
class LineQuote:
    """One side of one strikeout line, priced from the outing distribution."""

    line: float
    side: str
    win_probability: float
    push_probability: float
    conditional_probability: float
    fair_decimal_odds: float
    correlation_group: Hashable
    market_decimal_odds: float | None = None
    market_fair_probability: float | None = None
    expected_value: float | None = None

    @property
    def has_edge(self) -> bool:
        return self.expected_value is not None and self.expected_value > 0.0

    def to_bet(self, timestamp, outcome: int, closing_fair_probability=None) -> Bet:
        """Convert to a settled :class:`~sweetbear.backtest.Bet`.

        The correlation group travels with it, so a backtest containing a full
        ladder is scored as one observation rather than as five.
        """
        if self.market_decimal_odds is None:
            raise ValueError("cannot settle a quote with no market price")
        return Bet(
            timestamp=timestamp,
            probability=self.conditional_probability,
            decimal_odds=self.market_decimal_odds,
            outcome=outcome,
            closing_fair_probability=closing_fair_probability,
            market_id=f"{self.correlation_group}:{self.side}{self.line}",
            label=f"{self.side} {self.line} K",
            correlation_group=self.correlation_group,
        )


@dataclass(frozen=True)
class StrikeoutDistribution:
    """``P(K = k)`` for one start, and every line derived from it."""

    pmf: NDArray[np.float64]
    correlation_group: Hashable
    pitcher: str = ""
    opponent: str = ""

    def __post_init__(self) -> None:
        if self.pmf.ndim != 1 or self.pmf.size == 0:
            raise ValueError("pmf must be a non-empty one-dimensional array")
        if np.any(self.pmf < 0.0):
            raise ValueError("pmf contains negative mass")
        total = float(self.pmf.sum())
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"pmf must sum to 1, got {total}")

    @property
    def mean(self) -> float:
        return float(np.dot(np.arange(self.pmf.size), self.pmf))

    @property
    def variance(self) -> float:
        k = np.arange(self.pmf.size)
        return float(np.dot((k - self.mean) ** 2, self.pmf))

    @property
    def standard_deviation(self) -> float:
        return float(np.sqrt(self.variance))

    def probability_exactly(self, k: int) -> float:
        if k < 0 or k >= self.pmf.size:
            return 0.0
        return float(self.pmf[k])

    def probability_over(self, line: float) -> float:
        """``P(K > line)``. Strictly greater, so a whole line excludes the push."""
        k = np.arange(self.pmf.size)
        return float(self.pmf[k > line].sum())

    def probability_under(self, line: float) -> float:
        """``P(K < line)``. Strictly less, so a whole line excludes the push."""
        k = np.arange(self.pmf.size)
        return float(self.pmf[k < line].sum())

    def probability_push(self, line: float) -> float:
        """``P(K == line)``, which is zero unless the line is a whole number."""
        if float(line).is_integer():
            return self.probability_exactly(int(line))
        return 0.0

    def quote(self, line: float, side: str) -> LineQuote:
        """Price one side of one line, correctly handling pushes.

        On a whole-number line the stake is returned when the total lands
        exactly on it. The fair price is therefore conditional on the bet
        resolving at all -- ``P(win) / (P(win) + P(lose))`` -- because the push
        mass is not lost, merely refunded.
        """
        side = side.lower()
        if side not in ("over", "under"):
            raise ValueError(f"side must be 'over' or 'under', got {side!r}")

        push = self.probability_push(line)
        if side == "over":
            win = self.probability_over(line)
            lose = self.probability_under(line)
        else:
            win = self.probability_under(line)
            lose = self.probability_over(line)

        resolved = win + lose
        conditional = win / resolved if resolved > 0.0 else 0.0
        fair = 1.0 / conditional if conditional > 0.0 else float("inf")

        return LineQuote(
            line=float(line),
            side=side,
            win_probability=win,
            push_probability=push,
            conditional_probability=conditional,
            fair_decimal_odds=fair,
            correlation_group=self.correlation_group,
        )

    def ladder(self, lines: Sequence[float] | None = None) -> list[LineQuote]:
        """Both sides of every line, all sharing one correlation group.

        The shared group is the point. These quotes are not independent bets
        and must never be counted as such.
        """
        if lines is None:
            lines = standard_ladder(self.mean)
        out: list[LineQuote] = []
        for line in lines:
            out.append(self.quote(line, "over"))
            out.append(self.quote(line, "under"))
        return out

    def price_against_market(
        self,
        market: Mapping[float, tuple[float, float]],
        devig_method: str = "shin",
    ) -> list[LineQuote]:
        """Compare model prices to de-vigged market prices, per line.

        ``market`` maps each line to its ``(over_odds, under_odds)`` decimal
        pair. Both sides are required, because de-vigging a single side is
        impossible -- there is no way to know how much margin is in a price
        without seeing the rest of the book. Comparing a model probability
        against a raw one-sided implied probability is the single most reliable
        way to manufacture an edge that does not exist.

        Expected value is computed with the push mass held out, since a push
        returns the stake rather than losing it:

            EV = P(win) * (odds - 1) - P(lose)
        """
        quotes: list[LineQuote] = []
        for line, prices in market.items():
            if len(prices) != 2:
                raise ValueError(f"line {line} needs exactly two prices")
            over_odds, under_odds = (float(p) for p in prices)
            fair_probs = devig(np.array([over_odds, under_odds]), method=devig_method)

            for side, market_odds, fair_prob in (
                ("over", over_odds, float(fair_probs[0])),
                ("under", under_odds, float(fair_probs[1])),
            ):
                base = self.quote(line, side)
                win = base.win_probability
                lose = 1.0 - win - base.push_probability
                ev = win * (market_odds - 1.0) - lose
                quotes.append(
                    LineQuote(
                        line=base.line,
                        side=base.side,
                        win_probability=base.win_probability,
                        push_probability=base.push_probability,
                        conditional_probability=base.conditional_probability,
                        fair_decimal_odds=base.fair_decimal_odds,
                        correlation_group=base.correlation_group,
                        market_decimal_odds=market_odds,
                        market_fair_probability=fair_prob,
                        expected_value=float(ev),
                    )
                )
        return quotes

    def summary(self) -> str:
        lines = [
            f"pitcher           {self.pitcher or '(unnamed)'}",
            f"opponent          {self.opponent or '(unnamed)'}",
            f"expected K        {self.mean:.2f}  (sd {self.standard_deviation:.2f})",
            f"correlation group {self.correlation_group}",
            "",
            "  line     over     under     push",
        ]
        for line in standard_ladder(self.mean):
            lines.append(
                f"  {line:<5.1f}  {self.probability_over(line):6.2%}   "
                f"{self.probability_under(line):6.2%}   "
                f"{self.probability_push(line):6.2%}"
            )
        return "\n".join(lines)


def standard_ladder(expected_strikeouts: float, span: int = 4) -> list[float]:
    """Half- and whole-number lines bracketing the expected total.

    Books quote roughly this range; anything far outside it prices at odds
    where the model's tail assumptions dominate and the market's margin is
    widest.
    """
    centre = int(round(expected_strikeouts))
    lo = max(0, centre - span)
    hi = centre + span
    out: list[float] = []
    for k in range(lo, hi + 1):
        out.append(float(k))
        out.append(k + 0.5)
    return [x for x in out if x >= 0.5]


def build_strikeout_distribution(
    pitcher: PitcherProfile,
    opponent: OpponentProfile,
    league_strikeout_rate: float = 0.22,
    correlation_group: Hashable | None = None,
    faced_pmf: NDArray[np.float64] | None = None,
) -> StrikeoutDistribution:
    """Compound a binomial strikeout process over a random batters-faced count.

    ``P(K=k) = sum_n P(BF=n) * Binomial(k; n, p)``

    where ``p`` is the matchup strikeout rate. The compounding is what gives
    the distribution its realistic width: holding batters faced fixed would
    treat a five-inning start and an eight-inning start as the same
    opportunity, and understate variance enough to make every alternate line
    look mispriced.
    """
    rate = matchup_strikeout_rate(
        pitcher.strikeout_rate, opponent.strikeout_rate, league_strikeout_rate
    )

    if faced_pmf is None:
        faced_pmf = batters_faced_pmf(
            pitcher.expected_batters_faced, pitcher.batters_faced_sd
        )
    faced_pmf = np.asarray(faced_pmf, dtype=np.float64)
    if faced_pmf.ndim != 1:
        raise ValueError("faced_pmf must be one-dimensional")
    total = float(faced_pmf.sum())
    if total <= 0.0:
        raise ValueError("faced_pmf has no mass")
    faced_pmf = faced_pmf / total

    max_faced = faced_pmf.size - 1
    counts = np.arange(max_faced + 1)
    # rows: batters faced; columns: strikeouts
    grid = stats.binom.pmf(counts[None, :], counts[:, None], rate)
    grid = np.nan_to_num(grid, nan=0.0)
    pmf = faced_pmf @ grid

    pmf = np.clip(pmf, 0.0, None)
    pmf = pmf / pmf.sum()

    if correlation_group is None:
        correlation_group = f"{pitcher.name}-vs-{opponent.name}"

    return StrikeoutDistribution(
        pmf=pmf,
        correlation_group=correlation_group,
        pitcher=pitcher.name,
        opponent=opponent.name,
    )
