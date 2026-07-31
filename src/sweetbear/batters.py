"""Batter plate-appearance distribution, and every prop priced from it.

Same discipline as :mod:`sweetbear.strikeouts`: one distribution, every market
an integral of it. Here the distribution is over what happens in a single
plate appearance -- out, strikeout, walk, HBP, single, double, triple, home
run -- and every prop is a linear function of that outcome, compounded over a
random number of plate appearances.

    outcome | PA ~ Categorical(rates)      PA count ~ (supplied distribution)

Hits, total bases, home runs, walks, and strikeouts all fall out of the same
per-PA categorical distribution, so they cannot disagree with each other the
way five separately-fit classifiers could -- a batter's total-base ladder is
mechanically consistent with his hit ladder because both are read off the same
underlying process.

What is deliberately absent: runs and RBIs. Those depend on who is on base and
how the rest of the lineup hits, which this module does not model. Faking them
from isolated batter rates would produce a number with no defensible basis, so
:data:`STAT_VALUE_MAPS` does not include them, and :meth:`BatterOutcomeDistribution.pmf_for`
raises if asked for them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from .margin import devig
from .strikeouts import LineQuote, standard_ladder

__all__ = [
    "CATEGORIES",
    "LEAGUE_PA_OUTCOME_RATES",
    "STAT_VALUE_MAPS",
    "BatterProfile",
    "PitcherAllowedProfile",
    "BatterOutcomeDistribution",
    "plate_appearances_pmf",
    "matchup_batter_rates",
    "build_batter_distribution",
]

#: The eight mutually exclusive, collectively exhaustive outcomes of a plate
#: appearance that everything else in this module is built from.
CATEGORIES: tuple[str, ...] = (
    "strikeout",
    "walk",
    "hbp",
    "single",
    "double",
    "triple",
    "home_run",
    "out_in_play",
)

#: Rough major-league averages. Callers modelling a real slate should supply
#: their own league table for the relevant season rather than trust this.
LEAGUE_PA_OUTCOME_RATES: Mapping[str, float] = {
    "strikeout": 0.222,
    "walk": 0.083,
    "hbp": 0.010,
    "single": 0.140,
    "double": 0.044,
    "triple": 0.004,
    "home_run": 0.032,
    "out_in_play": 0.465,
}

#: Per-PA integer value induced by each outcome category, for every prop this
#: module knows how to price. A stat's per-PA distribution is the pushforward
#: of the categorical outcome distribution under this map; its compound
#: distribution is a mixture of that map's self-convolutions over the PA-count
#: distribution. Deliberately does not include "runs" or "rbi" -- see module
#: docstring.
STAT_VALUE_MAPS: Mapping[str, Mapping[str, int]] = {
    "hits": {"single": 1, "double": 1, "triple": 1, "home_run": 1},
    "total_bases": {"single": 1, "double": 2, "triple": 3, "home_run": 4},
    "home_runs": {"home_run": 1},
    # Combines walk and hit-by-pitch into one count. Real books usually quote
    # a "walks" prop as bases on balls only, with HBP excluded. Treat this
    # stat as "times reached on a non-batted-ball event", and check a
    # candidate market's own settlement rules before pricing against it --
    # otherwise a real BB-only line will be priced against the wrong number.
    "walks": {"walk": 1, "hbp": 1},
    "strikeouts": {"strikeout": 1},
    "singles": {"single": 1},
    "doubles": {"double": 1},
    "triples": {"triple": 1},
}


def _validate_rate_dict(rates: Mapping[str, float], name: str) -> None:
    missing = set(CATEGORIES) - set(rates)
    extra = set(rates) - set(CATEGORIES)
    if missing or extra:
        raise ValueError(
            f"{name} must have exactly the categories {CATEGORIES}; "
            f"missing={sorted(missing)} extra={sorted(extra)}"
        )
    for cat in CATEGORIES:
        v = rates[cat]
        if not 0.0 <= v < 1.0:
            raise ValueError(f"{name}[{cat!r}] out of range [0, 1): {v}")
    total = sum(rates.values())
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to 1, got {total}")


@dataclass(frozen=True)
class BatterProfile:
    """A batter's own rate for each plate-appearance outcome."""

    name: str
    rates: Mapping[str, float]
    expected_plate_appearances: float = 4.3
    plate_appearances_sd: float = 0.9

    def __post_init__(self) -> None:
        _validate_rate_dict(self.rates, "rates")
        if self.expected_plate_appearances <= 0.0:
            raise ValueError("expected_plate_appearances must be positive")
        if self.plate_appearances_sd <= 0.0:
            raise ValueError("plate_appearances_sd must be positive")


@dataclass(frozen=True)
class PitcherAllowedProfile:
    """What a pitcher (or bullpen) allows for each plate-appearance outcome."""

    name: str
    rates: Mapping[str, float]

    def __post_init__(self) -> None:
        _validate_rate_dict(self.rates, "rates")


def plate_appearances_pmf(
    mean: float, sd: float, min_pa: int = 0, max_pa: int = 9
) -> NDArray[np.float64]:
    """Discrete distribution over plate appearances in a game, indexed from zero.

    Same construction and the same caveat as
    :func:`sweetbear.strikeouts.batters_faced_pmf`: a discretised, truncated,
    renormalised normal. It is symmetric where real plate-appearance counts are
    not -- a batter pulled for a pinch hitter or removed for injury shortens
    the tail on one side without a matching extension on the other. Callers
    with real game logs should estimate this empirically.
    """
    if mean <= 0.0:
        raise ValueError("mean plate appearances must be positive")
    if sd <= 0.0:
        raise ValueError("sd must be positive")
    if not 0 <= min_pa < max_pa:
        raise ValueError("require 0 <= min_pa < max_pa")

    support = np.arange(min_pa, max_pa + 1, dtype=np.float64)
    weights = stats.norm.pdf(support, loc=mean, scale=sd)
    total = weights.sum()
    if total <= 0.0:
        raise ValueError("plate-appearance distribution collapsed; check mean and sd")

    pmf = np.zeros(max_pa + 1, dtype=np.float64)
    pmf[min_pa : max_pa + 1] = weights / total
    return pmf


def matchup_batter_rates(
    batter_rates: Mapping[str, float],
    pitcher_rates: Mapping[str, float],
    league_rates: Mapping[str, float] = LEAGUE_PA_OUTCOME_RATES,
) -> dict[str, float]:
    """Combine batter and pitcher per-category rates via odds ratios.

    Each category is combined independently, log5-style, against the league
    rate for that category, then the eight results are renormalised to sum to
    one:

        odds_c = odds(batter_c) * odds(pitcher_c) / odds(league_c)

    This is a known approximation, not a proper joint (Dirichlet) combination:
    treating eight mutually exclusive outcomes as eight independent binary
    events and renormalising afterward ignores the covariance between
    categories a fully consistent model would carry. It is the standard
    shortcut in projection systems because it is simple, respects the (0, 1)
    boundary per category, and its error is small next to the uncertainty in
    the input rates themselves. Callers who need better should replace this
    function; nothing downstream depends on the combination method, only on
    receiving eight rates that sum to one.

    When ``pitcher_rates`` equals ``league_rates`` exactly, this returns
    ``batter_rates`` unchanged -- a league-average opponent should not move a
    batter's own established rates.
    """
    _validate_rate_dict(batter_rates, "batter_rates")
    _validate_rate_dict(pitcher_rates, "pitcher_rates")
    _validate_rate_dict(league_rates, "league_rates")

    eps = 1e-9
    raw: dict[str, float] = {}
    for cat in CATEGORIES:
        b = min(max(batter_rates[cat], eps), 1.0 - eps)
        p = min(max(pitcher_rates[cat], eps), 1.0 - eps)
        league = min(max(league_rates[cat], eps), 1.0 - eps)
        odds = (b / (1.0 - b)) * (p / (1.0 - p)) / (league / (1.0 - league))
        raw[cat] = odds / (1.0 + odds)

    total = sum(raw.values())
    if total <= 0.0:
        raise ValueError("combined rates collapsed to zero; check input rates")
    return {cat: v / total for cat, v in raw.items()}


def _self_convolve(pmf: NDArray[np.float64], n: int) -> NDArray[np.float64]:
    """The n-fold convolution of ``pmf`` with itself: the distribution of a
    sum of ``n`` iid draws. ``n = 0`` is a point mass at zero."""
    result = np.array([1.0])
    for _ in range(n):
        result = np.convolve(result, pmf)
    return result


def _compound(
    per_pa_pmf: NDArray[np.float64], count_pmf: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Mixture, over the plate-appearance count, of that many self-convolutions
    of the per-PA value distribution. This is what turns a per-PA rate into a
    realistic full-game distribution: fixing the PA count would understate
    variance for the same reason fixing batters faced would in the strikeout
    model.
    """
    max_n = count_pmf.size - 1
    per_pa_span = per_pa_pmf.size - 1
    max_len = max_n * per_pa_span + 1
    total = np.zeros(max_len, dtype=np.float64)
    for n, w in enumerate(count_pmf):
        if w <= 0.0:
            continue
        conv = _self_convolve(per_pa_pmf, n)
        total[: conv.size] += w * conv
    total = np.clip(total, 0.0, None)
    s = total.sum()
    if s <= 0.0:
        raise ValueError("compound distribution collapsed; check input rates")
    return total / s


def _per_pa_value_pmf(
    rates: Mapping[str, float], value_map: Mapping[str, int]
) -> NDArray[np.float64]:
    """Pushforward of the per-PA categorical distribution under a value map.

    Categories absent from ``value_map`` contribute to value 0 (e.g. an
    ordinary out contributes nothing to the total-bases count).
    """
    max_value = max(value_map.values(), default=0)
    pmf = np.zeros(max_value + 1, dtype=np.float64)
    for cat, rate in rates.items():
        value = value_map.get(cat, 0)
        pmf[value] += rate
    return pmf


@dataclass(frozen=True)
class BatterOutcomeDistribution:
    """The per-PA outcome distribution for one batter-game, and every prop
    priced from it.

    All props derived from one instance share ``correlation_group``: a
    batter's hits ladder and his total-bases ladder are not independent
    observations, since both are read off the same plate appearances in the
    same game.
    """

    per_pa_rates: Mapping[str, float]
    pa_pmf: NDArray[np.float64]
    correlation_group: Hashable
    batter: str = ""
    opponent: str = ""

    def __post_init__(self) -> None:
        _validate_rate_dict(self.per_pa_rates, "per_pa_rates")
        if not np.isclose(float(self.pa_pmf.sum()), 1.0, atol=1e-6):
            raise ValueError("pa_pmf must sum to 1")

    @property
    def expected_plate_appearances(self) -> float:
        return float(np.dot(np.arange(self.pa_pmf.size), self.pa_pmf))

    def pmf_for(self, stat: str) -> NDArray[np.float64]:
        """The full-game compound distribution for one prop.

        Raises for any stat not in :data:`STAT_VALUE_MAPS`, in particular
        "runs" and "rbi" -- see the module docstring for why those are not
        offered here.
        """
        if stat not in STAT_VALUE_MAPS:
            raise ValueError(
                f"unknown or unsupported stat {stat!r}; available: "
                f"{sorted(STAT_VALUE_MAPS)}. Runs and RBIs require game-state "
                "coupling and are deliberately not derivable from this engine."
            )
        per_pa = _per_pa_value_pmf(self.per_pa_rates, STAT_VALUE_MAPS[stat])
        return _compound(per_pa, self.pa_pmf)

    def mean(self, stat: str) -> float:
        pmf = self.pmf_for(stat)
        return float(np.dot(np.arange(pmf.size), pmf))

    def probability_over(self, stat: str, line: float) -> float:
        pmf = self.pmf_for(stat)
        k = np.arange(pmf.size)
        return float(pmf[k > line].sum())

    def probability_under(self, stat: str, line: float) -> float:
        pmf = self.pmf_for(stat)
        k = np.arange(pmf.size)
        return float(pmf[k < line].sum())

    def probability_push(self, stat: str, line: float) -> float:
        if not float(line).is_integer():
            return 0.0
        pmf = self.pmf_for(stat)
        k = int(line)
        return float(pmf[k]) if 0 <= k < pmf.size else 0.0

    def quote(self, stat: str, line: float, side: str) -> LineQuote:
        """Price one side of one line for one prop.

        Pushes are handled exactly as in the strikeout ladder: the fair price
        is conditional on the bet resolving at all, since a push on a
        whole-number line refunds the stake rather than losing it.
        """
        side = side.lower()
        if side not in ("over", "under"):
            raise ValueError(f"side must be 'over' or 'under', got {side!r}")

        win = self.probability_over(stat, line) if side == "over" else (
            self.probability_under(stat, line)
        )
        lose = self.probability_under(stat, line) if side == "over" else (
            self.probability_over(stat, line)
        )
        push = self.probability_push(stat, line)

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

    def ladder(self, stat: str, lines: Sequence[float] | None = None) -> list[LineQuote]:
        """Both sides of every line for one prop, sharing this batter-game's
        correlation group."""
        if lines is None:
            lines = standard_ladder(self.mean(stat), span=3)
        out: list[LineQuote] = []
        for line in lines:
            if line < 0:
                continue
            out.append(self.quote(stat, line, "over"))
            out.append(self.quote(stat, line, "under"))
        return out

    def price_against_market(
        self,
        stat: str,
        market: Mapping[float, tuple[float, float]],
        devig_method: str = "shin",
    ) -> list[LineQuote]:
        """Compare model prices to de-vigged market prices for one prop.

        Same requirement as the strikeout model: both sides of each line are
        needed, because a single side cannot be de-vigged, and comparing
        against a raw vig-inclusive implied probability manufactures edges
        that are not real.
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
                base = self.quote(stat, line, side)
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

    def summary(self, stats: Sequence[str] | None = None) -> str:
        if stats is None:
            stats = ("hits", "total_bases", "home_runs", "walks", "strikeouts")
        lines = [
            f"batter            {self.batter or '(unnamed)'}",
            f"opponent          {self.opponent or '(unnamed)'}",
            f"expected PA       {self.expected_plate_appearances:.2f}",
            f"correlation group {self.correlation_group}",
            "",
        ]
        for stat in stats:
            lines.append(f"  {stat}:  mean {self.mean(stat):.3f}")
            for line in standard_ladder(self.mean(stat), span=2):
                if line < 0:
                    continue
                lines.append(
                    f"    {line:<5.1f}  over {self.probability_over(stat, line):6.2%}"
                    f"   under {self.probability_under(stat, line):6.2%}"
                    f"   push {self.probability_push(stat, line):6.2%}"
                )
        return "\n".join(lines)


def build_batter_distribution(
    batter: BatterProfile,
    pitcher: PitcherAllowedProfile,
    league_rates: Mapping[str, float] = LEAGUE_PA_OUTCOME_RATES,
    correlation_group: Hashable | None = None,
    pa_pmf: NDArray[np.float64] | None = None,
) -> BatterOutcomeDistribution:
    """Combine a batter's and pitcher's rates into one outcome distribution
    for the game, ready to price every supported prop.
    """
    matchup = matchup_batter_rates(batter.rates, pitcher.rates, league_rates)

    if pa_pmf is None:
        pa_pmf = plate_appearances_pmf(
            batter.expected_plate_appearances, batter.plate_appearances_sd
        )
    pa_pmf = np.asarray(pa_pmf, dtype=np.float64)
    total = float(pa_pmf.sum())
    if total <= 0.0:
        raise ValueError("pa_pmf has no mass")
    pa_pmf = pa_pmf / total

    if correlation_group is None:
        correlation_group = f"{batter.name}-vs-{pitcher.name}"

    return BatterOutcomeDistribution(
        per_pa_rates=matchup,
        pa_pmf=pa_pmf,
        correlation_group=correlation_group,
        batter=batter.name,
        opponent=pitcher.name,
    )
