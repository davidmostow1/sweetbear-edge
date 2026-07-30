"""Chronological bankroll simulation that refuses to lie to you.

Most backtests are broken in one of three ways, and each is defended against
here explicitly:

1. **Lookahead.** Bets are sorted by timestamp and settled in order.
   ``walk_forward_calibrate`` fits a recalibrator only on bets that had already
   settled, so a calibration curve can never be informed by its own future.
2. **Turnover laundering.** Return is reported both per unit staked (yield) and
   per unit of bankroll, because a 3% yield on 10,000 bets and a 3% yield on 40
   are not the same claim.
3. **Variance mistaken for skill.** Every result carries a t-statistic and a
   bootstrap interval. A backtest without an error bar is a story, not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from .calibration import brier_score, expected_calibration_error, log_loss
from .edge import closing_line_value, fractional_kelly

__all__ = [
    "Bet",
    "BacktestResult",
    "run_backtest",
    "flat_stake",
    "percent_bankroll",
    "kelly_stake",
    "walk_forward_calibrate",
]

StakingPolicy = Callable[[float, "Bet"], float]


@dataclass(frozen=True)
class Bet:
    """A single settled wager.

    ``probability`` is the model's estimate at the time of the bet.
    ``decimal_odds`` is the price actually taken, vig included.
    ``closing_fair_probability`` is the de-vigged market probability at close,
    and is what makes CLV reporting possible.
    """

    timestamp: Any
    probability: float
    decimal_odds: float
    outcome: int
    closing_fair_probability: float | None = None
    market_id: str | None = None
    label: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError(f"probability out of range: {self.probability}")
        if self.decimal_odds <= 1.0:
            raise ValueError(f"decimal_odds must exceed 1.0: {self.decimal_odds}")
        if self.outcome not in (0, 1):
            raise ValueError(f"outcome must be 0 or 1: {self.outcome}")


@dataclass(frozen=True)
class BacktestResult:
    """Everything needed to decide whether a strategy is real."""

    n_bets: int
    n_wins: int
    total_staked: float
    profit: float
    initial_bankroll: float
    final_bankroll: float
    bankroll_curve: NDArray[np.float64]
    max_drawdown: float
    longest_losing_streak: int
    per_bet_returns: NDArray[np.float64]
    mean_clv: float | None
    clv_beat_rate: float | None
    brier: float
    log_loss: float
    ece: float
    went_broke: bool
    skipped: int = 0
    _rng_seed: int = field(default=0, repr=False)

    @property
    def hit_rate(self) -> float:
        return self.n_wins / self.n_bets if self.n_bets else 0.0

    @property
    def yield_(self) -> float:
        """Profit per unit staked. The standard betting-industry ROI."""
        return self.profit / self.total_staked if self.total_staked else 0.0

    @property
    def bankroll_growth(self) -> float:
        return self.final_bankroll / self.initial_bankroll - 1.0

    @property
    def t_statistic(self) -> float:
        """t-stat of per-unit-stake returns against zero.

        Assumes bets are independent. Correlated positions -- the same game
        from three angles, or a whole slate moving with one injury -- inflate
        this, so treat it as an optimistic bound rather than a p-value.
        """
        r = self.per_bet_returns
        if r.size < 2:
            return 0.0
        sd = float(np.std(r, ddof=1))
        if sd == 0.0:
            return 0.0
        return float(np.mean(r)) / (sd / np.sqrt(r.size))

    def yield_confidence_interval(
        self, level: float = 0.95, n_resamples: int = 10_000, seed: int = 0
    ) -> tuple[float, float]:
        """Bootstrap interval for yield. If it straddles zero, you have nothing."""
        r = self.per_bet_returns
        if r.size < 2:
            return (0.0, 0.0)
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, r.size, size=(n_resamples, r.size))
        means = r[draws].mean(axis=1)
        alpha = (1.0 - level) / 2.0
        return (
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1.0 - alpha)),
        )

    def summary(self) -> str:
        lo, hi = self.yield_confidence_interval()
        lines = [
            f"bets              {self.n_bets:,} ({self.skipped:,} skipped)",
            f"hit rate          {self.hit_rate:.2%}",
            f"turnover          {self.total_staked:,.2f}",
            f"profit            {self.profit:,.2f}",
            f"yield             {self.yield_:+.2%}  95% CI [{lo:+.2%}, {hi:+.2%}]",
            f"bankroll          {self.initial_bankroll:,.2f} -> {self.final_bankroll:,.2f}"
            f"  ({self.bankroll_growth:+.2%})",
            f"max drawdown      {self.max_drawdown:.2%}",
            f"losing streak     {self.longest_losing_streak}",
            f"t-statistic       {self.t_statistic:+.2f}",
            f"brier / logloss   {self.brier:.4f} / {self.log_loss:.4f}",
            f"calibration err   {self.ece:.4f}",
        ]
        if self.mean_clv is not None:
            lines.append(f"mean CLV          {self.mean_clv:+.2%}")
            lines.append(f"CLV beat rate     {self.clv_beat_rate:.2%}")
        if self.went_broke:
            lines.append("BANKRUPT          bankroll hit zero before the sample ended")
        return "\n".join(lines)


def flat_stake(unit: float) -> StakingPolicy:
    """Stake a fixed amount regardless of bankroll or edge."""
    if unit <= 0.0:
        raise ValueError("unit must be positive")
    return lambda bankroll, bet: unit


def percent_bankroll(pct: float) -> StakingPolicy:
    """Stake a fixed percentage of the current bankroll."""
    if not 0.0 < pct <= 1.0:
        raise ValueError("pct must lie in (0, 1]")
    return lambda bankroll, bet: bankroll * pct


def kelly_stake(
    fraction: float = 0.25, cap: float = 0.05, compounding: bool = True
) -> StakingPolicy:
    """Fractional Kelly sizing, optionally on a fixed rather than live bankroll.

    ``compounding=False`` sizes off the starting bankroll throughout, which
    isolates edge from the compounding that flatters any winning sequence.
    """

    def policy(bankroll: float, bet: Bet) -> float:
        base = bankroll if compounding else policy.reference  # type: ignore[attr-defined]
        f = float(fractional_kelly(bet.probability, bet.decimal_odds, fraction, cap))
        return base * f

    policy.reference = 0.0  # type: ignore[attr-defined]
    return policy


def run_backtest(
    bets: Iterable[Bet],
    staking: StakingPolicy | None = None,
    initial_bankroll: float = 1000.0,
    min_edge: float = 0.0,
    stop_on_ruin: bool = True,
) -> BacktestResult:
    """Settle bets in chronological order and report what actually happened.

    ``min_edge`` filters out bets whose expected value per unit staked falls
    below the threshold; those are counted as skipped rather than silently
    dropped, so selectivity stays visible in the output.
    """
    ordered = sorted(bets, key=lambda b: b.timestamp)
    if not ordered:
        raise ValueError("No bets to simulate")
    if initial_bankroll <= 0.0:
        raise ValueError("initial_bankroll must be positive")

    if staking is None:
        staking = kelly_stake()
    if hasattr(staking, "reference"):
        staking.reference = initial_bankroll  # type: ignore[attr-defined]

    bankroll = initial_bankroll
    curve = [initial_bankroll]
    staked_total = 0.0
    profit_total = 0.0
    wins = 0
    skipped = 0
    returns: list[float] = []
    clvs: list[float] = []
    used_probs: list[float] = []
    used_outcomes: list[int] = []
    losing_streak = 0
    longest_streak = 0
    went_broke = False

    for bet in ordered:
        ev = bet.probability * bet.decimal_odds - 1.0
        if ev < min_edge:
            skipped += 1
            continue

        stake = min(max(float(staking(bankroll, bet)), 0.0), bankroll)
        if stake <= 0.0:
            skipped += 1
            continue

        payout = stake * (bet.decimal_odds - 1.0) if bet.outcome == 1 else -stake
        bankroll += payout
        staked_total += stake
        profit_total += payout
        returns.append(payout / stake)
        used_probs.append(bet.probability)
        used_outcomes.append(bet.outcome)
        curve.append(bankroll)

        if bet.outcome == 1:
            wins += 1
            losing_streak = 0
        else:
            losing_streak += 1
            longest_streak = max(longest_streak, losing_streak)

        if bet.closing_fair_probability is not None:
            clvs.append(
                float(closing_line_value(bet.decimal_odds, bet.closing_fair_probability))
            )

        if bankroll <= 0.0:
            went_broke = True
            if stop_on_ruin:
                break

    if not returns:
        raise ValueError("Every bet was filtered out; nothing to report")

    curve_arr = np.asarray(curve, dtype=np.float64)
    peak = np.maximum.accumulate(curve_arr)
    drawdown = np.where(peak > 0.0, (peak - curve_arr) / peak, 1.0)

    probs = np.asarray(used_probs, dtype=np.float64)
    outcomes = np.asarray(used_outcomes, dtype=np.float64)

    return BacktestResult(
        n_bets=len(returns),
        n_wins=wins,
        total_staked=staked_total,
        profit=profit_total,
        initial_bankroll=initial_bankroll,
        final_bankroll=float(curve_arr[-1]),
        bankroll_curve=curve_arr,
        max_drawdown=float(np.max(drawdown)),
        longest_losing_streak=longest_streak,
        per_bet_returns=np.asarray(returns, dtype=np.float64),
        mean_clv=float(np.mean(clvs)) if clvs else None,
        clv_beat_rate=float(np.mean(np.asarray(clvs) > 0.0)) if clvs else None,
        brier=brier_score(outcomes, probs),
        log_loss=log_loss(outcomes, probs),
        ece=expected_calibration_error(outcomes, probs),
        went_broke=went_broke,
        skipped=skipped,
    )


def walk_forward_calibrate(
    bets: Sequence[Bet],
    calibrator_factory: Callable[[], Any],
    min_train: int = 200,
    refit_every: int = 50,
) -> list[Bet]:
    """Recalibrate each bet's probability using only bets that settled earlier.

    This is the only defensible way to report a calibrated backtest. Fitting a
    calibrator on the whole sample and then scoring that same sample produces a
    reliability curve that looks immaculate and means nothing.

    The first ``min_train`` bets are returned with probabilities untouched,
    since there is nothing yet to learn from. The calibrator is refit every
    ``refit_every`` bets rather than on every single one, which is both faster
    and closer to how a model is actually maintained in production.
    """
    if min_train < 2:
        raise ValueError("min_train must be at least 2")
    if refit_every < 1:
        raise ValueError("refit_every must be at least 1")

    ordered = sorted(bets, key=lambda b: b.timestamp)
    out: list[Bet] = []
    calibrator: Any = None
    fitted_at = 0

    for i, bet in enumerate(ordered):
        if i >= min_train and (calibrator is None or i - fitted_at >= refit_every):
            history = ordered[:i]
            y = np.array([b.outcome for b in history], dtype=np.float64)
            p = np.array([b.probability for b in history], dtype=np.float64)
            if 0.0 < y.mean() < 1.0:
                calibrator = calibrator_factory().fit(y, p)
                fitted_at = i

        if calibrator is None:
            out.append(bet)
        else:
            adjusted = float(np.clip(calibrator.transform([bet.probability])[0], 0.0, 1.0))
            out.append(replace(bet, probability=adjusted))

    return out
