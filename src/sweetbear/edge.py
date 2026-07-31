"""Turning a probability into a position size.

Two distinct quantities get confused constantly, so they are kept apart here:

*Expected value* is computed at the price you actually receive -- the posted,
vig-inclusive odds. That is the money.

*Edge versus the fair line* compares your probability to the de-vigged market
consensus. That is the diagnostic. It tells you whether your number disagrees
with the sharpest available estimate, and by how much. A large EV that comes
with a large disagreement against a sharp closing line is usually a modelling
error, not an opportunity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

__all__ = [
    "expected_value",
    "breakeven_probability",
    "kelly_fraction",
    "fractional_kelly",
    "shrink_toward_market",
    "simultaneous_kelly",
    "closing_line_value",
    "growth_rate",
]


def expected_value(probability: ArrayLike, decimal_odds: ArrayLike) -> NDArray[np.float64]:
    """Expected profit per unit staked at the offered price.

    ``p * d - 1``. Zero means the bet is a coin flip against the price; the
    break-even probability is ``1 / d``.
    """
    p = np.asarray(probability, dtype=np.float64)
    d = np.asarray(decimal_odds, dtype=np.float64)
    if np.any(d <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.0")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Probabilities must lie in [0, 1]")
    return p * d - 1.0


def breakeven_probability(decimal_odds: ArrayLike) -> NDArray[np.float64]:
    """Win rate required to break even at a given price."""
    d = np.asarray(decimal_odds, dtype=np.float64)
    if np.any(d <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.0")
    return 1.0 / d


def kelly_fraction(probability: ArrayLike, decimal_odds: ArrayLike) -> NDArray[np.float64]:
    """Full-Kelly stake as a fraction of bankroll, floored at zero.

    ``(p * d - 1) / (d - 1)``. Maximizes long-run log growth *given that ``p``
    is exactly right*. It never is, which is why nobody sane bets full Kelly --
    see ``fractional_kelly``.
    """
    ev = expected_value(probability, decimal_odds)
    b = np.asarray(decimal_odds, dtype=np.float64) - 1.0
    return np.maximum(ev / b, 0.0)


def fractional_kelly(
    probability: ArrayLike,
    decimal_odds: ArrayLike,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> NDArray[np.float64]:
    """Scaled-down Kelly stake, additionally capped as a share of bankroll.

    Estimation error in ``p`` enters the growth rate quadratically, so
    over-betting is punished far more than under-betting. Quarter Kelly gives
    up roughly 25% of theoretical growth while cutting variance by about 75%
    and surviving a badly mis-specified model. The ``cap`` is the backstop for
    the case where the model is not merely wrong but broken.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if cap <= 0.0:
        raise ValueError("cap must be positive")
    return np.minimum(kelly_fraction(probability, decimal_odds) * fraction, cap)


def shrink_toward_market(
    model_probability: ArrayLike,
    market_probability: ArrayLike,
    confidence: float = 0.5,
) -> NDArray[np.float64]:
    """Blend the model with the de-vigged market line in log-odds space.

    ``confidence`` is the weight on the model. Shrinking in log-odds rather
    than probability space keeps the blend sane in the tails, where a linear
    average of 1% and 5% badly misstates the implied price. Use the blended
    number for staking and keep the raw model number for scoring, otherwise
    you can no longer tell whether the model is adding anything.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must lie in [0, 1]")
    eps = 1e-12
    p_model = np.clip(np.asarray(model_probability, dtype=np.float64), eps, 1.0 - eps)
    p_market = np.clip(np.asarray(market_probability, dtype=np.float64), eps, 1.0 - eps)
    logit_model = np.log(p_model / (1.0 - p_model))
    logit_market = np.log(p_market / (1.0 - p_market))
    blended = confidence * logit_model + (1.0 - confidence) * logit_market
    return 1.0 / (1.0 + np.exp(-blended))


def simultaneous_kelly(
    probabilities: ArrayLike,
    decimal_odds: ArrayLike,
    fraction: float = 1.0,
    max_total_stake: float = 1.0,
) -> NDArray[np.float64]:
    """Optimal simultaneous stakes across mutually exclusive outcomes.

    Betting several outcomes of the same market is not the same problem as
    sizing each in isolation. Because the outcomes are mutually exclusive the
    stakes partially hedge one another -- whichever way the market lands, one
    ticket is live -- so per-bet Kelly, which prices each leg as a standalone
    all-or-nothing risk, typically *under*stakes here. (The familiar warning
    that per-bet Kelly overstakes applies to simultaneous bets on *different*
    events, where the risks add instead of offsetting.) This maximizes expected
    log wealth jointly, subject to non-negative stakes and an exposure cap.

    ``probabilities`` need not sum to 1 -- any residual mass is treated as the
    "none of these" outcome, which is correct for partial markets.
    """
    p = np.asarray(probabilities, dtype=np.float64).ravel()
    d = np.asarray(decimal_odds, dtype=np.float64).ravel()
    if p.shape != d.shape:
        raise ValueError("probabilities and decimal_odds must have the same length")
    if np.any(d <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.0")
    if np.any(p < 0.0) or p.sum() > 1.0 + 1e-9:
        raise ValueError("Probabilities must be non-negative and sum to at most 1")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")

    n = p.size
    residual = max(0.0, 1.0 - float(p.sum()))
    floor = 1e-9

    def wealth(f: NDArray[np.float64]) -> NDArray[np.float64]:
        cash = 1.0 - f.sum()
        return np.maximum(cash + f * d, floor)

    def objective(f: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
        w = wealth(f)
        cash = max(1.0 - f.sum(), floor)
        # Negated: scipy minimizes, we want maximum log growth.
        value = -(float(np.sum(p * np.log(w))) + residual * np.log(cash))
        common = float(np.sum(p / w)) + residual / cash
        grad = -(p * d / w - common)
        return value, grad

    constraints = [{"type": "ineq", "fun": lambda f: max_total_stake - f.sum()}]
    result = minimize(
        objective,
        x0=np.full(n, min(0.01, max_total_stake / max(n, 1))),
        jac=True,
        bounds=[(0.0, max_total_stake)] * n,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-12},
    )
    stakes = np.maximum(result.x, 0.0)
    stakes[stakes < 1e-9] = 0.0
    return stakes * fraction


def closing_line_value(
    bet_decimal_odds: ArrayLike, closing_fair_probability: ArrayLike
) -> NDArray[np.float64]:
    """Expected value of the bet measured against the closing fair line.

    The single most honest short-horizon metric available. Results take
    thousands of bets to separate skill from variance; CLV converges in
    hundreds, because the closing line is the market's best estimate and beating
    it consistently is the definition of an edge. Pass a *de-vigged* closing
    probability -- comparing against a raw closing price silently books the
    bookmaker's margin as your skill.
    """
    d = np.asarray(bet_decimal_odds, dtype=np.float64)
    p_close = np.asarray(closing_fair_probability, dtype=np.float64)
    if np.any(d <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.0")
    if np.any(p_close <= 0.0) or np.any(p_close >= 1.0):
        raise ValueError("Closing probabilities must lie strictly inside (0, 1)")
    return d * p_close - 1.0


def growth_rate(
    probability: ArrayLike, decimal_odds: ArrayLike, stake_fraction: ArrayLike
) -> NDArray[np.float64]:
    """Expected log-growth per bet at a given stake fraction.

    Plot this against ``stake_fraction`` to see the asymmetry that justifies
    fractional Kelly: the curve is nearly flat to the left of the optimum and
    falls off a cliff to the right, crossing zero at twice full Kelly.
    """
    p = np.asarray(probability, dtype=np.float64)
    d = np.asarray(decimal_odds, dtype=np.float64)
    f = np.asarray(stake_fraction, dtype=np.float64)
    if np.any(f < 0.0) or np.any(f >= 1.0):
        raise ValueError("stake_fraction must lie in [0, 1)")
    if np.any(d <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.0")
    return p * np.log1p(f * (d - 1.0)) + (1.0 - p) * np.log1p(-f)
