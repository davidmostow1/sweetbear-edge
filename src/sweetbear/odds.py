"""Conversions between odds formats and implied probability.

Decimal odds are the internal canonical representation: a decimal price ``d``
returns ``d`` units total (stake included) for a 1-unit winning stake.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "american_to_decimal",
    "decimal_to_american",
    "fractional_to_decimal",
    "decimal_to_fractional",
    "decimal_to_implied",
    "implied_to_decimal",
    "american_to_implied",
    "implied_to_american",
    "overround",
    "vig",
]

_MIN_PROB = 1e-12


def _as_float_array(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def american_to_decimal(american: ArrayLike) -> NDArray[np.float64]:
    """Convert American (moneyline) odds to decimal odds.

    ``+150`` -> 2.5, ``-200`` -> 1.5. Values in ``(-100, 100)`` are not
    representable moneylines and raise.
    """
    a = _as_float_array(american)
    if np.any(np.abs(a) < 100.0):
        raise ValueError("American odds must satisfy |odds| >= 100")
    return np.where(a > 0, 1.0 + a / 100.0, 1.0 + 100.0 / np.abs(a))


def decimal_to_american(decimal: ArrayLike) -> NDArray[np.float64]:
    """Convert decimal odds to American (moneyline) odds."""
    d = _validate_decimal(decimal)
    return np.where(d >= 2.0, (d - 1.0) * 100.0, -100.0 / (d - 1.0))


def fractional_to_decimal(numerator: ArrayLike, denominator: ArrayLike) -> NDArray[np.float64]:
    """Convert fractional odds (``numerator/denominator``) to decimal odds."""
    num = _as_float_array(numerator)
    den = _as_float_array(denominator)
    if np.any(num <= 0) or np.any(den <= 0):
        raise ValueError("Fractional odds require positive numerator and denominator")
    return 1.0 + num / den


def decimal_to_fractional(decimal: ArrayLike, max_denominator: int = 100) -> list[tuple[int, int]]:
    """Convert decimal odds to the nearest fractional odds.

    Returns ``(numerator, denominator)`` pairs approximated with a bounded
    denominator, which is how books actually quote fractions.
    """
    from fractions import Fraction

    d = np.atleast_1d(_validate_decimal(decimal))
    out: list[tuple[int, int]] = []
    for value in d:
        frac = Fraction(float(value) - 1.0).limit_denominator(max_denominator)
        out.append((frac.numerator, frac.denominator))
    return out


def decimal_to_implied(decimal: ArrayLike) -> NDArray[np.float64]:
    """Implied (vig-inclusive) probability of a decimal price."""
    return 1.0 / _validate_decimal(decimal)


def implied_to_decimal(probability: ArrayLike) -> NDArray[np.float64]:
    """Decimal price corresponding to a probability."""
    return 1.0 / _validate_probability(probability)


def american_to_implied(american: ArrayLike) -> NDArray[np.float64]:
    """Implied (vig-inclusive) probability of an American price."""
    return decimal_to_implied(american_to_decimal(american))


def implied_to_american(probability: ArrayLike) -> NDArray[np.float64]:
    """American price corresponding to a probability."""
    return decimal_to_american(implied_to_decimal(probability))


def overround(decimal_odds: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Sum of implied probabilities across a market.

    A fair market sums to 1.0; anything above is the book's margin. Also known
    as the "book sum" or "total book".
    """
    return np.sum(decimal_to_implied(decimal_odds), axis=axis)


def vig(decimal_odds: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Bookmaker margin as a fraction of the total book.

    ``vig = (overround - 1) / overround``: the share of every unit wagered the
    book keeps if bettors distribute stakes proportionally to the prices.
    """
    book = overround(decimal_odds, axis=axis)
    return (book - 1.0) / book


def _validate_decimal(decimal: ArrayLike) -> NDArray[np.float64]:
    d = _as_float_array(decimal)
    if np.any(d <= 1.0):
        raise ValueError("Decimal odds must be greater than 1.0")
    return d


def _validate_probability(probability: ArrayLike) -> NDArray[np.float64]:
    p = _as_float_array(probability)
    if np.any(p <= _MIN_PROB) or np.any(p >= 1.0):
        raise ValueError("Probabilities must lie strictly inside (0, 1)")
    return p
