"""Removing bookmaker margin to recover fair probabilities.

Posted prices are not probabilities. A market quoted at -110/-110 implies
52.38% on both sides, summing to 104.76%. That 4.76% overround is the book's
margin, and how you strip it materially changes the fair line you compare your
model against.

Four standard estimators are provided. They disagree most on longshots, which
is exactly where mis-attributed edge tends to appear:

``multiplicative``
    Scale every price by the same factor. Simple, but it assumes margin is
    applied proportionally, which understates the true probability of favorites
    and overstates longshots.
``additive``
    Subtract an equal share of the margin from each outcome. Fails outright
    when an outcome's raw price is smaller than its share of the margin.
``power``
    Solve for the exponent ``k`` where ``sum(q_i ** k) == 1``. Applies more
    margin to longshots than favorites.
``shin``
    Models the margin as the book's defense against insider money, solving for
    the implied insider fraction ``z``. Generally the best-behaved estimator on
    real books, and the default here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from .odds import decimal_to_implied

__all__ = ["devig", "devig_probabilities", "shin_z", "power_exponent", "METHODS"]

METHODS = ("multiplicative", "additive", "power", "shin")

_TOL = 1e-12
_MAX_EXPONENT = 1024.0


def devig(
    decimal_odds: ArrayLike,
    method: str = "shin",
    axis: int = -1,
) -> NDArray[np.float64]:
    """Strip margin from decimal odds, returning fair probabilities.

    ``decimal_odds`` holds the complete set of mutually exclusive, collectively
    exhaustive outcomes for a market along ``axis``. Partial markets produce
    meaningless results, because the estimator has no way to know how much of
    the book it is seeing.
    """
    raw = decimal_to_implied(decimal_odds)
    return devig_probabilities(raw, method=method, axis=axis)


def devig_probabilities(
    raw_probabilities: ArrayLike,
    method: str = "shin",
    axis: int = -1,
) -> NDArray[np.float64]:
    """Strip margin from vig-inclusive implied probabilities."""
    if method not in METHODS:
        raise ValueError(f"Unknown de-vig method {method!r}; expected one of {METHODS}")
    q = np.asarray(raw_probabilities, dtype=np.float64)
    if q.ndim == 0:
        raise ValueError("A market needs at least two outcomes")
    if np.any(q <= 0.0):
        raise ValueError("Implied probabilities must be positive")
    solver = _SOLVERS[method]
    return np.apply_along_axis(solver, axis, q)


def shin_z(raw_probabilities: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Estimated insider-money fraction implied by each market's prices.

    Rising ``z`` across a book is a signal that the market is defending against
    informed flow, which usually means the line is sharper than it looks.
    """
    q = np.asarray(raw_probabilities, dtype=np.float64)
    return np.apply_along_axis(_solve_shin_z, axis, q)


def power_exponent(raw_probabilities: ArrayLike, axis: int = -1) -> NDArray[np.float64]:
    """Exponent ``k`` satisfying ``sum(q ** k) == 1`` for each market."""
    q = np.asarray(raw_probabilities, dtype=np.float64)
    return np.apply_along_axis(_solve_power_k, axis, q)


def _multiplicative(q: NDArray[np.float64]) -> NDArray[np.float64]:
    return q / q.sum()


def _additive(q: NDArray[np.float64]) -> NDArray[np.float64]:
    excess = (q.sum() - 1.0) / q.size
    p = q - excess
    if np.any(p <= 0.0):
        raise ValueError(
            "Additive de-vig produced a non-positive probability; the margin "
            "share exceeds a longshot's raw price. Use 'shin' or 'power'."
        )
    return p


def _power(q: NDArray[np.float64]) -> NDArray[np.float64]:
    k = _solve_power_k(q)
    p = q**k
    return p / p.sum()


def _shin(q: NDArray[np.float64]) -> NDArray[np.float64]:
    z = _solve_shin_z(q)
    if z == 0.0:
        return _multiplicative(q)
    p = _shin_probabilities(q, z)
    return p / p.sum()


def _solve_power_k(q: NDArray[np.float64]) -> float:
    if q.size < 2:
        raise ValueError("A market needs at least two outcomes")

    def objective(k: float) -> float:
        return float(np.sum(q**k) - 1.0)

    at_one = objective(1.0)
    if abs(at_one) < _TOL:
        return 1.0
    if at_one > 0.0:
        lo, hi = 1.0, 2.0
        while objective(hi) > 0.0:
            hi *= 2.0
            if hi > _MAX_EXPONENT:
                raise ValueError("Power de-vig failed to bracket a solution")
    else:
        lo, hi = 0.5, 1.0
        while objective(lo) < 0.0:
            lo /= 2.0
            if lo < 1.0 / _MAX_EXPONENT:
                raise ValueError("Power de-vig failed to bracket a solution")
    return float(brentq(objective, lo, hi, xtol=1e-14, rtol=1e-15))


def _shin_probabilities(q: NDArray[np.float64], z: float) -> NDArray[np.float64]:
    total = q.sum()
    inner = z * z + 4.0 * (1.0 - z) * q * q / total
    return (np.sqrt(inner) - z) / (2.0 * (1.0 - z))


def _solve_shin_z(q: NDArray[np.float64]) -> float:
    """Solve ``sum(shin_probabilities(q, z)) == 1`` for the insider fraction.

    At ``z = 0`` the sum is ``sqrt(book)``, so the objective is positive for any
    overround book. As ``z -> 1`` the sum tends to ``sum(q**2) / book``, which is
    bounded above by ``max(q) < 1``, so the bracket is always valid.
    """
    if q.size < 2:
        raise ValueError("A market needs at least two outcomes")
    total = q.sum()
    if total < 1.0 - _TOL:
        raise ValueError(
            "Shin de-vig requires a book summing to at least 1.0; got "
            f"{total:.6f}. An underround book has no margin to strip."
        )
    if abs(total - 1.0) < _TOL:
        return 0.0

    def objective(z: float) -> float:
        return float(np.sum(_shin_probabilities(q, z)) - 1.0)

    return float(brentq(objective, 0.0, 1.0 - 1e-9, xtol=1e-14, rtol=1e-15))


_SOLVERS = {
    "multiplicative": _multiplicative,
    "additive": _additive,
    "power": _power,
    "shin": _shin,
}
