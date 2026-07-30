"""Scoring and repairing the honesty of predicted probabilities.

A model that says "70%" must be right about 70% of the time. Discrimination
without calibration is worthless for staking: Kelly sizing on an overconfident
model does not merely underperform, it goes broke. Everything here exists to
answer one question -- do these numbers mean what they claim to mean?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

__all__ = [
    "brier_score",
    "log_loss",
    "brier_decomposition",
    "BrierDecomposition",
    "reliability_curve",
    "ReliabilityCurve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "PlattScaler",
    "IsotonicCalibrator",
]

_EPS = 1e-15


def _validate(y_true: ArrayLike, y_prob: ArrayLike) -> tuple[NDArray, NDArray]:
    y = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_prob, dtype=np.float64).ravel()
    if y.shape != p.shape:
        raise ValueError(f"Shape mismatch: {y.shape} outcomes vs {p.shape} predictions")
    if y.size == 0:
        raise ValueError("Need at least one observation")
    if not np.all((y == 0.0) | (y == 1.0)):
        raise ValueError("Outcomes must be binary 0/1")
    if np.any(p < 0.0) or np.any(p > 1.0):
        raise ValueError("Probabilities must lie in [0, 1]")
    return y, p


def brier_score(y_true: ArrayLike, y_prob: ArrayLike) -> float:
    """Mean squared error of probabilistic forecasts. Lower is better.

    A model that always predicts the base rate scores ``p*(1-p)``; beating that
    is the minimum bar for the model being worth anything.
    """
    y, p = _validate(y_true, y_prob)
    return float(np.mean((p - y) ** 2))


def log_loss(y_true: ArrayLike, y_prob: ArrayLike, eps: float = _EPS) -> float:
    """Negative log likelihood per observation. Lower is better.

    Punishes confident mistakes far harder than Brier does, which is the right
    bias when the downstream consumer is a Kelly staking rule.
    """
    y, p = _validate(y_true, y_prob)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy's three-way split of the Brier score.

    ``brier == reliability - resolution + uncertainty`` up to binning error.
    """

    reliability: float
    resolution: float
    uncertainty: float

    @property
    def brier(self) -> float:
        return self.reliability - self.resolution + self.uncertainty

    @property
    def skill_score(self) -> float:
        """Fraction of the climatological Brier score removed by the model."""
        if self.uncertainty == 0.0:
            return 0.0
        return (self.resolution - self.reliability) / self.uncertainty


def brier_decomposition(
    y_true: ArrayLike, y_prob: ArrayLike, n_bins: int = 10
) -> BrierDecomposition:
    """Split Brier into reliability, resolution, and uncertainty.

    ``reliability`` is calibration error (lower is better). ``resolution`` is
    how far predictions move away from the base rate in the right direction
    (higher is better). ``uncertainty`` is the irreducible variance of the
    outcome itself and is a property of the data, not the model.
    """
    y, p = _validate(y_true, y_prob)
    base_rate = float(np.mean(y))
    uncertainty = base_rate * (1.0 - base_rate)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    reliability = 0.0
    resolution = 0.0
    n = y.size
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        mean_pred = float(np.mean(p[mask]))
        observed = float(np.mean(y[mask]))
        weight = count / n
        reliability += weight * (mean_pred - observed) ** 2
        resolution += weight * (observed - base_rate) ** 2

    return BrierDecomposition(reliability, resolution, uncertainty)


@dataclass(frozen=True)
class ReliabilityCurve:
    """Binned agreement between predicted and realized frequency."""

    mean_predicted: NDArray[np.float64]
    observed_frequency: NDArray[np.float64]
    counts: NDArray[np.int64]

    @property
    def gaps(self) -> NDArray[np.float64]:
        """Signed miss per bin. Positive means the model is overconfident."""
        return self.mean_predicted - self.observed_frequency


def reliability_curve(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> ReliabilityCurve:
    """Bin predictions and compare each bin's mean forecast to its hit rate.

    ``strategy='uniform'`` uses equal-width bins, which leaves sparse bins in
    the tails. ``strategy='quantile'`` uses equal-count bins, which gives every
    point on the curve comparable statistical weight -- usually what you want
    when judging whether a gap is real or noise.
    """
    y, p = _validate(y_true, y_prob)
    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
        if edges.size < 2:
            edges = np.array([0.0, 1.0])
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'")

    bin_ids = np.clip(np.digitize(p, edges[1:-1], right=False), 0, edges.size - 2)

    means, observed, counts = [], [], []
    for b in range(edges.size - 1):
        mask = bin_ids == b
        count = int(np.count_nonzero(mask))
        if count == 0:
            continue
        means.append(float(np.mean(p[mask])))
        observed.append(float(np.mean(y[mask])))
        counts.append(count)

    return ReliabilityCurve(
        np.asarray(means, dtype=np.float64),
        np.asarray(observed, dtype=np.float64),
        np.asarray(counts, dtype=np.int64),
    )


def expected_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Count-weighted mean absolute gap between forecast and reality."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    weights = curve.counts / curve.counts.sum()
    return float(np.sum(weights * np.abs(curve.gaps)))


def maximum_calibration_error(
    y_true: ArrayLike,
    y_prob: ArrayLike,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> float:
    """Worst single-bin calibration gap. The number that bankrupts you."""
    curve = reliability_curve(y_true, y_prob, n_bins=n_bins, strategy=strategy)
    return float(np.max(np.abs(curve.gaps)))


def _logit(p: NDArray[np.float64], eps: float = 1e-12) -> NDArray[np.float64]:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class PlattScaler:
    """Two-parameter logistic recalibration: ``sigmoid(a * logit(p) + b)``.

    Corrects systematic over/underconfidence and bias with almost no capacity
    to overfit, which makes it the safe default on small samples. ``a < 1``
    means the raw model was overconfident.
    """

    def __init__(self, regularize: bool = True) -> None:
        self.regularize = regularize
        self.a_: float | None = None
        self.b_: float | None = None

    def fit(self, y_true: ArrayLike, y_prob: ArrayLike) -> PlattScaler:
        y, p = _validate(y_true, y_prob)
        x = _logit(p)

        if self.regularize:
            # Platt's smoothed targets pull the fit off the 0/1 boundary so a
            # perfectly separable sample cannot drive the slope to infinity.
            n_pos = float(np.sum(y))
            n_neg = float(y.size - n_pos)
            hi = (n_pos + 1.0) / (n_pos + 2.0)
            lo = 1.0 / (n_neg + 2.0)
            target = np.where(y > 0.5, hi, lo)
        else:
            target = y

        def objective(params: NDArray[np.float64]) -> tuple[float, NDArray[np.float64]]:
            a, b = params
            z = a * x + b
            # log(1 + exp(z)) evaluated stably for large |z|
            softplus = np.logaddexp(0.0, z)
            loss = float(np.mean(softplus - target * z))
            sigma = 1.0 / (1.0 + np.exp(-z))
            residual = sigma - target
            grad = np.array([np.mean(residual * x), np.mean(residual)])
            return loss, grad

        result = minimize(objective, x0=np.array([1.0, 0.0]), jac=True, method="L-BFGS-B")
        self.a_, self.b_ = float(result.x[0]), float(result.x[1])
        return self

    def transform(self, y_prob: ArrayLike) -> NDArray[np.float64]:
        if self.a_ is None or self.b_ is None:
            raise RuntimeError("PlattScaler must be fit before transform")
        z = self.a_ * _logit(np.asarray(y_prob, dtype=np.float64)) + self.b_
        return 1.0 / (1.0 + np.exp(-z))

    def fit_transform(self, y_true: ArrayLike, y_prob: ArrayLike) -> NDArray[np.float64]:
        return self.fit(y_true, y_prob).transform(y_prob)


class IsotonicCalibrator:
    """Non-parametric monotone recalibration by pool-adjacent-violators.

    Strictly more flexible than Platt and therefore strictly more dangerous:
    it will happily memorize a few hundred observations. Fit it on a held-out
    calibration split, never on the data the model was trained on.
    """

    def __init__(self, out_of_bounds: str = "clip") -> None:
        if out_of_bounds != "clip":
            raise ValueError("only out_of_bounds='clip' is supported")
        self.x_: NDArray[np.float64] | None = None
        self.y_: NDArray[np.float64] | None = None

    def fit(self, y_true: ArrayLike, y_prob: ArrayLike) -> IsotonicCalibrator:
        y, p = _validate(y_true, y_prob)
        order = np.argsort(p, kind="mergesort")
        x_sorted = p[order]
        y_sorted = y[order]

        fitted = _pool_adjacent_violators(y_sorted)

        # Collapse ties in x so the step function is single-valued.
        unique_x, inverse = np.unique(x_sorted, return_inverse=True)
        sums = np.bincount(inverse, weights=fitted, minlength=unique_x.size)
        counts = np.bincount(inverse, minlength=unique_x.size)
        self.x_ = unique_x
        self.y_ = np.maximum.accumulate(sums / counts)
        return self

    def transform(self, y_prob: ArrayLike) -> NDArray[np.float64]:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("IsotonicCalibrator must be fit before transform")
        p = np.asarray(y_prob, dtype=np.float64)
        return np.interp(p, self.x_, self.y_, left=self.y_[0], right=self.y_[-1])

    def fit_transform(self, y_true: ArrayLike, y_prob: ArrayLike) -> NDArray[np.float64]:
        return self.fit(y_true, y_prob).transform(y_prob)


def _pool_adjacent_violators(y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Least-squares projection of ``y`` onto the non-decreasing cone."""
    sums: list[float] = []
    counts: list[int] = []
    for value in y:
        block_sum = float(value)
        block_count = 1
        while sums and sums[-1] / counts[-1] >= block_sum / block_count:
            block_sum += sums.pop()
            block_count += counts.pop()
        sums.append(block_sum)
        counts.append(block_count)

    out = np.empty(y.size, dtype=np.float64)
    position = 0
    for block_sum, block_count in zip(sums, counts):
        out[position : position + block_count] = block_sum / block_count
        position += block_count
    return out
