"""Inference that survives correlated bets and repeated searching.

The plain t-statistic on ``BacktestResult`` assumes every bet is an independent
draw. Betting almost never satisfies that. Three bets on the same game share one
outcome. A whole Sunday slate moves together when the weather turns. A model
that likes home favorites has a position on the same latent factor forty times
in a weekend. Treating those as forty observations rather than the handful of
independent bets they really are inflates the t-statistic by roughly the square
root of the design effect -- routinely a factor of two.

That is not a rounding error. It is the difference between "publishable" and
"noise", and it always errs toward telling you the strategy works.

This module provides the corrections:

* :func:`clustered_significance` -- cluster-robust standard errors and a
  cluster bootstrap, so the error bar reflects the number of independent
  *situations* rather than the number of tickets.
* :func:`required_sample_size` -- how many bets it actually takes to detect a
  given edge. Use this before adopting a promotion threshold, not after.
* :func:`multiple_testing_threshold` -- what a t-statistic must clear once you
  admit how many strategies you tried.

The governing asymmetry: correlation, selection, and optional stopping all bias
results in the same direction. Every one of them makes a dead strategy look
alive. None of them makes a live strategy look dead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "SignificanceResult",
    "clustered_significance",
    "cluster_codes",
    "intracluster_correlation",
    "design_effect",
    "effective_sample_size",
    "cluster_bootstrap_ci",
    "required_sample_size",
    "detectable_edge",
    "multiple_testing_threshold",
    "deflate_t_statistic",
]


@dataclass(frozen=True)
class SignificanceResult:
    """A verdict on whether a measured return is distinguishable from luck."""

    mean: float
    n_observations: int
    n_clusters: int
    naive_standard_error: float
    clustered_standard_error: float
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    confidence_interval: tuple[float, float]
    bootstrap_interval: tuple[float, float]
    intracluster_correlation: float
    design_effect: float
    effective_sample_size: float

    @property
    def inflation_factor(self) -> float:
        """How much the naive t-statistic overstates significance.

        A value of 2.0 means the independence assumption doubled your apparent
        edge quality -- a t of 4.0 that is really a t of 2.0.
        """
        if self.clustered_standard_error <= 0.0:
            return 1.0
        return self.clustered_standard_error / self.naive_standard_error

    @property
    def significant(self) -> bool:
        """Whether the clustered interval excludes zero at the stated level."""
        lo, hi = self.confidence_interval
        return lo > 0.0 or hi < 0.0

    def summary(self) -> str:
        lo, hi = self.confidence_interval
        blo, bhi = self.bootstrap_interval
        verdict = "SIGNIFICANT" if self.significant else "NOT DISTINGUISHABLE FROM ZERO"
        return "\n".join(
            [
                f"mean              {self.mean:+.4%}",
                f"observations      {self.n_observations:,} in {self.n_clusters:,} clusters",
                f"naive SE          {self.naive_standard_error:.4%}",
                f"clustered SE      {self.clustered_standard_error:.4%}"
                f"  ({self.inflation_factor:.2f}x wider)",
                f"t-statistic       {self.t_statistic:+.2f}  (df={self.degrees_of_freedom})",
                f"p-value           {self.p_value:.4f}",
                f"95% CI            [{lo:+.4%}, {hi:+.4%}]",
                f"bootstrap CI      [{blo:+.4%}, {bhi:+.4%}]",
                f"ICC               {self.intracluster_correlation:+.4f}",
                f"design effect     {self.design_effect:.2f}",
                f"effective n       {self.effective_sample_size:,.0f}"
                f" (of {self.n_observations:,})",
                f"verdict           {verdict}",
            ]
        )


def cluster_codes(labels: Sequence[Hashable]) -> NDArray[np.int64]:
    """Map arbitrary cluster labels to dense integer codes, order preserved."""
    seen: dict[Hashable, int] = {}
    out = np.empty(len(labels), dtype=np.int64)
    for i, label in enumerate(labels):
        if label not in seen:
            seen[label] = len(seen)
        out[i] = seen[label]
    return out


def _grouped(values: NDArray[np.float64], codes: NDArray[np.int64]):
    """Return (group_sums, group_counts, group_means) indexed by cluster code."""
    n_groups = int(codes.max()) + 1 if codes.size else 0
    sums = np.bincount(codes, weights=values, minlength=n_groups)
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    return sums, counts, means


def intracluster_correlation(
    values: Sequence[float] | NDArray[np.float64],
    clusters: Sequence[Hashable],
) -> float:
    """One-way ANOVA estimate of the intracluster correlation.

    Roughly: what fraction of total variance is shared within a cluster. Zero
    means clustering costs you nothing. One means every bet in a cluster is the
    same bet wearing a different hat.

    The estimate can come out slightly negative when clusters are more varied
    internally than externally; that is sampling noise, not a real negative
    correlation, and callers should floor it at zero before using it to widen
    an interval.
    """
    v = np.asarray(values, dtype=np.float64)
    codes = cluster_codes(clusters)
    if v.size != codes.size:
        raise ValueError("values and clusters must have equal length")

    n = v.size
    _, counts, means = _grouped(v, codes)
    n_groups = counts.size
    if n_groups < 2 or n <= n_groups:
        return 0.0

    grand = float(v.mean())
    ss_between = float(np.sum(counts * (means - grand) ** 2))
    ss_within = float(np.sum((v - means[codes]) ** 2))

    ms_between = ss_between / (n_groups - 1)
    ms_within = ss_within / (n - n_groups)

    # Average cluster size adjusted for imbalance (Fisher's m0).
    m0 = (n - float(np.sum(counts**2)) / n) / (n_groups - 1)
    denom = ms_between + (m0 - 1.0) * ms_within
    if denom <= 0.0:
        return 0.0
    return float(np.clip((ms_between - ms_within) / denom, -1.0, 1.0))


def design_effect(
    values: Sequence[float] | NDArray[np.float64],
    clusters: Sequence[Hashable],
) -> float:
    """Variance inflation from clustering: ``1 + (m0 - 1) * ICC``.

    A design effect of 4 means your 1,000 bets carry the information of 250.
    """
    v = np.asarray(values, dtype=np.float64)
    codes = cluster_codes(clusters)
    _, counts, _ = _grouped(v, codes)
    if counts.size < 2:
        return 1.0
    n = v.size
    m0 = (n - float(np.sum(counts**2)) / n) / (counts.size - 1)
    icc = max(0.0, intracluster_correlation(v, clusters))
    return float(max(1.0, 1.0 + (m0 - 1.0) * icc))


def effective_sample_size(
    values: Sequence[float] | NDArray[np.float64],
    clusters: Sequence[Hashable],
) -> float:
    """Number of independent observations your correlated sample is worth."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return 0.0
    return float(v.size / design_effect(v, clusters))


def _clustered_variance_of_mean(
    v: NDArray[np.float64], codes: NDArray[np.int64]
) -> float:
    """CR1 cluster-robust variance of the sample mean.

    The mean is the OLS coefficient on a constant, so the sandwich estimator
    reduces to the sum of squared within-cluster residual totals, with the
    usual finite-cluster correction ``G / (G - 1)``.
    """
    n = v.size
    resid = v - v.mean()
    n_groups = int(codes.max()) + 1
    cluster_totals = np.bincount(codes, weights=resid, minlength=n_groups)
    meat = float(np.sum(cluster_totals**2))
    correction = n_groups / (n_groups - 1.0) if n_groups > 1 else 1.0
    return correction * meat / (n**2)


def cluster_bootstrap_ci(
    values: Sequence[float] | NDArray[np.float64],
    clusters: Sequence[Hashable],
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap interval that resamples whole clusters, not individual bets.

    Resampling individual bets would rebuild the very independence the data
    does not have, reproducing the naive interval with extra steps.
    """
    v = np.asarray(values, dtype=np.float64)
    codes = cluster_codes(clusters)
    if v.size != codes.size:
        raise ValueError("values and clusters must have equal length")
    n_groups = int(codes.max()) + 1 if codes.size else 0
    if n_groups < 2:
        return (float("-inf"), float("inf"))

    order = np.argsort(codes, kind="stable")
    sorted_vals = v[order]
    boundaries = np.searchsorted(codes[order], np.arange(n_groups + 1))
    groups = [sorted_vals[boundaries[g] : boundaries[g + 1]] for g in range(n_groups)]
    sums = np.array([g.sum() for g in groups], dtype=np.float64)
    sizes = np.array([g.size for g in groups], dtype=np.float64)

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_groups, size=(n_resamples, n_groups))
    boot_sums = sums[picks].sum(axis=1)
    boot_sizes = sizes[picks].sum(axis=1)
    means = np.divide(
        boot_sums, boot_sizes, out=np.zeros_like(boot_sums), where=boot_sizes > 0
    )

    alpha = (1.0 - level) / 2.0
    return (float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha)))


def clustered_significance(
    values: Sequence[float] | NDArray[np.float64],
    clusters: Sequence[Hashable] | None = None,
    level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> SignificanceResult:
    """Test whether a mean return differs from zero, honouring correlation.

    ``values`` are per-bet returns per unit staked; ``clusters`` labels which
    bets share fate -- game id, slate, or correlated-factor bucket. Passing
    ``None`` treats every bet as its own cluster, which reproduces the naive
    result and is provided only so callers can measure the gap.

    Inference uses the t distribution with ``G - 1`` degrees of freedom, where
    ``G`` is the cluster count. That is deliberately harsh with few clusters:
    ten correlated weekends is ten observations, and the critical value should
    say so.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        raise ValueError("need at least two observations")
    if clusters is None:
        clusters = list(range(v.size))
    codes = cluster_codes(clusters)
    if v.size != codes.size:
        raise ValueError("values and clusters must have equal length")

    n = v.size
    n_groups = int(codes.max()) + 1
    mean = float(v.mean())

    naive_se = float(np.std(v, ddof=1) / np.sqrt(n))

    if n_groups < 2:
        # A single cluster carries no information about between-cluster spread.
        return SignificanceResult(
            mean=mean,
            n_observations=n,
            n_clusters=n_groups,
            naive_standard_error=naive_se,
            clustered_standard_error=float("inf"),
            t_statistic=0.0,
            p_value=1.0,
            degrees_of_freedom=0,
            confidence_interval=(float("-inf"), float("inf")),
            bootstrap_interval=(float("-inf"), float("inf")),
            intracluster_correlation=0.0,
            design_effect=float(n),
            effective_sample_size=1.0,
            )

    var = _clustered_variance_of_mean(v, codes)
    clustered_se = float(np.sqrt(var))
    df = n_groups - 1

    if clustered_se > 0.0:
        t_stat = mean / clustered_se
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        crit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df))
        ci = (mean - crit * clustered_se, mean + crit * clustered_se)
    else:
        t_stat, p_value, ci = 0.0, 1.0, (mean, mean)

    return SignificanceResult(
        mean=mean,
        n_observations=n,
        n_clusters=n_groups,
        naive_standard_error=naive_se,
        clustered_standard_error=clustered_se,
        t_statistic=float(t_stat),
        p_value=p_value,
        degrees_of_freedom=df,
        confidence_interval=ci,
        bootstrap_interval=cluster_bootstrap_ci(v, clusters, level, n_resamples, seed),
        intracluster_correlation=intracluster_correlation(v, clusters),
        design_effect=design_effect(v, clusters),
        effective_sample_size=effective_sample_size(v, clusters),
    )


def required_sample_size(
    edge: float,
    decimal_odds: float = 2.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
    design_effect: float = 1.0,
) -> int:
    """Bets needed to detect ``edge`` (profit per unit staked) at given power.

    This is the calculation that should govern any promotion threshold. A bet
    at decimal odds ``d`` on a true probability ``p`` returns ``d - 1`` or
    ``-1``, so its return variance is ``p(1-p)d^2`` -- close to 1.0 for
    even-money markets and far larger for longshots. Against a 2-3% edge, the
    noise dwarfs the signal, and the required sample runs into the thousands.

    A threshold of a few hundred bets does not test anything. It certifies
    that paperwork was filed.

    ``design_effect`` scales the answer for correlated bets: pass the value
    from :func:`design_effect` to learn how many *tickets* you need, as opposed
    to independent situations.
    """
    if not 0.0 < power < 1.0:
        raise ValueError("power must lie in (0, 1)")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must exceed 1.0")
    if edge <= 0.0:
        raise ValueError("edge must be positive")
    if design_effect < 1.0:
        raise ValueError("design_effect cannot be below 1.0")

    p = (1.0 + edge) / decimal_odds
    if not 0.0 < p < 1.0:
        raise ValueError(f"edge {edge} is unattainable at odds {decimal_odds}")

    sd = float(np.sqrt(p * (1.0 - p) * decimal_odds**2))
    z_alpha = float(stats.norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha)))
    z_beta = float(stats.norm.ppf(power))

    n = ((z_alpha + z_beta) * sd / edge) ** 2
    return int(np.ceil(n * design_effect))


def detectable_edge(
    n_bets: int,
    decimal_odds: float = 2.0,
    power: float = 0.80,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Smallest edge a sample of ``n_bets`` can reliably detect.

    The inverse of :func:`required_sample_size`, and the more uncomfortable
    direction to read it in: given the sample you actually have, anything
    smaller than this is invisible to you.
    """
    if n_bets < 2:
        raise ValueError("n_bets must be at least 2")

    z_alpha = float(stats.norm.ppf(1.0 - (alpha / 2.0 if two_sided else alpha)))
    z_beta = float(stats.norm.ppf(power))
    # Variance depends on the edge itself; iterate from a zero-edge guess.
    edge = 0.0
    for _ in range(64):
        p = (1.0 + edge) / decimal_odds
        p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
        sd = float(np.sqrt(p * (1.0 - p) * decimal_odds**2))
        updated = (z_alpha + z_beta) * sd / np.sqrt(n_bets)
        if abs(updated - edge) < 1e-12:
            edge = updated
            break
        edge = updated
    return float(edge)


def multiple_testing_threshold(
    n_trials: int, alpha: float = 0.05, two_sided: bool = True
) -> float:
    """t-statistic a strategy must clear after ``n_trials`` searched variants.

    Šidák correction. Try forty feature sets and the best one clears t=2 by
    construction; the threshold rises to roughly t=3.2. Count every variant you
    looked at, including the ones abandoned early -- especially those, since
    abandoning on a bad start is itself a selection rule.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    adjusted = 1.0 - (1.0 - alpha) ** (1.0 / n_trials)
    tail = adjusted / 2.0 if two_sided else adjusted
    return float(stats.norm.ppf(1.0 - tail))


def deflate_t_statistic(t_statistic: float, n_trials: int) -> float:
    """Shrink an observed t-statistic by the expected maximum under the null.

    Searching ``N`` independent dead strategies produces a best t-statistic of
    about ``sqrt(2 ln N)`` from noise alone. Subtracting that is a blunt
    instrument, but it answers the right question: how much of this result
    survives the fact that it was chosen for being the best?
    """
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if n_trials == 1:
        return float(t_statistic)
    expected_max = float(np.sqrt(2.0 * np.log(n_trials)))
    return float(t_statistic - expected_max)
