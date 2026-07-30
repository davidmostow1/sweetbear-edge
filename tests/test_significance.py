"""Tests for correlation-aware inference.

The important cases here are the ones where the naive statistic and the
clustered statistic disagree. Anything else is bookkeeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from sweetbear.backtest import Bet, flat_stake, run_backtest
from sweetbear.significance import (
    cluster_bootstrap_ci,
    cluster_codes,
    clustered_significance,
    deflate_t_statistic,
    design_effect,
    detectable_edge,
    effective_sample_size,
    intracluster_correlation,
    multiple_testing_threshold,
    required_sample_size,
)


def test_cluster_codes_are_dense_and_order_preserving():
    codes = cluster_codes(["b", "a", "b", "c", "a"])
    assert list(codes) == [0, 1, 0, 2, 1]


def test_independent_data_matches_naive_standard_error():
    rng = np.random.default_rng(0)
    values = rng.normal(0.02, 1.0, size=2000)
    res = clustered_significance(values, clusters=list(range(values.size)))
    # With one observation per cluster the sandwich reduces to the usual SE,
    # up to the finite-cluster correction.
    assert res.clustered_standard_error == pytest.approx(
        res.naive_standard_error, rel=0.02
    )
    assert res.n_clusters == 2000
    assert res.design_effect == pytest.approx(1.0, abs=0.15)


def test_perfectly_correlated_clusters_destroy_significance():
    """Ten distinct outcomes copied 50 times each is ten observations."""
    rng = np.random.default_rng(1)
    per_cluster = rng.normal(0.05, 1.0, size=10)
    values = np.repeat(per_cluster, 50)
    clusters = np.repeat(np.arange(10), 50)

    naive = clustered_significance(values, clusters=list(range(values.size)))
    clustered = clustered_significance(values, clusters=clusters)

    assert clustered.n_clusters == 10
    assert clustered.n_observations == 500
    # The naive SE is inflated by sqrt(50) relative to the truth.
    assert clustered.clustered_standard_error > 5.0 * naive.clustered_standard_error
    assert clustered.inflation_factor > 5.0
    assert clustered.effective_sample_size < 20.0
    assert clustered.intracluster_correlation > 0.95


def test_clustering_widens_the_interval():
    rng = np.random.default_rng(2)
    shared = rng.normal(0.0, 1.0, size=40)
    noise = rng.normal(0.03, 0.3, size=(40, 25))
    values = (shared[:, None] + noise).ravel()
    clusters = np.repeat(np.arange(40), 25)

    naive = clustered_significance(values, clusters=list(range(values.size)))
    clustered = clustered_significance(values, clusters=clusters)

    naive_width = naive.confidence_interval[1] - naive.confidence_interval[0]
    clustered_width = (
        clustered.confidence_interval[1] - clustered.confidence_interval[0]
    )
    assert clustered_width > naive_width
    assert clustered.design_effect > 2.0


def test_single_cluster_yields_no_information():
    values = np.array([0.1, 0.2, 0.3, 0.4])
    res = clustered_significance(values, clusters=["g"] * 4)
    assert res.n_clusters == 1
    assert res.confidence_interval == (float("-inf"), float("inf"))
    assert not res.significant
    assert res.p_value == 1.0


def test_zero_icc_when_clusters_are_exchangeable():
    rng = np.random.default_rng(3)
    values = rng.normal(0.0, 1.0, size=1000)
    clusters = np.repeat(np.arange(100), 10)  # labels unrelated to values
    icc = intracluster_correlation(values, clusters)
    assert abs(icc) < 0.1
    assert design_effect(values, clusters) == pytest.approx(1.0, abs=0.6)


def test_effective_sample_size_never_exceeds_n():
    rng = np.random.default_rng(4)
    values = rng.normal(0.0, 1.0, size=300)
    clusters = np.repeat(np.arange(30), 10)
    assert effective_sample_size(values, clusters) <= 300.0


def test_cluster_bootstrap_resamples_groups():
    rng = np.random.default_rng(5)
    per_cluster = rng.normal(0.0, 1.0, size=12)
    values = np.repeat(per_cluster, 40)
    clusters = np.repeat(np.arange(12), 40)

    lo, hi = cluster_bootstrap_ci(values, clusters, n_resamples=2000, seed=7)
    naive_lo, naive_hi = cluster_bootstrap_ci(
        values, list(range(values.size)), n_resamples=2000, seed=7
    )
    assert (hi - lo) > 5.0 * (naive_hi - naive_lo)


def test_bootstrap_is_deterministic_under_seed():
    rng = np.random.default_rng(6)
    values = rng.normal(0.01, 1.0, size=400)
    clusters = np.repeat(np.arange(40), 10)
    a = cluster_bootstrap_ci(values, clusters, seed=11, n_resamples=500)
    b = cluster_bootstrap_ci(values, clusters, seed=11, n_resamples=500)
    assert a == b


# --- sample size -----------------------------------------------------------


def test_required_sample_size_for_realistic_edge_is_in_the_thousands():
    """The claim that motivates the whole module.

    A 3% edge at even money is not detectable in a few hundred bets, and any
    promotion gate set at that scale is theatre.
    """
    n = required_sample_size(edge=0.03, decimal_odds=2.0, power=0.80, alpha=0.05)
    assert 8000 < n < 9500
    one_sided = required_sample_size(
        edge=0.03, decimal_odds=2.0, power=0.80, alpha=0.05, two_sided=False
    )
    assert 6000 < one_sided < 7500
    assert one_sided < n


def test_required_sample_size_shrinks_with_bigger_edge():
    big = required_sample_size(edge=0.10)
    small = required_sample_size(edge=0.02)
    assert big < small


def test_longshots_need_far_more_bets():
    even = required_sample_size(edge=0.03, decimal_odds=2.0)
    longshot = required_sample_size(edge=0.03, decimal_odds=10.0)
    assert longshot > 3 * even


def test_design_effect_scales_required_sample():
    base = required_sample_size(edge=0.03)
    clustered = required_sample_size(edge=0.03, design_effect=4.0)
    assert clustered == pytest.approx(4 * base, rel=0.01)


def test_detectable_edge_inverts_required_sample_size():
    n = required_sample_size(edge=0.03, decimal_odds=2.0)
    recovered = detectable_edge(n, decimal_odds=2.0)
    assert recovered == pytest.approx(0.03, rel=0.05)


def test_detectable_edge_at_500_bets_is_uselessly_large():
    """The old promotion floor, stated as what it can actually see."""
    edge = detectable_edge(500, decimal_odds=2.0)
    assert edge > 0.10  # cannot resolve anything below a 10% edge


def test_required_sample_size_rejects_impossible_edge():
    with pytest.raises(ValueError):
        required_sample_size(edge=2.0, decimal_odds=1.5)
    with pytest.raises(ValueError):
        required_sample_size(edge=-0.01)


# --- multiple testing ------------------------------------------------------


def test_multiple_testing_threshold_rises_with_trials():
    one = multiple_testing_threshold(1)
    forty = multiple_testing_threshold(40)
    assert one == pytest.approx(1.96, abs=0.01)
    assert forty > 3.0
    assert forty > one


def test_deflate_t_statistic_penalises_search():
    assert deflate_t_statistic(4.0, 1) == 4.0
    assert deflate_t_statistic(4.0, 100) < 4.0
    assert deflate_t_statistic(4.0, 1000) < deflate_t_statistic(4.0, 100)


# --- integration with the backtest ----------------------------------------


def _bets_with_groups(n_groups: int, per_group: int, seed: int = 0):
    """Bets where every ticket in a group settles the same way."""
    rng = np.random.default_rng(seed)
    bets = []
    t = 0
    for g in range(n_groups):
        outcome = int(rng.random() < 0.55)
        for _ in range(per_group):
            bets.append(
                Bet(
                    timestamp=t,
                    probability=0.55,
                    decimal_odds=2.0,
                    outcome=outcome,
                    correlation_group=f"game-{g}",
                )
            )
            t += 1
    return bets


def test_backtest_carries_cluster_labels():
    bets = _bets_with_groups(20, 5)
    res = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    assert len(res.cluster_labels) == res.n_bets
    assert len(set(res.cluster_labels)) == 20


def test_backtest_significance_is_harsher_than_naive_t():
    bets = _bets_with_groups(20, 5, seed=3)
    res = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    sig = res.significance()
    assert sig.n_clusters == 20
    assert sig.n_observations == 100
    assert abs(sig.t_statistic) < abs(res.t_statistic)
    assert sig.effective_sample_size < res.n_bets


def test_unlabelled_bets_fall_back_to_naive():
    bets = [
        Bet(timestamp=i, probability=0.55, decimal_odds=2.0, outcome=i % 2)
        for i in range(50)
    ]
    res = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    sig = res.significance()
    assert sig.n_clusters == sig.n_observations


def test_market_id_is_used_as_fallback_cluster():
    bets = [
        Bet(
            timestamp=i,
            probability=0.55,
            decimal_odds=2.0,
            outcome=i % 2,
            market_id=f"m{i // 10}",
        )
        for i in range(50)
    ]
    res = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    assert len(set(res.cluster_labels)) == 5


def test_clv_significance_available_when_closing_prices_present():
    bets = [
        Bet(
            timestamp=i,
            probability=0.55,
            decimal_odds=2.0,
            outcome=i % 2,
            closing_fair_probability=0.53,
            correlation_group=f"g{i // 5}",
        )
        for i in range(60)
    ]
    res = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    clv_sig = res.clv_significance()
    assert clv_sig is not None
    assert clv_sig.n_observations == res.n_bets
    assert clv_sig.n_clusters == 12
    # Consistently beating a 0.53 close at 2.0 is a real, detectable edge.
    assert clv_sig.mean > 0.0


def test_clv_significance_absent_without_closing_prices():
    bets = _bets_with_groups(10, 3)
    res = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    assert res.clv_significance() is None


def test_summary_reports_clustered_verdict():
    bets = _bets_with_groups(20, 5)
    res = run_backtest(bets, staking=flat_stake(1.0), initial_bankroll=10_000.0)
    text = res.summary()
    assert "clustered t" in text
    assert "effective n" in text
