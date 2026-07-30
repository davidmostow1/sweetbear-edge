import numpy as np
import pytest

from sweetbear import backtest as bt
from sweetbear.calibration import PlattScaler


def make_bets(n=1500, true_edge=0.03, seed=13):
    """Bets priced at 2.0 where the model is right and holds a real edge."""
    rng = np.random.default_rng(seed)
    p = 0.5 + true_edge
    outcomes = (rng.uniform(size=n) < p).astype(int)
    return [
        bt.Bet(timestamp=i, probability=p, decimal_odds=2.0, outcome=int(o))
        for i, o in enumerate(outcomes)
    ]


def test_bet_validates_its_fields():
    with pytest.raises(ValueError):
        bt.Bet(timestamp=0, probability=1.5, decimal_odds=2.0, outcome=1)
    with pytest.raises(ValueError):
        bt.Bet(timestamp=0, probability=0.5, decimal_odds=1.0, outcome=1)
    with pytest.raises(ValueError):
        bt.Bet(timestamp=0, probability=0.5, decimal_odds=2.0, outcome=2)


def test_flat_staking_arithmetic_is_exact():
    bets = [
        bt.Bet(timestamp=1, probability=0.6, decimal_odds=2.0, outcome=1),
        bt.Bet(timestamp=2, probability=0.6, decimal_odds=2.0, outcome=0),
        bt.Bet(timestamp=3, probability=0.6, decimal_odds=3.0, outcome=1),
    ]
    result = bt.run_backtest(bets, staking=bt.flat_stake(10.0), initial_bankroll=100.0)
    assert result.n_bets == 3
    assert result.n_wins == 2
    assert result.total_staked == pytest.approx(30.0)
    assert result.profit == pytest.approx(10.0 - 10.0 + 20.0)
    assert result.final_bankroll == pytest.approx(120.0)
    assert result.yield_ == pytest.approx(20.0 / 30.0)


def test_bets_are_settled_in_chronological_order():
    scrambled = [
        bt.Bet(timestamp=3, probability=0.9, decimal_odds=2.0, outcome=1),
        bt.Bet(timestamp=1, probability=0.9, decimal_odds=2.0, outcome=0),
        bt.Bet(timestamp=2, probability=0.9, decimal_odds=2.0, outcome=0),
    ]
    result = bt.run_backtest(
        scrambled, staking=bt.percent_bankroll(0.5), initial_bankroll=100.0
    )
    # Losing twice then winning half the remaining bankroll: 100 -> 50 -> 25 -> 37.5.
    assert result.bankroll_curve == pytest.approx([100.0, 50.0, 25.0, 37.5])


def test_percent_bankroll_compounds():
    bets = [
        bt.Bet(timestamp=i, probability=0.9, decimal_odds=2.0, outcome=1)
        for i in range(3)
    ]
    result = bt.run_backtest(
        bets, staking=bt.percent_bankroll(0.1), initial_bankroll=100.0
    )
    assert result.final_bankroll == pytest.approx(100.0 * 1.1**3)


def test_max_drawdown_and_losing_streak():
    outcomes = [1, 0, 0, 0, 1]
    bets = [
        bt.Bet(timestamp=i, probability=0.6, decimal_odds=2.0, outcome=o)
        for i, o in enumerate(outcomes)
    ]
    result = bt.run_backtest(bets, staking=bt.flat_stake(10.0), initial_bankroll=100.0)
    # Curve: 100, 110, 100, 90, 80, 90. Peak 110, trough 80.
    assert result.max_drawdown == pytest.approx((110.0 - 80.0) / 110.0)
    assert result.longest_losing_streak == 3


def test_min_edge_filters_and_counts_skips():
    bets = [
        bt.Bet(timestamp=1, probability=0.6, decimal_odds=2.0, outcome=1),
        bt.Bet(timestamp=2, probability=0.50, decimal_odds=1.9090909, outcome=1),
    ]
    result = bt.run_backtest(
        bets, staking=bt.flat_stake(10.0), initial_bankroll=100.0, min_edge=0.01
    )
    assert result.n_bets == 1
    assert result.skipped == 1


def test_ruin_is_detected_and_halts_the_simulation():
    bets = [
        bt.Bet(timestamp=i, probability=0.9, decimal_odds=2.0, outcome=0)
        for i in range(5)
    ]
    result = bt.run_backtest(
        bets, staking=bt.percent_bankroll(1.0), initial_bankroll=100.0
    )
    assert result.went_broke
    assert result.n_bets == 1
    assert result.final_bankroll == pytest.approx(0.0)


def test_kelly_staking_sizes_off_the_live_bankroll_when_compounding():
    bets = [
        bt.Bet(timestamp=i, probability=0.6, decimal_odds=2.0, outcome=1)
        for i in range(2)
    ]
    compounding = bt.run_backtest(
        bets, staking=bt.kelly_stake(0.25, cap=1.0), initial_bankroll=100.0
    )
    fixed = bt.run_backtest(
        bets,
        staking=bt.kelly_stake(0.25, cap=1.0, compounding=False),
        initial_bankroll=100.0,
    )
    # Quarter Kelly on a 20% full-Kelly edge is a 5% stake.
    assert compounding.final_bankroll == pytest.approx(100.0 * 1.05**2)
    assert fixed.final_bankroll == pytest.approx(110.0)
    assert compounding.final_bankroll > fixed.final_bankroll


def test_a_real_edge_shows_a_positive_t_statistic():
    result = bt.run_backtest(
        make_bets(n=4000, true_edge=0.04), staking=bt.flat_stake(1.0)
    )
    assert result.yield_ > 0.0
    assert result.t_statistic > 2.0
    lo, hi = result.yield_confidence_interval(seed=1)
    assert lo > 0.0
    assert lo < result.yield_ < hi


def test_a_coin_flip_does_not_look_like_skill():
    rng = np.random.default_rng(99)
    bets = [
        bt.Bet(
            timestamp=i,
            probability=0.55,
            decimal_odds=2.0,
            outcome=int(rng.uniform() < 0.5),
        )
        for i in range(4000)
    ]
    result = bt.run_backtest(bets, staking=bt.flat_stake(1.0))
    lo, hi = result.yield_confidence_interval(seed=2)
    assert lo < 0.0 < hi
    assert abs(result.t_statistic) < 2.0


def test_clv_is_reported_when_closing_lines_are_supplied():
    bets = [
        bt.Bet(
            timestamp=i,
            probability=0.55,
            decimal_odds=2.0,
            outcome=i % 2,
            closing_fair_probability=0.54,
        )
        for i in range(10)
    ]
    result = bt.run_backtest(bets, staking=bt.flat_stake(1.0))
    assert result.mean_clv == pytest.approx(2.0 * 0.54 - 1.0)
    assert result.clv_beat_rate == pytest.approx(1.0)


def test_clv_is_absent_without_closing_lines():
    result = bt.run_backtest(make_bets(n=50), staking=bt.flat_stake(1.0))
    assert result.mean_clv is None
    assert result.clv_beat_rate is None


def test_summary_renders_every_headline_number():
    bets = [
        bt.Bet(
            timestamp=i,
            probability=0.6,
            decimal_odds=2.0,
            outcome=i % 2,
            closing_fair_probability=0.55,
        )
        for i in range(20)
    ]
    text = bt.run_backtest(bets, staking=bt.flat_stake(1.0)).summary()
    for token in ("bets", "yield", "max drawdown", "t-statistic", "mean CLV"):
        assert token in text


def test_empty_or_fully_filtered_input_is_rejected():
    with pytest.raises(ValueError, match="No bets"):
        bt.run_backtest([], staking=bt.flat_stake(1.0))
    with pytest.raises(ValueError, match="filtered out"):
        bt.run_backtest(
            make_bets(n=10), staking=bt.flat_stake(1.0), min_edge=10.0
        )


def test_walk_forward_leaves_the_warmup_window_untouched():
    bets = make_bets(n=500)
    out = bt.walk_forward_calibrate(bets, PlattScaler, min_train=200, refit_every=50)
    assert len(out) == len(bets)
    for original, adjusted in zip(bets[:200], out[:200]):
        assert adjusted.probability == original.probability


def test_walk_forward_never_uses_future_outcomes():
    """Flipping outcomes strictly after index k must not change output at k."""
    bets = make_bets(n=600, seed=4)
    baseline = bt.walk_forward_calibrate(bets, PlattScaler, min_train=100, refit_every=25)

    k = 400
    tampered = list(bets)
    for i in range(k + 1, len(tampered)):
        tampered[i] = bt.Bet(
            timestamp=tampered[i].timestamp,
            probability=tampered[i].probability,
            decimal_odds=tampered[i].decimal_odds,
            outcome=1 - tampered[i].outcome,
        )
    perturbed = bt.walk_forward_calibrate(
        tampered, PlattScaler, min_train=100, refit_every=25
    )

    for i in range(k + 1):
        assert perturbed[i].probability == pytest.approx(baseline[i].probability)


def test_walk_forward_pulls_an_overconfident_model_back_toward_reality():
    rng = np.random.default_rng(17)
    n = 3000
    true_p = 0.52
    claimed_p = 0.75
    bets = [
        bt.Bet(
            timestamp=i,
            probability=claimed_p,
            decimal_odds=2.0,
            outcome=int(rng.uniform() < true_p),
        )
        for i in range(n)
    ]
    out = bt.walk_forward_calibrate(bets, PlattScaler, min_train=300, refit_every=100)
    tail = np.array([b.probability for b in out[-500:]])
    assert np.all(tail < claimed_p)
    assert np.mean(tail) == pytest.approx(true_p, abs=0.05)


def test_walk_forward_validates_its_arguments():
    with pytest.raises(ValueError):
        bt.walk_forward_calibrate(make_bets(n=10), PlattScaler, min_train=1)
    with pytest.raises(ValueError):
        bt.walk_forward_calibrate(make_bets(n=10), PlattScaler, refit_every=0)
