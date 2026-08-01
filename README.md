# sweetbear-edge

A calibrated probability and edge-detection engine for sports, markets, and games.

This is the measurement layer of a prediction system: the part that decides
whether a forecast is any good, what it is worth against a price, how much to
stake, and whether an apparent edge is real or is variance wearing a disguise.
It does not contain a model of any particular sport or market. That is
deliberate — a model is only as trustworthy as the scaffolding used to judge it,
and this is that scaffolding.

```bash
pip install -e ".[dev]"
pytest
python examples/end_to_end.py
```

## What it does

| Module | Responsibility |
| --- | --- |
| `odds` | Conversions between American, decimal, fractional, and implied probability. Overround and vig. |
| `margin` | De-vigging: recovering fair probabilities from posted prices via multiplicative, additive, power, or Shin estimators. |
| `calibration` | Brier, log loss, ECE/MCE, reliability curves, Murphy decomposition, Platt and isotonic recalibration. |
| `edge` | Expected value, break-even prices, Kelly and fractional Kelly, joint Kelly across mutually exclusive outcomes, closing line value. |
| `backtest` | Chronological bankroll simulation with drawdown, bootstrap confidence intervals, and walk-forward recalibration. |
| `significance` | Correlation-aware inference: cluster-robust errors, wild cluster bootstrap, effective sample size, required sample size, multiple-testing correction. |
| `strikeouts` | Pitcher strikeout distribution and every main and alternate line priced from it. |
| `batters` | Batter plate-appearance distribution; hits, total bases, home runs, walks, strikeouts, and singles/doubles/triples all priced from it. Deliberately excludes runs and RBIs, which need game-state coupling this module does not model. |
| `portfolio` | Staking a set of positions that share fate. Joint Kelly over an explicit scenario distribution, plus the effective position count of a correlated group. |

## The three claims it is built to defend against

**"My model is 68% accurate."** Accuracy is not the metric. A forecast that says
70% must be right 70% of the time, and `calibration` is what establishes that.
`brier_decomposition` splits performance into calibration error, genuine
discrimination, and the irreducible noise in the outcome itself, so you can tell
which one is carrying the result.

**"My backtest returned 340%."** Backtests lie in three specific ways, and each
has a countermeasure here. Lookahead is prevented by `walk_forward_calibrate`,
which fits a recalibrator only on bets that had already settled — a calibration
curve can never be informed by its own future. Turnover laundering is prevented
by reporting yield and bankroll growth separately. Variance mistaken for skill is
addressed by `significance`, and the detail there matters more than it sounds:
a plain t-statistic assumes bets are independent, and betting almost never is.
Measured on simulated data with a **true zero edge**, the naive statistic
reports significance **54.5% of the time** against a nominal 5%, and its 95%
interval covers the truth only 42.5% of the time. `clustered_significance`
lands at 5.8% and 94.8%. Use it, not `BacktestResult.t_statistic`, which is
retained only so the gap between the two can be measured.

**"It hit on 8 of my last 10 alternate strikeout lines."** Every rung of a
ladder is a view of one random variable, so five lines on one start is one
observation, not five. `strikeouts` gives every quote from an outing the same
`correlation_group`, and `significance` counts it accordingly. Skipping this is
not a rounding error: on longshot ladders, uncorrected inference called dead
strategies profitable in 35-41% of simulated null samples.

**"I'm up 12 units this month."** Results take thousands of bets to separate
skill from luck. `required_sample_size` puts a number on it — roughly 8,700
bets to detect a 3% edge at even money — and `detectable_edge` runs it backward
to tell you what your actual sample can and cannot see. Closing line value converges in hundreds, because the closing
line is the market's best estimate and consistently beating it is what an edge
*is*. `run_backtest` reports mean CLV and beat rate whenever closing prices are
supplied, and `examples/end_to_end.py` demonstrates that CLV calls the outcome
correctly in both directions well before the bankroll curve is significant.

## Things that are easy to get wrong

**Expected value and edge are different quantities.** EV is computed at the
price you actually receive, vig included — that is the money. Edge versus the
de-vigged consensus is a diagnostic: it tells you how far your number sits from
the sharpest available estimate. A large EV paired with a large disagreement
against a sharp closing line is usually a modelling error, not an opportunity.

**How you de-vig changes your answer.** The four estimators disagree most on
longshots, which is exactly where spurious edge tends to appear. Shin is the
default because it models margin as the book's defence against informed money
rather than assuming it is spread proportionally, and it behaves best on real
books. Its fitted `z` is independently useful: a market defending hard against
insiders is sharper than its prices alone suggest.

**Per-bet Kelly is wrong for multi-outcome markets.** Mutually exclusive
outcomes partially hedge each other, so sizing each leg as a standalone
all-or-nothing risk *under*stakes. `simultaneous_kelly` solves the joint
problem. (The familiar warning that Kelly overstakes applies to simultaneous
bets on *different* events, where risks add instead of offsetting.)

**Per-bet Kelly is wrong for correlated positions too, in the opposite
direction.** Positions that lose together — a ladder, a same-game group — are
one risk wearing several tickets, and staking them rung by rung *over*stakes by
roughly the number of rungs. This is not a marginal effect. On a five-rung
ladder where **every rung carries a genuine 6% edge**, per-rung full Kelly
stakes 1.007 of bankroll and goes bust in **200 of 200** simulated runs, while
the joint solution compounds to a median 965× over the same 400 outings. The
edge was real in both cases; only the staking differed. `portfolio.joint_kelly`
works from an explicit joint scenario distribution rather than from marginals,
so correlation is carried exactly instead of estimated. Reproduce with
`validation/correlated_staking_audit.py`.

The joint solution's answer on that ladder is worth stating plainly, because it
is not what ladder bettors do: it backs **one** rung and declines the other
four. Rungs correlated near 1 are substitutes, not a portfolio — the extra ones
add variance without adding edge. `effective_position_count` puts a number on
it, and it is the count you should report rather than the ticket count.

**Nobody sane bets full Kelly.** Estimation error enters the growth rate
quadratically, so overbetting is punished far harder than underbetting — growth
is nearly flat below the optimum and falls off a cliff above it, crossing zero
just under twice full Kelly. `fractional_kelly` defaults to a quarter, which
surrenders roughly 25% of theoretical growth to cut variance by about 75% and
survive a mis-specified model.

**Isotonic recalibration is not free.** It is strictly more flexible than Platt
and therefore strictly more dangerous: it will memorise a few hundred
observations happily. Fit it on held-out data, and check that it actually helps
— in `examples/end_to_end.py` it improves calibration on the badly-calibrated
model and slightly *degrades* an already-calibrated one.

## What is not here yet

No data ingestion, no feature engineering, no live odds feeds, no execution or
bankroll persistence, no market models outside baseball props.

Correlation is now handled in both places it matters — `significance` for
inference, `portfolio` for staking — but with one gap that is worth naming
precisely rather than leaving implied. `portfolio.combine_independent` assumes
independence *across* games, which is defensible for unrelated games and wrong
for anything sharing a weather system, an umpire, or a postponement risk. And
it cannot couple a pitcher's strikeout total to the opposing batters'
strikeout totals in the same game, even though those count the same events
twice: `strikeouts` and `batters` do not share a plate-appearance-level game
state, so the coupling is not available to be extracted. Betting both sides of
that pair will understate the true correlation.

Everything here remains simulation-validated, not market-validated. No claim in
this repository has been tested against real closing lines or real settlement.
