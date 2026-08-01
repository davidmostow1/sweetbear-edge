# Verification package

Every claim below is falsifiable and reproducible from this repository. It is
written to be handed to an adversarial reviewer — human or model — whose job is
to break it. Nothing here asks to be taken on trust.

```bash
pip install -e ".[dev]"
pytest -q                                       # 239 tests
python validation/coverage_audit.py             # claims 1-3
python validation/longshot_audit.py             # claims 4-5
python validation/correlated_staking_audit.py   # claim 7
```

## What is actually built

A calibrated probability and edge-detection engine in Python: odds conversion,
de-vigging (multiplicative, power, Shin), calibration (Platt, isotonic,
reliability curves, Brier decomposition), Kelly staking, a walk-forward
backtest harness, correlation-aware significance testing, and correlation-aware
portfolio staking.

**Two baseball prop models are now implemented** — `strikeouts` (a pitcher's
strikeout distribution) and `batters` (a batter's plate-appearance outcome
distribution) — each pricing every line it quotes as an integral of one
distribution. Earlier revisions of this document stated that no sport model
existed; that was true when written and is no longer true. There is still no
game-line model, no market model outside baseball props, and no data ingestion.

**No model here has been validated against a market.** Both prop models are
built from supplied rates and checked for internal coherence, not against real
settlement.

## Falsifiable claims

**Claim 1 — The naive t-statistic reports significance on pure noise more than
half the time when bets are correlated.** Under a true null with 30 clusters of
20 correlated bets, measured false positive rate is 54.5% against a nominal 5%.
*Falsified if* `coverage_audit.py` reports a naive rate near 5%.

**Claim 2 — Its 95% confidence interval covers the truth 42.5% of the time.**
The clustered interval covers 94.8%. *Falsified if* naive coverage approaches
95%.

**Claim 3 — Detecting a 3% edge at even money requires ~8,700 bets** (two-sided,
80% power, α=0.05); ~6,900 one-sided. Empirical power at the prescribed n is
80.5% against the 80% target. A 500-bet sample cannot resolve any edge below
10%. *Falsified if* simulated power at n=8,714 departs materially from 80%.

**Claim 4 — Cluster-robust CR1 standard errors are overconfident on longshot
markets.** Under a true null with whole-cluster shared outcomes:

| clusters | odds | CR1 false positive rate |
|---|---|---|
| 10 | 10.0 | 34.7% even / 41.0% uneven |
| 20 | 10.0 | 12.2% even / 21.8% uneven |
| 40 | 10.0 | 7.7% even / 13.8% uneven |
| 40 | 2.0 | 3.0% even / 5.0% uneven |

Even money is fine; longshots are not. *Falsified if* longshot rates come in
near 5%.

**Claim 5 — The wild cluster bootstrap plus an effective-cluster gate repairs
it without gutting power.** Across 16 configurations the gated false positive
rate never exceeds 4%. Power remains real: a 15% edge is detected 99.3% of the
time at 1,000 clusters and 100% at 3,000. *Falsified if* any configuration
exceeds 5% false positives, or if power collapses on large samples.

**Claim 6 — At 300 correlated games, a 5% edge is detected 13% of the time.**
This is the empirical case against any few-hundred-bet promotion threshold. The
sample size is not conservative; it is blind.

**Claim 7 — Staking a correlated ladder rung by rung destroys a real edge.**
Every rung below carries a genuine +6% edge, priced from the model's own
distribution, so there is no pricing error anywhere and stake sizing is the only
variable. At full Kelly:

| rungs | effective positions | joint stake | per-rung stake | joint growth | per-rung growth |
|---|---|---|---|---|---|
| 1 | 1.00 | 0.481 | 0.481 | +0.01718 | +0.01718 |
| 2 | 1.23 | 0.481 | 0.748 | +0.01718 | +0.00099 |
| 3 | 1.40 | 0.481 | 0.890 | +0.01718 | −0.03920 |
| 4 | 1.54 | 0.481 | 0.966 | +0.01718 | −0.10699 |
| 5 | 1.67 | 0.481 | 1.007 | +0.01718 | −∞ |

Compounded over 400 outings across 200 simulated bankrolls, joint staking
returns a median 965× and never busts; per-rung staking busts 200 out of 200.
*Falsified if* per-rung growth matches joint growth at two or more rungs, or if
the simulated per-rung bankroll survives.

A second, less obvious result from the same run: the joint solution stakes
**one** rung and declines the other four. Rungs correlated near 1 are
substitutes rather than a portfolio. *Falsified if* the optimiser spreads stake
across rungs — verified additionally by random perturbation, where 4,000
perturbations of the returned vector failed to find higher expected log growth.

**Claim 7a — the comparison is only valid at full Kelly.** Scaling both rules
by the same fraction does not preserve the ranking: at quarter Kelly the
per-rung vector on the three-rung ladder scores *higher* growth than the scaled
joint vector, because a quarter of an overstaked vector can land nearer the
optimum than a quarter of the optimum does. This is stated because it would
otherwise look like a contradiction of Claim 7 to anyone who reran it at the
default fraction. The joint solution is optimal at full Kelly; fractional
scaling is a separate defence against estimation error and is applied to the
joint vector, not used to compare rules.

## Known limitations, stated rather than discovered

- **Zero-variance samples return p=1.0.** Identical observations carry no
  information about sampling variability, so the estimator declines to reject.
  Conservative and deliberate, but it is a refusal, not a result.
- **The effective-cluster threshold of 12 is a judgement call**, informed by the
  CRVE literature placing plain CR1 at roughly 40 clusters and the wild
  bootstrap at about a dozen. It is not derived from first principles.
- **`intracluster_correlation` can return slightly negative values** under
  sampling noise. Callers floor it at zero before widening intervals.
- **Cluster labels are supplied by the caller.** Mislabelled correlation
  structure produces wrong inference, and nothing here can detect that. An
  unlabelled sample deliberately falls back to the naive result rather than
  inventing a structure.
- **`deflate_t_statistic` is blunt.** Subtracting `sqrt(2 ln N)` assumes
  independent trials, which searched model variants are not.
- **Everything is simulation-validated, not market-validated.** No claim here
  has been tested against real closing lines or real settlement.
- **Joint staking fixes correlation error, not estimation error.** The scenario
  matrix carries the correlation exactly, but it is built from model
  probabilities that are themselves wrong by an unknown amount. Fractional
  Kelly is still required and is still the larger of the two defences.
- **`combine_independent` assumes what its name says.** Positions in different
  games are joined by independent resampling. Shared weather, umpires, or
  postponement risk violate this and nothing detects it.
- **Pitcher and batter strikeouts in the same game cannot be coupled.** They
  count the same plate appearances, but the two pricing modules do not share a
  PA-level game state, so the correlation is unavailable and would be
  understated if both were staked together.
- **`batter_scenarios` is Monte Carlo.** Its marginals agree with the analytic
  compound distribution to sampling error (tested to ±0.01 at 200k draws), but
  long-odds positions depend on rare scenarios and need more draws than the
  default to stabilise.

## What must be true before any of this touches money

1. A sport model exists as executable code with tests present in the repo.
2. Every alternate-line ladder shares one correlation group — for inference,
   and for staking. A ladder is sized jointly or not at all.
3. Model probabilities are compared against **de-vigged** market prices.
4. Promotion gates on prospective CLV, not on backtested profit.
5. The count of searched variants is recorded, and thresholds adjusted for it.
