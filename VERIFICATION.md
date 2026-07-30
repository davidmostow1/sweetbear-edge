# Verification package

Every claim below is falsifiable and reproducible from this repository. It is
written to be handed to an adversarial reviewer — human or model — whose job is
to break it. Nothing here asks to be taken on trust.

```bash
pip install -e ".[dev]"
pytest -q                            # 131 tests
python validation/coverage_audit.py  # claims 1-3
python validation/longshot_audit.py  # claims 4-5
```

## What is actually built

A calibrated probability and edge-detection engine in Python: odds conversion,
de-vigging (multiplicative, power, Shin), calibration (Platt, isotonic,
reliability curves, Brier decomposition), Kelly staking, a walk-forward
backtest harness, and correlation-aware significance testing.

**No sport-specific model is implemented.** There is no MLB model, no
strikeout model, no batter model, no game-line model in this repository. This
is the statistical substrate those models would be judged by.

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

## What must be true before any of this touches money

1. A sport model exists as executable code with tests present in the repo.
2. Every alternate-line ladder shares one correlation group.
3. Model probabilities are compared against **de-vigged** market prices.
4. Promotion gates on prospective CLV, not on backtested profit.
5. The count of searched variants is recorded, and thresholds adjusted for it.
