---
name: prediction-model-audit
description: Audit a prediction or betting model before it is allowed to risk money. Use when someone claims a model is profitable, wants to promote a model from research to production, presents backtest results, proposes a sample-size or promotion threshold, or asks whether an edge is real. Also use when reviewing alternate-line ladders, parlays, or any set of correlated positions.
---

# Prediction model audit

A backtest is a claim, not evidence. This skill is the standing procedure for
deciding whether a claimed edge is real enough to risk money on.

The governing asymmetry: **correlation, selection, optional stopping, and
survivorship all bias results in the same direction.** Every one of them makes
a dead strategy look alive. None makes a live strategy look dead. So when a
result is ambiguous, the prior is that it is noise.

## Step 1 — Verify the artifacts exist

Before evaluating any claim, confirm the code is physically present and runs.

- Does the executable file named by the registry/manifest exist on disk?
- Do its tests exist, and do they pass?
- A model described in a config but absent from the filesystem is **not
  implemented**, regardless of what the registry says.

Never accept a description of a file as a substitute for the file. If asked to
audit something not provided, say so plainly and stop — do not infer its
contents from a summary and then reason as though you had read it.

## Step 2 — Establish the correlation structure

This is where most betting backtests die, and it is the step most often skipped.

Ask what makes two bets share fate:
- Same game, same team, same slate, same weather system, same injury report.
- **Alternate-line ladders are the worst case.** O/U 4.5, 5, 5.5, 6, 6.5 derived
  from one distribution are not five bets — they are five views of one random
  variable, correlated near 1. The same applies to any main line and its alts.
- Parlays and same-game parlays inherit every leg's correlation.

Every set of positions sharing a latent outcome must carry **one shared
correlation group**. Then:

- Report the **effective sample size**, not the ticket count.
- Use **cluster-robust standard errors** and a **cluster bootstrap** that
  resamples whole groups.
- With skewed payoffs (longshots), add a **wild cluster bootstrap** — normal
  theory critical values are far too small when a bet pays +9 or −1.
- Check the **effective cluster count**. If a handful of clusters supply most of
  the variance, decline to certify. "This data cannot answer the question" is a
  legitimate and often correct verdict.

## Step 3 — Check the sample against what it can actually detect

Compute the detectable edge for the sample size on hand, and compare it to the
edge being claimed. Per-bet return variance is `p(1-p)d²` — near 1.0 at even
money, much larger on longshots.

Rules of thumb worth stating out loud:
- Detecting a 3% edge at even money needs roughly **8,700 bets** (two-sided,
  80% power). A few hundred bets cannot resolve anything below a 10% edge.
- Correlation multiplies this by the design effect. Five-rung ladders on 300
  games are not 1,500 observations.

A promotion threshold of a few hundred bets is governance paperwork, not
statistical proof. Say so directly when you see one.

## Step 4 — Count the searches

Ask how many variants were tried — including ones abandoned early, since
abandoning on a bad start is itself a selection rule. Then:

- Raise the significance threshold accordingly (Šidák, or deflate the observed
  t-statistic by the expected maximum under the null, about `sqrt(2 ln N)`).
- Treat any result whose reported t-statistic sits just above 2.0 after a wide
  search as unproven.

## Step 5 — Prefer CLV over profit as the promotion gate

Profit is edge plus a large amount of settlement noise. Closing line value is
edge measured against the sharpest available estimate, with much of the noise
removed, so it converges in hundreds of bets where profit needs thousands.

Two requirements:
- Compare against **de-vigged** closing prices. Raw implied probabilities sum
  to more than 1, and comparing against them manufactures phantom edges —
  especially on alternate lines, which carry much higher hold.
- Confirm the price is from the **book you actually execute on**. An edge
  measured on one venue and executed on another is partly a spread you never
  capture.

## Step 6 — Check for lookahead

Chronological ordering (`evidence_cutoff ≤ predicted_at ≤ event_start`) is
necessary but **not sufficient**. It cannot catch evidence whose timestamp is
honest while its content leaked backward: a revised stat line, a backfilled
lineup feed, a corrected injury report.

Anti-lookahead requires the evidence store to be **immutable and
content-addressed**, not merely ordered. Also verify calibration is fit
walk-forward — a calibrator fit on the whole sample and scored on that same
sample produces a reliability curve that looks immaculate and means nothing.

## Step 7 — Write the verdict

State one of exactly three outcomes:

1. **RESEARCH_ONLY** — default. Anything not prospectively validated.
2. **INCONCLUSIVE** — the sample cannot answer the question. Give the effective
   sample size and the detectable edge so the gap is explicit.
3. **VALIDATED** — survived every check above. Name the sample size, effective
   cluster count, CLV significance, and the number of variants searched.

Report the actual numbers, including the ones that undercut the conclusion. If
a check was skipped, say it was skipped rather than implying it passed.

## Failure modes to name out loud

- Reporting a t-statistic without saying whether it assumes independence.
- Treating an alternate-line ladder as independent observations.
- Promoting on backtested profit instead of prospective CLV.
- Comparing model probability against vig-inclusive market prices.
- Quoting a sample-size floor without stating the edge it can detect.
- Accepting that a model exists because a config file names it.
