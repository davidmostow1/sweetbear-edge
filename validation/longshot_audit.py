"""Does cluster-robust inference survive longshot payoffs?

CR1 sandwich standard errors assume roughly symmetric residuals. Betting
violates that: a bet at decimal odds 10 pays +9 or -1, and when a whole
cluster shares one outcome the cluster totals are violently skewed.

This script measures the false positive rate of both procedures on data
generated under a TRUE NULL (zero edge). Anything materially above the
nominal 5% means the method will authorise dead strategies.
"""

from __future__ import annotations

import numpy as np

from sweetbear.significance import clustered_significance, wild_cluster_bootstrap_p


def false_positive_rates(
    n_groups: int, decimal_odds: float, unbalanced: bool, n_trials: int = 600
) -> tuple[float, float]:
    p = 1.0 / decimal_odds  # zero edge by construction
    cr1_hits = wild_hits = 0
    for s in range(n_trials):
        rng = np.random.default_rng(s + n_groups * 7919 + int(decimal_odds * 13))
        sizes = (
            rng.integers(1, 40, size=n_groups)
            if unbalanced
            else np.full(n_groups, 20)
        )
        values, labels = [], []
        for g in range(n_groups):
            outcome = rng.random() < p  # whole cluster shares fate
            values.append(np.where(np.full(sizes[g], outcome), decimal_odds - 1.0, -1.0))
            labels.append(np.full(sizes[g], g))
        v = np.concatenate(values)
        c = np.concatenate(labels)
        if v.size < 3:
            continue
        cr1_hits += clustered_significance(v, c, n_resamples=1).significant
        wild_hits += wild_cluster_bootstrap_p(v, c, n_resamples=399, seed=s) < 0.05
    return cr1_hits / n_trials, wild_hits / n_trials


def main() -> None:
    print("False positive rate under a TRUE NULL (nominal 5%)")
    print("  clusters  odds  balance   CR1      wild bootstrap")
    for n_groups in (10, 20, 40):
        for odds in (2.0, 10.0):
            for unbalanced in (False, True):
                cr1, wild = false_positive_rates(n_groups, odds, unbalanced)
                tag = "uneven" if unbalanced else "even  "
                mark = lambda x: f"{x:6.1%}" + ("  FAIL" if x > 0.10 else "      ")
                print(
                    f"  {n_groups:8d}  {odds:4.1f}  {tag}  {mark(cr1)} {mark(wild)}"
                )


if __name__ == "__main__":
    main()
