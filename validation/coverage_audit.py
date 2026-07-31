"""Adversarial audit: does the clustered estimator actually deliver what it claims?"""
import numpy as np
from sweetbear.significance import clustered_significance, required_sample_size

rng = np.random.default_rng(42)

# --- TEST 1: False positive rate under the null, with correlated data ---
# Truth: zero edge. Correlated bets. How often does each method cry "significant"?
def trial(seed, n_groups=30, per_group=20, icc_shared=1.0):
    r = np.random.default_rng(seed)
    shared = r.normal(0.0, icc_shared, size=n_groups)          # common shock
    idio   = r.normal(0.0, 1.0, size=(n_groups, per_group))    # idiosyncratic
    vals = (shared[:, None] + idio).ravel()
    clusters = np.repeat(np.arange(n_groups), per_group)
    naive = clustered_significance(vals, list(range(vals.size)), n_resamples=1)
    clust = clustered_significance(vals, clusters, n_resamples=1)
    return naive.significant, clust.significant

N = 400
naive_fp = clust_fp = 0
for s in range(N):
    a, b = trial(s)
    naive_fp += a; clust_fp += b
print(f"TEST 1 -- false positive rate under TRUE NULL (nominal 5%)")
print(f"  naive     : {naive_fp/N:6.1%}   <- how often you'd be fooled")
print(f"  clustered : {clust_fp/N:6.1%}")

# --- TEST 2: coverage of the clustered CI when there IS a real edge ---
TRUE = 0.05
cov_n = cov_c = 0
for s in range(N):
    r = np.random.default_rng(10_000+s)
    shared = r.normal(0.0, 1.0, size=30)
    idio = r.normal(TRUE, 1.0, size=(30, 20))
    vals = (shared[:, None] + idio).ravel()
    clusters = np.repeat(np.arange(30), 20)
    nlo, nhi = clustered_significance(vals, list(range(vals.size)), n_resamples=1).confidence_interval
    clo, chi = clustered_significance(vals, clusters, n_resamples=1).confidence_interval
    cov_n += (nlo <= TRUE <= nhi); cov_c += (clo <= TRUE <= chi)
print(f"\nTEST 2 -- 95% CI coverage of the TRUE mean (should be 95%)")
print(f"  naive     : {cov_n/N:6.1%}   <- badly undercovers = overconfident")
print(f"  clustered : {cov_c/N:6.1%}")

# --- TEST 3: is required_sample_size empirically right? ---
edge, d = 0.03, 2.0
n = required_sample_size(edge=edge, decimal_odds=d, power=0.80, alpha=0.05)
p = (1+edge)/d
hits = 0
M = 600
for s in range(M):
    r = np.random.default_rng(50_000+s)
    outcomes = r.random(n) < p
    ret = np.where(outcomes, d-1.0, -1.0)
    t = ret.mean()/(ret.std(ddof=1)/np.sqrt(n))
    hits += abs(t) > 1.96
print(f"\nTEST 3 -- empirical power at the prescribed n={n:,} (target 80%)")
print(f"  achieved power: {hits/M:6.1%}")
