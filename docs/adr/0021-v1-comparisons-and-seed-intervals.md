# ADR-0021: Comparisons by paired bootstrap; seed intervals for stochastic policies only (evaluator v1)

Status: **proposed**; supersedes ADR-0009's interval section on approval

## Decision
- `harness.compare(a, b, metric)` gives the paired bootstrap over users of mean_u[a(u) − b(u)] for `cum_reward`, `session_length` or `mean_reward`. It uses 10,000 resamples, `paired_bootstrap_seed = 0` and the configured confidence. It refuses unequal user sets.
- `harness.run` returns per-user values, averaged over runs.
- `ci_seeds` is reported only for stochastic policies (None otherwise). The user-cluster bootstrap stays as a descriptive per-policy interval.
- **What "stochastic" means:** per-user results may depend on the seed, through internal randomness or through state carried across users (the seed sets user order). A policy without randomness that learns across users is therefore stochastic. Every policy must declare a boolean `stochastic`. One declaring False runs on seeds[0] and is **re-run on seeds[1]**, and `run()` raises if any user's values differ. `tests/test_harness_v1.py` checks this with an order-dependent policy.
- Metric renames: `mean_watch_ratio` becomes `mean_reward`; `missing_pair_rate` is removed (ADR-0020).

## Why the re-run
Without it, declaring a stateful policy deterministic would silently drop its order variance. It doubles the cost only for deterministic policies, which are the cheap ones.
