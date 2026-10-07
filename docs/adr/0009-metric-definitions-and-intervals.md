# ADR-0009: Metric definitions and confidence intervals

Status: accepted (Phase 0)

Pending: the evaluator v1 proposal supersedes the interval section (paired bootstrap, seed intervals for stochastic policies only; ADR-0021) and renames mean_watch_ratio to mean_reward.

## Definitions (per seed, pooled over the seed's sessions)
- `mean_watch_ratio`: mean reward over recommendations whose pair is recorded. Each recommendation has equal weight, so longer sessions weigh more.
- `cum_reward`: mean over sessions of the summed reward.
- `session_length`: mean recommendations per session, including the quitting step.
- `top_category_share`: max over categories c of (recommendations whose video carries c) / (all recommendations). A video with k tags counts toward each of its k categories, so shares across categories can sum to more than 1.
- `catalog_coverage`: distinct videos recommended in the seed run / 3,327.
- Diagnostics: `missing_pair_rate`, `boredom_quit_rate`, `max_turn_rate`.

## Intervals
- **Required by the brief:** a two-sided 95% Student-t interval over the S seed-level values (S−1 dof). For rates near 0 it can extend below 0 (e.g. random `max_turn_rate` [−0.0001, 0.0006]). We report it as computed, unclipped.
- **Added:** a 95% user-cluster bootstrap (2,000 resamples, rng seed 0) for `mean_watch_ratio`, `cum_reward` and `session_length`. Each user's value is averaged over seeds before resampling. This captures which-users-you-drew variance, which the seed interval cannot, especially for deterministic policies. Coverage and category share are run-level and have no per-user form, so they get no bootstrap interval.

## Alternatives
Per-session means of `mean_watch_ratio` (weights short sessions up). First-tag-only categories (arbitrary, because tag order has no documented meaning).
