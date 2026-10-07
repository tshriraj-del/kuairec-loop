# ADR-0004: The 17,827 unrecorded small-matrix cells

Status: accepted (Phase 0)

Superseded by ADR-0020 (evaluator v1): unrecorded pairs leave the action set.

## Context
The small matrix is 99.62% dense, so 1,411 × 3,327 − 4,676,570 = 17,827 (user, video) cells have no recorded watch_ratio (`scripts/data_report.py`). The brief says step() "returns the recorded watch_ratio for that pair" and is silent about cells with no record. EasyRL4Rec builds a dense matrix from a `csr_matrix`, so missing cells silently become 0.0.

## Decision
- Every catalog video is a legal action for every user. Masking per user would tell the policy which cells are missing.
- An unrecorded cell returns `missing_pair_reward = 0.0` (config), matching EasyRL4Rec, with `StepResult.observed = False`.
- These steps are excluded from `mean_watch_ratio` because there is no recorded value. They are included in `cum_reward` (adding 0) and in session length. The harness reports `missing_pair_rate`: 0.39% for random and 0.21% for popularity on search.

## Alternatives
Forbid unrecorded actions (leaks the mask). Impute the user or video mean (invents data). Raise an error (makes random policies crash).
