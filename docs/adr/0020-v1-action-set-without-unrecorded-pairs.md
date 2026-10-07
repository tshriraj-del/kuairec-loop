# ADR-0020: Unrecorded pairs leave the action set (evaluator v1)

Status: **proposed**; supersedes ADR-0004 on approval

## Decision (your decision; the delivery mechanism is mine)
- **Action set:** for user u, the 3,327 small-matrix videos minus u's unrecorded pairs. Search users have 12.56 on average and at most 32.
- **Delivery:** `reset(u)` returns `{"user", "t", "excluded": frozenset}` and the harness passes the set to `policy.begin_session(user, excluded)`. Policies need the set, or a random policy would hit a rejection on 0.39% of steps (v0 measurement).
- **Rejection:** `step()` raises `UnavailableActionError`, matching the repeat ban. A rejected call does not consume a step.
- `StepResult.observed` and `missing_pair_reward` are removed, because every legal step now has a recorded reward.

## Consequence: the mask is visible to policies, and it carries a weak signal
Spearman correlations on search users (`scripts/audit_data_checks.py`):

| | vs mean watch_ratio | vs mean duration_pct |
|---|---|---|
| per user, # unrecorded | +0.078 (p 0.023) | −0.091 (p 0.0079) |
| per video, # unrecorded | +0.030 (p 0.081) | −0.044 (p 0.011) |

These are small but detectable. OPEN_QUESTIONS A4.
