# ADR-0024: Missing small-matrix timestamps need no action

Status: accepted (evaluator v1)

181,992 of 4,676,570 small-matrix rows have no timestamp, while all 12,530,806 big-matrix rows have one (`scripts/audit_data_checks.py`). The environment never needs time order:
1. **Reward is a static lookup.** Each (user, video) pair has exactly one row (0 duplicates), and the environment reads only `watch_ratio` and `play_duration` from it. Both are present on all rows.
2. **Sessions are synthetic.** Their order comes from the policy's recommendations, never from logged interaction order. The quit rule depends only on that recommended sequence.
3. **duration_pct uses no time.** Its buckets and reference sets come from big-matrix durations and play times.
4. **Policies that want time order** can use the big matrix, which has complete timestamps.

**Assumption this makes explicit:** the small matrix is treated as one static preference snapshot, and preference drift during the collection window is ignored. Revisit if a later phase needs time-ordered replay or temporal splits of the small matrix.
