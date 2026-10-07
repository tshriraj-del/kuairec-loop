# ADR-0027: Acceptance test v1, a two-sided, session-length-controlled duration test

Status: accepted (evaluator v1); supersedes T1 of ADR-0018, which stays in reports as a secondary line

## Decision (yours; the operational details are mine)
- **T1:** inside category rotation, the shortest-video probe (`rotation_shortest`) and the longest-video probe (`rotation_longest`) must each differ from `rotation_random` by less than **5% of the personalization gap**. All three run 30-step, quit-free sessions, so session length is controlled. Metric: cumulative reward per session, paired bootstrap over users. Each probe reports the point estimate, its 95% interval, and both as a share of the gap.
- **T2 (unchanged):** the personalized oracle beats the non-personalized one, with a 95% interval above 0.
- **Verdict:** T1 and T2 both pass.
- **Secondary, not in the verdict:** the legacy T1, shortest-first minus random with "interval not entirely above 0".

## Choices you did not specify
- **"Differ by less than" uses the interval.** A probe passes only if its whole 95% interval lies strictly inside (−margin, +margin), an equivalence-style test. A point estimate inside the margin with an interval crossing it fails (`tests/test_reporting.py`).
- **Margin** = 0.05 × the point estimate of the gap, measured under the same reward and quit profile in the same run. Uncertainty in the gap is not propagated, but the gap's interval is reported next to it.

## The tolerance was set after seeing the audit numbers
You chose 5% after reading the v0 audit (`runs/audit/`, commit `c8a9a0d`). Under duration_pct, the numbers you had seen were:
- personalization gap 11.37;
- `rotation_longest − rotation_random` = +0.187 [0.106, 0.268], which is +1.6% [+0.9%, +2.4%] of the gap;
- `rotation_shortest − rotation_random` = −0.030 [−0.117, 0.058], which is −0.3% [−1.0%, +0.5%].

Under raw and capped5, `rotation_shortest − rotation_random` was +25.80 and +24.35, against gaps of 133.15 and 46.75. So 5% is a post-hoc tolerance, not a pre-registered one. Any future reward candidate should be judged knowing that.
