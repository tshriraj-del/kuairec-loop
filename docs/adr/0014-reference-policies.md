# ADR-0014: Reference policies (a)-(d), one extra and two probes

Status: accepted (evaluator audit, 2026-10-06)

## Context
The audit brief named four public-data reference policies. Their mechanics were left open: what "duration" means, how a rotation avoids the quit rule without knowing its parameters, the category order, the view floor, and tie-breaks.

## Decision
- **Duration** is the canonical duration d(v): the median `video_duration` over v's big-matrix rows. `video_duration` varies across rows for 4,034 of 10,728 big-matrix videos. For catalog videos, the small-matrix median differs from the big-matrix median for 1 video, by 32 ms (`scripts/audit_data_checks.py`).
- **(a) ShortestFirst:** the shortest unseen catalog video, ties broken by `video_id`. No quit avoidance. Deterministic.
- **(b), (c) CategoryRotation:** shows only single-tag videos (2,209 of 3,327) and cycles through the 29 categories that have one. Categories 23 and 24 have none; 0, 13 and 29 have exactly one. A category recurs no sooner than 29 steps later, and exhausted categories are skipped. The policy never reads N or n_Q. It cannot trigger the rule when N < 29, for any threshold ≥ 0. (b) draws a random category order per session and a uniform video inside each category. (c) uses a fixed order, by each category's shortest single-tag video, and picks the shortest unseen video.
- **(d) BigMeanLabel("raw"):** mean raw watch_ratio over all big-matrix rows (repeat views included). Only videos with at least **100** big-matrix rows are eligible, which excludes 14 of 3,327. The 1st percentile is 368 rows, so the floor removes only near-unobserved videos. Ties broken by `video_id`.
- **Extra, not requested: (e) BigMeanLabel(active label).** The big-matrix mean of the active reward's own label, with the same floor. It is shown only for capped5 and duration_pct, because under raw it is (d). It is the public-data estimate of what the non-personalized oracle knows.
- **Probes, not requested: LongestFirst and CategoryRotation(pick="longest").** These mirror (a) and (c). I added them after duration_pct showed a per-video Spearman of +0.23 with duration (ADR-0017), to test whether that lean can be exploited.

## Consequences
On search, (b) and (c) run 30 steps with 0 boredom quits (`scripts/audit_rewards.py`; `tests/test_audit.py`). Under v1 they stay quit-free under all three quit profiles (`tests/test_harness_v1.py` in the v1 patch).

## Amendment (evaluator v1 approval, 2026-10-06)
The four duration probes, plain and inside rotation, and rotation-random are permanent rows of every results table, and their code is frozen (ADR-0029). The probes are no longer "extra": they are the measuring sticks of the acceptance test (ADR-0027).
