# ADR-0005: Raw watch_ratio as reward; repeats raise

Status: accepted (Phase 0)

## Decision
- **Reward is the raw recorded watch_ratio**, as the brief specifies. EasyRL4Rec caps it at 5 (`KuaiData.load_mat`). 22,790 small-matrix cells exceed 5 and the maximum is 571.52 (`scripts/data_report.py`). This is the only reward difference in the differential test (cause `cap_at_5`).
- **A repeat recommendation raises `RepeatRecommendationError`.** The brief says "forbids", and an exception makes a policy bug loud. A zero reward or a forced quit would hide it. EasyRL4Rec allows repeats (its repeat check is commented out). A new session (reset) clears the history.

## Consequences
Means of raw watch_ratio are sensitive to a few large outliers. The 99.9th percentile is 12.15 and the maximum is 571.5, so one lucky recommendation can move a session's cumulative reward a lot. Whether to cap, log-transform or keep raw for later phases is OPEN_QUESTIONS Q3.
