# ADR-0010: Global popularity = distinct big-matrix users per video

Status: accepted (Phase 0)

## Context
The brief fixes the source (big matrix only) but not the measure. The big matrix has 4,071,381 rows in repeated (user, video) pairs (`scripts/data_report.py`), so "interaction count" and "user count" differ.

## Decision
popularity(v) = number of distinct big-matrix users with at least one row for v. Videos are ranked descending with ties broken by ascending `video_id`. A catalog video absent from the big matrix would score 0, but none is: all 3,327 have big-matrix rows. The policy recommends the highest-ranked unseen video, the same sequence for every user.

## Alternatives
- Row count including repeats: rewards rewatch loops of a few users.
- EasyRL4Rec's `get_item_popularity`: the share of users with watch_ratio ≥ 1. That mixes the outcome into popularity.
- Mean watch_ratio in the big matrix: that is a quality score, not popularity.

## Measured consequence
`.venv/bin/python scripts/popularity_diagnostic.py` gives a Spearman correlation between this popularity and search users' median small-matrix watch_ratio of −0.0115 (p = 0.51). Popularity carries no watch_ratio signal here, so its mean watch_ratio (0.772) falls below random (0.916). Its first 5 videos include three with category 28, so every session quits at step 5.
