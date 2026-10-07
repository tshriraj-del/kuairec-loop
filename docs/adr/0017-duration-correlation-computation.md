# ADR-0017: How the duration correlation is computed

Status: accepted (evaluator audit, 2026-10-06)

## The Phase 0 figure
−0.9967 came from `scripts/popularity_diagnostic.py`: per video, over search users only, Spearman between the video's `video_duration` *from its first small-matrix row* and the video's *median* watch_ratio, across 3,327 videos. `scripts/audit_duration_correlation.py` replicates it exactly (−0.9967).

## Three versions, all on search users and all using canonical duration (ADR-0014)
- **Per video:** Spearman over the 3,327 catalog videos between duration and the video's mean (and median) reward.
- **Per interaction:** Spearman over all 2,807,328 search-user rows between the duration of the row's video and the row's reward.
- **Across buckets:** 20 equal-frequency big-matrix duration buckets. Spearman and Pearson over the 20 buckets between the bucket's median catalog-video duration and the mean reward of its interactions. I used 20 rather than the reward's 5 so the rank correlation has more than 5 points.

| Reward | per video, mean | per video, median | per interaction | buckets, Spearman | buckets, Pearson |
|---|---|---|---|---|---|
| raw | −0.9848 | −0.9966 | −0.6200 | −1.0000 | −0.6758 |
| capped5 | −0.9931 | −0.9966 | −0.6200 | −1.0000 | −0.6831 |
| duration_pct | +0.2262 | +0.2205 | +0.0183 | +0.2090 | +0.4986 |
| play_duration in ms (not a candidate) | +0.1659 | −0.1238 | −0.0044 | +0.8827 | +0.4465 |

Aggregating per video removes user noise, so per-video correlations are much stronger than per-interaction ones. The Phase 0 figure is the strongest of the three versions.
