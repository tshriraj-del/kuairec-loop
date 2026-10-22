# ADR-0038: The offline ranking check (brief item 7)

Status: accepted (Phase 1)

- **Data:** search users only. Their rows are sliced from the reward matrix at once, and the holdout rows are never used. For each user, the action set is every catalog video with a recorded reward, ranked by the scorer with ties to lower video id.
- **NDCG@10:** graded gains = the recorded duration_pct reward.
- **Recall@10:** a video is relevant when its reward is **at least 0.75**, the top quarter of big-matrix play times in its duration bucket. Recall = hits in the top 10 / min(10, number relevant). The cap keeps the figure readable, because users have hundreds of relevant videos.
- **Reference rows:**
  - random scores (`default_rng(0)`);
  - the non-personalized oracle's common values, in-sample;
  - the true rewards, which score 1.0 by construction.
