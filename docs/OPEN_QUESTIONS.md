# Open questions

## A. From the evaluator audit (2026-10-06): open

These are decisions the audit made provisionally, each with an ADR. Nothing below was guessed silently.

- **A1. duration_pct still leans slightly toward long videos.** On search users, its per-video Spearman with duration is +0.226. It stays at +0.22 to +0.23 for every bucket count from 5 to 100, while big-matrix rows give about 0, so it isn't a bucketing artifact (`scripts/audit_duration_correlation.py`). It can be exploited, but only a little: rotation-longest beats rotation-random by +0.187 [0.106, 0.268] cumulative reward (+1.2%) and by +0.0072 per recommendation (`scripts/audit_rewards.py`). That is 1.6% of the 11.37 personalization gap. A fix that uses small-matrix statistics is ruled out by your "big matrix only" constraint. Do you accept the lean, or add a mirror acceptance test (longest-first must not beat random) before freezing? (ADR-0016, ADR-0018)
- **A2. Which axis for the quit profiles?** The rule I fixed beforehand (vary N, threshold 1) found a looser setting at N=3 but could not produce a tighter one: every N ≥ 6 has a random median of 8. I therefore propose the threshold axis at N=9: looser (9, 2), tighter (9, 0). The documented alternative is the N-axis looser (3, 1). Your pick. (ADR-0019)
- **A3. How should T1 be stated?** I read "must not score above random" as "the paired 95% interval for cumulative reward does not lie entirely above 0". Under duration_pct, shortest-first is significantly *below* random on cumulative reward only because its sessions are shorter (8.00 vs 8.28); per recommendation the two are tied (−0.0025 [−0.0091, 0.0039]). Do you want T1 stated on per-recommendation reward, or on the rotation pair (c) − (b), which isolates duration? Every reading passes duration_pct and fails raw and capped5. (ADR-0018)
- **A4. The visible mask carries a weak reward signal.** Removing unrecorded pairs from the action set (your decision) shows each user's 12.6 unrecorded videos to policies. Their count correlates weakly with reward: per user, −0.091 (p 0.008) with duration_pct (`scripts/audit_data_checks.py`). I've flagged it rather than acted on it. (ADR-0020)
- **A5. What duration_pct means as an objective.** It ranks play time against big-matrix (organic) viewing. Small-matrix exposure is forced, so random already scores 0.528 per recommendation, not 0.5. Should a capped watch_ratio be reported next to it as a secondary metric? (ADR-0016)
- **A6. The non-personalized oracle in v1.** It uses one common value vector, applied within each user's action set, because only 2,319 of 3,327 videos are recorded for every search user. Do you accept that definition? (ADR-0015)
- **A7. Deterministic policies get a second, checking run.** It verifies the `stochastic=False` declaration and doubles the cost of deterministic policies only. Accept? (ADR-0021)
- **A8. Commit order.** The audit work is uncommitted, because this brief only asked for the phase-0 commit. My suggestion: commit the audit as is and tag it, so the v0 audit commands stay reproducible. Then apply `docs/proposals/evaluator-v1.patch`, run `scripts/freeze.py --refreeze`, and commit that separately.

## B. Phase 0 questions: status after the audit brief

1. Quit parameters. **Resolved:** keep N=9, n_Q=1, max_turn=30 as primary, with profiles proposed (ADR-0019, A2).
2. Correct or EasyRL4Rec window. **Resolved:** the intended window is primary; the slice is kept only for the differential check (ADR-0019).
3. Reward scale and target. **Audited:** duration_pct is recommended, pending your approval (ADR-0018, A1, A3, A5).
4. Unrecorded cells. **Resolved:** removed from the action set (ADR-0020, A4).
5. Popularity measure. **Kept.** Reference policies (a)-(d) now give stronger non-learning baselines (ADR-0014).
6. Which interval decides comparisons. **Resolved:** paired bootstrap; seed intervals for stochastic policies only (ADR-0021).
7. Process isolation. **Resolved:** not now; it is an entry condition for the agent-loop phase (ADR-0023).
8. Catalog. **Resolved:** the 3,327 small-matrix videos (ADR-0022).
9. Sessions per user. **Resolved:** one per user per seed; state across users is allowed (ADR-0022).
10. Missing timestamps. **Resolved:** no action, and ADR-0024 records why.
