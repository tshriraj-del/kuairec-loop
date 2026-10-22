# Open questions

## P. Phase 1, the base recommender (2026-10-22): open

Figures from `docs/PHASE1.md` (generated) and the ADRs cited.

- **P1. Two selections sit on the edge of the declared grid.** item-kNN chose k = 400 and EASE chose λ = 10,000, the largest values tried, and both validation curves were still rising (ADR-0032, ADR-0033). Widening the grid needs a newly declared environment grid. Is that worth doing, given P2?
- **P2. The big-matrix validation ranked the models backwards.** On the brief's time-split validation the three ranker variants (NDCG@10 0.577 to 0.586) beat EASE (0.554). In the environment, EASE beats the best ranker by 0.393 [0.260, 0.531] per session. The full-catalog offline check on search (item 7) agreed with the environment: EASE 0.598, best ranker 0.586. Ranking a user's own exposures does not predict choosing from the whole catalog. Should later tuning use the full-catalog check on search, which is development data, instead of the big-matrix split?
- **P3. Duration features hurt.** Allowing duration moves the ranker to short videos (mean catalog duration percentile 0.179, median 5.7 s, against the catalog's 0.500 and 9.6 s) and costs 0.387 [0.265, 0.510] per session. Big-matrix exposures reward short videos relative to their bucket. The forced-exposure reward does not, and even leans long (ADR-0016). Should Phase 2 drop duration from the ranker's inputs?
- **P4. Caption text adds nothing detectable** on top of duration: +0.063 [-0.042, +0.167] per session. Keep it or drop it?
- **P5. The non-personalized oracle is in-sample on each partition.** EASE beats it on search (+0.214 [0.064, 0.369]) but not on shadow (-0.094 [-0.345, +0.158]). An out-of-sample reference, common values chosen on search and applied to shadow, would separate the oracle's in-sample advantage from personalization. Shadow has now been used, so this would need a new partition plan.
- **P6. Gap captured is about zero.** The best public-data scorer captures -0.9% [-3.2%, +1.4%] of the personalization gap on shadow. Public data gets to the non-personalized ceiling and no further. The remaining 11 points of reward per session are preferences that a user's big-matrix history on non-catalog videos does not reveal. The obvious next source is in-session feedback (Phase 2). Is that the intended Phase 2 scope?
- **P7. Category affinity does not transfer.** Scorer (a) gets -5.1% [-11.8%, +1.5%] of the category-affinity oracle's lift and ties rotation-random. The oracle knows each user's category means under forced exposure. Big-matrix category means from non-catalog videos do not predict them. Worth a direct measurement in Phase 2: the rank correlation of the two affinity vectors per user.

## D. For Phase 2

- **D1. What does a policy tuned to the primary setting lose under the tighter setting?** **Measured with the oracle (2026-10-06).** The free category-affinity oracle built for primary, run under tighter, loses 15.477 [15.327, 15.627] per session against the same oracle built for tighter. That is 140.0% of the tighter personalization gap. It ends every session by boredom, after 2.89 steps on average, and scores 14.013 below the full-cycle baseline (ADR-0028; `docs/AUDIT.md` section 3.1; `scripts/audit_diversity_level.py`). Phase 2 should repeat the measurement with learned policies.

## C. After the evaluator-v1 freeze: resolved

- C1, the reading of "strict rotation": **the full-cycle policy is the baseline** (your decision, made after the check), and both readings are recorded. Claim 2 stands, reworded: "from the session-ended signal alone, the system learns how soon a category can be revisited and how many repeats are tolerated, measured against the full-cycle policy". Split of the looser setting's 19.2%: 12.8% from revisiting after the window and 6.4% from tolerating repeats; tolerating repeats adds 4.7% under the primary setting (ADR-0028).

## A. From the evaluator audit: resolved by the v1 approval

- A1, the long-video lean: **accepted.** The duration probes are permanent rows (ADR-0016, ADR-0029).
- A2, the quit profiles: **N = 9 with thresholds 2, 1 and 0** (ADR-0019).
- A3, the T1 statement: **replaced** by the two-sided, session-length-controlled probe test with a 5% tolerance set after seeing the audit numbers. The old T1 stays as a secondary line (ADR-0027).
- A4 (the mask signal), A5 (a secondary metric), A6 (the non-personalized oracle definition) and A7 (deterministic re-runs): **approved as proposed** (ADR-0020, ADR-0016, ADR-0015, ADR-0021). No secondary metric was added.
- A8, commit order: **done.** The audit is commit `c8a9a0d`; v1 is the commit tagged `evaluator-v1`.

## B. Phase 0 questions

All resolved by the audit brief. See ADR-0019 to ADR-0024.
