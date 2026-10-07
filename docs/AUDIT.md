# Evaluator audit (2026-10-06)

This is a proposal only. Nothing was re-frozen: `kuairec_eval/FROZEN_MANIFEST.json` still verifies with the phase-0 digest `83457af4e4f78e6c...`. The holdout was not read. Phase 0 is committed as `c643479` and tagged `phase-0`. The audit work itself is uncommitted (OPEN_QUESTIONS A8).

## Verdict

- **I recommend `duration_pct`.** It is the only candidate that passes both acceptance tests, on the frozen v0 evaluator and again under the proposed v1 evaluator. Under raw and capped5, shortest-first beats random by about +12 cumulative reward per session.
- **Personalization can add at most 11.37 [11.17, 11.56] cumulative reward per session under duration_pct.** The personalized oracle reaches 28.55 and the non-personalized one 17.18. Category rotation alone already reaches 15.84, because avoiding the quit rule is worth more than any non-personalized choice of videos.
- **The v1 patch is ready and validated** in a disposable copy: 65 tests pass, the acceptance outcome is unchanged, and the differential check has 0 unexplained differences. It is not frozen.
- **Three decisions are yours** before freezing: A1 (a small long-video lean in duration_pct), A2 (which axis for the quit profiles), A3 (how T1 is stated).

## 1. How the -0.997 was computed, and the other two versions (item 4)

Phase 0's figure came from `scripts/popularity_diagnostic.py`. It is a **per-video** Spearman correlation over 3,327 videos, search users only, between each video's `video_duration` *from its first small-matrix row* and the video's *median* watch_ratio. `scripts/audit_duration_correlation.py` replicates it exactly (-0.9967). Every version below uses search users and canonical duration (the big-matrix median per video; ADR-0014, ADR-0017).

| Reward | per video, mean | per video, median | per interaction (2,807,328 rows) | 20 duration buckets, Spearman | 20 buckets, Pearson |
|---|---|---|---|---|---|
| raw | -0.9848 | -0.9966 | -0.6200 | -1.0000 | -0.6758 |
| capped5 | -0.9931 | -0.9966 | -0.6200 | -1.0000 | -0.6831 |
| duration_pct | +0.2262 | +0.2205 | +0.0183 | +0.2090 | +0.4986 |

Aggregating per video averages away user noise, which makes the per-video figure the strongest of the three versions. For duration_pct, the per-video +0.23 is stable across bucket counts 5 to 100, while big-matrix rows give about 0 for every K (ADR-0016).

## 2. Results per reward candidate (items 2, 3, 6)

Command: `.venv/bin/python scripts/audit_rewards.py`. The frozen v0 evaluator is used for dynamics, with the candidate reward applied by `audit/reward_env.py`. Search partition, 847 users. Stochastic policies run on seeds 0-9, deterministic ones on seed 0. Comparisons are paired bootstraps over users (10,000 resamples). Rows marked [extra] were not requested (ADR-0014).

### raw watch_ratio

Per-video Spearman(mean reward over search users, canonical duration) = -0.9848

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate | Unrecorded-pair rate |
|---|---|---|---|---|---|---|---|
| random | 7.558 {7.406, 7.706} [7.455, 7.661] | 0.9162 | 8.28 | 0.256 | 0.8805 | 1.000 | 0.0039 |
| popularity (big-matrix distinct users) | 3.854 {3.758, 3.958} | 0.7724 | 5.00 | 0.600 | 0.0015 | 1.000 | 0.0021 |
| (a) shortest first | 20.410 {19.971, 20.884} | 2.5795 | 8.00 | 0.375 | 0.0024 | 1.000 | 0.0109 |
| (b) category rotation, random pick | 22.965 {22.630, 23.317} [22.824, 23.106] | 0.7668 | 30.00 | 0.035 | 0.6348 | 0.000 | 0.0017 |
| (c) category rotation, shortest pick | 48.763 {47.877, 49.766} | 1.6326 | 30.00 | 0.067 | 0.0090 | 0.000 | 0.0044 |
| (d) big-matrix mean watch_ratio, >=100 rows | 15.687 {15.279, 16.118} | 2.6326 | 6.00 | 0.500 | 0.0018 | 1.000 | 0.0069 |
| probe: longest first [extra] | 0.201 {0.189, 0.214} | 0.0402 | 5.00 | 0.600 | 0.0015 | 1.000 | 0.0009 |
| probe: category rotation, longest pick [extra] | 6.158 {6.026, 6.288} | 0.2060 | 30.00 | 0.067 | 0.0090 | 0.000 | 0.0034 |
| oracle: non-personalized fixed sequence (greedy, search-user means) | 67.588 {66.200, 69.177} | 2.2650 | 30.00 | 0.167 | 0.0090 | 0.000 | 0.0054 |
| oracle: personalized greedy | 200.734 {190.375, 212.365} | 6.6911 | 30.00 | 0.192 | 0.9068 | 0.000 | 0.0000 |

Quit-free top-30 bounds: personalized 204.393, non-personalized 67.818.

| Paired comparison (A - B), per-user, 10,000 resamples | Metric | Mean | 95% interval |
|---|---|---|---|
| **T1_shortest_not_above_random** a_shortest_first - random -> FAIL | cum_reward | 12.8524 | [12.4298, 13.2856] |
| **T2_personalized_beats_nonpersonalized** oracle_personalized - oracle_nonpersonalized -> PASS | cum_reward | 133.1459 | [122.8698, 143.6890] |
| a_shortest_first - random | mean_reward | 1.6619 | [1.6101, 1.7153] |
| c_rotation_shortest - b_rotation_random | cum_reward | 25.7978 | [25.0402, 26.6442] |
| c_rotation_shortest - b_rotation_random | mean_reward | 0.8656 | [0.8403, 0.8937] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 4.4259 | [4.0845, 4.7776] |
| probe_longest_first - random | cum_reward | -7.3571 | [-7.5073, -7.2105] |
| probe_rotation_longest - b_rotation_random | cum_reward | -16.8073 | [-17.0920, -16.5352] |
| probe_rotation_longest - b_rotation_random | mean_reward | -0.5609 | [-0.5703, -0.5518] |
| b_rotation_random - random | cum_reward | 15.4070 | [15.1464, 15.6797] |
| d_big_mean_watch_ratio - random | cum_reward | 8.1287 | [7.7351, 8.5344] |

### watch_ratio capped at 5

Per-video Spearman(mean reward over search users, canonical duration) = -0.9931

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate | Unrecorded-pair rate |
|---|---|---|---|---|---|---|---|
| random | 7.245 {7.131, 7.355} [7.169, 7.322] | 0.8783 | 8.28 | 0.256 | 0.8805 | 1.000 | 0.0039 |
| popularity (big-matrix distinct users) | 3.802 {3.724, 3.889} | 0.7621 | 5.00 | 0.600 | 0.0015 | 1.000 | 0.0021 |
| (a) shortest first | 19.265 {18.994, 19.529} | 2.4347 | 8.00 | 0.375 | 0.0024 | 1.000 | 0.0109 |
| (b) category rotation, random pick | 22.238 {21.963, 22.516} [22.201, 22.275] | 0.7426 | 30.00 | 0.035 | 0.6348 | 0.000 | 0.0017 |
| (c) category rotation, shortest pick | 46.591 {46.019, 47.170} | 1.5599 | 30.00 | 0.067 | 0.0090 | 0.000 | 0.0044 |
| (d) big-matrix mean watch_ratio, >=100 rows | 14.690 {14.465, 14.919} | 2.4652 | 6.00 | 0.500 | 0.0018 | 1.000 | 0.0069 |
| (e) big-matrix mean capped5 label, >=100 rows [extra] | 19.481 {19.218, 19.754} | 2.4620 | 8.00 | 0.375 | 0.0024 | 1.000 | 0.0109 |
| probe: longest first [extra] | 0.201 {0.189, 0.214} | 0.0402 | 5.00 | 0.600 | 0.0015 | 1.000 | 0.0009 |
| probe: category rotation, longest pick [extra] | 6.080 {5.974, 6.184} | 0.2033 | 30.00 | 0.067 | 0.0090 | 0.000 | 0.0034 |
| oracle: non-personalized fixed sequence (greedy, search-user means) | 63.444 {62.734, 64.178} | 2.1242 | 30.00 | 0.167 | 0.0090 | 0.000 | 0.0044 |
| oracle: personalized greedy | 110.193 {108.389, 111.997} | 3.6731 | 30.00 | 0.193 | 0.9005 | 0.000 | 0.0000 |

Quit-free top-30 bounds: personalized 111.998, non-personalized 63.715.

| Paired comparison (A - B), per-user, 10,000 resamples | Metric | Mean | 95% interval |
|---|---|---|---|
| **T1_shortest_not_above_random** a_shortest_first - random -> FAIL | cum_reward | 12.0195 | [11.7783, 12.2662] |
| **T2_personalized_beats_nonpersonalized** oracle_personalized - oracle_nonpersonalized -> PASS | cum_reward | 46.7489 | [45.0004, 48.4825] |
| a_shortest_first - random | mean_reward | 1.5555 | [1.5258, 1.5853] |
| c_rotation_shortest - b_rotation_random | cum_reward | 24.3522 | [23.9610, 24.7298] |
| c_rotation_shortest - b_rotation_random | mean_reward | 0.8172 | [0.8042, 0.8298] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 1.5489 | [1.4905, 1.6066] |
| probe_longest_first - random | cum_reward | -7.0445 | [-7.1561, -6.9330] |
| probe_rotation_longest - b_rotation_random | cum_reward | -16.1588 | [-16.3708, -15.9504] |
| probe_rotation_longest - b_rotation_random | mean_reward | -0.5392 | [-0.5462, -0.5323] |
| b_rotation_random - random | cum_reward | 14.9930 | [14.7895, 15.1971] |
| e_big_mean_label - random | cum_reward | 12.2358 | [11.9946, 12.4871] |
| d_big_mean_watch_ratio - random | cum_reward | 7.4442 | [7.2339, 7.6585] |

### duration-adjusted (play-time percentile within big-matrix duration bucket, K = 5)

Per-video Spearman(mean reward over search users, canonical duration) = +0.2262

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate | Unrecorded-pair rate |
|---|---|---|---|---|---|---|---|
| random | 4.357 {4.298, 4.417} [4.316, 4.398] | 0.5282 | 8.28 | 0.256 | 0.8805 | 1.000 | 0.0039 |
| popularity (big-matrix distinct users) | 2.612 {2.569, 2.656} | 0.5236 | 5.00 | 0.600 | 0.0015 | 1.000 | 0.0021 |
| (a) shortest first | 4.160 {4.096, 4.224} | 0.5257 | 8.00 | 0.375 | 0.0024 | 1.000 | 0.0109 |
| (b) category rotation, random pick | 15.844 {15.679, 16.014} [15.815, 15.873] | 0.5291 | 30.00 | 0.035 | 0.6348 | 0.000 | 0.0017 |
| (c) category rotation, shortest pick | 15.814 {15.615, 16.006} | 0.5295 | 30.00 | 0.067 | 0.0090 | 0.000 | 0.0044 |
| (d) big-matrix mean watch_ratio, >=100 rows | 3.144 {3.093, 3.198} | 0.5277 | 6.00 | 0.500 | 0.0018 | 1.000 | 0.0069 |
| (e) big-matrix mean duration_pct label, >=100 rows [extra] | 3.287 {3.238, 3.333} | 0.5480 | 6.00 | 0.500 | 0.0018 | 1.000 | 0.0002 |
| probe: longest first [extra] | 2.688 {2.644, 2.731} | 0.5381 | 5.00 | 0.600 | 0.0015 | 1.000 | 0.0009 |
| probe: category rotation, longest pick [extra] | 16.031 {15.857, 16.208} | 0.5362 | 30.00 | 0.067 | 0.0090 | 0.000 | 0.0034 |
| oracle: non-personalized fixed sequence (greedy, search-user means) | 17.182 {16.985, 17.384} | 0.5734 | 30.00 | 0.200 | 0.0090 | 0.000 | 0.0011 |
| oracle: personalized greedy | 28.548 {28.426, 28.668} | 0.9516 | 30.00 | 0.194 | 0.9940 | 0.000 | 0.0000 |

Quit-free top-30 bounds: personalized 28.642, non-personalized 17.186.

| Paired comparison (A - B), per-user, 10,000 resamples | Metric | Mean | 95% interval |
|---|---|---|---|
| **T1_shortest_not_above_random** a_shortest_first - random -> PASS | cum_reward | -0.1966 | [-0.2603, -0.1318] |
| **T2_personalized_beats_nonpersonalized** oracle_personalized - oracle_nonpersonalized -> PASS | cum_reward | 11.3660 | [11.1667, 11.5648] |
| a_shortest_first - random | mean_reward | -0.0025 | [-0.0091, 0.0039] |
| c_rotation_shortest - b_rotation_random | cum_reward | -0.0299 | [-0.1170, 0.0579] |
| c_rotation_shortest - b_rotation_random | mean_reward | 0.0004 | [-0.0025, 0.0034] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 0.3783 | [0.3717, 0.3849] |
| probe_longest_first - random | cum_reward | -1.6687 | [-1.7268, -1.6107] |
| probe_rotation_longest - b_rotation_random | cum_reward | 0.1871 | [0.1063, 0.2680] |
| probe_rotation_longest - b_rotation_random | mean_reward | 0.0072 | [0.0045, 0.0099] |
| b_rotation_random - random | cum_reward | 11.4875 | [11.3584, 11.6181] |
| e_big_mean_label - random | cum_reward | -1.0695 | [-1.1259, -1.0119] |
| d_big_mean_watch_ratio - random | cum_reward | -1.2125 | [-1.2734, -1.1504] |

## 3. Recommendation and evidence (item 7)

| Reward | per-video rho with duration | T1 (a) - random, cum reward | T2 P - NP (personalization gap) | Both |
|---|---|---|---|---|
| raw | -0.9848 | +12.85 [12.43, 13.29] FAIL | +133.15 [122.87, 143.69] pass | FAIL |
| capped5 | -0.9931 | +12.02 [11.78, 12.27] FAIL | +46.75 [45.00, 48.48] pass | FAIL |
| duration_pct | +0.2262 | -0.197 [-0.260, -0.132] pass | +11.37 [11.17, 11.56] pass | **PASS** |

The same outcome under the v1 preview (`runs/audit/v1_preview/acceptance.json`): duration_pct T1 -0.218 [-0.281, -0.152], T2 +11.35 [11.15, 11.55]. raw and capped5 fail T1 at +12.81 and +11.99.

**Why duration_pct and not just "it passed":**
- It removes the short-video advantage. Shortest-first and random are tied per recommendation (-0.0025 [-0.0091, 0.0039]). Rotation-shortest and rotation-random, which share the same 30-step sessions, are tied too: -0.030 [-0.117, 0.058]. Under raw, the same rotation pair differs by +25.80 [25.04, 26.64].
- It keeps personal signal. The personalized oracle beats the non-personalized one by 0.378 per recommendation [0.372, 0.385].
- A policy can learn it from public data. The label is computable on big-matrix rows (`kuairec_eval/labels.py`).

**Caveats you should weigh:**
- T1 passes partly because shortest-first's sessions are shorter (8.00 vs 8.28 steps). Per recommendation it is tied with random, not below it (A3).
- A small long-video lean remains: rotation-longest beats rotation-random by +0.187 [0.106, 0.268] cumulative reward (+1.2%) and +0.0072 per recommendation. That is 1.6% of the personalization gap (A1).
- The gap is an in-sample greedy ceiling. Greedy reaches 28.548 of the 28.642 quit-free personalized bound, so optimiser slack is negligible.

## 4. Evaluator v1 proposal (item 8)

**The diff:** `docs/proposals/evaluator-v1.patch` (42 files, 630 insertions, 713 deletions). It applies cleanly to the current working tree (`git apply --check`). To adopt it after approval:

```bash
git apply docs/proposals/evaluator-v1.patch
```

```bash
.venv/bin/python scripts/freeze.py --refreeze
```

| Your decision | What the patch does | ADR |
|---|---|---|
| Correct window primary; EasyRL4Rec switch for the differential check only | `harness.run` raises on `window_mode != "intended"`; `scripts/differential.py` still builds slice-mode environments directly | 0019 |
| N=9, threshold 1, 30 turns primary; one looser, one tighter from the random survival curve; policies never see them | `[env] quit_profile` with `[quit_profiles]` primary (9,1), looser (9,2), tighter (9,0); policies get only PolicyContext, the excluded set and StepResults | 0019 |
| Unrecorded pairs leave the action set and are rejected | `reset()` returns `excluded`; harness passes it to `begin_session(user, excluded)`; `step()` raises `UnavailableActionError`; `observed` and `missing_pair_reward` removed | 0020 |
| Action set = 3,327 videos; one session per user per seed; state across users allowed | unchanged mechanics, recorded; stateful policies must declare `stochastic = True` | 0022 |
| Comparisons by paired bootstrap; seed intervals for stochastic policies only | `harness.compare()`; per-user values in every result; `ci_seeds` is None for deterministic policies, whose declaration is checked by a second run | 0021 |
| Process isolation not now; entry condition for the agent-loop phase | recorded with four concrete entry conditions | 0023 |
| Missing timestamps: no action | recorded why the environment never needs time order | 0024 |
| (from item 7) reward switch | `[env] reward = "duration_pct"` with raw and capped5 available | 0016, 0018 |
| (integrity) | labels, rewards and oracles frozen; `big_matrix.csv` frozen; any unlisted module in `kuairec_eval/` fails verification | 0025 |

**Quit profiles, from the random policy's survival curve** (`scripts/audit_survival.py`, 100,000 Monte Carlo sequences). The rule I fixed beforehand (vary N) could not produce a tighter setting, so the proposal uses the threshold at N=9:

| N=9 | median length | mean length | P(reach 30) | S(8) | S(15) |
|---|---|---|---|---|---|
| tighter, n_Q=0 | 4 (0.50x) | 4.16 (0.50x) | 0.0000 | 0.0377 | 0.0000 |
| primary, n_Q=1 | 8 | 8.26 | 0.0004 | 0.5360 | 0.0512 |
| looser, n_Q=2 | 13 (1.62x) | 15.01 (1.82x) | 0.0929 | 0.8922 | 0.4131 |

**Validation** in a disposable copy with the patch applied and a throwaway manifest, since deleted (`runs/audit/v1_preview/README.md`, `validation.log`):
- 65 of 65 tests passed.
- The reward audit reproduced the acceptance outcome.
- The differential check ran 5,929 sequences and 64,508 reference steps with 0 unexplained differences.
- Every ported script ran.

The v1 preview table for the recommended reward:

Per-video Spearman(mean reward over search users, canonical duration) = +0.2262

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate |
|---|---|---|---|---|---|---|
| random | 4.381 {4.322, 4.441} [4.340, 4.421] | 0.5282 | 8.29 | 0.254 | 0.8811 | 1.000 |
| popularity (big-matrix distinct users) | 2.616 {2.572, 2.660} | 0.5231 | 5.00 | 0.600 | 0.0018 | 1.000 |
| (a) shortest first | 4.163 {4.098, 4.227} | 0.5258 | 7.92 | 0.379 | 0.0036 | 1.000 |
| (b) category rotation, random pick | 15.880 {15.713, 16.046} [15.854, 15.905] | 0.5293 | 30.00 | 0.035 | 0.6341 | 0.000 |
| (c) category rotation, shortest pick | 15.881 {15.683, 16.070} | 0.5294 | 30.00 | 0.067 | 0.0117 | 0.000 |
| (d) big-matrix mean watch_ratio, >=100 rows | 3.145 {3.093, 3.199} | 0.5277 | 5.96 | 0.503 | 0.0021 | 1.000 |
| (e) big-matrix mean duration_pct label, >=100 rows [extra] | 3.287 {3.238, 3.333} | 0.5480 | 6.00 | 0.500 | 0.0018 | 1.000 |
| probe: longest first [extra] | 2.707 {2.661, 2.756} | 0.5379 | 5.03 | 0.596 | 0.0039 | 1.000 |
| probe: category rotation, longest pick [extra] | 16.087 {15.913, 16.264} | 0.5362 | 30.00 | 0.067 | 0.0123 | 0.000 |
| oracle: non-personalized (common values, greedy) | 17.202 {17.002, 17.402} | 0.5734 | 30.00 | 0.200 | 0.0105 | 0.000 |
| oracle: personalized greedy | 28.548 {28.426, 28.668} | 0.9516 | 30.00 | 0.194 | 0.9940 | 0.000 |

Quit-free versions: personalized bound 28.642, non-personalized 17.205.

| Paired comparison (A - B), per-user | Metric | Mean | 95% interval |
|---|---|---|---|
| **T1_shortest_not_above_random** a_shortest_first - random -> PASS | cum_reward | -0.2178 | [-0.2814, -0.1524] |
| **T2_personalized_beats_nonpersonalized** oracle_personalized - oracle_nonpersonalized -> PASS | cum_reward | 11.3458 | [11.1462, 11.5453] |
| a_shortest_first - random | mean_reward | -0.0026 | [-0.0091, 0.0038] |
| c_rotation_shortest - b_rotation_random | cum_reward | 0.0018 | [-0.0862, 0.0893] |
| c_rotation_shortest - b_rotation_random | mean_reward | 0.0001 | [-0.0029, 0.0030] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 0.3782 | [0.3715, 0.3848] |
| probe_longest_first - random | cum_reward | -1.6734 | [-1.7330, -1.6128] |
| probe_rotation_longest - b_rotation_random | cum_reward | 0.2075 | [0.1267, 0.2884] |
| probe_rotation_longest - b_rotation_random | mean_reward | 0.0069 | [0.0042, 0.0096] |
| b_rotation_random - random | cum_reward | 11.4988 | [11.3697, 11.6288] |

## 5. Draft issue for EasyRL4Rec (item 9)

The draft is `docs/issues/easyrl4rec-negative-window.md`, not filed. The same `seq[t - N:t]` window bug is live in five environments: KuaiRec, KuaiRand, Coat, YahooR3 and MovieLens. The minimal reproduction (`docs/issues/easyrl4rec_window_repro.py`) was run against the cloned code at commit 95d0acf. It prints `window used: [3]`, `intended window: [0, 1, 2, 3]`, `terminated: False`. With the one-line `max(0, t - N)` fix patched in memory, it prints `terminated: True`.

## 6. ADRs

- **New:** 0014 reference policies, 0015 oracles, 0016 reward candidates, 0017 correlation computation, 0018 acceptance test and recommendation, 0026 audit execution.
- **Proposed for v1:** 0019 to 0025.
- **Phase 0 ADRs with a pending v1 note:** 0002, 0003, 0004, 0005, 0008, 0009, 0011, 0012.
- **Amended now:** 0007, for the static-scan fix.

## 7. Open questions

These are in `docs/OPEN_QUESTIONS.md`, section A:
- A1, the long-video lean;
- A2, the quit-profile axis;
- A3, the T1 statement;
- A4, the mask signal;
- A5, a secondary metric;
- A6, the non-personalized oracle definition;
- A7, deterministic re-runs;
- A8, commit order.

## Commands behind every number

| Numbers | Command | Output |
|---|---|---|
| duration constancy, floors, tags, unrecorded cells, mask signal, timestamps | `.venv/bin/python scripts/audit_data_checks.py --json runs/audit/data_checks.json` | `runs/audit/data_checks.txt` |
| K = 5 | `.venv/bin/python scripts/audit_bucket_choice.py --json runs/audit/bucket_choice.json` | `runs/audit/bucket_choice.txt` |
| section 1 | `.venv/bin/python scripts/audit_duration_correlation.py --json runs/audit/duration_correlation.json` | `runs/audit/duration_correlation.txt` |
| sections 2-3 | `.venv/bin/python scripts/audit_rewards.py` | `runs/audit/rewards_*.{md,json}`, `acceptance.json`, `per_user_*.csv` |
| quit profiles | `.venv/bin/python scripts/audit_survival.py --json runs/audit/survival.json` | `runs/audit/survival.txt` |
| v1 preview | see `runs/audit/v1_preview/README.md` | `runs/audit/v1_preview/` |
| tests (main tree: frozen v0 plus audit) | `.venv/bin/python -m pytest -v` | `docs/audit_test_output.txt` (57 passed) |
