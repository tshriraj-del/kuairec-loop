### Reward: `capped5`

Command: `.venv/bin/python scripts/audit_rewards.py `  
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
