### Reward: `duration_pct`

Command: `.venv/bin/python scripts/audit_rewards.py `  
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
