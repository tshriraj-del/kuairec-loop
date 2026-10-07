### Reward: `raw`

Command: `.venv/bin/python scripts/audit_rewards.py `  
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
