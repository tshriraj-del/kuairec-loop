### Reward: `duration_pct` (evaluator v1)

Command: `.venv/bin/python scripts/audit_rewards.py `  
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
