### Reward: `raw` (evaluator v1)

Command: `.venv/bin/python scripts/audit_rewards.py `  
Per-video Spearman(mean reward over search users, canonical duration) = -0.9848

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate |
|---|---|---|---|---|---|---|
| random | 7.608 {7.456, 7.767} [7.509, 7.707] | 0.9174 | 8.29 | 0.254 | 0.8811 | 1.000 |
| popularity (big-matrix distinct users) | 3.862 {3.766, 3.965} | 0.7723 | 5.00 | 0.600 | 0.0018 | 1.000 |
| (a) shortest first | 20.422 {19.985, 20.893} | 2.5794 | 7.92 | 0.379 | 0.0036 | 1.000 |
| (b) category rotation, random pick | 23.018 {22.688, 23.370} [22.909, 23.127] | 0.7673 | 30.00 | 0.035 | 0.6341 | 0.000 |
| (c) category rotation, shortest pick | 48.994 {48.097, 50.009} | 1.6331 | 30.00 | 0.067 | 0.0117 | 0.000 |
| (d) big-matrix mean watch_ratio, >=100 rows | 15.691 {15.283, 16.121} | 2.6327 | 5.96 | 0.503 | 0.0021 | 1.000 |
| probe: longest first [extra] | 0.203 {0.190, 0.216} | 0.0403 | 5.03 | 0.596 | 0.0039 | 1.000 |
| probe: category rotation, longest pick [extra] | 6.187 {6.053, 6.316} | 0.2062 | 30.00 | 0.067 | 0.0123 | 0.000 |
| oracle: non-personalized (common values, greedy) | 67.893 {66.508, 69.494} | 2.2631 | 30.00 | 0.168 | 0.0108 | 0.000 |
| oracle: personalized greedy | 200.734 {190.375, 212.365} | 6.6911 | 30.00 | 0.192 | 0.9068 | 0.000 |

Quit-free versions: personalized bound 204.393, non-personalized 68.242.

| Paired comparison (A - B), per-user | Metric | Mean | 95% interval |
|---|---|---|---|
| **T1_shortest_not_above_random** a_shortest_first - random -> FAIL | cum_reward | 12.8141 | [12.3916, 13.2523] |
| **T2_personalized_beats_nonpersonalized** oracle_personalized - oracle_nonpersonalized -> PASS | cum_reward | 132.8401 | [122.6019, 143.3911] |
| a_shortest_first - random | mean_reward | 1.6609 | [1.6092, 1.7144] |
| c_rotation_shortest - b_rotation_random | cum_reward | 25.9758 | [25.2164, 26.8187] |
| c_rotation_shortest - b_rotation_random | mean_reward | 0.8659 | [0.8405, 0.8940] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 4.4280 | [4.0867, 4.7797] |
| probe_longest_first - random | cum_reward | -7.4053 | [-7.5602, -7.2557] |
| probe_rotation_longest - b_rotation_random | cum_reward | -16.8316 | [-17.1150, -16.5560] |
| probe_rotation_longest - b_rotation_random | mean_reward | -0.5611 | [-0.5705, -0.5519] |
| b_rotation_random - random | cum_reward | 15.4099 | [15.1463, 15.6801] |
