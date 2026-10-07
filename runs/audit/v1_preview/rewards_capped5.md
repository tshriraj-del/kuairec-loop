### Reward: `capped5` (evaluator v1)

Command: `.venv/bin/python scripts/audit_rewards.py `  
Per-video Spearman(mean reward over search users, canonical duration) = -0.9931

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate |
|---|---|---|---|---|---|---|
| random | 7.288 {7.173, 7.399} [7.209, 7.366] | 0.8787 | 8.29 | 0.254 | 0.8811 | 1.000 |
| popularity (big-matrix distinct users) | 3.810 {3.733, 3.898} | 0.7620 | 5.00 | 0.600 | 0.0018 | 1.000 |
| (a) shortest first | 19.277 {19.006, 19.543} | 2.4347 | 7.92 | 0.379 | 0.0036 | 1.000 |
| (b) category rotation, random pick | 22.305 {22.029, 22.578} [22.275, 22.334] | 0.7435 | 30.00 | 0.035 | 0.6341 | 0.000 |
| (c) category rotation, shortest pick | 46.820 {46.245, 47.405} | 1.5607 | 30.00 | 0.067 | 0.0117 | 0.000 |
| (d) big-matrix mean watch_ratio, >=100 rows | 14.694 {14.470, 14.923} | 2.4654 | 5.96 | 0.503 | 0.0021 | 1.000 |
| (e) big-matrix mean capped5 label, >=100 rows [extra] | 19.497 {19.232, 19.774} | 2.4622 | 7.92 | 0.379 | 0.0033 | 1.000 |
| probe: longest first [extra] | 0.203 {0.190, 0.216} | 0.0403 | 5.03 | 0.596 | 0.0039 | 1.000 |
| probe: category rotation, longest pick [extra] | 6.108 {6.002, 6.212} | 0.2036 | 30.00 | 0.067 | 0.0123 | 0.000 |
| oracle: non-personalized (common values, greedy) | 63.752 {63.041, 64.486} | 2.1251 | 30.00 | 0.197 | 0.0099 | 0.000 |
| oracle: personalized greedy | 110.193 {108.389, 111.997} | 3.6731 | 30.00 | 0.193 | 0.9005 | 0.000 |

Quit-free versions: personalized bound 111.998, non-personalized 63.957.

| Paired comparison (A - B), per-user | Metric | Mean | 95% interval |
|---|---|---|---|
| **T1_shortest_not_above_random** a_shortest_first - random -> FAIL | cum_reward | 11.9891 | [11.7463, 12.2375] |
| **T2_personalized_beats_nonpersonalized** oracle_personalized - oracle_nonpersonalized -> PASS | cum_reward | 46.4405 | [44.6904, 48.1667] |
| a_shortest_first - random | mean_reward | 1.5551 | [1.5254, 1.5848] |
| c_rotation_shortest - b_rotation_random | cum_reward | 24.5148 | [24.1188, 24.9001] |
| c_rotation_shortest - b_rotation_random | mean_reward | 0.8172 | [0.8040, 0.8300] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 1.5480 | [1.4897, 1.6056] |
| probe_longest_first - random | cum_reward | -7.0848 | [-7.1978, -6.9725] |
| probe_rotation_longest - b_rotation_random | cum_reward | -16.1965 | [-16.4075, -15.9902] |
| probe_rotation_longest - b_rotation_random | mean_reward | -0.5399 | [-0.5469, -0.5330] |
| b_rotation_random - random | cum_reward | 15.0171 | [14.8142, 15.2204] |
