### Reward: `raw` (evaluator v1, primary quit profile)

Command: `.venv/bin/python scripts/audit_rewards.py `  
Manifest digest `dbd9cc3dc5d4cc33`. Per-video Spearman(mean reward over search users, canonical duration) = -0.9848

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate |
|---|---|---|---|---|---|---|
| random | 7.608 {7.456, 7.767} [7.509, 7.707] | 0.9174 | 8.29 | 0.254 | 0.8811 | 1.000 |
| popularity (big-matrix distinct users) | 3.862 {3.766, 3.965} | 0.7723 | 5.00 | 0.600 | 0.0018 | 1.000 |
| (b) category rotation, random pick [permanent] | 23.018 {22.688, 23.370} [22.909, 23.127] | 0.7673 | 30.00 | 0.035 | 0.6341 | 0.000 |
| (a) probe: shortest first [permanent] | 20.422 {19.985, 20.893} | 2.5794 | 7.92 | 0.379 | 0.0036 | 1.000 |
| probe: longest first [permanent] | 0.203 {0.190, 0.216} | 0.0403 | 5.03 | 0.596 | 0.0039 | 1.000 |
| (c) probe: category rotation, shortest pick [permanent] | 48.994 {48.097, 50.009} | 1.6331 | 30.00 | 0.067 | 0.0117 | 0.000 |
| probe: category rotation, longest pick [permanent] | 6.187 {6.053, 6.316} | 0.2062 | 30.00 | 0.067 | 0.0123 | 0.000 |
| (d) big-matrix mean watch_ratio, >=100 rows | 15.691 {15.283, 16.121} | 2.6327 | 5.96 | 0.503 | 0.0021 | 1.000 |
| oracle: category affinity, strict | 31.347 {30.824, 31.879} [31.122, 31.572] | 1.0449 | 30.00 | 0.100 | 0.5727 | 0.000 |
| oracle: category affinity, full cycle | 23.508 {23.126, 23.902} [23.336, 23.681] | 0.7836 | 30.00 | 0.043 | 0.6378 | 0.000 |
| oracle: category affinity, free | 33.903 {33.272, 34.562} [33.721, 34.086] | 1.1301 | 30.00 | 0.185 | 0.5566 | 0.000 |
| oracle: non-personalized (common values, greedy) | 67.893 {66.508, 69.494} | 2.2631 | 30.00 | 0.168 | 0.0108 | 0.000 |
| oracle: personalized greedy | 200.734 {190.375, 212.365} | 6.6911 | 30.00 | 0.192 | 0.9068 | 0.000 |

Quit-free versions: personalized bound 204.393, non-personalized 68.242.

**Acceptance: FAIL.** Personalization gap 132.840; T1 margin = 5% of the gap = +/-6.642.

| Test (paired bootstrap over users) | Mean | 95% interval | Share of gap | Result |
|---|---|---|---|---|
| T1 rotation_shortest - rotation_random (cum reward) | 25.9758 | [25.2164, 26.8187] | +19.6% [+19.0%, +20.2%] | FAIL |
| T1 rotation_longest - rotation_random (cum reward) | -16.8316 | [-17.1150, -16.5560] | -12.7% [-12.9%, -12.5%] | FAIL |
| T2 personalized - non-personalized oracle (cum reward) | 132.8401 | [122.6019, 143.3911] | 100% | pass |
| secondary: legacy T1 shortest_first - random (cum reward) | 12.8141 | [12.3916, 13.2523] | +9.6% | fail (not in verdict) |

| Other paired comparisons (A - B) | Metric | Mean | 95% interval |
|---|---|---|---|
| shortest_first - random | mean_reward | 1.6609 | [1.6092, 1.7144] |
| longest_first - random | cum_reward | -7.4053 | [-7.5602, -7.2557] |
| rotation_shortest - rotation_random | mean_reward | 0.8659 | [0.8405, 0.8940] |
| rotation_longest - rotation_random | mean_reward | -0.5611 | [-0.5705, -0.5519] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 4.4280 | [4.0867, 4.7797] |
| oracle_affinity_free - oracle_affinity_strict | cum_reward | 2.5566 | [2.2553, 2.8716] |
| rotation_random - random | cum_reward | 15.4099 | [15.1463, 15.6801] |
