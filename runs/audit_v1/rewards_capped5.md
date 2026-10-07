### Reward: `capped5` (evaluator v1, primary quit profile)

Command: `.venv/bin/python scripts/audit_rewards.py `  
Manifest digest `dbd9cc3dc5d4cc33`. Per-video Spearman(mean reward over search users, canonical duration) = -0.9931

Cum reward: mean per session over 847 search users {95% user bootstrap} [95% seed t-interval, stochastic policies only].

| Policy | Cum reward / session | Mean reward / rec | Session length | Top-category share | Coverage | Boredom-quit rate |
|---|---|---|---|---|---|---|
| random | 7.288 {7.173, 7.399} [7.209, 7.366] | 0.8787 | 8.29 | 0.254 | 0.8811 | 1.000 |
| popularity (big-matrix distinct users) | 3.810 {3.733, 3.898} | 0.7620 | 5.00 | 0.600 | 0.0018 | 1.000 |
| (b) category rotation, random pick [permanent] | 22.305 {22.029, 22.578} [22.275, 22.334] | 0.7435 | 30.00 | 0.035 | 0.6341 | 0.000 |
| (a) probe: shortest first [permanent] | 19.277 {19.006, 19.543} | 2.4347 | 7.92 | 0.379 | 0.0036 | 1.000 |
| probe: longest first [permanent] | 0.203 {0.190, 0.216} | 0.0403 | 5.03 | 0.596 | 0.0039 | 1.000 |
| (c) probe: category rotation, shortest pick [permanent] | 46.820 {46.245, 47.405} | 1.5607 | 30.00 | 0.067 | 0.0117 | 0.000 |
| probe: category rotation, longest pick [permanent] | 6.108 {6.002, 6.212} | 0.2036 | 30.00 | 0.067 | 0.0123 | 0.000 |
| (d) big-matrix mean watch_ratio, >=100 rows | 14.694 {14.470, 14.923} | 2.4654 | 5.96 | 0.503 | 0.0021 | 1.000 |
| (e) big-matrix mean capped5 label, >=100 rows | 19.497 {19.232, 19.774} | 2.4622 | 7.92 | 0.379 | 0.0033 | 1.000 |
| oracle: category affinity, strict | 29.696 {29.323, 30.062} [29.641, 29.750] | 0.9899 | 30.00 | 0.100 | 0.5375 | 0.000 |
| oracle: category affinity, full cycle | 22.664 {22.374, 22.947} [22.574, 22.754] | 0.7555 | 30.00 | 0.045 | 0.6387 | 0.000 |
| oracle: category affinity, free | 31.412 {31.027, 31.793} [31.331, 31.492] | 1.0471 | 30.00 | 0.190 | 0.4904 | 0.000 |
| oracle: non-personalized (common values, greedy) | 63.752 {63.041, 64.486} | 2.1251 | 30.00 | 0.197 | 0.0099 | 0.000 |
| oracle: personalized greedy | 110.193 {108.389, 111.997} | 3.6731 | 30.00 | 0.193 | 0.9005 | 0.000 |

Quit-free versions: personalized bound 111.998, non-personalized 63.957.

**Acceptance: FAIL.** Personalization gap 46.440; T1 margin = 5% of the gap = +/-2.322.

| Test (paired bootstrap over users) | Mean | 95% interval | Share of gap | Result |
|---|---|---|---|---|
| T1 rotation_shortest - rotation_random (cum reward) | 24.5148 | [24.1188, 24.9001] | +52.8% [+51.9%, +53.6%] | FAIL |
| T1 rotation_longest - rotation_random (cum reward) | -16.1965 | [-16.4075, -15.9902] | -34.9% [-35.3%, -34.4%] | FAIL |
| T2 personalized - non-personalized oracle (cum reward) | 46.4405 | [44.6904, 48.1667] | 100% | pass |
| secondary: legacy T1 shortest_first - random (cum reward) | 11.9891 | [11.7463, 12.2375] | +25.8% | fail (not in verdict) |

| Other paired comparisons (A - B) | Metric | Mean | 95% interval |
|---|---|---|---|
| shortest_first - random | mean_reward | 1.5551 | [1.5254, 1.5848] |
| longest_first - random | cum_reward | -7.0848 | [-7.1978, -6.9725] |
| rotation_shortest - rotation_random | mean_reward | 0.8172 | [0.8040, 0.8300] |
| rotation_longest - rotation_random | mean_reward | -0.5399 | [-0.5469, -0.5330] |
| oracle_personalized - oracle_nonpersonalized | mean_reward | 1.5480 | [1.4897, 1.6056] |
| oracle_affinity_free - oracle_affinity_strict | cum_reward | 1.7160 | [1.6008, 1.8314] |
| rotation_random - random | cum_reward | 15.0171 | [14.8142, 15.2204] |
