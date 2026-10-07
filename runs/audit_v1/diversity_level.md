### Diversity-level check (category-affinity oracle, duration_pct, search partition)

Command: `.venv/bin/python scripts/audit_diversity_level.py `

| Profile | N, n_Q | Affinity strict | Affinity free | Free - strict [95%] | Share of gap [95%] | Gap P - NP [95%] | Free - full cycle (share) |
|---|---|---|---|---|---|---|---|
| looser | 9, 2 | 17.472 | 18.197 | 0.725 [0.689, 0.763] | 6.4% [6.0%, 6.7%] | 11.414 [11.214, 11.611] | 2.190 (19.2%) |
| primary | 9, 1 | 17.472 | 18.003 | 0.531 [0.498, 0.565] | 4.7% [4.4%, 5.0%] | 11.346 [11.146, 11.545] | 1.996 (17.6%) |
| tighter | 9, 0 | 17.472 | 17.472 | 0.000 [0.000, 0.000] | 0.0% [0.0%, 0.0%] | 11.059 [10.862, 11.255] | 1.464 (13.2%) |

Decomposition of free minus full cycle (cumulative reward, share of gap):

| Profile | Concentrating on favourite categories: strict - full cycle | Tolerating repeats: free - strict |
|---|---|---|
| looser | 1.464 [1.416, 1.515] (12.8%) | 0.725 [0.689, 0.763] (6.4%) |
| primary | 1.464 [1.416, 1.515] (12.9%) | 0.531 [0.498, 0.565] (4.7%) |
| tighter | 1.464 [1.416, 1.515] (13.2%) | 0.000 [0.000, 0.000] (0.0%) |

**Decision (looser, strict reading, threshold 10%):** looser-setting difference is 6.4% of the personalization gap (< 10%): strict rotation is close to optimal everywhere; claim 2 returns to "a greedy personalized policy narrows and the diversity layer fixes it"; the three settings are robustness checks only.

**Flag:** the full-cycle reading of strict rotation gives 19.2% under looser, which lands on the other side of the threshold.

#### Ladder under `looser` (N=9, n_Q=2)

| Rung | Cum reward / session {95% users} [95% seeds] | Mean reward / rec | Session length | Boredom-quit rate |
|---|---|---|---|---|
| random | 7.987 {7.879, 8.102} [7.902, 8.071] | 0.5273 | 15.15 | 0.914 |
| (b) category rotation, random pick [permanent] | 15.880 {15.713, 16.046} [15.854, 15.905] | 0.5293 | 30.00 | 0.000 |
| (a) probe: shortest first [permanent] | 13.685 {13.438, 13.916} | 0.5328 | 25.69 | 1.000 |
| probe: longest first [permanent] | 9.757 {9.638, 9.881} | 0.5423 | 17.99 | 1.000 |
| (c) probe: category rotation, shortest pick [permanent] | 15.881 {15.683, 16.070} | 0.5294 | 30.00 | 0.000 |
| probe: category rotation, longest pick [permanent] | 16.087 {15.913, 16.264} | 0.5362 | 30.00 | 0.000 |
| oracle: category affinity, strict | 17.472 {17.300, 17.645} [17.452, 17.492] | 0.5824 | 30.00 | 0.000 |
| oracle: category affinity, full cycle | 16.008 {15.840, 16.174} [15.979, 16.036] | 0.5336 | 30.00 | 0.000 |
| oracle: category affinity, free | 18.197 {18.024, 18.373} [18.182, 18.213] | 0.6066 | 30.00 | 0.000 |
| oracle: non-personalized (common values, greedy) | 17.204 {17.008, 17.402} | 0.5735 | 30.00 | 0.000 |
| oracle: personalized greedy | 28.618 {28.501, 28.734} | 0.9539 | 30.00 | 0.000 |

#### Ladder under `primary` (N=9, n_Q=1)

| Rung | Cum reward / session {95% users} [95% seeds] | Mean reward / rec | Session length | Boredom-quit rate |
|---|---|---|---|---|
| random | 4.381 {4.322, 4.441} [4.340, 4.421] | 0.5282 | 8.29 | 1.000 |
| (b) category rotation, random pick [permanent] | 15.880 {15.713, 16.046} [15.854, 15.905] | 0.5293 | 30.00 | 0.000 |
| (a) probe: shortest first [permanent] | 4.163 {4.098, 4.227} | 0.5258 | 7.92 | 1.000 |
| probe: longest first [permanent] | 2.707 {2.661, 2.756} | 0.5379 | 5.03 | 1.000 |
| (c) probe: category rotation, shortest pick [permanent] | 15.881 {15.683, 16.070} | 0.5294 | 30.00 | 0.000 |
| probe: category rotation, longest pick [permanent] | 16.087 {15.913, 16.264} | 0.5362 | 30.00 | 0.000 |
| oracle: category affinity, strict | 17.472 {17.300, 17.645} [17.452, 17.492] | 0.5824 | 30.00 | 0.000 |
| oracle: category affinity, full cycle | 16.008 {15.840, 16.174} [15.979, 16.036] | 0.5336 | 30.00 | 0.000 |
| oracle: category affinity, free | 18.003 {17.828, 18.179} [17.978, 18.029] | 0.6001 | 30.00 | 0.000 |
| oracle: non-personalized (common values, greedy) | 17.202 {17.002, 17.402} | 0.5734 | 30.00 | 0.000 |
| oracle: personalized greedy | 28.548 {28.426, 28.668} | 0.9516 | 30.00 | 0.000 |

#### Ladder under `tighter` (N=9, n_Q=0)

| Rung | Cum reward / session {95% users} [95% seeds] | Mean reward / rec | Session length | Boredom-quit rate |
|---|---|---|---|---|
| random | 2.200 {2.169, 2.232} [2.178, 2.222] | 0.5293 | 4.16 | 1.000 |
| (b) category rotation, random pick [permanent] | 15.880 {15.713, 16.046} [15.854, 15.905] | 0.5293 | 30.00 | 0.000 |
| (a) probe: shortest first [permanent] | 2.640 {2.591, 2.688} | 0.5322 | 4.96 | 1.000 |
| probe: longest first [permanent] | 1.073 {1.049, 1.097} | 0.5361 | 2.00 | 1.000 |
| (c) probe: category rotation, shortest pick [permanent] | 15.881 {15.683, 16.070} | 0.5294 | 30.00 | 0.000 |
| probe: category rotation, longest pick [permanent] | 16.087 {15.913, 16.264} | 0.5362 | 30.00 | 0.000 |
| oracle: category affinity, strict | 17.472 {17.300, 17.645} [17.452, 17.492] | 0.5824 | 30.00 | 0.000 |
| oracle: category affinity, full cycle | 16.008 {15.840, 16.174} [15.979, 16.036] | 0.5336 | 30.00 | 0.000 |
| oracle: category affinity, free | 17.472 {17.300, 17.645} [17.452, 17.492] | 0.5824 | 30.00 | 0.000 |
| oracle: non-personalized (common values, greedy) | 17.152 {16.961, 17.339} | 0.5717 | 30.00 | 0.000 |
| oracle: personalized greedy | 28.210 {28.074, 28.338} | 0.9403 | 30.00 | 0.000 |
