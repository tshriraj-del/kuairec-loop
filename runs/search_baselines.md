Command: `.venv/bin/python scripts/run_baselines.py --partition search`  
Partition: search (847 users), harness seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], manifest digest `dbd9cc3dc5d4cc33`, env {'quit_profile': 'primary', 'quit_window': 9, 'quit_threshold': 1, 'max_turn': 30, 'window_mode': 'intended', 'reward': 'duration_pct', 'reward_cap': 5.0, 'duration_buckets': 5}

Each cell: mean over runs, [95% t-interval over seeds, stochastic policies only], {95% user-bootstrap interval where defined}. Compare policies with harness.compare (paired bootstrap).

| Policy | Mean reward per recommendation | Cumulative reward per session | Session length | Share in most-recommended category | Catalog coverage | (diag) sessions ended by boredom | (diag) sessions ended by max_turn |
|---|---|---|---|---|---|---|---|
| random | 0.5282 [0.5252, 0.5312] {0.5226, 0.5339} | 4.381 [4.340, 4.421] {4.322, 4.441} | 8.293 [8.238, 8.348] {8.221, 8.366} | 0.2541 [0.2507, 0.2575] | 0.8811 [0.8756, 0.8866] | 0.9998 [0.9994, 1.0001] | 0.0002 [-0.0001, 0.0006] |
| popularity | 0.5231 {0.5145, 0.5320} | 2.616 {2.572, 2.660} | 5.000 {5.000, 5.000} | 0.6000 | 0.0018 | 1.0000 | 0.0000 |
| rotation_random | 0.5293 [0.5285, 0.5302] {0.5238, 0.5349} | 15.880 [15.854, 15.905] {15.713, 16.046} | 30.000 [30.000, 30.000] {30.000, 30.000} | 0.0351 [0.0350, 0.0352] | 0.6341 [0.6327, 0.6355] | 0.0000 [0.0000, 0.0000] | 1.0000 [1.0000, 1.0000] |
| shortest_first | 0.5258 {0.5178, 0.5338} | 4.163 {4.098, 4.227} | 7.917 {7.895, 7.939} | 0.3789 | 0.0036 | 1.0000 | 0.0000 |
| longest_first | 0.5379 {0.5292, 0.5466} | 2.707 {2.661, 2.756} | 5.033 {5.008, 5.074} | 0.5961 | 0.0039 | 1.0000 | 0.0000 |
| rotation_shortest | 0.5294 {0.5228, 0.5357} | 15.881 {15.683, 16.070} | 30.000 {30.000, 30.000} | 0.0667 | 0.0117 | 0.0000 | 1.0000 |
| rotation_longest | 0.5362 {0.5304, 0.5421} | 16.087 {15.913, 16.264} | 30.000 {30.000, 30.000} | 0.0667 | 0.0123 | 0.0000 | 1.0000 |
