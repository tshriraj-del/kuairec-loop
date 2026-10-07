Command: `.venv/bin/python scripts/run_baselines.py --partition search`  
Partition: search (847 users), seeds [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], manifest digest `83457af4e4f78e6c`, env {'quit_window': 9, 'quit_threshold': 1, 'max_turn': 30, 'window_mode': 'intended', 'missing_pair_reward': 0.0}

Each cell: mean over seeds, [95% t-interval over seeds], {95% user-bootstrap interval where defined}.

| Metric | random | popularity |
|---|---|---|
| Mean watch_ratio of recommended videos | 0.9162 [0.9077, 0.9247] {0.9002, 0.9337} | 0.7724 [0.7724, 0.7724] {0.7533, 0.7931} |
| Cumulative reward per session | 7.558 [7.455, 7.661] {7.406, 7.706} | 3.854 [3.854, 3.854] {3.758, 3.958} |
| Session length | 8.281 [8.228, 8.335] {8.209, 8.355} | 5.000 [5.000, 5.000] {5.000, 5.000} |
| Share in most-recommended category | 0.2556 [0.2521, 0.2591] | 0.6000 [0.6000, 0.6000] |
| Catalog coverage | 0.8805 [0.8754, 0.8857] | 0.0015 [0.0015, 0.0015] |
| (diag) unrecorded-pair rate | 0.00393 [0.00336, 0.00451] | 0.00213 [0.00213, 0.00213] |
| (diag) sessions ended by boredom | 0.9998 [0.9994, 1.0001] | 1.0000 [1.0000, 1.0000] |
| (diag) sessions ended by max_turn | 0.0002 [-0.0001, 0.0006] | 0.0000 [0.0000, 0.0000] |
