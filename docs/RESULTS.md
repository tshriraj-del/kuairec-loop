# Phase 0 results

Each number below comes from the command shown with it. Env config for all runs: N=9, n_Q=1, max_turn=30, window_mode=intended, missing_pair_reward=0.0. Manifest digest `83457af4e4f78e6c99435b1386fa26dfb3ee71dfa17eee96effd63a7f10bda06`.

## 1. Matrix statistics vs published

`.venv/bin/python scripts/data_report.py --json data/data_report.json` (full output: `docs/data_report_output.txt`)

| Matrix | Stat | Measured | Published | Match |
|---|---|---|---|---|
| small | users | 1,411 | 1,411 | yes |
| small | videos | 3,327 | 3,327 | yes |
| small | interactions | 4,676,570 | 4,676,570 | yes |
| small | density | 99.62% | 99.6% | yes |
| big | users | 7,176 | 7,176 | yes |
| big | videos | 10,728 | 10,728 | yes |
| big | interactions | 12,530,806 | 12,530,806 | yes |
| big | density | 16.28% | 16.3% | yes |

All published figures match. Things the published figures don't show:
- Big-matrix density counts rows. There are only 10,300,969 unique (user, video) pairs (13.38%), because repeat views are separate rows.
- The small matrix has 17,827 unrecorded cells and 181,992 rows with no timestamp or date.
- Small-matrix watch_ratio: p50 0.769, p99 3.585, p99.9 12.153, max 571.521. 22,790 cells are above 5.

## 2. Overlap

Same command.

| Check | Value |
|---|---|
| (user, video) pairs in both matrices | **0** |
| Small users present in big | 1,411 / 1,411 |
| Small videos present in big | 3,327 / 3,327 |
| Big rows inside the small-user × small-video block | 0 |
| Big rows from small users (all on non-small videos) | 571,061 |
| Big rows on small videos | 10,272,525, from 5,765 big users |

The big matrix contains the small users and the small videos but excludes their cross block entirely. Big-matrix popularity therefore uses no small-matrix observation.

## 3. Baselines on the search partition (847 users, seeds 0-9)

`.venv/bin/python scripts/run_baselines.py --partition search` (machine-readable: `runs/search_baselines.json`)

Cells show: mean over 10 seeds, [95% t-interval over seeds], {95% user-cluster bootstrap where defined}.

| Metric | Uniform random | Global popularity (big matrix) |
|---|---|---|
| Mean watch_ratio of recommended videos | 0.9162 [0.9077, 0.9247] {0.9002, 0.9337} | 0.7724 [0.7724, 0.7724] {0.7533, 0.7931} |
| Cumulative reward per session | 7.558 [7.455, 7.661] {7.406, 7.706} | 3.854 [3.854, 3.854] {3.758, 3.958} |
| Session length | 8.281 [8.228, 8.335] {8.209, 8.355} | 5.000 [5.000, 5.000] {5.000, 5.000} |
| Share in most-recommended category | 0.2556 [0.2521, 0.2591] | 0.6000 [0.6000, 0.6000] |
| Catalog coverage | 0.8805 [0.8754, 0.8857] | 0.0015 [0.0015, 0.0015] |
| (diag) unrecorded-pair rate | 0.00393 [0.00336, 0.00451] | 0.00213 [0.00213, 0.00213] |
| (diag) sessions ended by boredom | 0.9998 [0.9994, 1.0001] | 1.0000 [1.0000, 1.0000] |
| (diag) sessions ended by max_turn | 0.0002 [−0.0001, 0.0006] | 0.0000 [0.0000, 0.0000] |

How to read this:
- **Popularity is deterministic.** Every user gets the same 5 videos and quits at step 5, because three of the first five carry category 28. So its seed interval has zero width, coverage is 5/3,327 and the top-category share is 3/5. Only the user bootstrap shows its uncertainty.
- **Popularity scores below random on watch_ratio.** `.venv/bin/python scripts/popularity_diagnostic.py`: the Spearman correlation between big-matrix distinct-user count and search users' median watch_ratio is −0.0115 (p = 0.51). Video duration vs median watch_ratio is −0.9967, so watch_ratio here mostly measures how short a video is.
- **The quit rule dominates.** 99.98% of random sessions end by boredom, and max_turn = 30 almost never binds.

## 4. Differential check vs EasyRL4Rec KuaiEnv (commit 95d0acf)

`.venv/bin/python scripts/differential.py` on all 847 search users: 5,929 sequences, 64,468 reference steps. Every difference is listed in `runs/differential_differences.csv` (9,506 rows); the summary is in `runs/differential_summary.json`.

| Our window mode | Kind | Cause | Count | By source |
|---|---|---|---|---|
| intended | quit step | negative_slice_window | 1,348 | random 1,025 (of 2,541 sequences), capped 323 |
| intended | reward | cap_at_5 | 3,863 | capped 3,596, random 107, diverse 132, popularity 17, stress 11 |
| easyrl4rec_slice | quit step | none | **0** | |
| easyrl4rec_slice | reward | cap_at_5 | 4,295 | capped 3,996, random 139, diverse 132, popularity 17, stress 11 |
| either | anything | UNEXPLAINED | **0** | |

- All 1,348 quit-step differences fall at t ∈ {5, 6, 7, 8}, the only steps where `seq[t-9:t]` wraps. Ours quits earlier in all 1,348 (`runs/differential_differences.csv`).
- Reference sessions: all 847 `diverse` sequences reach max_turn and every `stress`/`popularity`/`capped` sequence ends by boredom. Both termination paths are therefore exercised.
- In slice mode, the only difference left is the reference capping watch_ratio at 5. The quit logic, max_turn handling, reward lookup and the zero for unrecorded pairs agree on all 64,468 steps.

## 5. Tests

`.venv/bin/python -m pytest -v` (full output: `docs/test_output.txt`): **37 passed, 0 skipped, 0 failed.**

| Brief item | Tests |
|---|---|
| Small-matrix values only via step() | `test_leakage.py` (4) |
| Policies cannot import the small matrix | `test_policy_isolation.py` (5: AST scan per file plus runtime guard) |
| Same seed gives identical numbers | `test_determinism.py` (2) |
| Differential check vs EasyRL4Rec | `test_differential.py` (first 40 search users; fails on any UNEXPLAINED) |
| Frozen hash check | `test_frozen.py` (12: every frozen file changed in turn, plus data) |
| Holdout guard and partitions | `test_partitions.py` (4) |
| Quit rule, repeat ban, rewards, max_turn | `test_env.py` (9) |

Each leak and isolation guard was also checked by planting a violation (an ndarray attribute, a float attribute, `pd.read_csv('…small_matrix…')`, an import of `load_small_matrix`, an import from `kuairec_eval.env`, `e._lookup`). All six were caught.

## Not measured

- Holdout numbers: `scripts/final_report.py` has never been run.
- Shadow-partition numbers: not requested for Phase 0. Run `.venv/bin/python scripts/run_baselines.py --partition shadow`.
- Variance across partition seeds (a different 60/20/20 draw): the partition is frozen, so only one draw exists.
