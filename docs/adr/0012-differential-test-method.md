# ADR-0012: Differential test against EasyRL4Rec's KuaiEnv

Status: accepted (Phase 0)

## Decision
- **Reference:** EasyRL4Rec cloned at commit `95d0acfa55e2150d0775ebd41de0826272edcc94` into `third_party/` (gitignored). Its `preprocessing_kuairec.py`, `KuaiData.load_mat`, `KuaiData.load_category` and `KuaiEnv` run unmodified in `data/cache/easyrl4rec_ws/`.
- **Shims, none touching logic:** `src.core` is registered as a bare package so its `__init__` (torch, tianshou, policy zoo) does not run. `torch` is stubbed with `manual_seed` and a placeholder `Tensor`. `numba.njit` is an identity decorator, and the decorated functions are not on the KuaiEnv path. `df_dist_small=None` avoids its O(n²) pure-Python similarity matrix, which `KuaiEnv.step` never reads.
- **Sequences (no repeats, length 30)** per search user: 3 random, the popularity ranking, a "diverse" sequence that avoids recent categories (drives sessions to max_turn), a "stress" sequence from one category (quits immediately), and a "capped" sequence that starts with the user's watch_ratio > 5 videos.
- **Comparison:** per-step reward over the common prefix and the terminal step, for both our window modes. Causes are classified as `cap_at_5`, `negative_slice_window` or `UNEXPLAINED`. The script exits non-zero on any unexplained difference.
- **pytest** runs it on the first 40 search users. The full 847-user run is a script: `.venv/bin/python scripts/differential.py`.
