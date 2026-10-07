# ADR-0016: Reward candidates and the duration-adjusted reward

Status: accepted (evaluator audit, 2026-10-06)

## Decision
- **Three candidates behind one switch.** In the audit, the switch is `config/rewards.toml [reward] active` (not frozen). In v1 (proposed), it moves to `config/evaluator.toml [env] reward`.
  - `raw`: the recorded watch_ratio.
  - `capped5`: min(watch_ratio, 5), as in EasyRL4Rec `KuaiData.load_mat`.
  - `duration_pct`: defined below.
- **duration_pct(u, v)** is the mid-rank percentile of the small-matrix row's `play_duration` within R_b, where b is the duration bucket of v (`kuairec_eval/labels.py`, `kuairec_eval/rewards.py`):
  - Duration is the canonical big-matrix median duration (ADR-0014).
  - The buckets are K equal-frequency buckets over the canonical durations of the 10,728 big-matrix videos.
  - R_b is the `play_duration` of every big-matrix row whose video falls in bucket b, repeat views included.
  - Mid-rank means (#below + 0.5 × #equal) / |R_b|. This keeps the mean at exactly 0.5 within each reference set, and stops the 240,151 zero-play big-matrix rows from piling up at 0.
  - Everything except the small row's play time comes from the big matrix, so a policy can compute the same label for training (`labels.big_matrix_label`).
- **K = 5**, chosen by a rule fixed before any small-matrix evaluation (`scripts/audit_bucket_choice.py`): the smallest K in {5, 10, 20, 50, 100} with |Spearman| ≤ 0.05 between per-video mean percentile, computed on big-matrix rows, and duration over the catalog. Every K qualified (|ρ| ≤ 0.0062), so the rule picks 5.
- **Application in the audit:** `audit/reward_env.py` wraps the frozen v0 environment and replaces only the reward of recorded pairs. Under raw, every replaced value is asserted equal to the frozen environment's own reward.

## Consequences
On search users, duration_pct has a per-video Spearman of **+0.226** with duration. It stays between +0.22 and +0.23 for every K from 5 to 100, while big-matrix rows give about 0 for every K (`scripts/audit_duration_correlation.py`). So the lean isn't a bucketing artifact. Relative to big-matrix viewing of the same videos, small-matrix (forced-exposure) users give longer videos systematically higher play-time percentiles. OPEN_QUESTIONS A1.
