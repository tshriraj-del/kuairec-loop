# ADR-0030: Phase 1 training data and the validation protocol

Status: accepted (Phase 1, 2026-10-22)

## Training data (brief item 1)
- **Rows:** one per (user, video) pair, keeping the **first** big-matrix view. Its label is `kuairec_eval.labels.DurationBuckets.percentile(video, play_duration of that view)` with K = 5, the public definition of the frozen reward (`config/phase1.toml [label]`). A training row is then one exposure, like one environment step, and repeat views cannot leak a later re-watch into an earlier label.
- **Measured** (`.venv/bin/python scripts/phase1_data.py`): 12,530,806 big-matrix rows, all with a finite label and a timestamp, so 0 dropped as unusable. Keeping first views drops 2,229,837 repeat-view rows and leaves 10,300,969 pairs.

## Validation split (brief item 4: "by time where timestamps exist")
- Pairs are split by first-view timestamp quantile (`[split]`): T_fit [0, 70%), T_rank [70%, 85%), T_val [85%, 100%]. Dates: T_fit ends 2020-08-09 07:24 UTC, T_rank 2020-08-29 13:41 UTC, T_val 2020-09-09.
- **Catalog-cold histories (my choice).** In the big matrix, the 1,411 users the environment serves have no catalog video at all; their histories are entirely non-catalog. Validation copies that situation. Validation users are the 5,464 users with at least 10 catalog pairs in T_val, and each is scored from their **non-catalog** pairs before T_val only (median 193 pairs, versus 348 for the served users). Models are still fitted on everything before T_val.
- **Metric:** NDCG@10 of the scorer over the user's T_val catalog pairs, gains = the label, ties broken by lower video id.
- **The metric's range is narrow, and that is a finding.** Constant scores get 0.5418, and the best tuned scorer 0.5543. A user's own exposures were already chosen by the production recommender, and the label averages 0.5, so the floor is high. Tuning on it is weakly informed. The environment run decides; the full-catalog offline check (ADR-0038) is the closer proxy.

## Catalog
Taken from the environment's `PolicyContext`, the object policies receive. Policy code never reads it from the evaluator's data.
