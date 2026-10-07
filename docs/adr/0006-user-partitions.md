# ADR-0006: Partition procedure and holdout guard

Status: accepted (Phase 0)

## Decision
- The 1,411 small-matrix users are sorted, permuted with `numpy.random.default_rng(20261005)`, and cut at `round(0.6n) = 847` and `round(0.2n) = 282`. The remaining 282 form the holdout. The seed is in `config/evaluator.toml [partition]`.
- Written once to `data/partitions/{search,shadow,holdout}.json`. `partitions.write` refuses to overwrite, and all three files are in the frozen manifest.
- `partitions.load("holdout")` succeeds only when `__main__.__file__` resolves to `scripts/final_report.py`. The harness, tests and every other script get `HoldoutAccessError`. `final_report.py` also refuses to run without `--confirm-final`. It has **not** been run.
- The split is by user only, so search, shadow and holdout users all see the same 3,327-video catalog.

## Limits
The guard stops accidental reads. It does not stop deliberate ones: anyone can `json.load` the file directly. Holdout user ids are not secret, but their outcomes stay unread until the final report.
