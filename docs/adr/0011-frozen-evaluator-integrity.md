# ADR-0011: What is frozen and when it is checked

Status: accepted (Phase 0)

Amended by ADR-0025 (evaluator v1): more frozen files, big_matrix.csv frozen, unlisted modules fail verification.

## Decision
- `kuairec_eval/FROZEN_MANIFEST.json` stores sha256 hashes for `kuairec_eval/{__init__,_integrity,data,env,partitions,harness}.py`, `config/evaluator.toml`, the three partition files, and the data files `small_matrix.csv` and `item_categories.csv`.
- `_integrity.verify()` runs in `KuaiEnv.__init__`, `partitions.load` and `harness.run`. Any mismatch raises `FrozenEvaluatorError` naming the file. Code and config are rehashed on every call. The data CSVs are hashed once per process.
- The check runs at those entry points, not at package import, so `scripts/data_report.py` can load data before the first freeze.
- Re-freezing requires `scripts/freeze.py --refreeze`. Every results file records `manifest_digest`, so a re-freeze is visible in results (current digest `83457af4e4f78e6c…`).
- Policies, scripts and tests are not frozen.

## Evidence
`tests/test_frozen.py` changes each of the 10 frozen code/config/partition files and one data file in a copy, and checks that each change is rejected.
