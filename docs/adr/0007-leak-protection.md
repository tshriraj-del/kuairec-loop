# ADR-0007: How "small-matrix values only through step()" is enforced

Status: accepted (Phase 0)

## Decision
Three layers:
1. **Storage.** The reward matrix lives only inside the closure returned by `_build_oracle`. It is never an instance attribute. `tests/test_leakage.py` walks every attribute of the env and fails on any array, DataFrame, Series or stray float, and pins the public surface to `{catalog, categories, config, history, manifest_digest, reset, step, users}`.
2. **Runtime guard.** `data.load_small_matrix()` inspects the call stack and raises `PolicyDataAccessError` if any frame's file is under `policies/`.
3. **Static scan.** `tests/test_policy_isolation.py` parses every file in `policies/` and fails on imports of `kuairec_eval.env/harness/partitions/_integrity`, the names `load_small_matrix`/`_build_oracle`/`_lookup`/`_read_matrix`, star imports, or any string containing `small_matrix`.

Each layer was checked by planting a violation: an ndarray attribute, a float attribute, `pd.read_csv('data/small_matrix.csv')`, an import of `load_small_matrix`, an import from `kuairec_eval.env`, and `e._lookup`. All six were caught.

## Limits
Python can't enforce this against deliberate evasion (`env._lookup.__closure__`, building the filename at runtime, `importlib`). The goal is to make accidental leakage impossible and deliberate leakage obvious in review. A process boundary (env in a subprocess) would close this. Deferred: OPEN_QUESTIONS Q7.
