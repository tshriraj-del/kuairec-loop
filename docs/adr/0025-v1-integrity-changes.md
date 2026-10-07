# ADR-0025: Integrity changes for evaluator v1

Status: accepted (evaluator v1)

- **Newly frozen:** `kuairec_eval/labels.py`, `kuairec_eval/rewards.py`, `kuairec_eval/diagnostics/__init__.py`, `kuairec_eval/diagnostics/oracles.py`.
- **`big_matrix.csv` joins the frozen data**, because the duration_pct buckets and reference sets come from it. It is about 1.0 GB and is hashed once per process.
- **`verify()` now fails on any `.py` file inside `kuairec_eval/` that is not in the manifest.** The audit showed why this matters: its own unfrozen modules sat inside the package, and the v0 check could not see them. `tests/test_frozen.py` plants `kuairec_eval/sneaky.py` and expects a failure.
- `config/rewards.toml` is deleted; its switch moves into `config/evaluator.toml`.
- `audit/` is deleted, since the environment and harness now do what it did. The v0 audit is reproducible from the commit that contains it.

## Amendment (evaluator v1 approval, 2026-10-06)
`policies/baselines.py` and `policies/reference.py` are also frozen (ADR-0029). `kuairec_eval/diagnostics/oracles.py`, already frozen, now includes the category-affinity oracle (ADR-0028).
