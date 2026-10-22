# ADR-0036: The environment-run grid and the run ledger

Status: accepted (Phase 1)

- **Declared before the first run.** `scripts/phase1_declare.py` refuses if the ledger has any entry, and writes `runs/phase1/env_grid.json`:
  - search: the 10 ladder rows and the 6 scorers at their offline-selected configurations;
  - shadow: the 10 ladder rows and one scorer, the best of the six on search cumulative reward.
- **Enforced.** `scripts/phase1_common.run()` raises for an undeclared row, a config hash that differs from the declared one, or a second shadow run of the same row (`tests/test_phase1.py`).
- **Config hash:**
  - ladder rows: row name plus manifest digest;
  - base scorers: name, slate rule, label definition, pairs-file hash, selected parameters and artifact hash;
  - rankers: the booster file hash, the base artifacts and, for the text variant, the embedding file.
- **Ledger.** `runs/phase1/ledger.jsonl` gets one line per seed executed: time, policy, config hash, seed, partition, role (`reported` or `determinism_check`), grid hash, manifest digest, git commit and command line.
