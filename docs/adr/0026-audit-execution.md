# ADR-0026: How the audit ran on the frozen v0 evaluator

Status: accepted (evaluator audit, 2026-10-06)

- **Nothing frozen was changed.** `kuairec_eval/FROZEN_MANIFEST.json` still verifies with the phase-0 digest `83457af4e4f78e6c...`. New modules inside `kuairec_eval/` (`labels.py`, `rewards.py`, `diagnostics/`) are outside the v0 manifest, which v1 fixes (ADR-0025).
- **The runner is the frozen harness.** `audit/runner.py` calls the frozen `verify`, `partitions.load`, `run_seed`, `seed_metrics`, `t_interval` and `user_bootstrap`. `tests/test_audit.py` checks that its summary equals `harness.run`'s exactly. It additionally keeps per-user values for paired comparisons.
- **Scope:** search partition only. The shadow partition was not used. The holdout was not read (`scripts/final_report.py` was not run).
- **Seeds:** stochastic policies use the harness seeds 0-9; deterministic ones use seed 0. Per-user values of deterministic policies are identical across seeds (`tests/test_audit.py`), but pooled per-seed means can differ in the last float bit, because user order changes the summation order.
- **Static-scan fix:** the Phase 0 AST scan missed `from kuairec_eval import env`. `tests/test_policy_isolation.py` now matches banned modules by prefix and by `module.name`, and bans `kuairec_eval.rewards`, `kuairec_eval.diagnostics` and `audit` for policies. These are tests, which are not frozen.
- **v1 validation:** the proposed patch was applied in a disposable copy of the repository with a throwaway manifest. There its tests ran, along with the reward audit and the differential check (`runs/audit/v1_preview/README.md`). The copy was deleted.
