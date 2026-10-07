# ADR-0029: Permanent reference rows, frozen

Status: accepted (evaluator v1)

## Decision
- **Permanent rows (your decision, A1):** `policies/reference.py:PERMANENT_ROWS` holds the four duration probes, plain (`shortest_first`, `longest_first`) and inside rotation (`rotation_shortest`, `rotation_longest`). It also holds `rotation_random`, because the session-length-controlled test (ADR-0027) compares the rotation probes against it. That addition is mine.
- **Every results table includes them:** `scripts/run_baselines.py` always appends them (`--policies` only picks extra baselines), and so do `scripts/audit_rewards.py`, `scripts/audit_diversity_level.py` and `scripts/final_report.py`. `tests/test_reporting.py` fails if any script that calls `harness.run(` does not reference `PERMANENT_ROWS`.
- **Frozen (my choice):** `policies/baselines.py` and `policies/reference.py` are in the manifest. The acceptance test and every results table depend on their exact behaviour, so a silent edit to a probe would change the measuring stick. New policies go in new files, which are not frozen. Changing a reference policy requires `scripts/freeze.py --refreeze`, which then shows up as a new manifest digest in every results file.
