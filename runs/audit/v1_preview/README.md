These files were produced by the PROPOSED evaluator v1, not by the frozen v0 evaluator.

How: a disposable copy of this repository (phase-0 + audit working tree) with
docs/proposals/evaluator-v1.patch applied and a throwaway manifest written by
`scripts/freeze.py --refreeze` inside that copy only. The copy was deleted afterwards;
nothing in this repository was re-frozen (kuairec_eval/FROZEN_MANIFEST.json still carries
the phase-0 digest 83457af4e4f78e6c...).

Commands run inside the copy:
  .venv/bin/python scripts/audit_rewards.py      -> rewards_*.{md,json}, acceptance.json
  .venv/bin/python scripts/differential.py       -> differential_summary.json, differential_stdout_v1.txt
  .venv/bin/python -m pytest -q                  -> 65 passed (validation.log)
