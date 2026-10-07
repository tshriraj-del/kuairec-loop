# kuairec-loop

Closed-loop short-video recommender on KuaiRec 2.0 (CC BY-SA 4.0). **Evaluator v1** (tag `evaluator-v1`): data loading, the environment, the frozen evaluator, reference policies and oracle ceilings. No recommender models yet.

```
kuairec_eval/          FROZEN evaluator (hash-checked): data.py env.py labels.py rewards.py partitions.py
                       harness.py _integrity.py diagnostics/ (oracles; policies may not import)
config/evaluator.toml  FROZEN reward switch (duration_pct), quit profiles, partition seed, harness settings
data/partitions/       FROZEN search / shadow / holdout user lists
policies/              FROZEN baselines.py and reference.py (permanent reference rows, ADR-0029);
                       v1 protocol: `stochastic`, begin_session(user, excluded), act(), observe()
scripts/               data_report, freeze, run_baselines, differential, final_report (holdout, guarded),
                       audit_* (reward audit, diversity-level check, correlations, survival), audit_report
tests/                 pytest suite (docs/v1_test_output.txt)
docs/                  AUDIT.md (generated), OPEN_QUESTIONS.md, adr/, issues/ (draft, not filed)
```

Setup (the dataset and EasyRL4Rec are gitignored, see ADR-0001 and ADR-0012):

```bash
uv sync --all-groups
```

Reproduce:

```bash
.venv/bin/python scripts/data_report.py
.venv/bin/python scripts/run_baselines.py --partition search
.venv/bin/python scripts/audit_rewards.py
.venv/bin/python scripts/audit_diversity_level.py
.venv/bin/python scripts/differential.py
.venv/bin/python -m pytest -v
.venv/bin/python scripts/audit_report.py
```

Every results table includes the permanent rows: rotation-random and the four duration probes, plain and inside rotation (ADR-0029). Compare policies with `harness.compare(result_a, result_b, metric)`, a paired bootstrap over users (ADR-0021).

Changing anything frozen (`kuairec_eval/`, `config/evaluator.toml`, `data/partitions/`, `policies/baselines.py`, `policies/reference.py`, the three data CSVs), or adding a module inside `kuairec_eval/`, makes every later run fail until `scripts/freeze.py --refreeze` is run on purpose.
