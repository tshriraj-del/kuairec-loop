# kuairec-loop

Closed-loop short-video recommender on KuaiRec 2.0 (CC BY-SA 4.0). **Phase 0:** data loading, the environment, the frozen evaluator, and two baselines.

```
kuairec_eval/          FROZEN evaluator (hash-checked): data.py env.py partitions.py harness.py _integrity.py
config/evaluator.toml  FROZEN quit-rule parameters, partition seed, harness seeds
data/partitions/       FROZEN search / shadow / holdout user lists
policies/baselines.py  uniform random, global popularity (big matrix only)
scripts/               data_report, freeze, run_baselines, differential, popularity_diagnostic, final_report
tests/                 37 tests
docs/                  RESULTS.md, OPEN_QUESTIONS.md, adr/0001-0013
```

Setup (the dataset and EasyRL4Rec are gitignored, see ADR-0001 and ADR-0012):

```bash
uv sync --all-groups
```

Reproduce:

```bash
.venv/bin/python scripts/data_report.py
.venv/bin/python scripts/run_baselines.py --partition search
.venv/bin/python scripts/differential.py
.venv/bin/python -m pytest -v
```

Changing anything under `kuairec_eval/`, `config/` or `data/partitions/` makes every later run fail until `scripts/freeze.py --refreeze` is run on purpose.
