# ADR-0013: Tooling

Status: accepted (Phase 0)

- Python 3.12 (venv via `uv`). The system Python is 3.14, which several scientific wheels and EasyRL4Rec's dependency tree lag behind.
- Core dependencies: numpy, pandas, scipy (t quantiles). The dependency group `reference` (scikit-learn, gymnasium, tqdm, logzero) is needed only to import EasyRL4Rec. `dev` contains pytest. No torch.
- Config in TOML (`tomllib` from the standard library).
- Git initialised; nothing committed yet (commits wait for your go-ahead).
