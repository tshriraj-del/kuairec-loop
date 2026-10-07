# ADR-0022: Sessions and the catalog (evaluator v1)

Status: **proposed** (records your decisions; no code change beyond ADR-0020/0021)

- **Catalog:** the 3,327 small-matrix videos. Policies trained on the big matrix know 10,728, but recommending outside the 3,327 is impossible (`KeyError`), not penalised.
- **Sessions:** one per user per run (= seed). Users are visited once each, in `default_rng(seed)` order.
- **State:** a policy may keep state across users within a run. It is rebuilt between runs. Because state makes results order-dependent, such a policy must declare `stochastic = True` (ADR-0021).
