# ADR-0039: The shadow run and the claim-1 gate

Status: accepted (Phase 1)

- **One shadow run.** The ladder is regenerated on shadow, and only the best search scorer runs there, once (enforced, ADR-0036).
- **The non-personalized oracle on shadow** takes its common values from the shadow users, in-sample. That is its definition as a ceiling for the evaluated population, so the gap on shadow is measured on shadow.
- **Claim 1** passes if the best scorer beats rotation-random **and** the stronger duration probe on shadow, each with a paired 95% interval above zero. Whether it beats the non-personalized oracle is reported separately and does not enter the gate. If the gate fails, Phase 1 stops and reports why, without adding models.
