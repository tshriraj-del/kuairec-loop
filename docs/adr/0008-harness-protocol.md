# ADR-0008: Harness protocol (sessions, seeds, policy interface)

Status: accepted (Phase 0)

Pending: the evaluator v1 proposal amends the policy interface (stochastic declaration, begin_session(user, excluded)) and the seed rules (ADR-0021, ADR-0022).

## Decision
- **Policy interface:** `factory(context, seed) -> policy` with `begin_session(user)`, `act() -> video`, `observe(video, StepResult)`. `PolicyContext` holds only `catalog` (the 3,327 small-matrix video ids) and `categories`.
- **Per seed:** a fresh policy, every partition user visited exactly once in `default_rng(seed).permutation` order, one session per user. The policy keeps state across users within a seed, which a learning policy in later phases will need, and is rebuilt between seeds.
- **Default S = 10 seeds (0-9)**, in config. The brief left S open.
- **The policy sees `StepResult.observed`.** That is part of the step() feedback for the video it chose, not information about other cells.

## Consequences
For a deterministic, stateless policy (popularity), seeds change nothing. Its seed CI has zero width, which is correct but uninformative (ADR-0009).
