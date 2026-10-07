# ADR-0023: Process isolation is deferred; it is an entry condition for the agent-loop phase

Status: accepted (evaluator v1)

## Decision
Phase 1 keeps the in-process protections of ADR-0007 and ADR-0015. Before the first evaluation of a policy that can execute arbitrary code or tools, which is the agent-loop phase, the following must hold:
1. The environment and its reward matrix run in a separate OS process (or container), and only `reset`/`step` cross the boundary.
2. The policy process cannot read `small_matrix.csv` or the reward tables. Its filesystem view excludes `data/raw/` (container mount or OS permissions).
3. A conformance test, run from the policy side, tries to read the small matrix and to import `kuairec_eval.rewards`/`diagnostics`, and expects both to fail.
4. The holdout guard moves to the environment process.

## Why not now
Phase 1 policies are written in this repository and reviewed. The in-process layers catch accidents, and each was tested by planting a violation. Deliberate evasion (`env._lookup.__closure__`) needs code that review would see.
