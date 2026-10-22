# ADR-0035: The rule-aware slate wrapper

Status: accepted (Phase 1); Phase 2 removes the rule knowledge

- One wrapper for every scorer (`policies/phase1/slate.py`). The scorer's catalog vector is sorted descending, ties broken by lower video id, and the user's excluded videos are dropped.
- At each step the wrapper shows the first video that does not trigger the **primary** quit rule: no category of the video may occur more than once in the last 9 recommendations.
- The rule is a policy-side copy, because policies may not import the evaluator. Its parameters are written in `config/phase1.toml [slate]`, not read from the frozen config. `tests/test_phase1.py` checks it against `kuairec_eval.env.boredom_quit` on 3,000 random sequences.
- If no safe video remains, the best one is shown and the session may end. This has not occurred in 30 steps.
- It is deterministic (`stochastic = False`), so the harness checks it on a second seed.
