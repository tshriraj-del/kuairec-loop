# ADR-0002: KuaiEnv quit rule, exact definition and parameters

Status: accepted (Phase 0)

## Context
The CIRS paper (arXiv 2204.01266, Sec 5.1.2) says: "for the most recent N recommended items, if there are more than n_Q items in the N items [that] have at least one attribute of the current recommended target, then the user ... ends the interaction". It reports N = 1, n_Q = 1 for KuaiEnv. As worded, that rule can never fire, because one window item cannot be "more than 1" items.

EasyRL4Rec (commit `95d0acf`, `KuaiEnv._determine_whether_to_leave`) implements something different. It counts each category across the window as a multiset and quits if *any category of the target* has count > `leave_threshold`. When window items carry one tag each, this means "more than n_Q window items share one particular category with the target", not "share any category". The parameters also disagree across three places:

| Source | N (window) | n_Q (threshold) | max_turn |
|---|---|---|---|
| CIRS paper, Sec 5.2 | 1 | 1 | 100 (30 in the limited-rounds study) |
| `KuaiEnv.__init__` defaults | 5 | 1 | 100 |
| `src/core/util/data.py` defaults for `KuaiEnv-v0` (used by every run script) | 9 | 1 | 30 |

## Decision
- Implement the code's per-category count. Its exact statement is in the `kuairec_eval/env.py` docstring (steps 1-6).
- Parameters: N = 9, n_Q = 1, max_turn = 30, the EasyRL4Rec `KuaiEnv-v0` defaults that its published run scripts actually use. All three are in `config/evaluator.toml [env]`.
- Termination happens at step t when `t >= max_turn - 1` (at most 30 recommendations), matching `BaseEnv.step`.
- The reward of the step that triggers the quit is returned and counted. This matches the code, where `terminated` is computed first and `mat[user, a_t]` is still returned. The paper's text ("no reward can be added anymore") is ambiguous about the triggering item. Session length counts that step.

## Alternatives
- Paper wording with N=1, n_Q=1: never terminates, which makes the rule decorative.
- Paper wording read as "≥ n_Q": a different rule from the code, and not comparable to EasyRL4Rec results.

## Consequences
Users quit very early under random recommendation: mean session length 8.28 on search, 99.98% of sessions ending by boredom (`runs/search_baselines.md`). That is the regime the published code creates. See OPEN_QUESTIONS Q1.
