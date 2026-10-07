# ADR-0015: Oracle ceilings

Status: accepted (evaluator audit, 2026-10-06); v1 amendment accepted

## Decision
- **Location and isolation:** `kuairec_eval/diagnostics/`. Policies may not use it, enforced three ways: an import-time guard, a call-time guard in every factory and constructor (Python caches modules, so the import guard fires only once), and the AST scan in `tests/test_policy_isolation.py`. Each was checked by a test that plants a violation.
- **Greedy step (shared by both oracles):** show the highest-value unseen video that does not trigger the boredom rule at this step, ties broken by lower `video_id`. Using the same optimiser for both means their gap measures personalization, not optimiser quality.
- **Personalized oracle:** the values are the user's own row of the active reward matrix.
- **Non-personalized oracle:** the values are each video's mean reward over the **search** users, and it is evaluated on the same users, so it is in-sample by construction, as the brief asked.
  - v0 audit: unrecorded cells count as 0, the value the frozen environment returns. Every user gets the same fixed sequence.
  - v1 (proposed): unrecorded pairs leave the action set. Only 2,319 of 3,327 catalog videos are recorded for every search user, so a literal single sequence would have to drop 1,008 videos. Instead, one common value vector (the mean over recorded cells) is applied greedily within each user's action set. Values stay non-personalized; only availability differs.
- **Quit-free versions:** the sum of the 30 best values. For the personalized oracle this is a true upper bound on any policy's cumulative reward.
- **Personalization gap:** per-user P(u) − NP(u) cumulative reward, compared by paired bootstrap (ADR-0018).

## Evidence
Greedy is not proven optimal under the category constraint, but the slack is small. The personalized oracle reaches 200.73 of a 204.39 bound under raw and 28.548 of 28.642 under duration_pct (`scripts/audit_rewards.py`).

## Amendment (evaluator v1 approval, 2026-10-06)
The diagnostics module gains the category-affinity oracle (ADR-0028), with the same import-time and call-time guards and the same AST ban (`tests/test_policy_isolation.py`).
