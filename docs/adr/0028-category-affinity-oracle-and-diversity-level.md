# ADR-0028: Category-affinity oracle and the diversity-level check (claim 2)

Status: accepted (evaluator v1). The outcome was set by your C1 decision on 2026-10-06, made after the check.

## Oracle (your definition; operational details are mine)
`kuairec_eval/diagnostics/oracles.py:CategoryAffinityOracle` knows each user's mean reward per category and, at each step, shows a uniformly random available video of the chosen category. It is stochastic and runs on seeds 0-9.
- **Affinity:** a(u, c) is the mean reward over u's recorded **single-tag** videos of category c. That is the expected reward of the oracle's only action inside c (a random pick). 29 categories have single-tag videos (ADR-0014).
- **Choice:** the highest-affinity *allowed* category, ties broken by lower category id. Myopic, with static affinities, like the other greedy oracles.
- **Allowed categories, by mode.** Single-tag videos make the window counts exact.
  - `full_cycle`: the category has not yet been shown in the current cycle through all of the user's rotation categories, as in policies (b)/(c), ordered by affinity. **This is the baseline (C1).**
  - `strict`: the category is absent from the last N recommendations. A category may be revisited as soon as it leaves the window.
  - `free`: the category occurs at most n_Q times in the last N recommendations. This is as freely as the quit setting allows, and it never triggers the rule.

## Choices you did not specify (mine)
- Single-tag videos only, and affinity over that pickable set.
- The gap is measured per profile: personalized minus non-personalized oracle under the same profile.
- Share of gap: the point estimate of the difference divided by the gap's point estimate, with the interval scaled the same way. The 10% rule is applied to the point estimate.

## Which reading of "strict rotation" (your decision C1)
- **The baseline is `full_cycle`.** Your reason: the quit parameters are hidden from policies, so a policy that has learned nothing cannot run window-based rotation. Its only safe option is the full cycle through all 29 categories.
- **Timing:** the 10% rule was set before the check; the choice of reading was made after it. Before running the check I had proposed the window reading (`strict`) as the baseline. Under that reading the result is 6.4%, which would have reverted claim 2.
- **Both readings are recorded:**

| Profile | free − full_cycle (baseline reading) | free − strict (window reading) | gap |
|---|---|---|---|
| looser (9, 2) | 2.190 [2.128, 2.252], **19.2%** [18.6%, 19.7%] | 0.725 [0.689, 0.763], 6.4% [6.0%, 6.7%] | 11.414 [11.214, 11.611] |
| primary (9, 1) | 1.996 [1.938, 2.055], 17.6% [17.1%, 18.1%] | 0.531 [0.498, 0.565], 4.7% [4.4%, 5.0%] | 11.346 [11.146, 11.545] |
| tighter (9, 0) | 1.464 [1.416, 1.515], 13.2% [12.8%, 13.7%] | 0.000 (identical by construction) | 11.059 [10.862, 11.255] |

Command: `.venv/bin/python scripts/audit_diversity_level.py` (duration_pct, 847 search users). The cumulative reward per session is 16.008 for full_cycle and 17.472 for strict under every profile. Free scores 18.197 under looser, 18.003 under primary and 17.472 under tighter.

## Recorded outcome
**Claim 2 stands**, reworded: *"from the session-ended signal alone, the system learns how soon a category can be revisited and how many repeats are tolerated, measured against the full-cycle policy."* Under looser, the value of that knowledge is 19.2% of the personalization gap, above the 10% rule.

## The split, as it is
- **12.8% of the gap comes from revisiting after the window** (strict − full_cycle = 1.464 [1.416, 1.515] under looser; the same 1.464 is 12.9% of the primary gap).
- **Tolerating repeats adds 6.4% under the looser setting** (free − strict = 0.725) **and 4.7% under the primary setting** (0.531).

## What the check measures, and what it does not
It measures headroom: for an oracle that already knows each user's category affinities, the value of knowing how soon a category can be revisited and how many repeats are tolerated. It does not show that a policy learns these from the session-ended signal. That remains to be measured with learned policies (see also OPEN_QUESTIONS D1).

## Implementation status (text-only change, by instruction)
`scripts/audit_diversity_level.py` still computes its `decision` field under the window reading (`runs/audit_v1/diversity_level.json`, `runs/audit_v1/diversity_level.md`). `scripts/audit_report.py` prints that field as the report's outcome. This decision was recorded without code changes, so the claim-2 text in `docs/AUDIT.md` (sections 1, 3 and 9) was edited by hand. Running `scripts/audit_report.py` again would restore the window-reading wording until those two scripts adopt the full-cycle baseline.

## Side observation
The strict affinity oracle (17.472) beats the non-personalized video-level oracle (17.202 under primary). Knowing a user's categories, while picking videos at random inside them, is worth more than the best fixed choice of individual videos. The full-cycle oracle (16.008) sits barely above rotation-random (15.880), because it visits every category whatever the affinities.
