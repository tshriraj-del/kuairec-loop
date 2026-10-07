# ADR-0028: Category-affinity oracle and the diversity-level check (claim 2)

Status: accepted (evaluator v1)

## Oracle (your definition; operational details are mine)
`kuairec_eval/diagnostics/oracles.py:CategoryAffinityOracle` knows each user's mean reward per category and, at each step, shows a uniformly random available video of the chosen category. It is stochastic and runs on seeds 0-9.
- **Affinity:** a(u, c) is the mean reward over u's recorded **single-tag** videos of category c. That is the expected reward of the oracle's only action inside c (a random pick). 29 categories have single-tag videos (ADR-0014).
- **Choice:** the highest-affinity *allowed* category, ties broken by lower category id. Myopic, with static affinities, like the other greedy oracles.
- **Allowed categories, by mode.** Single-tag videos make the window counts exact.
  - `free`: the category occurs at most n_Q times in the last N recommendations. This is as freely as the quit setting allows, and it never triggers the rule.
  - `strict`: the category is absent from the last N recommendations. This is strict rotation: the freedom of n_Q = 0, safe under every profile.
  - `full_cycle`: the category has not yet been shown in the current cycle through all of the user's rotation categories, as in policies (b)/(c), ordered by affinity. This is the **alternative reading** of "strict rotation" and is reported as an extra.

## Choices you did not specify
- **"Strict rotation" means `strict`.** Both `free` and `strict` can then concentrate on a user's favourite categories, so their difference isolates the value of tolerating repeats, which is what "learning the diversity level" means. `full_cycle` mixes that with the value of concentration.
- **The gap is measured per profile:** personalized minus non-personalized oracle under the same profile.
- **Share of gap:** the point estimate of the difference divided by the gap's point estimate, with the interval scaled the same way. The 10% rule is applied to the point estimate. The interval is reported, and here it does not straddle 10%.

## Results (`.venv/bin/python scripts/audit_diversity_level.py`, duration_pct, 847 search users)

| Profile | strict | free | free − strict | share of gap | gap |
|---|---|---|---|---|---|
| looser (9, 2) | 17.472 | 18.197 | 0.725 [0.689, 0.763] | **6.4%** [6.0%, 6.7%] | 11.414 |
| primary (9, 1) | 17.472 | 18.003 | 0.531 [0.498, 0.565] | 4.7% [4.4%, 5.0%] | 11.346 |
| tighter (9, 0) | 17.472 | 17.472 | 0.000 (identical by construction) | 0.0% | 11.059 |

## Recorded outcome (your rule, applied to looser with the strict reading)
The looser-setting difference is 6.4% of the personalization gap, below 10%. **Strict rotation is close to optimal everywhere. Claim 2 returns to "a greedy personalized policy narrows and the diversity layer fixes it". The three settings are robustness checks only.**

## Flag: the reading of "strict rotation" decides the outcome
With `full_cycle` as the restricted run, free − full_cycle under looser is 2.190 [2.128, 2.252], or **19.2%** of the gap. That clears 10% and would record claim 2 as standing. The decomposition shows why the readings disagree:

| Profile | concentration on favourite categories: strict − full_cycle | tolerating repeats: free − strict |
|---|---|---|
| looser | 1.464 [1.416, 1.515] (12.8%) | 0.725 [0.689, 0.763] (6.4%) |
| primary | 1.464 [1.416, 1.515] (12.9%) | 0.531 [0.498, 0.565] (4.7%) |

Most of the free oracle's edge over a full-cycle rotation comes from concentrating on favourite categories, which a rotation that never repeats within the window already does. Tolerating repeats adds 4.7-6.4%. If you meant the full cycle, only the recorded text changes (this ADR and `docs/AUDIT.md`). The frozen code and the outputs report both readings.

## Side observation
The strict affinity oracle (17.472) beats the non-personalized video-level oracle (17.202 under primary). Knowing a user's categories, while picking videos at random inside them, is worth more than the best fixed choice of individual videos. The full-cycle variant (16.008) sits barely above rotation-random (15.880), because it visits every category whatever the affinities.
