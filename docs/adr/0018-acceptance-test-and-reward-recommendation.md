# ADR-0018: Acceptance test procedure and reward recommendation

Status: accepted (evaluator v1): recommendation adopted; **T1 superseded by ADR-0027** (kept in reports as a secondary line)

## Procedure (what the brief left open)
- **Score:** cumulative reward per session. Per user, it is averaged over seeds for stochastic policies and is the single session for deterministic ones.
- **Paired bootstrap:** per-user differences on the same 847 search users, users resampled with replacement, 10,000 resamples (numpy `default_rng(0)`), 95% percentile interval.
- **T1, shortest-first must not score above random:** passes unless the interval for mean(a − random) lies entirely above 0.
- **T2, personalized oracle beats non-personalized:** passes if the interval for mean(P − NP) lies entirely above 0.
- **Also reported:** the per-recommendation reward difference, (c) − (b) (same session dynamics, so it isolates duration), and the longest-video probes.

## Results
v0 audit: `scripts/audit_rewards.py`. v1 preview: `runs/audit/v1_preview/`.

| Reward | T1 (a) − random, v0 | T2 P − NP, v0 | T1, v1 preview | T2, v1 preview |
|---|---|---|---|---|
| raw | +12.85 [12.43, 13.29] FAIL | +133.15 [122.87, 143.69] pass | +12.81 FAIL | +132.84 pass |
| capped5 | +12.02 [11.78, 12.27] FAIL | +46.75 [45.00, 48.48] pass | +11.99 FAIL | +46.44 pass |
| duration_pct | −0.197 [−0.260, −0.132] pass | +11.37 [11.17, 11.56] pass | −0.218 [−0.281, −0.152] pass | +11.35 [11.15, 11.55] pass |

## Recommendation: duration_pct
It is the only candidate that passes both tests, under v0 and under the v1 preview. Evidence that it removes the short-video advantage and doesn't just hide it:
- Per recommendation, (a) − random = −0.0025 [−0.0091, 0.0039].
- (c) − (b) = −0.030 [−0.117, 0.058] on cumulative reward.
- Under raw, (c) − (b) = +25.80 [25.04, 26.64].

## Caveats
- T1 passes under duration_pct partly because (a) runs shorter sessions (8.00 vs 8.28 steps); per recommendation the two are tied.
- A small long-video lean remains. Rotation-longest minus rotation-random is +0.187 [0.106, 0.268] on cumulative reward (+1.2%) and +0.0072 [0.0045, 0.0099] per recommendation. That is 1.6% of the 11.37 personalization gap. OPEN_QUESTIONS A1.
