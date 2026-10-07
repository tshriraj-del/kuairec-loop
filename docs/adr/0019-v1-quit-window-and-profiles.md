# ADR-0019: Quit window and quit profiles (evaluator v1)

Status: accepted (evaluator v1)

## Decision
- The intended window is primary. `window_mode = "easyrl4rec_slice"` stays in `EnvConfig` for `scripts/differential.py`, but `harness.run` refuses it.
- `[env] quit_profile` selects from `[quit_profiles]`. **primary** (N=9, n_Q=1) is the default. **looser** (9, 2) and **tighter** (9, 0) are sensitivity settings. max_turn stays at 30. Policies receive none of these values.
- **How looser and tighter were chosen** (`scripts/audit_survival.py`, 100,000 Monte Carlo random sequences from the catalog, validated against `env.boredom_quit` on 2,000 sequences and against the harness's random runs: mean 8.264 vs 8.281):
  - The rule I fixed beforehand varied N with n_Q = 1. It found a looser N = 3 (median 15, mean 16.71). It **failed for tighter**: every N ≥ 6 has median 8, and the tie-break picked N = 15, whose survival curve equals the primary's through t = 11 (mean 8.05). Random sessions usually end before the window fills, so a longer window changes little.
  - So I switched to the threshold axis at the primary window. It moves the curve both ways with one knob:

| N=9 | median length | mean length | P(reach 30) |
|---|---|---|---|
| n_Q = 0 (tighter) | 4 (0.50×) | 4.16 (0.50×) | 0.0000 |
| n_Q = 1 (primary) | 8 | 8.26 | 0.0004 |
| n_Q = 2 (looser) | 13 (1.62×) | 15.01 (1.82×) | 0.0929 |

The N-axis looser setting (3, 1) is the documented alternative (OPEN_QUESTIONS A2).

## Decision on A2 (evaluator v1 approval, 2026-10-06)
Approved: N = 9 with thresholds 2 (looser), 1 (primary) and 0 (tighter). The N-axis alternative (3, 1) is not used. The role of the three profiles follows the claim-2 outcome in ADR-0028. Claim 2 stands against the full-cycle baseline (C1), so the earlier wording, "robustness checks only", no longer applies: the profiles are the repeat tolerances the system is meant to learn from the session-ended signal.
