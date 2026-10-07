# Open questions

## D. For Phase 2

- **D1. What does a policy tuned to the primary setting lose under the tighter setting?** **Measured with the oracle (2026-10-06).** The free category-affinity oracle built for primary, run under tighter, loses 15.477 [15.327, 15.627] per session against the same oracle built for tighter. That is 140.0% of the tighter personalization gap. It ends every session by boredom, after 2.89 steps on average, and scores 14.013 below the full-cycle baseline (ADR-0028; `docs/AUDIT.md` section 3.1; `scripts/audit_diversity_level.py`). Phase 2 should repeat the measurement with learned policies.

## C. After the evaluator-v1 freeze: resolved

- C1, the reading of "strict rotation": **the full-cycle policy is the baseline** (your decision, made after the check), and both readings are recorded. Claim 2 stands, reworded: "from the session-ended signal alone, the system learns how soon a category can be revisited and how many repeats are tolerated, measured against the full-cycle policy". Split of the looser setting's 19.2%: 12.8% from revisiting after the window and 6.4% from tolerating repeats; tolerating repeats adds 4.7% under the primary setting (ADR-0028).

## A. From the evaluator audit: resolved by the v1 approval

- A1, the long-video lean: **accepted.** The duration probes are permanent rows (ADR-0016, ADR-0029).
- A2, the quit profiles: **N = 9 with thresholds 2, 1 and 0** (ADR-0019).
- A3, the T1 statement: **replaced** by the two-sided, session-length-controlled probe test with a 5% tolerance set after seeing the audit numbers. The old T1 stays as a secondary line (ADR-0027).
- A4 (the mask signal), A5 (a secondary metric), A6 (the non-personalized oracle definition) and A7 (deterministic re-runs): **approved as proposed** (ADR-0020, ADR-0016, ADR-0015, ADR-0021). No secondary metric was added.
- A8, commit order: **done.** The audit is commit `c8a9a0d`; v1 is the commit tagged `evaluator-v1`.

## B. Phase 0 questions

All resolved by the audit brief. See ADR-0019 to ADR-0024.
