# Open questions

## D. For Phase 2: open

- **D1. What does a policy tuned to the primary setting lose when it runs under the tighter setting?** Claim 2 says the system learns how soon a category can be revisited and how many repeats are tolerated (ADR-0028). This item measures the cost of having learned the wrong tolerance. It can be measured with the frozen code: build the `free` category-affinity oracle with the primary configuration and run it in a tighter-profile environment. Compare it with the same oracle built for tighter and with the full-cycle baseline.

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
