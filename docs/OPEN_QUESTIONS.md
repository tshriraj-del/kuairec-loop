# Open questions

## C. After the evaluator-v1 freeze (2026-10-06): open

- **C1. Which reading of "strict rotation" should the claim-2 outcome use?** The recorded outcome reads strict rotation as "no category repeated within the 9-item window". On that reading, tolerating repeats is worth 6.4% of the gap under looser (< 10%), so claim 2 returns to "a greedy personalized policy narrows and the diversity layer fixes it". If strict rotation means a full cycle through all categories, as in policies (b)/(c), the figure is 19.2% and claim 2 stands. Of that 19.2%, 12.8% comes from concentrating on favourite categories, which window-strict rotation also does (ADR-0028). Switching changes only the recorded text in ADR-0028 and AUDIT.md. No re-freeze is needed, because both readings are computed and reported.

## A. From the evaluator audit: resolved by the v1 approval

- A1, the long-video lean: **accepted.** The duration probes are permanent rows (ADR-0016, ADR-0029).
- A2, the quit profiles: **N = 9 with thresholds 2, 1 and 0** (ADR-0019).
- A3, the T1 statement: **replaced** by the two-sided, session-length-controlled probe test with a 5% tolerance set after seeing the audit numbers. The old T1 stays as a secondary line (ADR-0027).
- A4 (the mask signal), A5 (a secondary metric), A6 (the non-personalized oracle definition) and A7 (deterministic re-runs): **approved as proposed** (ADR-0020, ADR-0016, ADR-0015, ADR-0021). No secondary metric was added.
- A8, commit order: **done.** The audit is commit `c8a9a0d`; v1 is the commit tagged `evaluator-v1`.

## B. Phase 0 questions

All resolved by the audit brief. See ADR-0019 to ADR-0024.
