# ADR-0037: Phase 1 evaluation, comparisons and the gap captured

Status: accepted (Phase 1)

- **Score:** cumulative reward per session. Per user it is the mean over seeds for stochastic rows; for deterministic rows it is the seed-0 session.
- **Duration profile:** each video's percentile of canonical duration within the catalog (mid-rank, so the catalog's own mean is 0.500), averaged over every recommendation of the reported seeds. The median recommended duration is shown next to the catalog's median. A recording wrapper captures recommendations without changing behaviour.
- **Stronger duration probe:** whichever of the four permanent probes has the higher search cumulative reward, fixed on search and reused on shadow.
- **Comparisons:** `harness.compare` (paired bootstrap over users, 10,000 resamples, seed 0) for each scorer against rotation-random, the stronger probe, the non-personalized oracle and, for (b)-(d), against scorer (a).
- **Gap captured** = mean_u[s(u) - NP(u)] / mean_u[P(u) - NP(u)]. Its interval comes from a paired **ratio** bootstrap that resamples users jointly for numerator and denominator (`phase1_common.ratio_bootstrap`). The same estimator gives scorer (a)'s share of the category-affinity oracle's lift over rotation-random, against both the window-rotation and the free oracle. The free oracle is the matching ceiling, because the wrapper may repeat a category within the rule as it does.
- **Best scorer** = the highest search cumulative reward.
