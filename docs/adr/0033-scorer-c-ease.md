# ADR-0033: Scorer (c), EASE, written by hand

Status: accepted (Phase 1)

- Steck (2019) closed form over all 10,728 items: P = (X'X + λI)^-1, B = I - P diag(1/diag(P)), diag(B) = 0. The inverse is computed in float64 with numpy, and only B's catalog columns are kept, in float32. A toy check against the dense closed form agrees to 9e-10.
- X is binary, or the centered label (y - 0.5) on observed pairs. A user's catalog scores are x_u B. For the served users x_u is their non-catalog history, so the B[non-catalog, catalog] block carries all the signal.
- **Grid:** value in {binary, centered}, λ in {100, 500, 2000, 10000}. **Selected: centered, λ = 10,000** (NDCG@10 0.5543). It sits on the grid's edge (OPEN_QUESTIONS P1).
