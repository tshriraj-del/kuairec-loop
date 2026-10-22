# ADR-0032: Scorer (b), item-kNN, written by hand

Status: accepted (Phase 1)

- **Similarity:** cosine on binary co-watching between each catalog item i and every item j, S_ij = co(i, j) / (sqrt(n_i n_j) + shrink), with shrink = 10. Self-similarity is set to 0, and 0/0 is guarded to 0.
- **Neighbourhood inside the user's history:** for catalog item i, the k history items most similar to i. A global top-k per item would fail for the served users. Their histories hold only non-catalog videos, while a catalog item's global nearest neighbours are mostly other catalog videos, so the intersection would usually be empty.
- **Value modes:**
  - binary: the sum of those similarities;
  - centered: sum s_ij (y_uj - 0.5) / (sum |s_ij| + 1e-6), the predicted relative engagement.
- **Grid:** value in {binary, centered}, k in {20, 100, 400}. **Selected: centered, k = 400** (NDCG@10 0.5529). It sits on the grid's edge (OPEN_QUESTIONS P1).
