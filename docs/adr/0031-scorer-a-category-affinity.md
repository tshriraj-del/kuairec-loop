# ADR-0031: Scorer (a), user-category affinity

Status: accepted (Phase 1)

- aff(u, c) = (sum of u's labels in category c + λ m_c) / (n_uc + λ). Here m_c is the mean label of category c over the fitting pairs and n_uc is the number of u's pairs that carry c. A multi-tag pair counts toward each of its tags.
- A video's score is the **mean** affinity over its categories. Ties are broken by lower video id in the slate.
- History is the user's non-catalog pairs (ADR-0030). A user with no history gets the category means.
- λ is tuned over {1, 5, 20, 50, 200} on the validation split; **λ = 20** was selected (NDCG@10 0.5429 against the 0.5418 floor).
- **Consequence:** scores are category-level, so many videos tie. In the serving smoke test one user's 3,327 catalog scores took 246 distinct values. Within a category the slate falls back to video-id order, so (a) measures category choice alone, like the category-affinity oracle, which also picks randomly within a category.
