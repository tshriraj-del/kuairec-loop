# ADR-0034: Scorer (d), the gradient-boosted ranker and its three variants

Status: accepted (Phase 1)

## Model
LightGBM 4.7.0 LambdaRank. Relevance grades are min(4, floor(5 x label)) with LightGBM's default gains, grouped by user. Early stopping (50 rounds) uses LightGBM's NDCG@10 on the validation groups. The validation figure reported is recomputed with the base scorers' NDCG@10 (gains = label). The declared grid is num_leaves in {31, 63}, learning rate 0.05, min_data_in_leaf 200, feature_fraction 0.8, seed 0, deterministic.

## Rows (stacking without leakage)
- **Training rows:** catalog pairs first seen in T_rank, for users with at least 10 of them.
- **Validation rows:** the base validation set.
- **Features of every row** come from (a)-(c) refitted on T_fit at their selected settings, the user's T_fit **non-catalog** history, and video statistics dated before T_fit's end. No row's features see its own label.

## Features
- **User:** `user_features.csv`, with categorical fields as codes.
- **Video:** per-day means and per-show rates from `item_daily_features`, codes, size, age, category levels, and the video's mean label and pair-count rank.
- **Category affinity:** mean, max, support share and category prior.
- **Base scores:** (b) and (c) as **percentile ranks within the user's catalog**, plus the user's history mean label.
- **Scale-free on purpose:** serving uses fits on all pairs and longer histories than training, so raw score magnitudes and counts shift between the two.

## Variants (nested, so each step's contribution is visible)
1. **no_duration** excludes video duration and every field the KuaiRec documentation defines through it: total play time, complete / valid / long-time / short-time plays (all thresholded on `video_duration`) and play progress. `comment_stay_duration` (time in the comments) stays. `tests/test_phase1.py` checks the column list.
2. **duration** adds log canonical duration, its bucket and the per-play rates of those fields.
3. **text** adds caption features. The text is caption, cover text and topic tags (category names left out, since they duplicate the category features), from `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 384 dims). 9,738 of 10,728 videos have text. The features are 16 PCA components fitted on all videos, a has_text flag, and two cosines: the video against the user's label-weighted taste vector and against the user's mean history vector.

## Serving
The boosters trained here, fed features from (a)-(c) refitted on all pairs (`scripts/phase1_fit.py`).

## torch is never in a training or serving process
Importing torch before LightGBM segfaults on this machine (exit 139; importing LightGBM first works), most likely two OpenMP runtimes. Captions are embedded once by `scripts/phase1_embed.py` in its own process and cached; nothing else imports torch.
