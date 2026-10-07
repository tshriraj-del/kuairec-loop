"""Why does big-matrix popularity score below random on watch_ratio? Search users only.

Usage: .venv/bin/python scripts/popularity_diagnostic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from kuairec_eval import partitions  # noqa: E402
from kuairec_eval.data import load_item_categories, load_small_matrix  # noqa: E402
from policies.baselines import GlobalPopularity, big_matrix_popularity  # noqa: E402

search = set(partitions.load("search"))
small = load_small_matrix()
small = small[small.user_id.isin(search)]
per_video = small.groupby("video_id").watch_ratio.agg(["mean", "median"])
pop = big_matrix_popularity().reindex(per_video.index, fill_value=0)
rho, p = stats.spearmanr(pop, per_video["median"])
print(f"videos: {len(per_video)}; Spearman(big distinct-user count, search-user median watch_ratio) = {rho:.4f} (p={p:.2g})")
rho_m, _ = stats.spearmanr(pop, per_video["mean"])
print(f"Spearman with mean watch_ratio = {rho_m:.4f}")
dur = small.groupby("video_id").video_duration.first()
rd, _ = stats.spearmanr(pop, dur.reindex(per_video.index))
rw, _ = stats.spearmanr(dur.reindex(per_video.index), per_video["median"])
print(f"Spearman(popularity, video_duration) = {rd:.4f}; Spearman(video_duration, median watch_ratio) = {rw:.4f}")
cats = load_item_categories()
ctx = type("C", (), {"catalog": tuple(sorted(per_video.index))})
ranking = GlobalPopularity(ctx, 0).ranking[:6]
print("top-6 popularity ranking (video, distinct users, categories, search mean wr):")
for v in ranking:
    print(f"  {v:>6} {int(pop[v]):>6} {cats[v]!s:>10} {per_video.loc[v, 'mean']:.3f}")
