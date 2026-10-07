"""Item 4 and the correlation column of item 6: rank correlation between video duration and
reward, computed three ways, for every reward candidate. Search users only (ADR-0017).

  duration       canonical duration d(v) = median video_duration over v's big-matrix rows
  per video      Spearman over the 3,327 catalog videos between d(v) and the video's mean
                 (and median) reward across search-user interactions
  per interaction Spearman over every search-user small-matrix row between d(v) of the
                 row's video and the row's reward
  across buckets 20 equal-frequency big-matrix duration buckets (labels.duration_buckets(20));
                 Spearman and Pearson over the 20 buckets between the bucket's median
                 catalog-video duration and the mean reward of its search-user interactions

Also replicates the Phase 0 figure exactly as scripts/popularity_diagnostic.py computed it.

Usage: .venv/bin/python scripts/audit_duration_correlation.py [--json runs/audit/duration_correlation.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from kuairec_eval import labels, partitions  # noqa: E402
from kuairec_eval.data import load_small_matrix  # noqa: E402
from kuairec_eval.rewards import REWARDS, RewardConfig, row_rewards  # noqa: E402

BUCKETS_FOR_BUCKET_VERSION = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    search = set(partitions.load("search"))
    small = load_small_matrix()
    s = small[small.user_id.isin(search)].reset_index(drop=True)
    dur = labels.canonical_durations()
    out = {"n_search_rows": int(len(s))}
    print(f"search users: {len(search)}; search-user small-matrix rows: {len(s):,}")

    # Phase 0 replication (popularity_diagnostic.py): first-row duration among search rows,
    # per-video median watch_ratio, Spearman over videos.
    first = s.groupby("video_id").video_duration.first()
    med = s.groupby("video_id").watch_ratio.median()
    rho_p0 = stats.spearmanr(first.reindex(med.index), med).statistic
    print(f"Phase 0 replication: Spearman(first-row duration, per-video median watch_ratio), "
          f"n={len(med)} videos -> {rho_p0:.4f}")
    out["phase0_replication"] = float(rho_p0)

    d_row = dur.reindex(s.video_id).to_numpy()
    b20 = labels.duration_buckets(BUCKETS_FOR_BUCKET_VERSION)
    row_bucket = b20.bucket_of(s.video_id.to_numpy())
    cat = np.sort(s.video_id.unique())
    cat_bucket = b20.bucket_of(cat)
    bucket_dur = pd.Series(dur.reindex(cat).to_numpy()).groupby(cat_bucket).median()

    series = {}
    for name in REWARDS:
        cfg = RewardConfig.from_toml(active=name)
        series[name] = row_rewards(cfg, s.video_id.to_numpy(), s.watch_ratio.to_numpy(), s.play_duration.to_numpy())
    series["play_duration_ms (extra, not a candidate)"] = s.play_duration.to_numpy().astype(float)

    print(f"\n{'reward':42s} {'per-video mean':>15s} {'per-video median':>17s} {'per-interaction':>16s}"
          f" {'buckets Spearman':>17s} {'buckets Pearson':>16s}")
    out["by_reward"] = {}
    for name, y in series.items():
        df = pd.DataFrame({"v": s.video_id.to_numpy(), "y": y})
        g = df.groupby("v").y
        mean_v, med_v = g.mean(), g.median()
        r_mean = stats.spearmanr(dur.reindex(mean_v.index), mean_v).statistic
        r_med = stats.spearmanr(dur.reindex(med_v.index), med_v).statistic
        r_int = stats.spearmanr(d_row, y).statistic
        bmean = pd.Series(y).groupby(row_bucket).mean()
        r_bs = stats.spearmanr(bucket_dur.reindex(bmean.index), bmean).statistic
        r_bp = stats.pearsonr(bucket_dur.reindex(bmean.index), bmean).statistic
        print(f"{name:42s} {r_mean:>15.4f} {r_med:>17.4f} {r_int:>16.4f} {r_bs:>17.4f} {r_bp:>16.4f}")
        out["by_reward"][name] = {"per_video_mean": float(r_mean), "per_video_median": float(r_med),
                                  "per_interaction": float(r_int), "buckets_spearman": float(r_bs),
                                  "buckets_pearson": float(r_bp), "n_videos": int(len(mean_v)),
                                  "n_interactions": int(len(y)), "n_buckets": int(len(bmean))}
    # Diagnostic only (K stays as config/rewards.toml says; the rule in audit_bucket_choice.py
    # fixed it on big-matrix data): does the duration_pct correlation on search users depend on K?
    print("\nduration_pct sensitivity to K (diagnostic; not used to choose K):")
    print(f"  {'K':>4s} {'per-video mean (search, small rows)':>36s} {'per-interaction':>16s} {'per-video mean (big rows)':>26s}")
    out["duration_pct_by_K"] = {}
    for k in (5, 10, 20, 50, 100):
        y = labels.duration_buckets(k).percentile(s.video_id.to_numpy(), s.play_duration.to_numpy())
        mean_v = pd.Series(y).groupby(s.video_id.to_numpy()).mean()
        r_v = stats.spearmanr(dur.reindex(mean_v.index), mean_v).statistic
        r_i = stats.spearmanr(d_row, y).statistic
        big_v = labels.big_matrix_video_stats("duration_pct", n_buckets=k)["mean"].reindex(mean_v.index)
        r_b = stats.spearmanr(dur.reindex(mean_v.index), big_v).statistic
        print(f"  {k:>4d} {r_v:>36.4f} {r_i:>16.4f} {r_b:>26.4f}")
        out["duration_pct_by_K"][k] = {"per_video_mean_search": float(r_v), "per_interaction_search": float(r_i),
                                       "per_video_mean_big": float(r_b)}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
