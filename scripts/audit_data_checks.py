"""Audit data checks that the reference policies and reward candidates depend on.

Big-matrix facts use the whole big matrix (public data). Facts about small-matrix
outcomes use search-partition users only; the holdout is never read.

Usage: .venv/bin/python scripts/audit_data_checks.py [--json runs/audit/data_checks.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from kuairec_eval import partitions  # noqa: E402
from kuairec_eval.data import load_big_matrix, load_item_categories, load_small_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    out = {}

    big = load_big_matrix()
    small = load_small_matrix()
    catalog = np.sort(small["video_id"].unique())
    search = set(partitions.load("search"))
    small_s = small[small.user_id.isin(search)]

    print("== video_duration constancy")
    for name, df in (("big", big), ("small (all rows; duration is a video attribute)", small)):
        n_dur = df.groupby("video_id").video_duration.nunique()
        print(f"  {name:48s} videos with >1 distinct duration: {int((n_dur > 1).sum()):>6,} of {len(n_dur):,}")
        out[f"duration_nunique_gt1_{name.split()[0]}"] = int((n_dur > 1).sum())
    big_med = big.groupby("video_id").video_duration.median()
    small_med = small.groupby("video_id").video_duration.median()
    diff = (small_med - big_med.reindex(small_med.index)).abs()
    print(f"  catalog videos whose small median duration != big median duration: {int((diff > 0).sum()):,} of {len(diff):,}"
          f" (max abs diff {diff.max():.0f} ms)")
    out["catalog_small_vs_big_duration_mismatch"] = int((diff > 0).sum())
    cd = big_med.reindex(catalog)
    print(f"  catalog canonical (big median) duration ms: min {cd.min():.0f}  p10 {cd.quantile(.1):.0f}"
          f"  median {cd.median():.0f}  p90 {cd.quantile(.9):.0f}  max {cd.max():.0f}")

    print("\n== timestamps (ADR-0024)")
    for name, df in (("big", big), ("small (all users)", small)):
        print(f"  {name:18s} rows without timestamp: {int(df.timestamp.isna().sum()):>9,} of {len(df):,}")
        out[f"rows_without_timestamp_{name.split()[0]}"] = int(df.timestamp.isna().sum())
    print(f"  small (all users)  duplicate (user, video) rows: {int(small.duplicated(['user_id', 'video_id']).sum()):,}")

    print("\n== play_duration vs watch_ratio * video_duration")
    for name, df in (("big", big), ("small/search", small_s)):
        rel = (df.play_duration - df.watch_ratio * df.video_duration).abs() / df.video_duration
        print(f"  {name:14s} max |play - wr*dur| / dur = {rel.max():.2e}; rows with play_duration == 0: {int((df.play_duration == 0).sum()):,}")
        out[f"play_vs_ratio_maxrel_{name.split('/')[0]}"] = float(rel.max())

    print("\n== big-matrix views per catalog video (for the policy (d) floor)")
    views = big[big.video_id.isin(catalog)].groupby("video_id").size().reindex(catalog, fill_value=0)
    users = big[big.video_id.isin(catalog)].groupby("video_id").user_id.nunique().reindex(catalog, fill_value=0)
    qs = [0, .01, .05, .1, .25, .5]
    print("  rows    quantiles " + "  ".join(f"q{int(q*100):02d}={views.quantile(q):.0f}" for q in qs))
    print("  users   quantiles " + "  ".join(f"q{int(q*100):02d}={users.quantile(q):.0f}" for q in qs))
    for floor in (10, 50, 100, 200, 500, 1000):
        print(f"  catalog videos with < {floor:>4} big rows: {int((views < floor).sum()):>5,}")
        out[f"catalog_videos_lt_{floor}_big_rows"] = int((views < floor).sum())

    print("\n== categories and single-tag videos in the catalog")
    cats = load_item_categories()
    single = {}
    for v in catalog:
        if len(cats[v]) == 1:
            single.setdefault(cats[v][0], []).append(int(v))
    all_cats = sorted({c for v in catalog for c in cats[v]})
    print(f"  categories present in catalog: {len(all_cats)}  (ids {all_cats[0]}..{all_cats[-1]})")
    n_tags = pd.Series([len(cats[v]) for v in catalog]).value_counts().sort_index()
    print(f"  catalog videos by number of tags: {n_tags.to_dict()}")
    out["catalog_videos_by_n_tags"] = {int(k): int(v) for k, v in n_tags.items()}
    print(f"  categories with >=1 single-tag catalog video: {len(single)}")
    sizes = sorted((len(vs), c) for c, vs in single.items())
    print(f"  single-tag videos per category, smallest five: {sizes[:5]}; largest: {sizes[-1]}")
    missing = [c for c in all_cats if c not in single]
    print(f"  categories with no single-tag catalog video: {missing}")
    out["categories_in_catalog"] = len(all_cats)
    out["categories_with_single_tag_video"] = len(single)
    out["categories_without_single_tag_video"] = missing
    dup_tags = sum(1 for v in catalog if len(set(cats[v])) != len(cats[v]))
    print(f"  catalog videos with a repeated tag inside their own tuple: {dup_tags}")

    print("\n== per-user unrecorded cells (small matrix, search users)")
    n_v = len(catalog)
    per_user = small_s.groupby("user_id").size()
    miss = n_v - per_user
    print(f"  search users: {len(per_user)}; unrecorded cells per user: mean {miss.mean():.2f}, median {miss.median():.0f},"
          f" max {miss.max()}, users with 0: {int((miss == 0).sum())}")
    complete = small_s.groupby("video_id").user_id.nunique()
    n_complete = int((complete.reindex(catalog, fill_value=0) == len(per_user)).sum())
    print(f"  catalog videos recorded for every search user: {n_complete:,} of {n_v:,};"
          f" videos unrecorded for at least one search user: {n_v - n_complete:,}")
    out["catalog_videos_recorded_for_all_search_users"] = n_complete
    out["search_unrecorded_per_user_mean"] = float(miss.mean())
    out["search_unrecorded_per_user_max"] = int(miss.max())

    print("\n== does the unrecorded-pair mask carry reward information? (search users; ADR-0020)")
    from scipy import stats
    from kuairec_eval import labels as L
    from kuairec_eval.env import EnvConfig
    k = EnvConfig.from_toml().duration_buckets
    pct = L.duration_buckets(k).percentile(small_s.video_id.to_numpy(), small_s.play_duration.to_numpy())
    rec = small_s.assign(pct=pct)
    v_missing = (len(per_user) - rec.groupby("video_id").user_id.nunique()).reindex(catalog, fill_value=len(per_user))
    v_raw = rec.groupby("video_id").watch_ratio.mean().reindex(catalog)
    v_pct = rec.groupby("video_id").pct.mean().reindex(catalog)
    u_raw = rec.groupby("user_id").watch_ratio.mean().reindex(per_user.index)
    u_pct = rec.groupby("user_id").pct.mean().reindex(per_user.index)
    for label, a, b in (("per video: #unrecorded vs mean watch_ratio", v_missing, v_raw),
                        ("per video: #unrecorded vs mean duration_pct", v_missing, v_pct),
                        ("per user:  #unrecorded vs mean watch_ratio", miss, u_raw),
                        ("per user:  #unrecorded vs mean duration_pct", miss, u_pct)):
        r = stats.spearmanr(a, b)
        print(f"  Spearman {label:44s} {r.statistic:+.4f} (p={r.pvalue:.2g})")
        out[f"mask_{label.split(':')[0].strip()}_{label.split('mean ')[1]}"] = float(r.statistic)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
