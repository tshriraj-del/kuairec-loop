"""Items 1 and 2: matrix statistics vs published figures, and the small/big overlap.

Exits with status 2 if any statistic differs from the published figure.
Usage: .venv/bin/python scripts/data_report.py [--json out.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402

from kuairec_eval import data as D  # noqa: E402


def pair_key(df):
    return df["user_id"].astype(np.int64) * 100_000 + df["video_id"].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    tables = {
        "big_matrix.csv": D.load_big_matrix(),
        "small_matrix.csv": D.load_small_matrix(),
        "item_categories.csv": D.load_item_categories(),
        "item_daily_features.csv": D.load_item_daily_features(),
        "user_features.csv": D.load_user_features(),
        "kuairec_caption_category.csv": D.load_caption_category(),
    }
    print("== loaded")
    for name, t in tables.items():
        print(f"  {name:32s} rows={len(t):>10,}")
    small, big = tables["small_matrix.csv"], tables["big_matrix.csv"]

    report = {"stats": {}, "mismatches": [], "overlap": {}}
    print("\n== item 1: matrix statistics vs published")
    print(f"  {'matrix':6s} {'stat':14s} {'measured':>14s} {'published':>12s}  match")
    for label, df in (("small", small), ("big", big)):
        st = D.matrix_stats(df)
        report["stats"][label] = st
        pub = D.PUBLISHED[label]
        for k in ("users", "videos", "interactions"):
            ok = st[k] == pub[k]
            print(f"  {label:6s} {k:14s} {st[k]:>14,} {pub[k]:>12,}  {'yes' if ok else 'NO'}")
            if not ok:
                report["mismatches"].append(f"{label}.{k}: {st[k]} vs {pub[k]}")
        ok = round(st["density_rows"], 3) == pub["density"]
        print(f"  {label:6s} {'density(rows)':14s} {st['density_rows']:>14.4%} {pub['density']:>12.1%}  {'yes' if ok else 'NO'}")
        if not ok:
            report["mismatches"].append(f"{label}.density: {st['density_rows']} vs {pub['density']}")
        print(f"  {label:6s} {'unique pairs':14s} {st['unique_pairs']:>14,} {'-':>12s}  (density {st['density_unique_pairs']:.4%})")
        print(f"  {label:6s} {'wr NaN/inf':14s} {st['watch_ratio_nan']:>7,}/{st['watch_ratio_inf']:<6,}")

    n_cells = small["user_id"].nunique() * small["video_id"].nunique()
    report["stats"]["small"]["missing_cells"] = int(n_cells - len(small))
    report["stats"]["small"]["rows_without_timestamp"] = int(small["timestamp"].isna().sum())
    report["stats"]["small"]["watch_ratio_gt_5"] = int((small["watch_ratio"] > 5).sum())
    report["stats"]["small"]["watch_ratio_max"] = float(small["watch_ratio"].max())
    print(f"  small  missing cells (user x video not recorded): {n_cells - len(small):,}")
    print(f"  small  rows with no timestamp/date: {report['stats']['small']['rows_without_timestamp']:,}")
    q = small["watch_ratio"].quantile([0.5, 0.99, 0.999])
    report["stats"]["small"]["watch_ratio_p50_p99_p999"] = [float(x) for x in q]
    print(f"  small  watch_ratio > 5: {report['stats']['small']['watch_ratio_gt_5']:,}  max {small['watch_ratio'].max():.3f}"
          f"  p50 {q.iloc[0]:.3f}  p99 {q.iloc[1]:.3f}  p99.9 {q.iloc[2]:.3f}")

    print("\n== item 2: overlap")
    su, sv = set(small.user_id), set(small.video_id)
    bu, bv = set(big.user_id), set(big.video_id)
    sk, bk = pair_key(small), pair_key(big)
    sk_set = set(sk)
    ov = {
        "pairs_in_both_unique": len(sk_set & set(bk)),
        "big_rows_on_small_pairs": int(bk.isin(sk_set).sum()),
        "small_users_in_big": len(su & bu),
        "small_videos_in_big": len(sv & bv),
        "big_rows_in_small_user_x_small_video_block": int((big.user_id.isin(su) & big.video_id.isin(sv)).sum()),
        "big_rows_from_small_users": int(big.user_id.isin(su).sum()),
        "big_rows_on_small_videos": int(big.video_id.isin(sv).sum()),
        "big_users_with_rows_on_small_videos": int(big.loc[big.video_id.isin(sv), "user_id"].nunique()),
        "small_videos_with_big_rows": int(big.loc[big.video_id.isin(sv), "video_id"].nunique()),
    }
    report["overlap"] = ov
    for k, v in ov.items():
        print(f"  {k:46s} {v:>12,}")
    print(f"  (small users: {len(su):,}; small videos: {len(sv):,}; big users: {len(bu):,}; big videos: {len(bv):,})")

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    if report["mismatches"]:
        print("\nSTOP: measured statistics differ from published figures:", *report["mismatches"], sep="\n  ")
        sys.exit(2)
    print("\nall published figures matched")


if __name__ == "__main__":
    main()
