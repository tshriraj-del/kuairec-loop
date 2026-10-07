"""Choose K, the number of duration buckets for the duration-adjusted reward, from the
big matrix only (ADR-0016). No small-matrix data is read.

Rule (fixed before any small-matrix evaluation of the candidate):
  K = smallest value in GRID whose per-video mean duration percentile, computed on
  big-matrix rows, has |Spearman rho| <= 0.05 with canonical duration over the 3,327
  catalog videos. If no K qualifies, take the K with the smallest |rho|.

Usage: .venv/bin/python scripts/audit_bucket_choice.py [--json runs/audit/bucket_choice.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from kuairec_eval import labels  # noqa: E402

GRID = (5, 10, 20, 50, 100)
THRESHOLD = 0.05


def catalog_videos() -> np.ndarray:
    # The catalog is the set of small-matrix video ids (what env.catalog exposes): a video
    # attribute, not an outcome. Only the video_id column is read.
    from kuairec_eval.data import data_dir
    ids = pd.read_csv(data_dir() / "small_matrix.csv", usecols=["video_id"])["video_id"].unique()
    return np.sort(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    dur = labels.canonical_durations()
    cat = catalog_videos()
    rows = []
    for k in GRID:
        b = labels.duration_buckets(k)
        stats_k = labels.big_matrix_video_stats("duration_pct", n_buckets=k)
        rho_cat = stats.spearmanr(dur.reindex(cat), stats_k["mean"].reindex(cat)).statistic
        rho_all = stats.spearmanr(dur.reindex(stats_k.index), stats_k["mean"]).statistic
        sizes = np.bincount(b.video_bucket, minlength=b.n_buckets)
        rows.append({"K_requested": k, "K_effective": b.n_buckets, "rho_catalog": float(rho_cat),
                     "rho_all_big_videos": float(rho_all), "min_videos_per_bucket": int(sizes.min()),
                     "min_rows_per_bucket": int(min(len(r) for r in b.reference))})
        print(f"K={k:>3} (effective {b.n_buckets:>3})  rho(catalog, big rows) = {rho_cat:+.4f}   "
              f"rho(all big videos) = {rho_all:+.4f}   min videos/bucket {sizes.min():>4}   "
              f"min rows/bucket {min(len(r) for r in b.reference):>9,}")
    ok = [r for r in rows if abs(r["rho_catalog"]) <= THRESHOLD]
    chosen = ok[0] if ok else min(rows, key=lambda r: abs(r["rho_catalog"]))
    print(f"\nchosen K = {chosen['K_requested']} ({'meets' if ok else 'no K meets'} |rho| <= {THRESHOLD})")
    if args.json:
        args.json.write_text(json.dumps({"grid": rows, "threshold": THRESHOLD, "chosen": chosen}, indent=2))


if __name__ == "__main__":
    main()
