"""Phase 1, item 1: training data. Big-matrix pairs with the public duration-adjusted label.

Reports label coverage, dropped rows, the time windows, and whether the validation users'
NON-catalog histories resemble the histories the environment's users are served from. The
catalog is taken from the environment's PolicyContext, the same object policies receive.

Usage: .venv/bin/python scripts/phase1_data.py
Writes data/cache/phase1/pairs.npz and runs/phase1/data_report.json
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from kuairec_eval import harness  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from policies.phase1 import pairs as P  # noqa: E402
from policies.phase1.config import load  # noqa: E402

OUT = ROOT / "runs" / "phase1"


def day(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def quartiles(x) -> dict:
    x = np.asarray(x)
    return {"n": int(len(x)), "zero": int((x == 0).sum()), "p25": float(np.percentile(x, 25)) if len(x) else None,
            "median": float(np.median(x)) if len(x) else None, "p75": float(np.percentile(x, 75)) if len(x) else None}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = np.asarray(harness.context_for(KuaiEnv()).catalog)
    pairs, report = P.build_pairs()
    P.save(pairs)
    cfg = load()
    cut = P.time_cutoffs(pairs)
    in_cat = np.isin(pairs.video, catalog)
    t_fit = pairs.ts < cut["fit_end"]
    t_val = pairs.ts >= cut["rank_end"]
    t_rank = ~t_fit & ~t_val
    report["time"] = {"first_view_min": day(pairs.ts.min()), "fit_end": day(cut["fit_end"]),
                      "rank_end": day(cut["rank_end"]), "first_view_max": day(pairs.ts.max())}
    report["windows"] = {name: {"pairs": int(m.sum()), "catalog_pairs": int((m & in_cat).sum())}
                         for name, m in (("T_fit", t_fit), ("T_rank", t_rank), ("T_val", t_val))}
    report["label"] = {"mean_all_pairs": float(pairs.label.mean()), "mean_catalog_pairs": float(pairs.label[in_cat].mean())}

    users_with_catalog = np.unique(pairs.user[in_cat])
    cold_users = np.setdiff1d(np.unique(pairs.user), users_with_catalog)
    val_counts = np.bincount(pairs.user[t_val & in_cat], minlength=pairs.user.max() + 1)
    val_users = np.flatnonzero(val_counts >= cfg["split"]["min_val_pairs"])
    hist_before_val = np.bincount(pairs.user[(~t_val) & ~in_cat], minlength=pairs.user.max() + 1)
    hist_all = np.bincount(pairs.user[~in_cat], minlength=pairs.user.max() + 1)
    report["users"] = {
        "all": int(len(np.unique(pairs.user))),
        "with_any_catalog_pair": int(len(users_with_catalog)),
        "catalog_cold (no catalog pair at all)": int(len(cold_users)),
        "validation_users (>= min_val_pairs catalog pairs in T_val)": int(len(val_users)),
        "validation_targets (their T_val catalog pairs)": int(val_counts[val_users].sum()),
    }
    report["history_sizes_non_catalog"] = {
        "validation_users_before_T_val": quartiles(hist_before_val[val_users]),
        "catalog_cold_users_all_time": quartiles(hist_all[cold_users]),
    }
    (OUT / "data_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
