"""Phase 1, item 2(d) and the offline part of item 4: train and validate the three ranker variants.

Base scorers (a)-(c) use the configurations selected in runs/phase1/tuning_base.json, refitted on
T_fit. Training rows: catalog pairs first seen in T_rank, users with >= min_val_pairs of them.
Validation rows: the base validation set (catalog pairs in T_val, users with >= min_val_pairs).
Every row's features come from the T_fit fits and the user's T_fit NON-catalog history, and the
video statistics from item_daily_features dated before T_fit's end. Variants are nested column
subsets of one feature build (policies/phase1/features.py).

Usage: .venv/bin/python scripts/phase1_rank.py
Writes runs/phase1/tuning_ranker.json and data/cache/phase1/final/ranker_<variant>.pkl
"""
import lightgbm  # noqa: F401  (imported first: see policies/phase1/ranker.py)
import gc
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from kuairec_eval import harness  # noqa: E402
from kuairec_eval.data import load_item_categories  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from policies.phase1 import ranker as R  # noqa: E402
from policies.phase1 import scorers as S  # noqa: E402
from policies.phase1.config import CACHE, load  # noqa: E402
from policies.phase1.features import VARIANTS, RowBuilder, columns, prior_table  # noqa: E402
from policies.phase1.pairs import History, load_pairs, time_cutoffs  # noqa: E402

OUT = ROOT / "runs" / "phase1"
MAKE = {"affinity": S.Affinity, "item_knn": S.ItemKNN, "ease": S.EASE}


def build_rows(builder, users, history, targets):
    xs, ys, vs, gs = [], [], [], []
    for u in users:
        tv, tl = targets.get(u)
        hv, hl = history.get(u)
        xs.append(builder.user_rows(u, hv, hl, tv))
        ys.append(tl)
        vs.append(tv)
        gs.append(len(tv))
    return np.vstack(xs), np.concatenate(ys), np.concatenate(vs), np.asarray(gs)


def main():
    cfg = load()
    catalog = np.asarray(harness.context_for(KuaiEnv()).catalog)
    cat_matrix = S.category_matrix(load_item_categories().to_dict())
    selected = json.loads((OUT / "tuning_base.json").read_text())["selected"]
    pairs = load_pairs()
    cut = time_cutoffs(pairs)
    in_cat = np.isin(pairs.video, catalog)
    fit_m = pairs.ts < cut["fit_end"]
    rank_m = (~fit_m) & (pairs.ts < cut["rank_end"]) & in_cat
    val_m = (pairs.ts >= cut["rank_end"]) & in_cat
    min_n = cfg["split"]["min_val_pairs"]
    n_users = int(pairs.user.max()) + 1
    users_tr = np.flatnonzero(np.bincount(pairs.user[rank_m], minlength=n_users) >= min_n)
    users_va = np.flatnonzero(np.bincount(pairs.user[val_m], minlength=n_users) >= min_n)
    fit = pairs.subset(fit_m)
    base = {name: MAKE[name](**selected[name]["params"]).fit(fit, cat_matrix, catalog) for name in MAKE}
    prior = prior_table(fit)
    history = History(pairs.subset(fit_m & ~in_cat))
    t_tr = History(pairs.subset(rank_m & np.isin(pairs.user, users_tr)))
    t_va = History(pairs.subset(val_m & np.isin(pairs.user, users_va)))
    del pairs, fit
    gc.collect()
    cutoff_date = int(datetime.fromtimestamp(cut["fit_end"], tz=timezone.utc).strftime("%Y%m%d"))
    t0 = time.time()
    builder = RowBuilder("text", cutoff_date, base, prior, catalog)
    x_tr, y_tr, v_tr, g_tr = build_rows(builder, users_tr, history, t_tr)
    x_va, y_va, v_va, g_va = build_rows(builder, users_va, history, t_va)
    print(f"rows: train {len(y_tr):,} in {len(g_tr):,} groups, validation {len(y_va):,} in {len(g_va):,} groups "
          f"({time.time() - t0:.0f}s)", flush=True)
    floor = R.grouped_ndcg(np.zeros(len(y_va)), y_va, v_va, g_va)
    out = {"video_stats_cutoff_date": cutoff_date, "train_rows": int(len(y_tr)), "train_groups": int(len(g_tr)),
           "val_rows": int(len(y_va)), "val_groups": int(len(g_va)), "constant_floor_ndcg_at_10": floor, "variants": {}}
    (CACHE / "final").mkdir(parents=True, exist_ok=True)
    for variant in VARIANTS:
        cols = columns(variant, builder.vtab, builder.utab, builder.text)
        idx = [builder.cols.index(c) for c in cols]
        trials = []
        for nl in cfg["tune"]["ranker"]["num_leaves"]:
            t0 = time.time()
            booster, info = R.train(x_tr[:, idx], y_tr, g_tr, x_va[:, idx], y_va, g_va, nl, cfg["tune"]["ranker"], cols)
            pred = booster.predict(x_va[:, idx], num_iteration=info["best_iteration"])
            ndcg = R.grouped_ndcg(pred, y_va, v_va, g_va)
            trials.append({"num_leaves": nl, **info, "ndcg_at_10": ndcg, "seconds": round(time.time() - t0, 1),
                           "booster": booster})
            print(f"{variant:12s} num_leaves={nl:3d} best_iter={info['best_iteration']:4d} NDCG@10 {ndcg:.4f} "
                  f"({trials[-1]['seconds']}s)", flush=True)
        best = max(trials, key=lambda t: t["ndcg_at_10"])
        imp = best["booster"].feature_importance("gain")
        top = sorted(zip(cols, (imp / imp.sum()).tolist()), key=lambda t: -t[1])[:15]
        with open(CACHE / "final" / f"ranker_{variant}.pkl", "wb") as fh:
            pickle.dump({"booster": best["booster"], "columns": cols, "best_iteration": best["best_iteration"],
                         "num_leaves": best["num_leaves"], "variant": variant}, fh)
        out["variants"][variant] = {"n_features": len(cols), "selected_num_leaves": best["num_leaves"],
                                    "trials": [{k: v for k, v in t.items() if k != "booster"} for t in trials],
                                    "top_gain_share": top}
    (OUT / "tuning_ranker.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({v: {"num_leaves": d["selected_num_leaves"], "ndcg": max(t["ndcg_at_10"] for t in d["trials"])}
                      for v, d in out["variants"].items()}, indent=2), f"\nconstant floor {floor:.4f}")


if __name__ == "__main__":
    main()
