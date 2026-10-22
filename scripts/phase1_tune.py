"""Phase 1, item 4 (offline part): tune scorers (a)-(c) on the big-matrix validation split.

Every grid point in config/phase1.toml [tune.*] is fitted on pairs first seen before rank_end and
scored on the validation users (policies/phase1/validate.py). No environment run happens here.
A no-skill reference (constant scores, ties broken by video id) is reported for scale.

Usage: .venv/bin/python scripts/phase1_tune.py
Writes runs/phase1/tuning_base.json
"""
import gc
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from kuairec_eval import harness  # noqa: E402
from kuairec_eval.data import load_item_categories  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from policies.phase1 import scorers as S  # noqa: E402
from policies.phase1 import validate as V  # noqa: E402
from policies.phase1.config import config_hash, load  # noqa: E402
from policies.phase1.pairs import load_pairs  # noqa: E402

OUT = ROOT / "runs" / "phase1"


class Constant:
    name = "constant"

    def __init__(self, n):
        self.n = n

    def score(self, hv, hl):
        return np.zeros(self.n, np.float32)


def grid(cfg: dict) -> list[tuple[str, dict]]:
    t = cfg["tune"]
    out = [("affinity", {"shrinkage": s}) for s in t["affinity"]["shrinkage"]]
    out += [("item_knn", {"value": v, "k": k, "shrink": sh})
            for v, k, sh in itertools.product(t["item_knn"]["value"], t["item_knn"]["k"], t["item_knn"]["shrink"])]
    out += [("ease", {"value": v, "l2": l2}) for v, l2 in itertools.product(t["ease"]["value"], t["ease"]["l2"])]
    return out


def make(name: str, params: dict):
    return {"affinity": S.Affinity, "item_knn": S.ItemKNN, "ease": S.EASE}[name](**params)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load()
    catalog = np.asarray(harness.context_for(KuaiEnv()).catalog)
    cat_matrix = S.category_matrix(load_item_categories().to_dict())
    pairs = load_pairs()
    vs = V.build(pairs, catalog)
    del pairs
    gc.collect()
    rows = [{"scorer": "constant", "params": {}, **V.evaluate(Constant(len(catalog)), vs, catalog)}]
    print(f"constant (no skill): NDCG@10 {rows[0]['ndcg_at_10']:.4f} over {rows[0]['users']} users", flush=True)
    for name, params in grid(cfg):
        t0 = time.time()
        sc = make(name, params).fit(vs.fit_pairs, cat_matrix, catalog)
        res = V.evaluate(sc, vs, catalog)
        del sc
        gc.collect()
        rows.append({"scorer": name, "params": params, "config_hash": config_hash({"scorer": name, **params}),
                     **res, "seconds": round(time.time() - t0, 1)})
        print(f"{name:9s} {json.dumps(params):45s} NDCG@10 {res['ndcg_at_10']:.4f}  ({rows[-1]['seconds']}s)", flush=True)
    best = {}
    for r in rows[1:]:
        if r["scorer"] not in best or r["ndcg_at_10"] > best[r["scorer"]]["ndcg_at_10"]:
            best[r["scorer"]] = r
    out = {"validation_users": rows[0]["users"], "rows": rows,
           "selected": {k: {"params": v["params"], "ndcg_at_10": v["ndcg_at_10"]} for k, v in best.items()}}
    (OUT / "tuning_base.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["selected"], indent=2))


if __name__ == "__main__":
    main()
