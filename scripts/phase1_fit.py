"""Phase 1: fit the serving artifacts on ALL big-matrix pairs (after offline tuning).

Refits (a) affinity, (b) item-kNN and (c) EASE at the configurations selected in
runs/phase1/tuning_base.json, and the video label prior, on every first-view pair. The ranker
boosters come from scripts/phase1_rank.py unchanged; at serving they read features built from
these all-pairs fits (policies/phase1/serving.py).

Usage: .venv/bin/python scripts/phase1_fit.py
Writes data/cache/phase1/final/{affinity,item_knn,ease,prior}.pkl and meta.json
"""
import gc
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from kuairec_eval import harness  # noqa: E402
from kuairec_eval.data import load_item_categories  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from policies.phase1 import scorers as S  # noqa: E402
from policies.phase1.config import CACHE, file_hash, load  # noqa: E402
from policies.phase1.features import prior_table  # noqa: E402
from policies.phase1.pairs import PAIRS_PATH, load_pairs  # noqa: E402

OUT = ROOT / "runs" / "phase1"
MAKE = {"affinity": S.Affinity, "item_knn": S.ItemKNN, "ease": S.EASE}


def main():
    final = CACHE / "final"
    final.mkdir(parents=True, exist_ok=True)
    catalog = np.asarray(harness.context_for(KuaiEnv()).catalog)
    cat_matrix = S.category_matrix(load_item_categories().to_dict())
    selected = json.loads((OUT / "tuning_base.json").read_text())["selected"]
    pairs = load_pairs()
    for name, cls in MAKE.items():
        model = cls(**selected[name]["params"]).fit(pairs, cat_matrix, catalog)
        with open(final / f"{name}.pkl", "wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        del model
        gc.collect()
        print("fitted", name, selected[name]["params"], flush=True)
    with open(final / "prior.pkl", "wb") as fh:
        pickle.dump(prior_table(pairs), fh)
    meta = {"catalog": catalog.tolist(), "selected": selected, "label": load()["label"],
            "pairs_file_sha16": file_hash(PAIRS_PATH), "fitted_at": datetime.now(timezone.utc).isoformat(),
            "artifacts_sha16": {p.name: file_hash(p) for p in sorted(final.glob("*.pkl"))}}
    (final / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: v for k, v in meta.items() if k != "catalog"}, indent=2))


if __name__ == "__main__":
    main()
