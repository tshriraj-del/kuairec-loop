"""Phase 1, item 7: offline ranking check on the SEARCH partition (evaluator-side script).

For each search user, each scorer ranks the user's action set (the catalog videos with a recorded
reward; ties broken by lower video id) against the user's recorded rewards:
  NDCG@10    graded gains = the recorded duration_pct reward
  Recall@10  relevant = reward >= relevance_threshold (config/phase1.toml [eval], 0.75: the top
             quarter of big-matrix play times in the video's duration bucket);
             hits in the top 10 / min(10, number of relevant videos)
Reference rows: random scores (numpy default_rng(0)), the non-personalized oracle's common values
(search-user means, in-sample) and the true rewards (an upper bound of 1.0 by construction).
Only search users' rows of the reward matrix are kept; the holdout is never touched.

Usage: .venv/bin/python scripts/phase1_offline.py
Writes runs/phase1/offline_search.json
"""
import lightgbm  # noqa: F401
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402

from kuairec_eval import partitions  # noqa: E402
from kuairec_eval.env import KuaiEnv  # noqa: E402
from kuairec_eval.rewards import reward_matrix  # noqa: E402
from policies.phase1 import serving  # noqa: E402
from policies.phase1.config import load  # noqa: E402
from policies.phase1.scorers import ndcg_at_k  # noqa: E402

OUT = ROOT / "runs" / "phase1"


def recall_at_k(scores, rewards, items, tau, k=10):
    rel = rewards >= tau
    if not rel.any():
        return None
    top = np.lexsort((items, -scores))[:k]
    return float(rel[top].sum() / min(k, rel.sum()))


def main():
    tau = load()["eval"]["relevance_threshold"]
    env = KuaiEnv()
    users_all, catalog = np.asarray(env.users), np.asarray(env.catalog)
    search = np.asarray(sorted(partitions.load("search")))
    full = reward_matrix(env.config.reward_config(), users_all, catalog)
    mat = full[np.searchsorted(users_all, search)]
    del full
    common = np.nanmean(mat, axis=0)
    rng = np.random.default_rng(0)
    scorers = {name: serving.score_function(name) for name in serving.SCORERS}
    results = {name: {"ndcg": [], "recall": []} for name in ["random", "common_values", "true_rewards", *scorers]}
    for i, u in enumerate(search):
        rec = ~np.isnan(mat[i])
        items, gains = catalog[rec], mat[i][rec]
        cand = {"random": rng.random(len(catalog)), "common_values": common, "true_rewards": np.nan_to_num(mat[i], nan=-1.0)}
        cand.update({name: fn(int(u)) for name, fn in scorers.items()})
        for name, s in cand.items():
            s = np.asarray(s, dtype=np.float64)[rec]
            results[name]["ndcg"].append(ndcg_at_k(s, gains, items))
            r = recall_at_k(s, gains, items, tau)
            if r is not None:
                results[name]["recall"].append(r)
    out = {"users": int(len(search)), "relevance_threshold": tau,
           "mean_relevant_per_user": float(np.mean([(row[~np.isnan(row)] >= tau).sum() for row in mat])),
           "rows": {name: {"ndcg_at_10": float(np.mean(v["ndcg"])), "recall_at_10": float(np.mean(v["recall"])),
                           "users_with_relevant": len(v["recall"])} for name, v in results.items()}}
    (OUT / "offline_search.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
