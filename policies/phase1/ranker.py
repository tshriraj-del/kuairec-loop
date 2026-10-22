"""(d) Gradient-boosted ranker: LightGBM LambdaRank on the duration-adjusted label (ADR-0034).

Relevance grades for LambdaRank: grade = min(4, floor(5 * label)), LightGBM's default gains.
Early stopping on LightGBM's NDCG@10 of the validation groups; the reported validation figure
is recomputed with the same NDCG@10 the base scorers use (gains = the label itself).
LightGBM must be imported before anything that imports torch (scripts/phase1_embed.py), so
Phase 1 keeps torch out of every process that trains or serves.
"""
from __future__ import annotations

import numpy as np

import lightgbm as lgb
from policies.phase1.scorers import ndcg_at_k


def grades(label: np.ndarray) -> np.ndarray:
    return np.minimum(4, np.floor(np.asarray(label) * 5)).astype(np.int32)


def train(x_tr, y_tr, g_tr, x_va, y_va, g_va, num_leaves: int, cfg: dict, columns: list[str]) -> tuple[lgb.Booster, dict]:
    params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10], "learning_rate": cfg["learning_rate"],
              "num_leaves": num_leaves, "min_data_in_leaf": cfg["min_data_in_leaf"],
              "feature_fraction": cfg["feature_fraction"], "seed": cfg["seed"], "deterministic": True,
              "force_row_wise": True, "num_threads": 8, "verbose": -1}
    dtr = lgb.Dataset(x_tr, grades(y_tr), group=g_tr, feature_name=columns, free_raw_data=False)
    dva = lgb.Dataset(x_va, grades(y_va), group=g_va, reference=dtr, free_raw_data=False)
    record: dict = {}
    booster = lgb.train(params, dtr, num_boost_round=cfg["max_rounds"], valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
                                   lgb.record_evaluation(record)])
    return booster, {"best_iteration": int(booster.best_iteration),
                     "lgb_val_ndcg_at_10": float(max(record["val"]["ndcg@10"]))}


def grouped_ndcg(pred: np.ndarray, label: np.ndarray, videos: np.ndarray, groups: np.ndarray, k: int = 10) -> float:
    out, start = [], 0
    for n in groups:
        sl = slice(start, start + n)
        out.append(ndcg_at_k(pred[sl], label[sl].astype(np.float64), videos[sl], k))
        start += n
    return float(np.mean(out))
