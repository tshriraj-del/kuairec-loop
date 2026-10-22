"""Offline validation on the big matrix (ADR-0030): a time split with catalog-cold histories.

Fit on every pair first seen before rank_end. Validation users have >= min_val_pairs catalog
pairs in T_val; each is scored from their NON-catalog pairs before rank_end only, because the
environment's users have no catalog video in their big-matrix history. Metric: mean NDCG@10 of
the scorer over the user's T_val catalog pairs, graded gains = the label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from policies.phase1.config import load
from policies.phase1.pairs import History, Pairs, time_cutoffs
from policies.phase1.scorers import ndcg_at_k


@dataclass
class ValidationSet:
    fit_pairs: Pairs
    users: np.ndarray
    history: History
    targets: History


def build(pairs: Pairs, catalog: np.ndarray) -> ValidationSet:
    cut = time_cutoffs(pairs)
    in_cat = np.isin(pairs.video, catalog)
    before = pairs.ts < cut["rank_end"]
    val = (~before) & in_cat
    counts = np.bincount(pairs.user[val], minlength=int(pairs.user.max()) + 1)
    users = np.flatnonzero(counts >= load()["split"]["min_val_pairs"])
    return ValidationSet(fit_pairs=pairs.subset(before), users=users,
                         history=History(pairs.subset(before & ~in_cat)),
                         targets=History(pairs.subset(val & np.isin(pairs.user, users))))


def evaluate(scorer, vs: ValidationSet, catalog: np.ndarray, k: int = 10) -> dict:
    pos = np.full(int(catalog.max()) + 1, -1)
    pos[catalog] = np.arange(len(catalog))
    scores = []
    for u in vs.users:
        hv, hl = vs.history.get(u)
        s = scorer.score(hv, hl)
        tv, tl = vs.targets.get(u)
        scores.append(ndcg_at_k(s[pos[tv]], tl.astype(np.float64), tv, k))
    scores = np.asarray(scores)
    return {"ndcg_at_10": float(scores.mean()), "users": int(len(scores))}
