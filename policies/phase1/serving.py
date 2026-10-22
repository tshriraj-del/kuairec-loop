"""Serving: turns the fitted Phase 1 artifacts (scripts/phase1_fit.py) into harness policies.

Every scorer serves from models fitted on ALL big-matrix pairs and scores a user from their
NON-catalog pairs, which for the environment's users is their whole big-matrix history.
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache

import numpy as np

from policies.phase1.config import CACHE, load
from policies.phase1.pairs import History, load_pairs
from policies.phase1.slate import RuleAwareSlate

SCORERS = ("affinity", "item_knn", "ease", "ranker_no_duration", "ranker_duration", "ranker_text")


@lru_cache(maxsize=1)
def _history_and_catalog() -> tuple[History, np.ndarray]:
    meta = json.loads((CACHE / "final" / "meta.json").read_text())
    catalog = np.asarray(meta["catalog"])
    pairs = load_pairs()
    return History(pairs.subset(~np.isin(pairs.video, catalog))), catalog


@lru_cache(maxsize=None)
def _artifact(name: str):
    with open(CACHE / "final" / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


@lru_cache(maxsize=1)
def _builder():
    from policies.phase1.features import RowBuilder
    _, catalog = _history_and_catalog()
    base = {k: _artifact(k) for k in ("affinity", "item_knn", "ease")}
    return RowBuilder("text", None, base, _artifact("prior"), catalog)


def score_function(name: str):
    history, catalog = _history_and_catalog()
    if name in ("affinity", "item_knn", "ease"):
        model = _artifact(name)
        return lambda user: model.score(*history.get(user))
    variant = name.removeprefix("ranker_")
    booster_cols = _artifact(name)
    builder = _builder()
    idx = [builder.cols.index(c) for c in booster_cols["columns"]]
    booster = booster_cols["booster"]

    def score(user):
        hv, hl = history.get(user)
        x = builder.user_rows(user, hv, hl, catalog)[:, idx]
        return booster.predict(x, num_iteration=booster_cols["best_iteration"])
    score.variant = variant
    return score


def factory(name: str):
    """Harness policy factory for scorer `name` behind the rule-aware slate."""
    if name not in SCORERS:
        raise KeyError(name)
    slate_cfg = load()["slate"]
    fn = score_function(name)
    _, catalog = _history_and_catalog()

    def make(context, seed):
        if not np.array_equal(np.asarray(context.catalog), catalog):
            raise ValueError("artifacts were fitted for a different catalog")
        return RuleAwareSlate(context, seed, fn, slate_cfg["quit_window"], slate_cfg["quit_threshold"])
    return make
