"""Phase 0 baselines. Policies see only PolicyContext, the per-session excluded set and the
StepResults they receive (v1 protocol: `stochastic`, begin_session(user, excluded)).

They must never import the small matrix; tests/test_policy_isolation.py enforces this
statically (AST) and kuairec_eval.data enforces it at runtime.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from kuairec_eval.data import load_big_matrix


class RankedPolicy:
    """Shows `self.ranking` in order, skipping videos outside the user's action set."""
    stochastic = False
    ranking: tuple[int, ...] = ()

    def begin_session(self, user: int, excluded: frozenset[int]) -> None:
        self._excluded, self._pos = excluded, 0

    def act(self) -> int:
        while self.ranking[self._pos] in self._excluded:
            self._pos += 1
        video = self.ranking[self._pos]
        self._pos += 1
        return video

    def observe(self, video, result) -> None:
        pass


class UniformRandom:
    """Uniform over the user's action set, without replacement within a session."""
    stochastic = True

    def __init__(self, context, seed: int):
        self._catalog = np.asarray(context.catalog, dtype=np.int64)
        self._rng = np.random.default_rng(seed)
        self._queue: list[int] = []

    def begin_session(self, user: int, excluded: frozenset[int]) -> None:
        self._queue = [v for v in self._rng.permutation(self._catalog).tolist() if v not in excluded]

    def act(self) -> int:
        return self._queue.pop()

    def observe(self, video, result) -> None:
        pass


@lru_cache(maxsize=1)
def big_matrix_popularity() -> pd.Series:
    """video_id -> number of distinct big-matrix users with at least one interaction (ADR-0010)."""
    big = load_big_matrix()
    return big.drop_duplicates(["user_id", "video_id"]).groupby("video_id").size().rename("n_users")


class GlobalPopularity(RankedPolicy):
    """Most popular available unseen video, popularity from the big matrix only.

    Ties broken by ascending video_id; identical ranking for every user, so it is
    deterministic and ignores `seed`.
    """

    def __init__(self, context, seed: int):
        pop = big_matrix_popularity().reindex(list(context.catalog), fill_value=0)
        self.ranking = tuple(sorted(context.catalog, key=lambda v: (-int(pop[v]), v)))


REGISTRY = {"random": UniformRandom, "popularity": GlobalPopularity}
