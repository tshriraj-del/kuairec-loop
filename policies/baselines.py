"""Phase 0 baselines. Policies see only PolicyContext and the StepResults they receive.

They must never import the small matrix; tests/test_policy_isolation.py enforces this
statically (AST) and kuairec_eval.data enforces it at runtime.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from kuairec_eval.data import load_big_matrix


class UniformRandom:
    """Uniform over catalog videos not yet shown in this session (sampling without replacement)."""

    def __init__(self, context, seed: int):
        self._catalog = np.asarray(context.catalog, dtype=np.int64)
        self._rng = np.random.default_rng(seed)
        self._queue: list[int] = []

    def begin_session(self, user: int) -> None:
        self._queue = self._rng.permutation(self._catalog).tolist()

    def act(self) -> int:
        return self._queue.pop()

    def observe(self, video, result) -> None:
        pass


@lru_cache(maxsize=1)
def big_matrix_popularity() -> pd.Series:
    """video_id -> number of distinct big-matrix users with at least one interaction (ADR-0010)."""
    big = load_big_matrix()
    return big.drop_duplicates(["user_id", "video_id"]).groupby("video_id").size().rename("n_users")


class GlobalPopularity:
    """Most popular unseen catalog video, popularity from the big matrix only.

    Ties broken by ascending video_id; catalog videos absent from the big matrix score 0.
    Identical ranking for every user, so it is deterministic and ignores `seed`.
    """

    def __init__(self, context, seed: int):
        pop = big_matrix_popularity().reindex(list(context.catalog), fill_value=0)
        ranked = sorted(context.catalog, key=lambda v: (-int(pop[v]), v))
        self.ranking: tuple[int, ...] = tuple(ranked)
        self._pos = 0

    def begin_session(self, user: int) -> None:
        self._pos = 0

    def act(self) -> int:
        video = self.ranking[self._pos]
        self._pos += 1
        return video

    def observe(self, video, result) -> None:
        pass


REGISTRY = {"random": UniformRandom, "popularity": GlobalPopularity}
