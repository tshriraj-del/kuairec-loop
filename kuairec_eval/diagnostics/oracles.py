"""Oracle ceilings (DIAGNOSTICS ONLY, see package docstring). ADR-0015 (v1 update).

Both oracles know the quit rule and the true small-matrix reward of the active candidate,
and both use the same greedy step, so their gap measures personalization rather than
optimiser quality:

  greedy step   given the session so far, show the highest-value video in the user's action
                set that is unseen and does not trigger the boredom rule at this step
                (ties: lower video_id). Unrecorded pairs are outside the action set.

  PersonalizedGreedyOracle    values = the user's own row of the reward matrix
  CommonValueGreedyOracle     values = one vector for everybody: each video's mean reward
                              over the search users who have it recorded. Applied within each
                              user's action set, so the sequence differs between users only
                              where a video is unavailable to them (in-sample by construction)

Greedy is not proven optimal under the category constraint. top_k() reports the quit-free
versions (the k best values in the action set), so the slack is visible; for the
personalized oracle that is a true upper bound on any policy's cumulative reward.

  CategoryAffinityOracle (ADR-0028) knows each user's mean reward per category and shows a
  uniformly random available single-tag video of the chosen category. Its modes differ only
  in which categories may be chosen next (single-tag videos make the window counts exact):
    free        count of the category in the last N recommendations <= n_Q: as freely as the
                quit setting allows, never triggering it
    strict      category absent from the last N recommendations: strict rotation, the freedom
                of n_Q = 0, safe under every profile
    full_cycle  category not yet shown in the current cycle through all of the user's rotation
                categories (the rotation of policies (b)/(c), ordered by affinity)
  Each step takes the highest-affinity allowed category (ties: lower category id).
"""
from __future__ import annotations

import warnings
from collections import Counter
from typing import Mapping

import numpy as np

from kuairec_eval.diagnostics import guard_against_policies
from kuairec_eval.env import EnvConfig, boredom_quit


def greedy_sequence(values: np.ndarray, videos: np.ndarray, categories: Mapping[int, tuple[int, ...]],
                    cfg: EnvConfig) -> list[int]:
    """Greedy no-quit sequence of up to cfg.max_turn videos; entries with non-finite value are unavailable."""
    order = [int(videos[i]) for i in np.lexsort((videos, -values)) if np.isfinite(values[i])]
    seq: list[int] = []
    while len(seq) < cfg.max_turn:
        for i, v in enumerate(order):
            if not boredom_quit(seq, v, categories, cfg.quit_window, cfg.quit_threshold, cfg.window_mode):
                seq.append(v)
                del order[i]
                break
        else:
            break
    chosen = set(seq)
    return seq + [v for v in order if v not in chosen]  # padding, only reached if no safe video remains


class _GreedyPlayer:
    stochastic = False

    def __init__(self, context, videos: np.ndarray, cfg: EnvConfig):
        guard_against_policies()
        self._videos, self._cfg, self._categories = videos, cfg, context.categories
        self._seq: list[int] = []
        self._pos = 0

    def _play(self, values: np.ndarray) -> None:
        self._seq, self._pos = greedy_sequence(values, self._videos, self._categories, self._cfg), 0

    def act(self) -> int:
        v = self._seq[self._pos]
        self._pos += 1
        return v

    def observe(self, video, result) -> None:
        pass


class PersonalizedGreedyOracle(_GreedyPlayer):
    def __init__(self, context, matrix: np.ndarray, user_index: Mapping[int, int], videos: np.ndarray, cfg: EnvConfig):
        super().__init__(context, videos, cfg)
        self._matrix, self._user_index = matrix, user_index

    def begin_session(self, user: int, excluded: frozenset[int]) -> None:
        row = self._matrix[self._user_index[user]]
        self._play(np.where(np.isnan(row), -np.inf, row))


class CommonValueGreedyOracle(_GreedyPlayer):
    def __init__(self, context, common: np.ndarray, videos: np.ndarray, cfg: EnvConfig):
        super().__init__(context, videos, cfg)
        self._common = common
        self._vidx = {int(v): j for j, v in enumerate(videos)}

    def begin_session(self, user: int, excluded: frozenset[int]) -> None:
        values = self._common.copy()
        values[[self._vidx[v] for v in excluded]] = -np.inf
        self._play(values)


def common_values(matrix: np.ndarray, users: np.ndarray, selection_users: list[int]) -> np.ndarray:
    rows = np.searchsorted(users, np.asarray(sorted(selection_users)))
    return np.nanmean(matrix[rows], axis=0)


def personalized_factory(matrix: np.ndarray, users: np.ndarray, videos: np.ndarray, cfg: EnvConfig):
    guard_against_policies()
    user_index = {int(u): i for i, u in enumerate(users)}

    def factory(context, seed: int):
        return PersonalizedGreedyOracle(context, matrix, user_index, videos, cfg)
    return factory


def nonpersonalized_factory(matrix: np.ndarray, users: np.ndarray, videos: np.ndarray, cfg: EnvConfig,
                            selection_users: list[int]):
    """Common value vector from `selection_users` (the search users), applied per user."""
    guard_against_policies()
    common = common_values(matrix, users, selection_users)

    def factory(context, seed: int):
        return CommonValueGreedyOracle(context, common, videos, cfg)
    factory.common_values = common
    return factory


def top_k(matrix: np.ndarray, users: np.ndarray, eval_users: list[int], k: int, common: np.ndarray) -> dict:
    """Quit-free versions over k steps without repeats, within each user's action set."""
    rows = matrix[np.searchsorted(users, np.asarray(sorted(eval_users)))]
    own = np.where(np.isnan(rows), -np.inf, rows)
    personalized = np.sort(own, axis=1)[:, -k:].sum(axis=1)
    pick = np.where(np.isnan(rows), -np.inf, common[None, :])
    top = np.argsort(-pick, axis=1, kind="stable")[:, :k]
    nonpersonalized = np.take_along_axis(rows, top, axis=1).sum(axis=1)
    return {"personalized_bound": float(personalized.mean()), "nonpersonalized_quit_free": float(nonpersonalized.mean())}


AFFINITY_MODES = ("free", "strict", "full_cycle")


class CategoryAffinityOracle:
    stochastic = True

    def __init__(self, context, seed: int, affinity: np.ndarray, user_index: Mapping[int, int],
                 categories: tuple[int, ...], single_videos: Mapping[int, tuple[int, ...]], cfg: EnvConfig,
                 mode: str):
        guard_against_policies()
        if mode not in AFFINITY_MODES:
            raise ValueError(f"mode must be one of {AFFINITY_MODES}")
        self._affinity, self._user_index, self._cats = affinity, user_index, categories
        self._single, self._cfg, self._mode = single_videos, cfg, mode
        self._rng = np.random.default_rng(seed)

    def begin_session(self, user: int, excluded: frozenset[int]) -> None:
        aff = self._affinity[self._user_index[user]]
        ranked = sorted((-aff[k], c) for k, c in enumerate(self._cats) if np.isfinite(aff[k]))
        self._order = [c for _, c in ranked]
        self._pool = {c: [v for v in self._single[c] if v not in excluded] for c in self._order}
        self._recent: list[int] = []
        self._cycle: set[int] = set()

    def _allowed(self, c: int, counts: Counter) -> bool:
        if not self._pool[c]:
            return False
        if self._mode == "full_cycle":
            return c not in self._cycle
        limit = self._cfg.quit_threshold if self._mode == "free" else 0
        return counts[c] <= limit

    def act(self) -> int:
        counts = Counter(self._recent[-self._cfg.quit_window:])
        chosen = next((c for c in self._order if self._allowed(c, counts)), None)
        if chosen is None and self._mode == "full_cycle":
            self._cycle = set()
            chosen = next((c for c in self._order if self._allowed(c, counts)), None)
        if chosen is None:  # nothing allowed is left: show the best non-empty category (may trigger)
            chosen = next(c for c in self._order if self._pool[c])
        pool = self._pool[chosen]
        video = pool.pop(int(self._rng.integers(len(pool))))
        self._recent.append(chosen)
        self._cycle.add(chosen)
        return video

    def observe(self, video, result) -> None:
        pass


def category_affinity_factory(matrix: np.ndarray, users: np.ndarray, videos: np.ndarray,
                              categories: Mapping[int, tuple[int, ...]], cfg: EnvConfig, mode: str):
    """Affinity(u, c) = mean reward over u's recorded single-tag videos of category c: the expected
    reward of a random pick from c, the oracle's only action inside a category."""
    guard_against_policies()
    cols_by_cat: dict[int, list[int]] = {}
    for j, v in enumerate(videos):
        tags = categories[int(v)]
        if len(tags) == 1:
            cols_by_cat.setdefault(tags[0], []).append(j)
    cats = tuple(sorted(cols_by_cat))
    affinity = np.full((len(users), len(cats)), -np.inf)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN rows: no recorded video
        for k, c in enumerate(cats):
            m = np.nanmean(matrix[:, cols_by_cat[c]], axis=1)
            affinity[:, k] = np.where(np.isnan(m), -np.inf, m)
    single_videos = {c: tuple(int(videos[j]) for j in cols) for c, cols in cols_by_cat.items()}
    user_index = {int(u): i for i, u in enumerate(users)}

    def factory(context, seed: int):
        return CategoryAffinityOracle(context, seed, affinity, user_index, cats, single_videos, cfg, mode)
    factory.affinity, factory.categories = affinity, cats
    return factory
