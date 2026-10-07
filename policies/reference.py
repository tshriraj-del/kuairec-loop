"""Reference policies (ADR-0014, ADR-0029). Public data only: the big matrix (through
kuairec_eval.labels) and item categories (through PolicyContext). FROZEN since evaluator v1:
the acceptance test and every results table depend on these exact definitions.

  (a) ShortestFirst          shortest unseen catalog video, canonical duration ascending
  (b) CategoryRotation       rotate categories; random single-tag video inside each category
  (c) CategoryRotation       rotate categories; shortest single-tag video inside each category
  (d) BigMeanLabel("raw")    big-matrix mean watch_ratio per video, minimum 100 big-matrix rows
  (e) BigMeanLabel(label)    (d) with the active reward's own label
  LongestFirst, CategoryRotation(pick="longest")   mirrors of (a) and (c)

PERMANENT_ROWS below must appear in every results table (tests/test_reporting.py).

The rotation never reads the quit parameters. It shows only single-tag videos and cycles
through the K = 29 categories that have one, so a category recurs no sooner than K steps
later; it can trigger the boredom rule only if the window N is at least K.
"""
from __future__ import annotations

import numpy as np

from kuairec_eval.labels import big_matrix_video_stats, canonical_durations
from policies.baselines import RankedPolicy

MIN_BIG_ROWS = 100


class ShortestFirst(RankedPolicy):
    """(a) Deterministic: same ranking for every user, ties broken by video_id."""
    longest = False

    def __init__(self, context, seed: int):
        dur = canonical_durations()
        sign = -1.0 if self.longest else 1.0
        self.ranking = tuple(sorted(context.catalog, key=lambda v: (sign * float(dur[v]), v)))


class LongestFirst(ShortestFirst):
    """Probe (not requested): longest unseen catalog video first."""
    longest = True


class CategoryRotation:
    """(b) pick="random": category order and video drawn at random per session.
    (c) pick="shortest": fixed order by each category's shortest single-tag video; shortest unseen video.
    probe pick="longest": the mirror of (c), longest first."""

    def __init__(self, context, seed: int, pick: str = "random"):
        if pick not in ("random", "shortest", "longest"):
            raise ValueError(pick)
        self.pick = pick
        self.stochastic = pick == "random"
        dur = canonical_durations()
        sign = -1.0 if pick == "longest" else 1.0
        by_cat: dict[int, list[int]] = {}
        for v in context.catalog:
            tags = context.categories[v]
            if len(tags) == 1:
                by_cat.setdefault(tags[0], []).append(v)
        self._by_cat = {c: sorted(vs, key=lambda v: (sign * float(dur[v]), v)) for c, vs in by_cat.items()}
        self.rotation_categories = tuple(sorted(self._by_cat))
        self._fixed_order = sorted(self.rotation_categories, key=lambda c: (sign * float(dur[self._by_cat[c][0]]), c))
        self._rng = np.random.default_rng(seed)
        self._order: list[int] = []
        self._pos = 0
        self._shown: set[int] = set()

    def begin_session(self, user: int, excluded: frozenset[int]) -> None:
        if self.pick == "random":
            self._order = [int(c) for c in self._rng.permutation(self.rotation_categories)]
        else:
            self._order = list(self._fixed_order)
        self._pos = 0
        self._shown = set(excluded)  # never shown, never eligible

    def act(self) -> int:
        k = len(self._order)
        for skip in range(k):
            c = self._order[(self._pos + skip) % k]
            avail = [v for v in self._by_cat[c] if v not in self._shown]
            if avail:
                self._pos += skip + 1
                v = int(avail[self._rng.integers(len(avail))]) if self.pick == "random" else avail[0]
                self._shown.add(v)
                return v
        raise RuntimeError("every rotation category is exhausted")

    def observe(self, video, result) -> None:
        pass


class BigMeanLabel(RankedPolicy):
    """(d) label="raw": rank catalog videos by big-matrix mean watch_ratio, among videos with
    at least MIN_BIG_ROWS big-matrix rows (14 of 3,327 fall below). (e) any other label."""

    def __init__(self, context, seed: int, label: str = "raw", n_buckets: int | None = None):
        st = big_matrix_video_stats(label, n_buckets).reindex(list(context.catalog))
        eligible = st[st["size"] >= MIN_BIG_ROWS]
        self.ranking = tuple(sorted(eligible.index, key=lambda v: (-float(eligible.at[v, "mean"]), v)))


def rotation_random(context, seed: int) -> CategoryRotation:
    return CategoryRotation(context, seed, "random")


def rotation_shortest(context, seed: int) -> CategoryRotation:
    return CategoryRotation(context, seed, "shortest")


def rotation_longest(context, seed: int) -> CategoryRotation:
    return CategoryRotation(context, seed, "longest")


# Every results table includes these rows (ADR-0029): the four duration probes, plain and inside
# rotation, and rotation-random, which the session-length-controlled acceptance test compares
# them against (ADR-0027).
PERMANENT_ROWS: tuple[tuple[str, str, object], ...] = (
    ("rotation_random", "(b) category rotation, random pick", rotation_random),
    ("shortest_first", "(a) probe: shortest first", ShortestFirst),
    ("longest_first", "probe: longest first", LongestFirst),
    ("rotation_shortest", "(c) probe: category rotation, shortest pick", rotation_shortest),
    ("rotation_longest", "probe: category rotation, longest pick", rotation_longest),
)
