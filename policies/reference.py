"""Reference policies for the evaluator audit (ADR-0014). Public data only: the big matrix
(through kuairec_eval.labels) and item categories (through PolicyContext).

  (a) ShortestFirst          shortest unseen catalog video, canonical duration ascending
  (b) CategoryRotation       rotate categories; random single-tag video inside each category
  (c) CategoryRotation       rotate categories; shortest single-tag video inside each category
  (d) BigMeanLabel("raw")    big-matrix mean watch_ratio per video, minimum 100 big-matrix rows
  (e) BigMeanLabel(label)    NOT REQUESTED: (d) with the active reward's own label
  probes, NOT REQUESTED      LongestFirst and CategoryRotation(pick="longest"): mirror (a) and
                             (c) to test for a long-video bias (ADR-0014)

The rotation never reads the quit parameters. It shows only single-tag videos and cycles
through the K = 29 categories that have one, so a category recurs no sooner than K steps
later; it can trigger the boredom rule only if the window N is at least K.
"""
from __future__ import annotations

import numpy as np

from kuairec_eval.labels import big_matrix_video_stats, canonical_durations

MIN_BIG_ROWS = 100


class ShortestFirst:
    """(a) Deterministic: same ranking for every user, ties broken by video_id."""
    stochastic = False
    longest = False

    def __init__(self, context, seed: int):
        dur = canonical_durations()
        sign = -1.0 if self.longest else 1.0
        self.ranking: tuple[int, ...] = tuple(sorted(context.catalog, key=lambda v: (sign * float(dur[v]), v)))
        self._pos = 0

    def begin_session(self, user: int) -> None:
        self._pos = 0

    def act(self) -> int:
        v = self.ranking[self._pos]
        self._pos += 1
        return v

    def observe(self, video, result) -> None:
        pass


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

    def begin_session(self, user: int) -> None:
        if self.pick == "random":
            self._order = [int(c) for c in self._rng.permutation(self.rotation_categories)]
        else:
            self._order = list(self._fixed_order)
        self._pos = 0
        self._shown = set()

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


class BigMeanLabel:
    """(d) label="raw": rank catalog videos by big-matrix mean watch_ratio, among videos with
    at least MIN_BIG_ROWS big-matrix rows (14 of 3,327 fall below). (e) any other label."""
    stochastic = False

    def __init__(self, context, seed: int, label: str = "raw", n_buckets: int | None = None):
        st = big_matrix_video_stats(label, n_buckets).reindex(list(context.catalog))
        eligible = st[st["size"] >= MIN_BIG_ROWS]
        self.ranking: tuple[int, ...] = tuple(sorted(eligible.index, key=lambda v: (-float(eligible.at[v, "mean"]), v)))
        self._pos = 0

    def begin_session(self, user: int) -> None:
        self._pos = 0

    def act(self) -> int:
        v = self.ranking[self._pos]
        self._pos += 1
        return v

    def observe(self, video, result) -> None:
        pass
