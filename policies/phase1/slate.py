"""The slate wrapper shared by every Phase 1 scorer (brief item 3, ADR-0035).

At each step it shows the highest-scored available video whose categories do not trigger the
PRIMARY quit rule. It is rule-aware on purpose in Phase 1, so scorers are compared on scoring
quality alone; Phase 2 removes the rule knowledge. The rule is a policy-side copy of the
environment's definition with the intended window (config/phase1.toml [slate]);
tests/test_phase1.py checks it against the frozen kuairec_eval.env.boredom_quit.
"""
from __future__ import annotations

import itertools
from collections import Counter
from typing import Callable

import numpy as np


def would_quit(seq: list[int], video: int, categories, window: int, threshold: int) -> bool:
    """True if showing `video` after `seq` makes the user quit: a category of `video` occurs more
    than `threshold` times among the last `window` recommendations."""
    if not seq:
        return False
    counts = Counter(itertools.chain.from_iterable(categories[w] for w in seq[-window:]))
    return any(counts[c] > threshold for c in categories[video])


class RuleAwareSlate:
    stochastic = False

    def __init__(self, context, seed: int, score_fn: Callable[[int], np.ndarray], window: int, threshold: int):
        self.catalog = np.asarray(context.catalog)
        self.categories = context.categories
        self.score_fn, self.window, self.threshold = score_fn, window, threshold

    def begin_session(self, user: int, excluded: frozenset[int]) -> None:
        scores = np.asarray(self.score_fn(user), dtype=np.float64)
        order = np.lexsort((self.catalog, -scores))
        self._queue = [int(self.catalog[i]) for i in order if int(self.catalog[i]) not in excluded]
        self._seq: list[int] = []

    def act(self) -> int:
        for i, v in enumerate(self._queue):
            if not would_quit(self._seq, v, self.categories, self.window, self.threshold):
                del self._queue[i]
                self._seq.append(v)
                return v
        v = self._queue.pop(0)  # no safe video left: show the best one (never observed in 30 steps)
        self._seq.append(v)
        return v

    def observe(self, video, result) -> None:
        pass
