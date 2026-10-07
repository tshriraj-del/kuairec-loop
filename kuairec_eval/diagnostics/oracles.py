"""Oracle ceilings (DIAGNOSTICS ONLY, see package docstring). ADR-0015.

Both oracles know the quit rule and the true small-matrix reward of the active candidate,
and both use the same greedy step, so their gap measures personalization rather than
optimiser quality:

  greedy step   given the session so far, show the highest-value unseen video that does
                not trigger the boredom rule at this step (ties: lower video_id). Values
                for unrecorded pairs are the environment's missing_pair_reward (0 in v0).

  PersonalizedGreedyOracle   values = the user's own row of the reward matrix
  FixedSequenceOracle        values = column means over the search users; one sequence,
                             built once, shown to every user (in-sample by construction)

Greedy is not proven optimal under the category constraint. top_k_bound() gives the
quit-free upper bound (sum of the k best values), so the slack is visible.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from kuairec_eval.diagnostics import guard_against_policies
from kuairec_eval.env import EnvConfig, boredom_quit


def _filled(values: np.ndarray, missing: float) -> np.ndarray:
    return np.where(np.isnan(values), missing, values)


def greedy_sequence(values: np.ndarray, videos: np.ndarray, categories: Mapping[int, tuple[int, ...]],
                    cfg: EnvConfig) -> list[int]:
    """Greedy no-quit sequence of up to cfg.max_turn videos for one value vector over `videos`."""
    order = [int(videos[i]) for i in np.lexsort((videos, -values))]
    seq: list[int] = []
    while len(seq) < cfg.max_turn:
        for i, v in enumerate(order):
            if not boredom_quit(seq, v, categories, cfg.quit_window, cfg.quit_threshold, cfg.window_mode):
                seq.append(v)
                del order[i]
                break
        else:
            break
    return seq


class _SequencePlayer:
    stochastic = False

    def __init__(self):
        self._seq: list[int] = []
        self._pos = 0

    def act(self) -> int:
        v = self._seq[self._pos]
        self._pos += 1
        return v

    def observe(self, video, result) -> None:
        pass


class PersonalizedGreedyOracle(_SequencePlayer):
    def __init__(self, context, matrix: np.ndarray, user_index: Mapping[int, int], videos: np.ndarray,
                 cfg: EnvConfig):
        guard_against_policies()
        super().__init__()
        self._matrix, self._user_index, self._videos, self._cfg = matrix, user_index, videos, cfg
        self._categories = context.categories

    def begin_session(self, user: int) -> None:
        row = _filled(self._matrix[self._user_index[user]], self._cfg.missing_pair_reward)
        seq = greedy_sequence(row, self._videos, self._categories, self._cfg)
        # Pad with the best remaining videos in case no non-triggering video was left (never observed).
        chosen = set(seq)
        rest = [int(self._videos[i]) for i in np.lexsort((self._videos, -row)) if int(self._videos[i]) not in chosen]
        self._seq, self._pos = seq + rest, 0


class FixedSequenceOracle(_SequencePlayer):
    def __init__(self, context, sequence: list[int]):
        guard_against_policies()
        super().__init__()
        self.sequence = tuple(sequence)
        chosen = set(sequence)
        rest = [v for v in context.catalog if v not in chosen]
        self._full = list(sequence) + rest

    def begin_session(self, user: int) -> None:
        self._seq, self._pos = self._full, 0


def column_means(matrix: np.ndarray, user_rows: np.ndarray, missing: float) -> np.ndarray:
    return _filled(matrix[user_rows], missing).mean(axis=0)


def personalized_factory(matrix: np.ndarray, users: np.ndarray, videos: np.ndarray, cfg: EnvConfig):
    guard_against_policies()
    user_index = {int(u): i for i, u in enumerate(users)}

    def factory(context, seed: int):
        return PersonalizedGreedyOracle(context, matrix, user_index, videos, cfg)
    return factory


def nonpersonalized_factory(matrix: np.ndarray, users: np.ndarray, videos: np.ndarray, cfg: EnvConfig,
                            selection_users: list[int], categories: Mapping[int, tuple[int, ...]]):
    """Fixed sequence chosen greedily on the mean value over `selection_users` (the search users)."""
    guard_against_policies()
    rows = np.searchsorted(users, np.asarray(sorted(selection_users)))
    seq = greedy_sequence(column_means(matrix, rows, cfg.missing_pair_reward), videos, categories, cfg)

    def factory(context, seed: int):
        return FixedSequenceOracle(context, seq)
    factory.sequence = tuple(seq)
    return factory


def top_k_bound(matrix: np.ndarray, users: np.ndarray, eval_users: list[int], k: int, missing: float) -> dict:
    """Quit-free upper bounds on cumulative reward over k steps without repeats."""
    rows = np.searchsorted(users, np.asarray(sorted(eval_users)))
    vals = _filled(matrix[rows], missing)
    per_user = np.sort(vals, axis=1)[:, -k:].sum(axis=1)
    means = vals.mean(axis=0)
    return {"personalized": float(per_user.mean()), "nonpersonalized": float(np.sort(means)[-k:].sum()),
            "per_user": per_user}
