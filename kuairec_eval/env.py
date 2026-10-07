"""KuaiRec closed-loop environment answering from the fully observed small matrix.

Quit rule implemented (the "feel bored then quit" exit of CIRS, arXiv 2204.01266,
Sec 5.1.2 and Fig. 6, as coded in EasyRL4Rec commit 95d0acf,
src/core/envs/KuaiRec/KuaiEnv.py::_determine_whether_to_leave and
src/core/envs/BaseEnv.py::step):

    Let a_0, a_1, ... be the videos recommended in the session and t the 0-based
    index of the current recommendation a_t. Let C(v) be the category tuple of
    video v from item_categories.csv.

    1. If t == 0 the user never quits on this step.
    2. Otherwise build the window W_t of earlier recommendations:
         window_mode = "intended":         W_t = (a_max(0, t-N), ..., a_{t-1})
         window_mode = "easyrl4rec_slice": W_t = seq[t-N:t] evaluated on the Python
             list seq = [a_0..a_{t-1}], so for 0 < t < N the negative start t-N
             counts from the end of the list and W_t can drop early items.
    3. count(c) = number of occurrences of category c across C(w) for w in W_t
       (multiset union, as collections.Counter over the chained tuples).
    4. The user quits at step t iff there is a category c in C(a_t) with
       count(c) > n_Q.
    5. Independently, the session ends at step t iff t >= max_turn - 1, so a
       session has at most max_turn recommendations.
    6. The reward of a_t is returned and counted even when step t is the quitting
       step: the quit takes effect after a_t is shown (EasyRL4Rec computes
       `terminated` first, then still returns mat[user, a_t]).

    N = config.quit_window, n_Q = config.quit_threshold, max_turn = config.max_turn.

The CIRS text says "more than n_Q items in the N items have at least one attribute
of the current recommended target" and reports N = 1, n_Q = 1 for KuaiEnv, which can
never trigger as written. The code's per-category count above is what we implement;
see ADR-0002.

Deviations from EasyRL4Rec's KuaiEnv, all deliberate and listed in ADR-0003/0004/0005:
  * reward is the raw recorded watch_ratio (EasyRL4Rec caps it at 5);
  * repeat recommendations raise RepeatRecommendationError (EasyRL4Rec allows them);
  * window_mode defaults to "intended" rather than reproducing the slice quirk.

Rewards are reachable only through step(): the matrix lives in a closure and is
never stored on the instance (ADR-0007).
"""
from __future__ import annotations

import itertools
import tomllib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from kuairec_eval import _integrity
from kuairec_eval.data import REPO_ROOT, load_item_categories, load_small_matrix

CONFIG_PATH = REPO_ROOT / "config" / "evaluator.toml"
WINDOW_MODES = ("intended", "easyrl4rec_slice")


class RepeatRecommendationError(ValueError):
    pass


class SessionClosedError(RuntimeError):
    pass


@dataclass(frozen=True)
class EnvConfig:
    quit_window: int
    quit_threshold: int
    max_turn: int
    window_mode: str = "intended"
    missing_pair_reward: float = 0.0

    def __post_init__(self):
        if self.window_mode not in WINDOW_MODES:
            raise ValueError(f"window_mode must be one of {WINDOW_MODES}")
        if self.quit_window < 1 or self.max_turn < 1 or self.quit_threshold < 0:
            raise ValueError("quit_window, max_turn must be >= 1 and quit_threshold >= 0")

    @classmethod
    def from_toml(cls, path: Path = CONFIG_PATH) -> "EnvConfig":
        with open(path, "rb") as f:
            return cls(**tomllib.load(f)["env"])


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    observed: bool            # False if the pair is absent from small_matrix.csv
    quit_reason: str | None   # "boredom", "max_turn" or None
    t: int                    # index of the recommendation just made


def _build_oracle(user_ids: np.ndarray, video_ids: np.ndarray, rewards: np.ndarray,
                  users: np.ndarray, videos: np.ndarray) -> Callable[[int, int], float]:
    uidx = {int(u): i for i, u in enumerate(users)}
    vidx = {int(v): j for j, v in enumerate(videos)}
    mat = np.full((len(users), len(videos)), np.nan, dtype=np.float64)
    mat[[uidx[int(u)] for u in user_ids], [vidx[int(v)] for v in video_ids]] = rewards

    def lookup(user: int, video: int) -> float:
        return float(mat[uidx[user], vidx[video]])

    return lookup


def boredom_quit(seq: list[int], action: int, categories: Mapping[int, tuple[int, ...]],
                 window: int, threshold: int, window_mode: str) -> bool:
    """Steps 1-4 of the module docstring; `seq` holds a_0..a_{t-1}."""
    t = len(seq)
    if t == 0:
        return False
    w = seq[t - window:t] if window_mode == "easyrl4rec_slice" else seq[max(0, t - window):t]
    counts = Counter(itertools.chain.from_iterable(categories[v] for v in w))
    return any(counts[c] > threshold for c in categories[action])


class KuaiEnv:
    def __init__(self, config: EnvConfig | None = None):
        self.manifest_digest = _integrity.verify()
        self.config = config or EnvConfig.from_toml()
        small = load_small_matrix()
        users = np.sort(small["user_id"].unique())
        videos = np.sort(small["video_id"].unique())
        self._lookup = _build_oracle(small["user_id"].to_numpy(), small["video_id"].to_numpy(),
                                     small["watch_ratio"].to_numpy(), users, videos)
        del small
        self.users: tuple[int, ...] = tuple(int(u) for u in users)
        self.catalog: tuple[int, ...] = tuple(int(v) for v in videos)
        cats = load_item_categories()
        self.categories: Mapping[int, tuple[int, ...]] = {v: cats[v] for v in self.catalog}
        self._user_set = frozenset(self.users)
        self._catalog_set = frozenset(self.catalog)
        self._user: int | None = None
        self._seq: list[int] = []
        self._shown: set[int] = set()
        self._done = True

    def reset(self, user: int) -> dict:
        if user not in self._user_set:
            raise KeyError(f"user {user} is not in the small matrix")
        self._user, self._seq, self._shown, self._done = int(user), [], set(), False
        return {"user": self._user, "t": 0}

    def step(self, video: int) -> StepResult:
        if self._done:
            raise SessionClosedError("call reset(user) before step()")
        video = int(video)
        if video not in self._catalog_set:
            raise KeyError(f"video {video} is not in the small-matrix catalog")
        if video in self._shown:
            raise RepeatRecommendationError(f"video {video} already shown to user {self._user}")
        c = self.config
        t = len(self._seq)
        bored = boredom_quit(self._seq, video, self.categories, c.quit_window,
                             c.quit_threshold, c.window_mode)
        at_cap = t >= c.max_turn - 1
        self._seq.append(video)
        self._shown.add(video)
        value = self._lookup(self._user, video)
        observed = not np.isnan(value)
        reward = value if observed else float(c.missing_pair_reward)
        self._done = bored or at_cap
        reason = "boredom" if bored else ("max_turn" if at_cap else None)
        return StepResult(reward=reward, done=self._done, observed=observed, quit_reason=reason, t=t)

    @property
    def history(self) -> tuple[int, ...]:
        return tuple(self._seq)
