"""Frozen v0 KuaiEnv dynamics with the reward swapped for a candidate (ADR-0026).

Quit rule, repeat ban, max_turn and unrecorded-pair handling all come from the frozen
environment, unchanged: RewardEnv calls KuaiEnv.step and replaces only `reward`, and only
for recorded pairs (unrecorded pairs keep missing_pair_reward = 0, as in v0). Under
"raw" every replaced value is asserted equal to the frozen environment's own reward.
The candidate matrix lives in a closure, never on the instance (same rule as ADR-0007).
"""
from __future__ import annotations

import dataclasses

import numpy as np

from kuairec_eval.env import KuaiEnv, StepResult
from kuairec_eval.rewards import RewardConfig, reward_matrix


class RewardEnv:
    def __init__(self, base: KuaiEnv, cfg: RewardConfig, matrix: np.ndarray | None = None):
        mat = reward_matrix(cfg, base.users, base.catalog) if matrix is None else matrix
        uidx = {u: i for i, u in enumerate(base.users)}
        vidx = {v: j for j, v in enumerate(base.catalog)}

        def lookup(user: int, video: int) -> float:
            return float(mat[uidx[user], vidx[video]])

        self._lookup = lookup
        self._base = base
        self._user: int | None = None
        self.reward_config = cfg
        self.users, self.catalog, self.categories = base.users, base.catalog, base.categories
        self.config, self.manifest_digest = base.config, base.manifest_digest

    def reset(self, user: int) -> dict:
        obs = self._base.reset(user)
        self._user = int(user)
        return obs

    def step(self, video: int) -> StepResult:
        r = self._base.step(video)
        if not r.observed:
            return r
        value = self._lookup(self._user, int(video))
        if self.reward_config.active == "raw" and value != r.reward:
            raise AssertionError(f"raw candidate {value} != frozen env reward {r.reward}")
        return dataclasses.replace(r, reward=value)

    @property
    def history(self) -> tuple[int, ...]:
        return self._base.history
