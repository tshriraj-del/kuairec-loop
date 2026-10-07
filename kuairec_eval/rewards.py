"""Reward candidates behind one config switch (config/rewards.toml [reward] active).

EVALUATOR-SIDE: builds (user x video) reward matrices from the small matrix, so policies
must not import it (tests/test_policy_isolation.py; data.load_small_matrix also refuses
policy callers). The public half of the duration-adjusted reward lives in labels.py.

  raw           recorded watch_ratio (what the frozen v0 environment returns)
  capped5       min(watch_ratio, cap), cap = 5 as in EasyRL4Rec's KuaiData.load_mat
  duration_pct  labels.DurationBuckets.percentile(video, play_duration of the small row),
                buckets and reference sets from the big matrix only (ADR-0016)

Not in the v0 manifest: the frozen v0 environment still returns raw watch_ratio. The
audit applies these through audit/reward_env.py; evaluator v1 would move the switch into
the environment (docs/proposals/evaluator-v1.patch).
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass

import numpy as np

from kuairec_eval import labels
from kuairec_eval.data import REPO_ROOT, load_small_matrix

REWARDS = ("raw", "capped5", "duration_pct")
CONFIG_PATH = REPO_ROOT / "config" / "rewards.toml"


@dataclass(frozen=True)
class RewardConfig:
    active: str
    cap: float
    duration_buckets: int

    def __post_init__(self):
        if self.active not in REWARDS:
            raise ValueError(f"reward must be one of {REWARDS}, got {self.active!r}")

    @classmethod
    def from_toml(cls, path=CONFIG_PATH, active: str | None = None) -> "RewardConfig":
        with open(path, "rb") as f:
            cfg = tomllib.load(f)["reward"]
        if active is not None:
            cfg = {**cfg, "active": active}
        return cls(**cfg)


def row_rewards(cfg: RewardConfig, video_ids, watch_ratio, play_duration) -> np.ndarray:
    """Reward for interaction rows under cfg.active."""
    if cfg.active == "raw":
        return np.asarray(watch_ratio, dtype=np.float64)
    if cfg.active == "capped5":
        return np.minimum(np.asarray(watch_ratio, dtype=np.float64), cfg.cap)
    return labels.duration_buckets(cfg.duration_buckets).percentile(video_ids, play_duration)


def reward_matrix(cfg: RewardConfig, users, videos) -> np.ndarray:
    """Dense (len(users) x len(videos)) float64 matrix; NaN where the pair is unrecorded."""
    small = load_small_matrix()
    users = np.asarray(users)
    videos = np.asarray(videos)
    ui = np.searchsorted(users, small["user_id"].to_numpy())
    vi = np.searchsorted(videos, small["video_id"].to_numpy())
    if not (np.array_equal(users[ui], small["user_id"].to_numpy()) and
            np.array_equal(videos[vi], small["video_id"].to_numpy())):
        raise ValueError("users/videos must be sorted and cover the small matrix")
    mat = np.full((len(users), len(videos)), np.nan)
    mat[ui, vi] = row_rewards(cfg, small["video_id"].to_numpy(), small["watch_ratio"].to_numpy(),
                              small["play_duration"].to_numpy())
    return mat
