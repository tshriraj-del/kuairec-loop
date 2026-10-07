"""Public labels computed from the big matrix only. Policies MAY import this module.

Everything here is a function of big_matrix.csv, so a policy can compute exactly the
label the evaluator will use for the duration-adjusted reward (ADR-0016):

  canonical duration  d(v) = median of video_duration over v's big-matrix rows
                      (video_duration varies across rows for 4,034 of 10,728 videos)
  duration buckets    K equal-frequency buckets over the canonical durations of the
                      10,728 big-matrix videos; inner edges are the j/K quantiles,
                      bucket(v) = number of inner edges <= d(v)
  reference sets      R_b = play_duration of every big-matrix row (repeat views included)
                      whose video falls in bucket b
  duration percentile pct(v, x) = (#{r in R_b : r < x} + 0.5 * #{r in R_b : r == x}) / |R_b|
                      with b = bucket(v); mid-rank, so ties at x = 0 do not pile up at 0
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from kuairec_eval.data import load_big_matrix


@lru_cache(maxsize=1)
def _big_columns() -> pd.DataFrame:
    return load_big_matrix()[["user_id", "video_id", "video_duration", "play_duration", "watch_ratio"]]


@lru_cache(maxsize=1)
def canonical_durations() -> pd.Series:
    """video_id -> median video_duration (ms) over the video's big-matrix rows."""
    return _big_columns().groupby("video_id").video_duration.median().rename("duration_ms")


@dataclass(frozen=True)
class DurationBuckets:
    inner_edges: np.ndarray            # K-1 ascending duration thresholds in ms
    video_ids: np.ndarray              # big-matrix videos, ascending
    video_bucket: np.ndarray           # bucket index per entry of video_ids
    reference: tuple[np.ndarray, ...]  # sorted big-matrix play_duration per bucket

    @property
    def n_buckets(self) -> int:
        return len(self.reference)

    def bucket_of(self, videos) -> np.ndarray:
        idx = np.searchsorted(self.video_ids, np.asarray(videos))
        return self.video_bucket[idx]

    def percentile(self, videos, play_ms) -> np.ndarray:
        videos = np.asarray(videos)
        play = np.asarray(play_ms, dtype=np.float64)
        buckets = self.bucket_of(videos)
        out = np.empty(len(play), dtype=np.float64)
        for b in np.unique(buckets):
            m = buckets == b
            ref = self.reference[b]
            lo = np.searchsorted(ref, play[m], side="left")
            hi = np.searchsorted(ref, play[m], side="right")
            out[m] = (lo + hi) / (2.0 * len(ref))
        return out


@lru_cache(maxsize=8)
def duration_buckets(n_buckets: int) -> DurationBuckets:
    big = _big_columns()
    dur = canonical_durations()
    qs = np.arange(1, n_buckets) / n_buckets
    inner = np.unique(np.quantile(dur.to_numpy(), qs))
    video_ids = dur.index.to_numpy()
    video_bucket = np.searchsorted(inner, dur.to_numpy(), side="right")
    row_bucket = video_bucket[np.searchsorted(video_ids, big["video_id"].to_numpy())]
    play = big["play_duration"].to_numpy()
    reference = tuple(np.sort(play[row_bucket == b]) for b in range(len(inner) + 1))
    return DurationBuckets(inner_edges=inner, video_ids=video_ids, video_bucket=video_bucket,
                           reference=reference)


def big_matrix_label(name: str, n_buckets: int | None = None, cap: float = 5.0) -> pd.Series:
    """Per big-matrix row label for reward candidate `name` (what a policy can train on)."""
    big = _big_columns()
    if name == "raw":
        values = big["watch_ratio"].to_numpy()
    elif name == "capped5":
        values = np.minimum(big["watch_ratio"].to_numpy(), cap)
    elif name == "duration_pct":
        values = duration_buckets(n_buckets).percentile(big["video_id"].to_numpy(), big["play_duration"].to_numpy())
    else:
        raise KeyError(name)
    return pd.Series(values, index=big.index, name=name)


def big_matrix_video_stats(name: str, n_buckets: int | None = None, cap: float = 5.0) -> pd.DataFrame:
    """video_id -> (mean label, row count) over the big matrix."""
    big = _big_columns()
    label = big_matrix_label(name, n_buckets, cap)
    return pd.DataFrame({"video_id": big["video_id"], "y": label}).groupby("video_id").y.agg(["mean", "size"])
