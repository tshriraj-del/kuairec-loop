"""First-view (user, video) pairs from the big matrix with the public duration-adjusted label.

Each pair keeps the FIRST big-matrix view of that user and video (ADR-0030): its timestamp and
its label = kuairec_eval.labels.DurationBuckets.percentile(video, play_duration of that view).
Repeat views are dropped, so a training row is one exposure, like one environment step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from kuairec_eval import labels
from kuairec_eval.data import data_dir
from policies.phase1.config import CACHE, load

PAIRS_PATH = CACHE / "pairs.npz"


@dataclass
class Pairs:
    user: np.ndarray    # int32
    video: np.ndarray   # int32
    ts: np.ndarray      # float64, first-view timestamp
    label: np.ndarray   # float32, duration-adjusted label of the first view
    n_views: np.ndarray  # int32

    def __len__(self) -> int:
        return len(self.user)

    def subset(self, mask: np.ndarray) -> "Pairs":
        return Pairs(self.user[mask], self.video[mask], self.ts[mask], self.label[mask], self.n_views[mask])


def build_pairs() -> tuple[Pairs, dict]:
    """Read the big matrix, label every row, keep the first view per pair. Returns the pairs and a report."""
    k = load()["label"]["duration_buckets"]
    raw = pd.read_csv(data_dir() / "big_matrix.csv", usecols=["user_id", "video_id", "timestamp", "play_duration"],
                      dtype={"user_id": "int32", "video_id": "int32", "timestamp": "float64", "play_duration": "int64"})
    n_rows = len(raw)
    label = labels.duration_buckets(k).percentile(raw["video_id"].to_numpy(), raw["play_duration"].to_numpy())
    ok = np.isfinite(label) & raw["timestamp"].notna().to_numpy()
    report = {"big_rows": n_rows, "rows_label_finite": int(np.isfinite(label).sum()),
              "rows_without_timestamp": int(raw["timestamp"].isna().sum()), "rows_dropped_unusable": int((~ok).sum())}
    raw = raw.loc[ok].assign(label=label[ok].astype(np.float32))
    raw = raw.sort_values(["user_id", "video_id", "timestamp"], kind="stable")
    first = ~raw.duplicated(["user_id", "video_id"]).to_numpy()
    views = raw.groupby(["user_id", "video_id"], sort=True).size().to_numpy().astype(np.int32)
    kept = raw.loc[first]
    pairs = Pairs(kept["user_id"].to_numpy(np.int32), kept["video_id"].to_numpy(np.int32),
                  kept["timestamp"].to_numpy(np.float64), kept["label"].to_numpy(np.float32), views)
    report.update({"pairs": len(pairs), "repeat_view_rows_dropped": int((~first).sum()),
                   "label_coverage_of_usable_rows": float(ok.mean())})
    return pairs, report


def save(pairs: Pairs) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(PAIRS_PATH, user=pairs.user, video=pairs.video, ts=pairs.ts, label=pairs.label, n_views=pairs.n_views)


def load_pairs() -> Pairs:
    z = np.load(PAIRS_PATH)
    return Pairs(z["user"], z["video"], z["ts"], z["label"], z["n_views"])


def time_cutoffs(pairs: Pairs) -> dict:
    """First-view timestamp cutoffs for T_fit / T_rank / T_val (config [split])."""
    s = load()["split"]
    return {"fit_end": float(np.quantile(pairs.ts, s["fit_end"])), "rank_end": float(np.quantile(pairs.ts, s["rank_end"]))}


class History:
    """Per-user pair lists (CSR over users), optionally filtered."""

    def __init__(self, pairs: Pairs):
        order = np.lexsort((pairs.video, pairs.user))
        self.user = pairs.user[order]
        self.video = pairs.video[order]
        self.label = pairs.label[order]
        self.users, starts = np.unique(self.user, return_index=True)
        self.bounds = dict(zip(self.users.tolist(), zip(starts.tolist(), np.r_[starts[1:], len(self.user)].tolist())))

    def get(self, user: int) -> tuple[np.ndarray, np.ndarray]:
        b = self.bounds.get(int(user))
        if b is None:
            return np.empty(0, np.int32), np.empty(0, np.float32)
        return self.video[b[0]:b[1]], self.label[b[0]:b[1]]
