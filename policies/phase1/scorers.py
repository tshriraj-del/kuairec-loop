"""Scorers (a) user-category affinity, (b) item-kNN and (c) EASE, written by hand (ADR-0031..0033).

Every scorer is fitted on a Pairs table and scores the catalog from one user's history:
    score(hist_videos, hist_labels) -> float32 vector aligned with `catalog`
Video ids index columns directly (big-matrix ids are 0..10727). Reused external functions are
limited to numpy and scipy.sparse primitives (docs/phase1/EXTERNAL_FUNCTIONS.md).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from policies.phase1.pairs import Pairs

N_VIDEOS = 10728


def category_matrix(categories) -> np.ndarray:
    """(N_VIDEOS x n_categories) multi-hot float32 from {video_id: tuple of category ids}."""
    n_cat = 1 + max(c for tags in categories.values() for c in tags)
    m = np.zeros((N_VIDEOS, n_cat), dtype=np.float32)
    for v, tags in categories.items():
        m[v, list(tags)] = 1.0
    return m


def user_item(pairs: Pairs, value: str, n_users: int) -> sp.csr_matrix:
    data = np.ones(len(pairs), np.float32) if value == "binary" else (pairs.label - 0.5).astype(np.float32)
    return sp.csr_matrix((data, (pairs.user, pairs.video)), shape=(n_users, N_VIDEOS))


class Affinity:
    """(a) aff(u, c) = (sum of u's labels in c + lambda * m_c) / (n_uc + lambda), m_c the global mean
    label of category c; a video scores the mean affinity over its categories."""
    name = "affinity"

    def __init__(self, shrinkage: float):
        self.shrinkage = float(shrinkage)

    def fit(self, pairs: Pairs, cat_matrix: np.ndarray, catalog: np.ndarray) -> "Affinity":
        self.cat = cat_matrix
        tags = cat_matrix[pairs.video]
        self.global_mean = (tags.T @ pairs.label.astype(np.float64)) / np.maximum(tags.sum(0), 1)
        self.catalog = np.asarray(catalog)
        self.cat_catalog = cat_matrix[self.catalog]
        self.n_tags = self.cat_catalog.sum(1)
        return self

    def user_affinity(self, hist_videos, hist_labels) -> tuple[np.ndarray, np.ndarray]:
        tags = self.cat[hist_videos]
        n = tags.sum(0)
        s = tags.T @ np.asarray(hist_labels, np.float64)
        return (s + self.shrinkage * self.global_mean) / (n + self.shrinkage), n

    def score(self, hist_videos, hist_labels) -> np.ndarray:
        aff, _ = self.user_affinity(hist_videos, hist_labels)
        return ((self.cat_catalog @ aff) / self.n_tags).astype(np.float32)


class ItemKNN:
    """(b) cosine similarity on binary co-watching, S[i, j] = co(i, j) / (sqrt(n_i n_j) + shrink),
    between catalog item i and any item j. A user's score for i uses the k items of the user's
    history most similar to i:
        binary:   sum of those similarities
        centered: sum s_ij (y_uj - 0.5) / (sum |s_ij| + 1e-6)
    """
    name = "item_knn"

    def __init__(self, value: str, k: int, shrink: float):
        self.value, self.k, self.shrink = value, int(k), float(shrink)

    def fit(self, pairs: Pairs, cat_matrix: np.ndarray, catalog: np.ndarray) -> "ItemKNN":
        self.catalog = np.asarray(catalog)
        x = user_item(pairs, "binary", int(pairs.user.max()) + 1)
        n = np.asarray(x.sum(0)).ravel()
        co = (x[:, self.catalog].T @ x).toarray().astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore"):
            sim = co / (np.sqrt(np.outer(n[self.catalog], n)).astype(np.float32) + self.shrink)
        sim = np.nan_to_num(sim, nan=0.0, posinf=0.0)  # only possible for a never-watched video with shrink 0
        sim[np.arange(len(self.catalog)), self.catalog] = 0.0
        self.sim = sim
        return self

    def score(self, hist_videos, hist_labels) -> np.ndarray:
        hist_videos = np.asarray(hist_videos)
        if len(hist_videos) == 0:
            return np.zeros(len(self.catalog), np.float32)
        s = self.sim[:, hist_videos]
        if s.shape[1] > self.k:
            idx = np.argpartition(-s, self.k - 1, axis=1)[:, :self.k]
            s = np.take_along_axis(s, idx, axis=1)
            w = (np.asarray(hist_labels, np.float32) - 0.5)[idx]
        else:
            w = np.broadcast_to(np.asarray(hist_labels, np.float32) - 0.5, s.shape)
        if self.value == "binary":
            return s.sum(1).astype(np.float32)
        return ((s * w).sum(1) / (np.abs(s).sum(1) + 1e-6)).astype(np.float32)


class EASE:
    """(c) Steck 2019: P = (X'X + l2 I)^-1, B = I - P diag(1/diag(P)), diag(B) = 0. X is binary or the
    centered label (y - 0.5) on observed pairs. Only the catalog columns of B are kept."""
    name = "ease"

    def __init__(self, value: str, l2: float):
        self.value, self.l2 = value, float(l2)

    def fit(self, pairs: Pairs, cat_matrix: np.ndarray, catalog: np.ndarray) -> "EASE":
        self.catalog = np.asarray(catalog)
        x = user_item(pairs, self.value, int(pairs.user.max()) + 1)
        g = (x.T @ x).toarray().astype(np.float64)
        g[np.diag_indices_from(g)] += self.l2
        p = np.linalg.inv(g)
        del g
        b = -p[:, self.catalog] / np.diag(p)[self.catalog]
        b[self.catalog, np.arange(len(self.catalog))] = 0.0
        self.b = b.astype(np.float32)
        return self

    def score(self, hist_videos, hist_labels) -> np.ndarray:
        hist_videos = np.asarray(hist_videos)
        if len(hist_videos) == 0:
            return np.zeros(len(self.catalog), np.float32)
        x = np.ones(len(hist_videos), np.float32) if self.value == "binary" else np.asarray(hist_labels, np.float32) - 0.5
        return (x @ self.b[hist_videos]).astype(np.float32)


def ndcg_at_k(scores: np.ndarray, gains: np.ndarray, items: np.ndarray, k: int = 10) -> float:
    """NDCG@k of `scores` against graded `gains` over the same `items`; ties broken by lower item id."""
    order = np.lexsort((items, -scores))[:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gains[order] * disc[:len(order)]).sum())
    ideal = np.sort(gains)[::-1][:k]
    idcg = float((ideal * disc[:len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else 0.0
