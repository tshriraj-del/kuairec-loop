"""Ranker features (ADR-0034). Three nested sets:

  no_duration  user features, history stats, video statistics that do not involve video
               duration, category ids, the video's label prior, category affinities and the
               scores of (a) affinity, (b) item-kNN, (c) EASE
  duration     + canonical duration, its bucket, and every field the KuaiRec documentation
               defines through video_duration (complete / valid / long-time / short-time plays,
               play progress, total play time)
  text         + caption-embedding features (16 PCA components, two user-video cosines, has_text)

Video statistics come from item_daily_features rows dated before `cutoff_date` (training and
validation) or from every row (serving), as per-day means and per-show rates so the two
windows stay comparable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scipy.stats import rankdata

from kuairec_eval import labels
from kuairec_eval.data import load_caption_category, load_item_daily_features, load_user_features
from policies.phase1.config import CACHE, load
from policies.phase1.scorers import N_VIDEOS

DURATION_DERIVED = ["video_duration", "play_duration", "complete_play_cnt", "complete_play_user_num",
                    "valid_play_cnt", "valid_play_user_num", "long_time_play_cnt", "long_time_play_user_num",
                    "short_time_play_cnt", "short_time_play_user_num", "play_progress"]
COUNT_FIELDS = ["show_cnt", "show_user_num", "play_cnt", "play_user_num", "like_cnt", "like_user_num",
                "click_like_cnt", "double_click_cnt", "cancel_like_cnt", "cancel_like_user_num", "comment_cnt",
                "comment_user_num", "direct_comment_cnt", "reply_comment_cnt", "delete_comment_cnt",
                "delete_comment_user_num", "comment_like_cnt", "comment_like_user_num", "follow_cnt",
                "follow_user_num", "cancel_follow_cnt", "cancel_follow_user_num", "share_cnt", "share_user_num",
                "download_cnt", "download_user_num", "report_cnt", "report_user_num", "reduce_similar_cnt",
                "reduce_similar_user_num", "collect_cnt", "collect_user_num", "cancel_collect_cnt",
                "cancel_collect_user_num", "comment_stay_duration"]
RATE_FIELDS = ["play_cnt", "like_cnt", "comment_cnt", "follow_cnt", "share_cnt", "download_cnt", "report_cnt",
               "reduce_similar_cnt", "collect_cnt", "cancel_like_cnt"]
STATIC_CODED = ["video_type", "upload_type", "visible_status", "video_tag_id"]
USER_CODED = ["user_active_degree"] + [f"onehot_feat{i}" for i in range(18)]
USER_NUMERIC = ["is_lowactive_period", "is_live_streamer", "is_video_author", "follow_user_num", "fans_user_num",
                "friend_user_num", "register_days"]
# Scale-free on purpose: training rows use scorers fitted on T_fit, serving uses all pairs, so raw
# score magnitudes and history sizes shift between the two (ADR-0034).
ROW_BASE = ["aff_mean", "aff_max", "aff_support_share", "cat_prior", "knn_rank", "ease_rank", "hist_mean_label"]
ROW_TEXT = ["cos_taste", "cos_mean"]
VARIANTS = ("no_duration", "duration", "text")


def _codes(series: pd.Series) -> np.ndarray:
    return pd.Categorical(series.astype("string").fillna("NA")).codes.astype(np.float32)


def video_table(cutoff_date: int | None, variant: str) -> pd.DataFrame:
    """One row per video id 0..10727."""
    daily = load_item_daily_features()
    if cutoff_date is not None:
        daily = daily[daily["date"] < cutoff_date]
    g = daily.groupby("video_id")
    out = pd.DataFrame(index=pd.RangeIndex(N_VIDEOS, name="video_id"))
    out["n_days"] = g.size()
    sums = g[COUNT_FIELDS + DURATION_DERIVED[1:-1]].sum()
    for f in COUNT_FIELDS:
        out[f"perday_{f}"] = sums[f] / out["n_days"]
    show = sums["show_cnt"].replace(0, np.nan)
    for f in RATE_FIELDS:
        out[f"rate_{f}"] = sums[f] / show
    out["comment_stay_per_play"] = sums["comment_stay_duration"] / sums["play_cnt"].replace(0, np.nan)
    last = daily.sort_values("date").groupby("video_id").last()
    for f in STATIC_CODED:
        out[f"code_{f}"] = pd.Series(_codes(last[f]), index=last.index)
    out["video_width"] = last["video_width"]
    out["video_height"] = last["video_height"]
    ref = pd.Timestamp(str(cutoff_date)) if cutoff_date is not None else pd.Timestamp(str(int(daily["date"].max()))) + pd.Timedelta(days=1)
    out["age_days"] = (ref - pd.to_datetime(last["upload_dt"], errors="coerce")).dt.days
    cc = load_caption_category().set_index("video_id")
    for lvl in ("first", "second", "third"):
        out[f"cat_{lvl}_level"] = cc[f"{lvl}_level_category_id"].astype(np.float32)
    if variant in ("duration", "text"):
        dur, bucket = durations()
        out["log_duration"] = np.log(dur)
        out["duration_bucket"] = bucket.astype(np.float32)
        plays = sums["play_cnt"].replace(0, np.nan)
        for f in ("complete_play_cnt", "valid_play_cnt", "long_time_play_cnt", "short_time_play_cnt", "play_duration"):
            out[f"perplay_{f}"] = sums[f] / plays
        out["mean_play_progress"] = g["play_progress"].mean()
    return out.astype(np.float32)


def durations() -> tuple[np.ndarray, np.ndarray]:
    """Canonical duration (ms) and duration bucket per video id, from kuairec_eval.labels, cached
    once because the label module loads the whole big matrix to compute them."""
    path = CACHE / "durations.npz"
    if not path.exists():
        k = load()["label"]["duration_buckets"]
        ids = np.arange(N_VIDEOS)
        dur = labels.canonical_durations().reindex(ids).to_numpy(np.float64)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez(path, duration_ms=dur, bucket=labels.duration_buckets(k).bucket_of(ids))
    z = np.load(path)
    return z["duration_ms"], z["bucket"]


def user_table() -> pd.DataFrame:
    u = load_user_features().set_index("user_id").sort_index()
    out = pd.DataFrame(index=u.index)
    for f in USER_CODED:
        out[f"u_{f}"] = _codes(u[f])
    for f in USER_NUMERIC:
        out[f"u_{f}"] = u[f].astype(np.float32)
    return out.astype(np.float32)


class Text:
    """Caption embeddings (scripts/phase1_embed.py) reduced by PCA, plus user-video cosines."""

    def __init__(self):
        z = np.load(CACHE / "caption_emb.npz")
        self.emb, self.has = z["emb"], z["has_text"]
        dims = load()["text"]["pca_dims"]
        x = self.emb[self.has]
        self.mean = x.mean(0)
        _, _, vt = np.linalg.svd(x - self.mean, full_matrices=False)
        self.components = vt[:dims]
        self.pca = np.zeros((len(self.emb), dims), np.float32)
        self.pca[self.has] = (x - self.mean) @ self.components.T

    def video_block(self) -> pd.DataFrame:
        cols = {f"text_pc{i}": self.pca[:, i] for i in range(self.pca.shape[1])}
        cols["has_text"] = self.has.astype(np.float32)
        return pd.DataFrame(cols, index=pd.RangeIndex(N_VIDEOS, name="video_id"))

    def user_cosines(self, hist_videos, hist_labels, targets) -> tuple[np.ndarray, np.ndarray]:
        hv = np.asarray(hist_videos)
        keep = self.has[hv]
        if not keep.any():
            z = np.zeros(len(targets), np.float32)
            return z, z
        e = self.emb[hv[keep]]
        w = np.asarray(hist_labels, np.float32)[keep] - 0.5
        taste = (w[:, None] * e).sum(0) / (np.abs(w).sum() + 1e-6)
        mean = e.mean(0)
        t = self.emb[np.asarray(targets)]

        def cos(vec):
            n = np.linalg.norm(vec)
            return (t @ vec / n).astype(np.float32) if n > 0 else np.zeros(len(t), np.float32)
        return cos(taste), cos(mean)


DURATION_COLUMNS = ["log_duration", "duration_bucket", "perplay_complete_play_cnt", "perplay_valid_play_cnt",
                    "perplay_long_time_play_cnt", "perplay_short_time_play_cnt", "perplay_play_duration",
                    "mean_play_progress"]


def columns(variant: str, vtab: pd.DataFrame, utab: pd.DataFrame, text: Text | None) -> list[str]:
    """Feature columns of `variant`, in a fixed order; nested: no_duration < duration < text."""
    if variant not in VARIANTS:
        raise KeyError(variant)
    video = [c for c in vtab.columns if variant != "no_duration" or c not in DURATION_COLUMNS]
    cols = list(utab.columns) + video + ["video_prior_mean", "video_prior_n_rank"] + ROW_BASE
    if variant == "text":
        cols += list(text.video_block().columns) + ROW_TEXT
    return cols


class RowBuilder:
    """Builds ranker feature rows for (user, video) pairs from fitted base scorers."""

    def __init__(self, variant: str, cutoff_date: int | None, base: dict, prior: pd.DataFrame, catalog: np.ndarray):
        self.variant = variant
        self.vtab = video_table(cutoff_date, variant)
        self.utab = user_table()
        self.text = Text() if variant == "text" else None
        vt = self.vtab.join(prior)
        if self.text is not None:
            vt = vt.join(self.text.video_block())
        self.vblock = vt
        self.base = base
        self.catalog = np.asarray(catalog)
        self.pos = np.full(N_VIDEOS, -1)
        self.pos[self.catalog] = np.arange(len(self.catalog))
        self.cols = columns(variant, self.vtab, self.utab, self.text)

    def user_rows(self, user: int, hist_videos, hist_labels, videos) -> np.ndarray:
        """Feature matrix for one user and the given catalog videos (rows in `videos` order)."""
        videos = np.asarray(videos)
        aff = self.base["affinity"]
        a, n_uc = aff.user_affinity(hist_videos, hist_labels)
        tags = aff.cat[videos]
        n_tags = tags.sum(1)
        n_hist = max(len(hist_videos), 1)
        row = {
            "aff_mean": (tags @ a) / n_tags,
            "aff_max": np.where(tags > 0, a[None, :], -np.inf).max(1),
            "aff_support_share": (tags @ n_uc) / n_hist,
            "cat_prior": (tags @ aff.global_mean) / n_tags,
            "knn_rank": catalog_rank(self.base["item_knn"].score(hist_videos, hist_labels))[self.pos[videos]],
            "ease_rank": catalog_rank(self.base["ease"].score(hist_videos, hist_labels))[self.pos[videos]],
            "hist_mean_label": np.full(len(videos), float(np.mean(hist_labels)) if len(hist_labels) else 0.5, np.float32),
        }
        if self.text is not None:
            row["cos_taste"], row["cos_mean"] = self.text.user_cosines(hist_videos, hist_labels, videos)
        dyn = pd.DataFrame(row, index=videos)
        u = self.utab.reindex([int(user)]).to_numpy(np.float32)
        ublock = np.repeat(u, len(videos), axis=0)
        vblock = self.vblock.reindex(videos)
        frame = pd.concat([pd.DataFrame(ublock, columns=self.utab.columns, index=videos), vblock, dyn], axis=1)
        return frame[self.cols].to_numpy(np.float32)


def prior_table(pairs) -> pd.DataFrame:
    """The video's mean label and log pair count over the fitting pairs (a public, non-personal prior)."""
    df = pd.DataFrame({"video": pairs.video, "label": pairs.label})
    g = df.groupby("video")["label"]
    out = pd.DataFrame(index=pd.RangeIndex(N_VIDEOS, name="video_id"))
    out["video_prior_mean"] = g.mean()
    n = g.size().reindex(out.index, fill_value=0)
    out["video_prior_n_rank"] = (rankdata(n.to_numpy(), method="average") - 1) / (len(n) - 1)
    return out.astype(np.float32)


def catalog_rank(scores: np.ndarray) -> np.ndarray:
    """Percentile rank of each score within one user's catalog vector, ties averaged, in [0, 1]."""
    return ((rankdata(scores, method="average") - 1) / (len(scores) - 1)).astype(np.float32)
