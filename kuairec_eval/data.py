"""Loaders for the six KuaiRec 2.0 tables.

Part of the frozen evaluator. The small matrix is the ground truth the
environment answers from, so `load_small_matrix` refuses to run when any frame
on the call stack belongs to the `policies` package (see ADR-0007).
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "raw" / "KuaiRec 2.0" / "data"
POLICIES_DIR = REPO_ROOT / "policies"

# Published figures from kuairec.com / the KuaiRec paper (CIKM'22).
PUBLISHED = {
    "small": {"users": 1411, "videos": 3327, "interactions": 4676570, "density": 0.996},
    "big": {"users": 7176, "videos": 10728, "interactions": 12530806, "density": 0.163},
}

MATRIX_DTYPES = {
    "user_id": "int32",
    "video_id": "int32",
    "play_duration": "int64",
    "video_duration": "int64",
    "time": "string",
    "date": "float64",  # small_matrix has missing dates, so it parses as float
    "timestamp": "float64",
    "watch_ratio": "float64",
}


class PolicyDataAccessError(RuntimeError):
    """Raised when code in the policies package tries to read the small matrix."""


def data_dir() -> Path:
    return Path(os.environ.get("KUAIREC_DATA_DIR", DEFAULT_DATA_DIR))


def _assert_not_called_from_policy() -> None:
    policies = str(POLICIES_DIR) + os.sep
    for frame in inspect.stack(context=0):
        if os.path.abspath(frame.filename).startswith(policies):
            raise PolicyDataAccessError(
                f"small matrix requested from policy code ({frame.filename}:{frame.lineno}); "
                "policies may only see rewards through Environment.step()"
            )


def _read_matrix(name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir() / name, dtype=MATRIX_DTYPES)


def load_small_matrix() -> pd.DataFrame:
    _assert_not_called_from_policy()
    return _read_matrix("small_matrix.csv")


def load_big_matrix() -> pd.DataFrame:
    return _read_matrix("big_matrix.csv")


def load_item_categories() -> pd.Series:
    """video_id -> tuple of category ids (1 to 4 tags per video)."""
    df = pd.read_csv(data_dir() / "item_categories.csv")
    feats = df["feat"].map(lambda s: tuple(int(x) for x in ast.literal_eval(s)))
    return pd.Series(feats.values, index=df["video_id"].astype("int32"), name="categories")


def load_item_daily_features() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "item_daily_features.csv", low_memory=False)


def load_user_features() -> pd.DataFrame:
    return pd.read_csv(data_dir() / "user_features.csv", low_memory=False)


def load_caption_category() -> pd.DataFrame:
    # Four captions contain a bare '\r'; the C parser treats it as a row break unless the
    # terminator is pinned to '\n'. With it: 10,728 rows, video_id unique 0..10727.
    return pd.read_csv(data_dir() / "kuairec_caption_category.csv", lineterminator="\n",
                       low_memory=False)


def matrix_stats(df: pd.DataFrame) -> dict:
    users = int(df["user_id"].nunique())
    videos = int(df["video_id"].nunique())
    rows = int(len(df))
    pairs = int(df[["user_id", "video_id"]].drop_duplicates().shape[0])
    return {
        "users": users,
        "videos": videos,
        "interactions": rows,
        "unique_pairs": pairs,
        "density_rows": rows / (users * videos),
        "density_unique_pairs": pairs / (users * videos),
        "watch_ratio_nan": int(df["watch_ratio"].isna().sum()),
        "watch_ratio_inf": int(np.isinf(df["watch_ratio"]).sum()),
    }
