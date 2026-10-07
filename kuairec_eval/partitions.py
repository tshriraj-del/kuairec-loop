"""Seeded 60/20/20 split of the 1,411 small-matrix users into search / shadow / holdout.

`load("holdout")` succeeds only when the running program is scripts/final_report.py
(ADR-0006). Everything else, including the harness and the tests, gets an error.
"""
from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path

import numpy as np

from kuairec_eval import _integrity
from kuairec_eval.data import REPO_ROOT

PARTITION_DIR = REPO_ROOT / "data" / "partitions"
CONFIG_PATH = REPO_ROOT / "config" / "evaluator.toml"
FINAL_REPORT_SCRIPT = (REPO_ROOT / "scripts" / "final_report.py").resolve()
NAMES = ("search", "shadow", "holdout")


class HoldoutAccessError(PermissionError):
    pass


def split(user_ids, seed: int, fractions: dict[str, float]) -> dict[str, list[int]]:
    """Sort, permute with numpy default_rng(seed), cut at round(n*0.6) and round(n*0.2)."""
    if abs(sum(fractions[n] for n in NAMES) - 1.0) > 1e-9:
        raise ValueError("fractions must sum to 1")
    users = np.sort(np.unique(np.asarray(user_ids, dtype=np.int64)))
    perm = np.random.default_rng(seed).permutation(users)
    n = len(users)
    n_search = int(round(n * fractions["search"]))
    n_shadow = int(round(n * fractions["shadow"]))
    parts = {
        "search": perm[:n_search],
        "shadow": perm[n_search:n_search + n_shadow],
        "holdout": perm[n_search + n_shadow:],
    }
    return {k: sorted(int(u) for u in v) for k, v in parts.items()}


def write(parts: dict[str, list[int]], seed: int, directory: Path = PARTITION_DIR) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        path = directory / f"{name}.json"
        if path.exists():
            raise FileExistsError(f"{path} exists; partitions are frozen and never rewritten")
        path.write_text(json.dumps({"name": name, "seed": seed, "n": len(parts[name]),
                                    "users": parts[name]}, indent=0) + "\n")


def _running_final_report() -> bool:
    main = sys.modules.get("__main__")
    main_file = getattr(main, "__file__", None)
    return main_file is not None and Path(main_file).resolve() == FINAL_REPORT_SCRIPT


def load(name: str) -> list[int]:
    if name not in NAMES:
        raise KeyError(f"unknown partition {name!r}; expected one of {NAMES}")
    if name == "holdout" and not _running_final_report():
        raise HoldoutAccessError("the holdout partition is readable only by scripts/final_report.py")
    _integrity.verify()
    return json.loads((PARTITION_DIR / f"{name}.json").read_text())["users"]


def config() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)["partition"]
