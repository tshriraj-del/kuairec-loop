"""Phase 1 configuration and cache locations (config/phase1.toml, data/cache/phase1/)."""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "phase1.toml"
CACHE = ROOT / "data" / "cache" / "phase1"


def load() -> dict:
    with open(CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


def config_hash(obj) -> str:
    """sha256 of the canonical JSON of a config object (first 16 hex chars)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()[:16]
