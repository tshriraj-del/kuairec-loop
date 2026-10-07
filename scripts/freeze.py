"""Create the user partitions (once) and write kuairec_eval/FROZEN_MANIFEST.json.

Usage:
  .venv/bin/python scripts/freeze.py            # first freeze
  .venv/bin/python scripts/freeze.py --refreeze # deliberate re-freeze after an evaluator change
Partitions are never rewritten once they exist.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kuairec_eval import _integrity, partitions  # noqa: E402
from kuairec_eval.data import REPO_ROOT, data_dir, load_small_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refreeze", action="store_true")
    args = ap.parse_args()
    if _integrity.MANIFEST_PATH.exists() and not args.refreeze:
        sys.exit(f"{_integrity.MANIFEST_PATH} exists; pass --refreeze to overwrite it deliberately")
    if not (partitions.PARTITION_DIR / "search.json").exists():
        cfg = partitions.config()
        users = load_small_matrix()["user_id"].unique()
        parts = partitions.split(users, cfg["seed"], {k: cfg[k] for k in partitions.NAMES})
        partitions.write(parts, cfg["seed"])
        print("partitions written:", {k: len(v) for k, v in parts.items()})
    manifest = _integrity.compute(REPO_ROOT, data_dir())
    _integrity.MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("manifest digest:", _integrity.manifest_digest(manifest))


if __name__ == "__main__":
    main()
